from collections import defaultdict
import csv
import time
import rclpy
from rclpy.task import Future
from std_msgs.msg import Empty
from threading import Event

from choirbot_interfaces.srv import TaskCompletionService
from ...optimizer.cbba_optimizer import CBBAOptimizer
from .. import RobotData
from .executor import TaskExecutor
from ..guidance import OptimizationGuidance
from ..optimization_thread import OptimizationThread
from ...utils import OrEvent

class CBBAGuidance(OptimizationGuidance):
    """Guidance class for CBBA optimization.
    This class is responsible for managing the optimization process,
    including triggering the optimization, executing tasks, and
    synchronizing with other agents.
    It inherits from the OptimizationGuidance class and uses the
    CBBAOptimizer for the optimization process.

    Args:
        OptimizationGuidance (Guidance): Base class for Optimization Guidance.
    """

    def __init__(self, optimizer: CBBAOptimizer, executor: TaskExecutor,
                 data: RobotData, pose_handler: str = None, pose_topic: str = None):
        super().__init__(optimizer, CBBAOptimizationThread, pose_handler, pose_topic)
        self.data = data
        self.task_executor = executor
        self.current_task = None
        self.task_queue = []
        self.completed_tasks = []

        self._local_optimization_complete = False
        self._neighbor_optimization_status = defaultdict(lambda: False)
        self._sync_timeout = 10.0
        self._sync_check_interval = 0.5
        self.start_time = None
        self.end_time = None
        self.log_data = dict()
        self.log_task_completion_time = dict()

        # triggering mechanism to start optimization
        self.opt_trigger_subscription = self.create_subscription(
            Empty, '/optimization_trigger', self.start_optimization, 10)

        # task list and task completion services
        self.task_list_client = self.create_client(
            executor.service, '/task_list')
        self.task_completion_client = self.create_client(
            TaskCompletionService, '/task_completion')

        # guard condition to start a new task
        self.task_gc = self.create_guard_condition(self.start_new_task)

        # initialize task executor
        self.task_executor.initialize(self)

        # wait for services
        self.task_list_client.wait_for_service()
        self.task_completion_client.wait_for_service()

        self.get_logger().info('Guidance {} started'.format(self.agent_id))

    def start_optimization(self, _):
        """Start the optimization process."""

        self.get_logger().info('Optimization triggered: requesting task list')
        self.start_time = self.get_clock().now()
        self._local_optimization_complete = False
        self._neighbor_optimization_status = defaultdict(lambda: False)

        # remove all enqueued tasks
        self.task_queue = []

        # # request updated task list
        request = self.task_executor.service.Request(agent_id=self.agent_id)
        future = self.task_list_client.call_async(request)

        self.optimization_thread.optimize(future)

    def _optimization_ended(self):
        """
        Callback for when the optimization ends.

        This method is called when the optimization process is complete.
        It logs the results and starts a new task.
        """

        self.end_time = self.get_clock().now()
        self.get_logger().warn(
            f"************* AGENT {self.agent_id} Entered _Optimization ended *************")
        self.get_logger().info('Optimization ended')

        # collect results
        result = self.optimizer.get_result()
        self.get_logger().info(
            f"AGENT {self.agent_id} Optimization result: {[t.id for t in result]}")
        self.log_data['agent_id'] = self.agent_id
        self.log_data['optimization_time'] = (
            self.end_time - self.start_time).nanoseconds / 1e9
        self.log_data['iterations'] = self.optimizer.get_iterations()
        self.log_data['tasks'] = [t.seq_num for t in result]
        self.task_queue = result
        self.get_logger().info(
            f"AGENT {self.agent_id} task queue: {[t.seq_num for t in self.task_queue]}")

        # log to csv for visualization
        with open(f'optimization_log_agents_{len(self.in_neighbors) + 1}{time.strftime("%m %d", time.localtime())}.csv', 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.log_data.keys())
            writer.writeheader()
            writer.writerow(self.log_data)

        # start new task
        self.task_gc.trigger()

    def start_new_task(self):
        """Start a new task if available."""

        # stop if there are no new tasks
        self.get_logger().warn(
            f"********AGENT {self.agent_id} Entered start_new_task******")
        if not self.task_queue:
            self.get_logger().warn(
                f"********AGENT {self.agent_id} start_new_task: Queue is EMPTY.******")
            return

        # get task
        task = self.task_queue.pop(0)

        # check if task was already performed
        if task.seq_num in self.completed_tasks:
            # trigger new task and exit
            self.task_gc.trigger()
            return

        # execute task
        self.get_logger().info(
            'Got new task (seq_num {}) - starting execution'.format(task.seq_num))
        self.current_task = task
        self.task_executor.execute_async(self.current_task, self.task_ended)

    def task_ended(self):
        """
        Callback for when a task ends.

        This method is called when a task execution is complete.
        It logs the results and notifies the task completion service.
        It also starts a new task if available.
        """
        # log to console
        self.get_logger().info('Task completed (seq_num {})'.format(self.current_task.seq_num))

        # log to csv for visualization
        self.task_completion_time = self.get_clock().now() - self.start_time
        self.log_task_completion_time[self.current_task.id] = self.task_completion_time.nanoseconds / 1e9
        with open(f'task_completion_log_agents_{len(self.in_neighbors)+1}_{time.strftime("%m %d", time.localtime())}.csv', 'a', newline='') as f:
            writer = csv.DictWriter(
                f, fieldnames=['agent_id', 'seq_num', 'completion_time'])
            writer.writerow({'agent_id': self.agent_id, 'seq_num': self.current_task.seq_num,
                            'completion_time': self.task_completion_time.nanoseconds / 1e9})

        # add task to list of completed tasks
        self.completed_tasks.append(self.current_task.id)
        # notify table that task execution has completed
        request = TaskCompletionService.Request()
        request.agent_id = self.agent_id
        request.task_seq_num = self.current_task.id
        self.task_completion_client.call_async(request)

        # start a new task
        self.current_task = None
        self.task_gc.trigger()

    def wait_for_all_optimizers(self):
        """
        Wait for all optimizers to complete their optimization process.

        This method checks the status of all neighbors and waits for them
        to complete their optimization before proceeding.
        It also handles synchronization timeouts and logs the status of
        the neighbors.

        Returns:
            bool: True if all neighbors completed optimization, False otherwise.
        """

        self._local_optimization_complete = True
        start_time = self.get_clock().now()

        # Checks neighbors' optimization status
        while rclpy.ok() and not (self.optimization_thread._halt_event and self.optimization_thread._halt_event.is_set()):
            all_neighbors_complete = True
            if not self.in_neighbors:
                self.get_logger().info(
                    f"AGENT {self.agent_id} No neighbors, skipping synchronization")
                return True

            for neighbor_id in self.in_neighbors:
                if not self._neighbor_optimization_status[neighbor_id]:
                    all_neighbors_complete = False
                    self.get_logger().info(
                        f"AGENT {self.agent_id} Waiting for neighbor {neighbor_id} to complete optimization")
                    break

            if all_neighbors_complete:
                self.get_logger().info(
                    f"AGENT {self.agent_id} All neighbors completed optimization")
                return True

            # check if timeout has been reached
            elapsed_time = (self.get_clock().now() -
                            start_time).nanoseconds / 1e9
            if elapsed_time > self._sync_timeout:
                self.get_logger().warn(
                    f"AGENT {self.agent_id} Synchronization timeout reached")
                missing_neighbors = [
                    n for n in self.in_neighbors if not self._neighbor_optimization_status[n]]
                self.get_logger().warn(
                    f"AGENT {self.agent_id} Missing neighbors: {missing_neighbors}")
                return False

            status_message = {
                'agent_id': self.agent_id,
                'status': 'opt_complete' if self._local_optimization_complete else 'opt_in_progress'
            }

            self.get_logger().debug(f"AGENT {self.agent_id} SENDING STATUS")

            # exchange status with neighbors and update completion status for each
            try:
                if not hasattr(self, 'communicator'):
                    self.get_logger().error(
                        f"AGENT {self.agent_id} No communicator; Cant Sync")
                    time.sleep(self._sync_check_interval)
                    continue

                responses = self.communicator.neighbors_exchange(
                    status_message, self.in_neighbors, self.out_neighbors, False, self.optimization_thread._halt_event)

                for neighbor_id, response in responses.items():
                    if response and isinstance(response, dict) and response.get('status') == 'opt_complete':
                        if not self._neighbor_optimization_status[neighbor_id]:
                            self.get_logger().info(
                                f"AGENT {self.agent_id} Neighbor {neighbor_id} completed optimization")
                            self._neighbor_optimization_status[neighbor_id] = True
            except Exception as e:
                self.get_logger().error(
                    f"AGENT {self.agent_id} Error during synchronization: {e}")

            time.sleep(self._sync_check_interval)

        self.get_logger().info(
            f"AGENT {self.agent_id} Synchronization loop exited")
        return False


class CBBAOptimizationThread(OptimizationThread):
    """Optimization thread for CBBA optimization.
    This class is responsible for handling the optimization process
    in a separate thread. It waits for the problem data to be ready
    and then starts the optimization process. It also handles the
    termination of the optimization thread.

    Args:
        OptimizationThread (OptimizationThread): Base class for optimization threads.
    """
    data_ready_event = None
    data_ready_future = None

    def optimize(self, future: Future):
        self.data_ready_future = future

        # prepare handling of asynchronous request
        self.data_ready_event = Event()

        def unblock(_):
            self.data_ready_event.set()

        self.data_ready_future.add_done_callback(unblock)

        # call method of parent class
        super().optimize()

    def do_optimize(self):
        # wait for problem data to be ready (or for halt event)
        OrEvent(self.data_ready_event, self._halt_event).wait()

        # exit on halt
        if self._halt_event.is_set():
            return

        # initialize and start optimization
        self.guidance.get_logger().info('Data received: starting optimization')
        data = self.data_ready_future.result().tasks
        self.optimizer.initialize(self.guidance, self._halt_event)
        self.optimizer.create_problem(data)
        self.optimizer.optimize()

        # wait for all neighbors to complete optimization before ending
        if not self._halt_event.is_set():
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Entering post-optimization phase")
            sync_success = self.guidance.wait_for_all_optimizers()
            if sync_success:
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Synchronization successful")
            else:
                self.guidance.get_logger().warn(
                    f"AGENT {self.guidance.agent_id} Synchronization failed")
        else:
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Optimization halted during synchronization")
