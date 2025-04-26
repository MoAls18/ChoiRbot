import copy
import numpy as np
from threading import Event, Lock
from .optimizer import Optimizer
from disropt.agents import Agent

# Internal Task and Task List representations of ROS2 messages
from collections import namedtuple
Task = namedtuple('Task', ['id', 'coordinates', 'value', 'seq_num'])


class TaskList():
    def __init__(self, tasks):
        self.tasks = tasks

    def __getitem__(self, index):
        return self.tasks[index]

    def __len__(self):
        return len(self.tasks)

    def __iter__(self):
        return iter(self.tasks)


class CBBAOptimizer(Optimizer):
    """
    Consensus-Based Bundle Algorithm (CBBA) for task assignment.
    This class implements the CBBA algorithm for multi-agent task assignment.
    It uses a greedy approach to build a bundle of tasks based on marginal scoring.
    The algorithm consists of two phases:

    1. Bundle Construction: Each agent builds a bundle of tasks based on marginal scoring.
    2. Consensus: Agents exchange messages to resolve conflicts and update their bundles.
    The algorithm continues until convergence or a maximum number of iterations is reached.
    The algorithm is designed to be used in a multi-agent system where agents communicate with each other
    to share their bundles and resolve conflicts.

    """

    def __init__(self, settings: dict = None):
        """ Initialize the CBBA optimizer class.
        This method initializes the CBBA optimizer with the given settings.

        Args:
            settings (dict, optional): Dictionary of settings to pass to the optimizer. Defaults to None.
        """

        # Call the parent class constructor
        super().__init__(settings)

        # Initialize the communication agent and task list fields
        self.agent = None
        self.task_list = None

        # Initialize bundle construction parameters
        self.bundle = []
        self.path = []

        self.bid_values = {}
        self.winners = {}
        self.winner_bids = {}
        self.timestamps = {}

        # Initialize convergence parameters with default values
        self.converged = False
        self.iterations_since_last_change = 0
        self.max_no_change_iterations = 3

        # Lock for thread safety
        # This lock is used to ensure that only one thread can access the CBBA data at a time minimizing race conditions.
        self.cbba_lock = Lock()

        self._read_settings(**settings if settings else {})

    def _read_settings(self, max_iterations=50, max_bundle_size=None, convergence_threshold=3, task_value_weight=0.8, **kwargs):
        """Read settings from the provided dictionary and set default values.

        Args:
            max_iterations (int, optional): Maximum number of convergence attempts. Defaults to 50.
            max_bundle_size (int, optional): Maximum bundle length of an agent. Defaults to None.
            convergence_threshold (int, optional): Number of iterations without a change to consider convergence complete. Defaults to 3.
            task_value_weight (float, optional): Diminishing task value used in task scoring. Defaults to 0.8.
        """
        self.max_iterations = max_iterations
        self.max_bundle_size = max_bundle_size
        self.max_no_change_iterations = convergence_threshold
        self.task_value_weight = task_value_weight

    def initialize(self, guidance, halt_event: Event = None):
        """Initialize the CBBA optimizer with the given guidance and halt event.

        Args:
            guidance (Guidance): The Guidance Node. 
            halt_event (Event, optional): Halt event used in asynchronous functions. Defaults to None.
        """
        super().initialize(guidance, halt_event)

        self.agent = Agent(in_neighbors=self.guidance.in_neighbors,
                           out_neighbors=self.guidance.out_neighbors, communicator=self.guidance.communicator)

    def create_problem(self, task_list):
        """Create the task assignment problem from the given task list.

        Args:
            task_list (TaskList): Task list to be used for the optimization.
        """
        self.task_list = TaskList([Task(task.id, task.coordinates, task.value, task.seq_num)
                                  for i, task in enumerate(task_list.tasks)])

        self.bundle = []
        self.path = []
        self.iter = 0
        self.bid_values = {task.id: 0 for task in task_list.tasks}
        self.winners = {task.id: -1 for task in task_list.tasks}
        self.winner_bids = {task.id: 0.0 for task in task_list.tasks}
        self.timestamps = {task.id: self.iter for task in task_list.tasks}

        # Reset Convergence checks
        self.converged = False
        self.iterations_since_last_change = 0
        self.iteration_count = 0

        # Default max bundle size to the number of tasks if not specified
        if self.max_bundle_size is None:
            self.max_bundle_size = len(task_list.tasks)

    def calculate_marginal_score(self, task_id):
        """Calculate the marginal score for a task based on its distance from the last task in the bundle.

        Args:
            task_id (int): id of task to calculate the marginal score for.

        Returns:
            float: Marginal score associated with the task.
        """

        task = next(
            (task for task in self.task_list.tasks if task.id == task_id), None)
        if not task:
            return float('-inf')  # Invalid task

        task_pos = np.array(task.coordinates)

        # If bundle is empty calculate the score from the current position
        if not self.path:
            current_pos = self.guidance.current_pose.position[:-1]
            distance = np.linalg.norm(task_pos - current_pos)
            return pow(self.task_value_weight, self.get_time_to_reach(task.id)) * 1 if hasattr(task, 'value') else -distance

        last_task_id = self.path[-1]
        last_task = next(
            (task for task in self.task_list.tasks if task.id == last_task_id), None)
        last_task_pos = np.array(last_task.coordinates)
        distance = np.linalg.norm(task_pos - last_task_pos)
        return pow(self.task_value_weight, self.get_time_to_reach(task.id)) * 1 if hasattr(task, 'value') else -distance

    def get_time_to_reach(self, task_id):
        """Calculate the time to reach a task based on the current position and velocity.

        Args:
            task_id (int): id of task.

        Returns:
            float: The time to reach the task.
        """

        task = next(
            (task for task in self.task_list.tasks if task.id == task_id), None)
        if not task:
            return float('inf')
        task_pos = np.array(task.coordinates)

        if not self.path:
            currrent_pos = self.guidance.current_pose.position[:-1]
            distance = np.linalg.norm(task_pos - currrent_pos)
            return distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance

        last_task_id = self.path[-1]
        last_task = next(
            (task for task in self.task_list.tasks if task.id == last_task_id), None)
        if not last_task:
            return float('inf')

        last_task_pos = np.array(last_task.coordinates)
        distance = np.linalg.norm(task_pos - last_task_pos)
        return distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance

    def build_bundle(self):
        """ Phase 1 of CBBA: Build the bundle of tasks based on marginal scoring.
        This method constructs a bundle of tasks for the agent based on the marginal scoring of each task.
        The bundle is built by selecting the task with the highest marginal score that is not already in the bundle.
        The method continues to add tasks to the bundle until the maximum bundle size is reached or no more tasks can be added.
        The method also updates the bid values, winners, and timestamps for each task in the bundle.
        """

        with self.cbba_lock:
            while len(self.bundle) < self.max_bundle_size:
                best_task_id = -1
                best_score = float('-inf')

                for task in self.task_list.tasks:
                    if task.id in self.bundle:
                        continue  # Already in bundle

                    score = self.calculate_marginal_score(task.id)
                    if score > best_score and score > self.winner_bids.get(task.id, float('-inf')):
                        best_task_id = task.id
                        best_score = score
                if best_task_id == -1:
                    break

                self.bundle.append(best_task_id)
                self.path.append(best_task_id)

                self.bid_values[best_task_id] = best_score
                self.winners[best_task_id] = self.guidance.agent_id
                self.winner_bids[best_task_id] = best_score
                self.timestamps[best_task_id] = self.iter

    def create_cbba_message(self):
        """ Create a message to be sent to neighbors containing the current state of the agent.
        This message includes the agent ID, winners, winner bids, and timestamps for the agent at the current iteration.
        The message is used for conflict resolution in the CBBA algorithm.

        Returns:
            message (dict): A dictionary containing the agent ID, winners, winner bids, and timestamps.
        """
        message = {
            'agent_id': self.guidance.agent_id,
            'winners': copy.deepcopy(self.winners),
            'winner_bids': copy.deepcopy(self.winner_bids),
            'timestamps': copy.deepcopy(self.timestamps)
        }
        return message

    def process_cbba_message(self, sender_id, message):
        """Phase 2 of CBBA: Conflict resolution based on received messages.
        This method processes the received message from a neighbor agent, checks for conflicts in winners and winning bids, and updates the local state of the agent
        based on the results of the conflict resolution rules. 

        Args:
            sender_id (int): id of the sender agent.
            message (dict): Message recieved from the sender agent.

        Returns:
            updated (bool): True if the local state was updated, False otherwise.
        """

        if not message or 'winners' not in message:
            print("EARLY RETURN")
            return False

        updated = False
        sender_winners = message['winners']
        sender_winner_bids = message['winner_bids']
        sender_timestamps = message['timestamps']

        with self.cbba_lock:
            for task_id in self.winners.keys():

                # Reciever Knowledge
                my_winner = self.winners[task_id]
                my_bid = self.winner_bids[task_id]
                my_timestamp = self.timestamps[task_id]

                # Sender Knowledge
                sender_winner = sender_winners.get(task_id, -1)
                sender_bid = sender_winner_bids.get(task_id, 0.0)
                sender_timestamp = sender_timestamps.get(task_id, 0)

                # Conflict resolution rule
                update_needed = False

                # RULE 1: If reciever believes sender is the winner and sender doesn't believe so.
                if my_winner == sender_id and sender_winner != sender_id:
                    update_needed = True

                # RULE 2: If sender believes reciever is the winner and reciever doesn't believe so.
                elif sender_winner == self.guidance.agent_id and my_winner != self.guidance.agent_id:
                    if my_winner == sender_id:
                        update_needed = True
                        self.winners[task_id] = -1
                        self.winner_bids[task_id] = 0.0
                    # If reciever winner is another neighbor, use most up-to-date information
                    elif my_winner != -1:
                        if sender_timestamp > my_timestamp:
                            update_needed = True
                            self.winners[task_id] = -1
                            self.winner_bids[task_id] = 0.0

                # RULE 3: If neither sender nor reciever is the winner, use bid values to determine winner.
                elif my_winner != sender_winner:
                    if my_winner != -1 and sender_winner != -1:
                        if sender_bid > my_bid:
                            update_needed = True
                        elif sender_bid < my_bid:
                            pass
                        else:
                            # Check timestamps if bids are equal
                            if sender_timestamp > my_timestamp:
                                update_needed = True
                            elif sender_timestamp < my_timestamp:
                                pass
                            else:
                                if sender_winner < my_winner:
                                    update_needed = True
                    elif sender_winner != -1:
                        update_needed = True

                # If conflict resolution is needed, update the local state
                if update_needed:
                    updated = True
                    self.iter += 1
                    self.winners[task_id] = sender_winner
                    self.winner_bids[task_id] = sender_bid
                    self.timestamps[task_id] = sender_timestamp
                    if my_winner == self.guidance.agent_id and sender_winner != self.guidance.agent_id:

                        # Remove this task and all tasks added after it
                        if task_id in self.bundle:
                            idx = self.bundle.index(task_id)
                            self.bundle = self.bundle[:idx]
                            self.path = []

                            # Remove tasks from the winners and winner_bids
                            for key in self.winners.keys():
                                if key > task_id and self.winners[key] != self.guidance.agent_id:
                                    self.winners[key] = -1
                                    self.timestamps[key] = self.iter
                            for key in self.winner_bids.keys():
                                if key > task_id and self.winner_bids[key] != self.guidance.agent_id:
                                    self.winner_bids[key] = 0.0
                                    self.timestamps[key] = self.iter

        return updated

    def optimize(self):
        """
        Run the CBBA optimization algorithm.

        This method runs the CBBA optimization algorithm for a maximum number of iterations or until convergence is reached.
        It's composed of two phases: bundle construction and consensus.
        In the first phase, each agent builds a bundle of tasks based on marginal scoring and exchanges messages with its neighbors.
        In the second phase, agents exchange messages to resolve conflicts and update their bundles.
        The algorithm continues until convergence or a maximum number of iterations is reached.
        The method also handles message communication and processing between agents.

        Returns:
            converged (bool): True if the optimization converged, False otherwise.
        """

        # Build initial bundle before beginning the consensus process
        self.build_bundle()
        self.converged = False
        self.iterations_since_last_change = 0

        for iteration in range(self.max_iterations):
            self.iteration_count += 1
            if self._halt_event and self._halt_event.is_set():
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Halting optimization")
                return False

            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} --- Iteration {iteration + 1}")
            self.guidance.get_logger().debug(
                f"AGENT {self.guidance.agent_id} Bundle: {self.bundle}")
            self.guidance.get_logger().debug(
                f"AGENT {self.guidance.agent_id} Path: {self.path}")
            self.guidance.get_logger().debug(
                f"AGENT {self.guidance.agent_id} Winners: {self.winners}")
            self.guidance.get_logger().debug(
                f"AGENT {self.guidance.agent_id} Winner Bids: {self.winner_bids}")

            # PHASE 1 MESSAGE COMMUNICATION
            message = self.create_cbba_message()
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Exchanging messages...")
            try:
                responses = self.agent.communicator.neighbors_exchange(
                    message, self.guidance.in_neighbors, self.guidance.out_neighbors, False, self._halt_event)
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Message exchange completed. Received {len(responses)} responses.")
            except Exception as e:
                self.guidance.get_logger().error(
                    f"Error in consensus communication: {e}")
                return False

            # PHASE 2 MESSAGE PROCESSING --- Consensus/Conflict Resolution
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Processing messages...")
            any_updates = False
            for neighbor_id, neighbor_msg in responses.items():
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} received message from {neighbor_id}")
                if neighbor_msg:
                    any_updates = self.process_cbba_message(
                        neighbor_id, neighbor_msg)
                else:
                    self.guidance.get_logger().warn(
                        f"AGENT {self.guidance.agent_id} Received empty message from neighbor {neighbor_id}")
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Message processing completed. Any updates: {any_updates}")

            # BUNDLE UPDATE AS NEEDED
            if any_updates:
                self.guidance.get_logger().info(
                    f"--UPDATE OCCURED-- AGENT {self.guidance.agent_id} Bundle updated.")
                self.build_bundle()
                self.iterations_since_last_change = 0
            else:
                self.guidance.get_logger().info(
                    f"--NO UPDATE-- AGENT {self.guidance.agent_id} No updates.")
                self.iterations_since_last_change += 1

            # Check for convergence
            if self.iterations_since_last_change >= self.max_no_change_iterations:
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Converged after {iteration + 1} iterations.")
                self.converged = True
                break

        # Finalize optimization by running a couple final consensus rounds if converged
        if self.converged:
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Running final consensus...")
            for _ in range(2):
                message = self.create_cbba_message()
                try:
                    responses = self.agent.communicator.neighbors_exchange(
                        message, self.guidance.in_neighbors, self.guidance.out_neighbors, False, self._halt_event)
                    for neighbor_id, neighbor_msg in responses.items():
                        if neighbor_msg:
                            self.process_cbba_message(
                                neighbor_id, neighbor_msg)
                except Exception as e:
                    self.guidance.get_logger().error(
                        f"Error in consensus communication: {e}")
        else:
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} Max iterations reached without convergence.")

        self.guidance.get_logger().info(
            f"AGENT {self.guidance.agent_id} Final Winners: {self.winners}")

        return self.converged

    def update_path_from_winners(self):
        """Construct the final path based on winning bids.
        """

        self.guidance.get_logger().info(
            f"AGENT {self.guidance.agent_id} ENTERING UPDATE")

        # Get all tasks assigned to this agent
        assigned_task_ids = []
        for task in self.task_list.tasks:
            if self.winners.get(task.id) == self.guidance.agent_id:
                assigned_task_ids.append(task.id)

        self.guidance.get_logger().info(
            f"AGENT {self.guidance.agent_id} ASSIGNED TASKS: {assigned_task_ids}")

        self.path = []

        if not assigned_task_ids:
            return

        remaining_tasks = copy.deepcopy(assigned_task_ids)

        # Construct path using CBBA marginal scoring logic
        while remaining_tasks:

            best_task_id = None
            best_score = float('-inf')
            for task_id in remaining_tasks:
                score = self.calculate_marginal_score(task_id)
                if score > best_score:
                    best_task_id = task_id
                    best_score = score

            if best_task_id is not None:
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} ADDING Best task: {best_task_id} with score: {best_score}")
                self.path.append(best_task_id)
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Removing from remaining tasks: {best_task_id}")
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Remaining tasks: {remaining_tasks}")
                remaining_tasks.remove(best_task_id)
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} Remaining tasks AFTER UPDATE: {remaining_tasks}")

    def get_result(self):
        """Return the final result of the optimization.
        This method returns the final assigned tasks based on the optimization results.

        Returns:
            assigned_tasks (list): List of assigned tasks.
        """
        self.guidance.get_logger().info(
            f"*******AGENT {self.guidance.agent_id} ENTERING get_result")
        assigned_tasks = []
        try:
            self.guidance.get_logger().info(
                f"AGENT {self.guidance.agent_id} entering locking")
            with self.cbba_lock:

                # Logging
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} get_result: FINAL BUNDLE: {self.bundle}")
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} get_result: PATH BEFORE UPDATE: {self.path}")
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} get_result: FINAL WINNERS: {self.winners}")

                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} get_result: UPDATING PATH")

                # Path update and task assignment
                self.update_path_from_winners()
                self.guidance.get_logger().info(
                    f"AGENT {self.guidance.agent_id} get_result: PATH AFTER UPDATE: {self.path}")

                task_map = {task.id: task for task in self.task_list.tasks}

                for task_id in self.path:
                    if task_id in task_map:
                        self.guidance.get_logger().info(
                            f"AGENT {self.guidance.agent_id} get_result: TASK {task_id} FOUND IN TASK MAP. NOW ADDING")
                        assigned_tasks.append(task_map[task_id])
                    else:
                        self.guidance.get_logger().error(
                            f"AGENT {self.guidance.agent_id} get_result: TASK {task_id} NOT FOUND IN TASK MAP. THIS SHOULD NOT HAPPEN")

            self.guidance.get_logger().warn(
                f"AGENT {self.guidance.agent_id} get_result: FINAL ASSIGNED TASKS: {[task.id for task in assigned_tasks]} NOW EXITING")
            return assigned_tasks
        except Exception as e:
            self.guidance.get_logger().error(
                f"!!!!!!!! AGENT {self.guidance.agent_id} EXCEPTION IN get_result: {e} !!!!!!!!", exc_info=True)
            return []

    def get_cost(self):
        """Calculate the total cost of the assigned tasks.

        Returns:
            total_cost (float): The total cost of the assigned tasks.
        """
        total_cost = 0.0

        for task_id in self.bundle:
            total_cost += self.bid_values.get(task_id, 0.0)

        return total_cost

    def get_task_list(self):
        """Get the task list.

        Returns:
            TaskList: The task list associated with the optimizer.
        """

        return self.task_list.tasks

    def get_iterations(self):
        """Get the number of iterations performed.

        Returns:
            int: The number of iterations performed.
        """
        return self.iteration_count
