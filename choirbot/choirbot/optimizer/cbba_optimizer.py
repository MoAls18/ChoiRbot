import copy
import time
import numpy as np
from threading import Event, Lock
from .optimizer import Optimizer
from disropt.agents import Agent
from choirbot_interfaces.msg import PositionTask
from choirbot_interfaces.msg import PositionTaskArray
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
    Conensus-Based Bundle Algorithm (CBBA) for distributed task assignment.

    """
    
    def __init__(self, settings: dict=None):
        super().__init__(settings)
        
        self.agent = None
        self.task_list = None
        
        self.bundle = []
        self.path = []  

        self.bid_values = {}
        self.winners = {}
        self.winner_bids = {}
        self.timestamps = {}

        self.converged = False
        self.iterations_since_last_change = 0
        self.max_no_change_iterations = 3
        
        self.cbba_lock = Lock()

        self._read_settings(**settings if settings else {})

    def _read_settings(self, max_iterations=50, max_bundle_size = None, convergence_threshold=3, task_value_weight=0.8, **kwargs):
        self.max_iterations = max_iterations
        self.max_bundle_size = max_bundle_size
        self.max_no_change_iterations = convergence_threshold
        self.task_value_weight = task_value_weight
        
    
    def initialize(self, guidance, halt_event:Event = None):
        super().initialize(guidance, halt_event)

        # Create aggent for communication
        self.agent = Agent(in_neighbors=self.guidance.in_neighbors, out_neighbors=self.guidance.out_neighbors,communicator=self.guidance.communicator)
        # self.guidance.get_loggger().info(f"AGENT {self.guidance.agent_id} Initialized with neighbors: {self.guidance.in_neighbors}")
        # self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Initialized with neighbors: {self.guidance.out_neighbors}")
    
    def create_problem(self, task_list):
        """ Initialize the problem with a list of tasks.

        """
        self.task_list = TaskList([Task(task.id, task.coordinates, task.value, task.seq_num) for i, task in enumerate(task_list.tasks)])
        # self.task_list = task_list
        
        

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

        # Default max bundle size to the number of tasks if not specified
        if self.max_bundle_size is None:
            self.max_bundle_size = len(task_list.tasks)


    def calculate_marginal_score(self, task_id):
        """Calculate the marginal score of a task."""

        task = next((task for task in self.task_list.tasks if task.id == task_id), None)
        if not task:
            return float('-inf') # Invalid
        
        task_pos = np.array(task.coordinates)
        
        # If bundle is empty calculate the score from the current position
        if not self.path:
            current_pos = self.guidance.current_pose.position[:-1]
            distance = np.linalg.norm(task_pos - current_pos)
            # print(f"AGENT: {self.guidance.agent_id} Distance from current position to task {task.id}: {distance}")
            return pow(self.task_value_weight,self.get_time_to_reach(task.id)) * 1 if hasattr(task, 'value') else -distance 
        
        last_task_id = self.path[-1]
        last_task = next((task for task in self.task_list.tasks if task.id == last_task_id), None)
        last_task_pos = np.array(last_task.coordinates)
        distance = np.linalg.norm(task_pos - last_task_pos)
        # print(f"AGENT: {self.guidance.agent_id} Distance from last task {last_task_id} to task {task.id}: {distance}")
        return pow(self.task_value_weight,self.get_time_to_reach(task.id)) * 1 if hasattr(task, 'value') else -distance 
    
    def get_time_to_reach(self, task_id):
        """Calculate the time to reach a task."""
        task = next((task for task in self.task_list.tasks if task.id == task_id), None)
        if not task:
            return float('inf')
        task_pos = np.array(task.coordinates)
        
        if not self.path:
            currrent_pos = self.guidance.current_pose.position[:-1]
            distance = np.linalg.norm(task_pos - currrent_pos)
            return distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance
        
        last_task_id = self.path[-1]
        last_task = next((task for task in self.task_list.tasks if task.id == last_task_id), None)
        if not last_task:
            return float('inf')
        
        last_task_pos = np.array(last_task.coordinates)
        distance = np.linalg.norm(task_pos - last_task_pos)
        return distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance
    
    def build_bundle(self):
        """
        Phase 1 of CBBA: Bundle Construction.
        Uses marginal scoring function to build a bundle of tasks.
        Aggents greedily add tasks to the bundle until the max bundle size is reached.
        """

        with self.cbba_lock:
            while len(self.bundle) < self.max_bundle_size:
                best_task_id = -1
                best_score = float('-inf')
                
                for task in self.task_list.tasks:
                    if task.id in self.bundle:
                        continue # Already in bundle
                    
                    # print(f"Agent {self.guidance.agent_id} checking task {task.id}")
                    score = self.calculate_marginal_score(task.id)
                    if score > best_score and score > self.winner_bids.get(task.id, float('-inf')):
                        best_task_id = task.id
                        best_score = score
                if best_task_id == -1:
                    break
                
                self.bundle.append(best_task_id)
                self.path.append(best_task_id)
                # print(f"Agent {self.guidance.agent_id} added task {best_task_id} to bundle with score {best_score}")
                
                self.bid_values[best_task_id] = best_score
                self.winners[best_task_id] = self.guidance.agent_id
                self.winner_bids[best_task_id] = best_score
                self.timestamps[best_task_id] = self.iter # position in bundle as timestamp 
    
    def create_cbba_message(self):
        """Create a CBBA message to share with neighbors."""
        message = {
            'agent_id': self.guidance.agent_id,
            'winners': copy.deepcopy(self.winners),
            'winner_bids': copy.deepcopy(self.winner_bids),
            'timestamps': copy.deepcopy(self.timestamps)
        }
        return message
    
    def process_cbba_message(self, sender_id, message):
        """ Processes CBBA message according to conflict resolution rules."""
        # self.guidance.get_logger().info(f"Agent {self.guidance.agent_id} received message from {sender_id}: {message}")
        if not message or 'winners' not in message:
            print("EARLY RETURN")
            return False
        
        updated = False
        sender_winners = message['winners']
        sender_winner_bids = message['winner_bids']
        sender_timestamps = message['timestamps']
        # self.guidance.get_logger().info(f"MY {self.guidance.agent_id} winners: {self.winners}")
        # self.guidance.get_logger().info(f"MY {self.guidance.agent_id} winner bids: {self.winner_bids}")
        # self.guidance.get_logger().info(f"Sender  {message['agent_id']} winners: {sender_winners}")
        # self.guidance.get_logger().info(f"Sender {message['agent_id']} winner bids: {sender_winner_bids}")
        with self.cbba_lock:
            for task_id in self.winners.keys():
                
                # Reciever Knowledge
                my_winner = self.winners[task_id]
                my_bid = self.winner_bids[task_id]
                my_timestamp = self.timestamps[task_id]

                # print(f"Reciever Knowledge: winner_agent_id: {my_winner}, winning_bid: {my_bid}, bid time{my_timestamp}")
                # Sender Knowledge
                sender_winner = sender_winners.get(task_id, -1)
                sender_bid = sender_winner_bids.get(task_id, 0.0)
                sender_timestamp = sender_timestamps.get(task_id, 0)
                # print("Sender knowledge: ", sender_winner, sender_bid, sender_timestamp)

                # Conflict resolution rule
                update_needed = False

                if my_winner == sender_id and sender_winner != sender_id:
                    # print(f"I agent {self.guidance.agent_id} need to update my knowledge")
                    update_needed = True
                
                
                elif sender_winner == self.guidance.agent_id and my_winner != self.guidance.agent_id:
                    # print("RULE 2 RULE 2")
                    if my_winner == sender_id:
                        update_needed = True
                        self.winners[task_id] = -1
                        self.winner_bids[task_id] = 0.0
                    elif my_winner != -1:
                        if sender_timestamp > my_timestamp:
                            update_needed = True
                            self.winners[task_id] = -1
                            self.winner_bids[task_id] = 0.0
                    
                
                elif my_winner != sender_winner:
                    # print("RULE 3 RULE 3")
                    if my_winner != -1 and sender_winner != -1:
                        # print("SENDER BID {} - MY BID {}".format(sender_bid, my_bid))
                        if sender_bid > my_bid:
                            update_needed = True
                        elif sender_bid < my_bid:
                            pass
                        else:
                            # print("BIDS ARE EQUAL")
                            # Check timestamps
                            # print(f"Sender timestamp: {sender_timestamp}, My timestamp: {my_timestamp}")
                            if sender_timestamp > my_timestamp:
                                update_needed = True
                            elif sender_timestamp < my_timestamp:
                                pass
                            else:
                                if sender_winner < my_winner:
                                    update_needed = True
                    elif sender_winner != -1:
                        update_needed = True
                
                # print(f"Update needed: {update_needed}")
                if update_needed:
                    updated = True
                    self.iter += 1
                    # update my knowledge
                    self.winners[task_id] = sender_winner
                    self.winner_bids[task_id] = sender_bid
                    self.timestamps[task_id] = sender_timestamp
                    # print(f"Updating my knowledge: winner_agent_id: {self.winners[task_id]}, winning_bid: {self.winner_bids[task_id]}, bid time{self.timestamps[task_id]}")
                    # print(f"my_winner({self.guidance.agent_id}): {my_winner}, sender_winner: {sender_winner}")
                    if my_winner == self.guidance.agent_id and sender_winner != self.guidance.agent_id:
                        # print("I need to remove the task from my bundle")
                        if task_id in self.bundle:
                            # print(f"removing tasks from agent {self.guidance.agent_id} from bundle: {self.bundle}")
                            idx = self.bundle.index(task_id)
                            # print(f"removing task {task_id} from bundle")
                            # Remove this task and all tasks added after it
                            removed_tasks = self.bundle[idx:]
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
                            # print(f"Updated bundle: {self.bundle}")
                            # Also remove these tasks from the path
                            # for removed_task in removed_tasks:
                            #     if removed_task in self.path:
                            #         self.path.remove(removed_task)
                                    # print(f"Updated path: {self.path}")
        return updated   
                

    def consensus(self):
        
        message = self.create_cbba_message()
        try:
            responses = self.agent.communicator.neighbors_exchange(message, self.guidance.in_neighbors, self.guidance.out_neighbors, False , self._halt_event)
            # self.guidance.get_logger().info(f"Agent {self.guidance.agent_id} received messages from neighbors: {self.guidance.in_neighbors}")
            # self.guidance.get_logger().info(f"Agent {self.guidance.agent_id} ")
        except Exception as e:
            self.guidance.get_logger().error(f"Error in consensus communication: {e}")
            return False
        
        any_updates = False
        for neighbor_id, neighbor_msg in responses.items():
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} received message from {neighbor_id}: {neighbor_msg}")
            if neighbor_msg and self.process_cbba_message(neighbor_id, neighbor_msg):
                any_updates = True
        
        return any_updates
    
    
    
    def optimize(self):

        """ 
        Run the CBBA algorithm until convergence or max iterations reached.

        This two-phase algorithm consists of:
        1. Bundle Construction: (task selection)
        2. Consensus: (conflict resolution)
        """
        self.build_bundle()
        # while len(self.guidance.in_neighbors) < 2:
        #     time.sleep(0.1)
        self.converged = False
        self.iterations_since_last_change = 0
        for iteration in range(self.max_iterations):
            # Add halt event
            if self._halt_event and self._halt_event.is_set():
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Halting optimization")
                return False

            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} --- Iteration {iteration + 1}")
            self.guidance.get_logger().debug(f"AGENT {self.guidance.agent_id} Bundle: {self.bundle}")
            self.guidance.get_logger().debug(f"AGENT {self.guidance.agent_id} Path: {self.path}")
            self.guidance.get_logger().debug(f"AGENT {self.guidance.agent_id} Winners: {self.winners}")
            self.guidance.get_logger().debug(f"AGENT {self.guidance.agent_id} Winner Bids: {self.winner_bids}")
            
            # PHASE 1 MESSAGE COMMUNICATION
            message = self.create_cbba_message()
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Exchanging messages...")
            try:
                responses = self.agent.communicator.neighbors_exchange(message, self.guidance.in_neighbors, self.guidance.out_neighbors, False , self._halt_event)
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Message exchange completed. Received {len(responses)} responses.")
            except Exception as e:
                self.guidance.get_logger().error(f"Error in consensus communication: {e}")
                return False
            
            # PHASE 2 MESSAGE PROCESSING --- Consensus/Conflict Resolution
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Processing messages...")
            any_updates = False
            for neighbor_id, neighbor_msg in responses.items():
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} received message from {neighbor_id}")
                if neighbor_msg:
                    any_updates = self.process_cbba_message(neighbor_id, neighbor_msg)
                else:
                    self.guidance.get_logger().warn(f"AGENT {self.guidance.agent_id} Received empty message from neighbor {neighbor_id}")
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Message processing completed. Any updates: {any_updates}")
            
            # BUNDLE UPDATE AS NEEDED
            if any_updates:
                self.guidance.get_logger().info(f"--UPDATE OCCURED-- AGENT {self.guidance.agent_id} Bundle updated.")
                self.build_bundle()
                self.iterations_since_last_change = 0
            else:
                self.guidance.get_logger().info(f"--NO UPDATE-- AGENT {self.guidance.agent_id} No updates.")
                self.iterations_since_last_change += 1
            
            if self.iterations_since_last_change >= self.max_no_change_iterations:
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Converged after {iteration + 1} iterations.")
                self.converged = True
                break
        
        if self.converged:
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Running final consensus...")
            for _ in range(2):
                message = self.create_cbba_message()
                try:
                    responses = self.agent.communicator.neighbors_exchange(message, self.guidance.in_neighbors, self.guidance.out_neighbors, False , self._halt_event)
                    for neighbor_id, neighbor_msg in responses.items():
                        if neighbor_msg:
                            self.process_cbba_message(neighbor_id, neighbor_msg)
                except Exception as e:
                    self.guidance.get_logger().error(f"Error in consensus communication: {e}")
        else:
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Max iterations reached without convergence.")
        
        self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Final Winners: {self.winners}")

        if self.guidance.agent_id == 0:
            agent_0_tasks = [task for task in self.task_list.tasks if self.winners.get(task.id) == 0]
            self.guidance.get_logger().warn(f"AGENT 0 FINAL CHECK - WINNING TASKS: {agent_0_tasks}")
            self.guidance.get_logger().warn(f"AGENT 0 FINAL CHECK - FINAL BUNDLE: {self.bundle}")
            self.guidance.get_logger().warn(f"AGENT 0 FINAL CHECK - FINAL PATH: {self.path}")
        

        # self.update_path_from_winners()
        return self.converged
            

        

    def update_path_from_winners(self):
        """Create path based on CBBA marginal score"""
        # with self.cbba_lock:
        self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} ENTERING UPDATE")
        # Get all tasks assigned to this agent
        assigned_task_ids = []
        for task in self.task_list.tasks:
            if self.winners.get(task.id) == self.guidance.agent_id:
                assigned_task_ids.append(task.id)
        self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} ASSIGNED TASKS: {assigned_task_ids}")
        #  Clear current path
        self.path = []
        
        if not assigned_task_ids:
            return
        
        remaining_tasks = copy.deepcopy(assigned_task_ids)
        current_pos = self.guidance.current_pose.position[:-1]
        
        # Construct path using CBBA marginal scoring logic
        while remaining_tasks:
            
            best_task_id = None
            best_score = float('-inf')
            for task_id in remaining_tasks:
                task = next((t for t in self.task_list.tasks if t.id == task_id), None)
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} checking task {task}")
                if task:
                    # Calculate score using same logic as calculate_marginal_score
                    task_pos = np.array(task.coordinates)
                    
                    if not self.path:
                        # If path is empty, calculate from current position
                        distance = np.linalg.norm(task_pos - current_pos)
                        time_to_reach = distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance
                    else:
                        # Otherwise, calculate from last task in path
                        last_task_id = self.path[-1]
                        last_task = next((t for t in self.task_list.tasks if t.id == last_task_id), None)
                        last_task_pos = np.array(last_task.coordinates)
                        distance = np.linalg.norm(task_pos - last_task_pos)
                        time_to_reach = distance / self.guidance.default_velocity if hasattr(self.guidance, 'default_velocity') else distance
                    
                    # Use same scoring function as in bundle construction
                    score = pow(self.task_value_weight, time_to_reach) * 1
                    self.guidance.get_logger().info(f"***AGENT {self.guidance.agent_id} Score for task {task_id}: {score}")
                    
                    if score > best_score:
                        best_score = score
                        best_task_id = task_id
            
            if best_task_id is not None:
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} ADDING Best task: {best_task_id} with score: {best_score}")
                self.path.append(best_task_id)
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Removing from remaining tasks: {best_task_id}")
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Remaining tasks: {remaining_tasks}")
                remaining_tasks.remove(best_task_id)
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} Remaining tasks AFTER UPDATE: {remaining_tasks}")
                # Update current position for next iteration
                task = next((t for t in self.task_list.tasks if t.id == best_task_id), None)
                current_pos = task.coordinates
            
        
    def get_result(self):
        """Get assigned tasks based on the bundle"""
        self.guidance.get_logger().info(f"*******AGENT {self.guidance.agent_id} ENTERING get_result")
        assigned_tasks = []
        try:
            self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} entering locking")
            with self.cbba_lock:
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: FINAL BUNDLE: {self.bundle}")
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: PATH BEFORE UPDATE: {self.path}")
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: FINAL WINNERS: {self.winners}")

                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: UPDATING PATH")
                self.update_path_from_winners()
                self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: PATH AFTER UPDATE: {self.path}")
                task_map = {task.id: task for task in self.task_list.tasks}
                
                for task_id in self.path:
                    if task_id in task_map:
                        self.guidance.get_logger().info(f"AGENT {self.guidance.agent_id} get_result: TASK {task_id} FOUND IN TASK MAP. NOW ADDING")
                        assigned_tasks.append(task_map[task_id])
                    else:
                        self.guidance.get_logger().error(f"AGENT {self.guidance.agent_id} get_result: TASK {task_id} NOT FOUND IN TASK MAP. THIS SHOULD NOT HAPPEN")
                # for task in self.path:
                #     task = next((t for t in self.task_list.tasks if t.id == task), None)
                #     if task:
                #         assigned_tasks.append(task)
                    
                
                # if assigned_tasks:
                #     task_order = {task_id: idx for idx, task_id in enumerate(self.path)}
                #     assigned_tasks.sort(key=lambda task: task_order.get(task.id, float('inf')))
                # print the step by step order of tasks
                # print(f"AGENT {self.guidance.agent_id} ASSIGNED TASKS: {[task.id for task in assigned_tasks]}")
                # for task in assigned_tasks:
                    # print(f"AGENT {self.guidance.agent_id} BID VALUE: {self.bid_values.get(task.id, 0.0)} for task {task.id}")
                
                # create PositionTaskArray message
                

                
            self.guidance.get_logger().warn(f"AGENT {self.guidance.agent_id} get_result: FINAL ASSIGNED TASKS: {[task.id for task in assigned_tasks]} NOW EXITING")
            return assigned_tasks
        except Exception as e:
            self.guidance.get_logger().error(f"!!!!!!!! AGENT {self.guidance.agent_id} EXCEPTION IN get_result: {e} !!!!!!!!", exc_info=True)
            return []

    def get_cost(self):
        """Get the total cost of the assignment"""
        total_cost = 0.0
        
        for task_id in self.bundle:
            total_cost += self.bid_values.get(task_id, 0.0)

        return total_cost
    def get_task_list(self):
        """Get the task list"""
        return self.task_list.tasks