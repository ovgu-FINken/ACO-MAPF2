from collections import deque
import networkx as nx
import random
import numpy as np
from tqdm import tqdm

class Agent:
    def __init__(
                    self,
                    agent_id,
                    start_position,
                    goal_position,
                    G,
                    nodelist,
                    time_horizon,
                    n_iterations=5,
                    n_episodes=10,
                    max_stored_iterations=1,
                    alpha=1,
                    beta=2,
                    gamma=4,
                    evaporation_rate=0.1,
                    dispersion_rate=0.1,
                    initial_epsilon=0.8,
                    collision_weight=0.3,
                    length_weight=None,
                    method="aco",
                ):
        self.agent_id = agent_id
        self.start_position = start_position
        self.goal_position = goal_position
        self.time_horizon = time_horizon
        self.G = G
        self.nodelist = nodelist
        self.G_t = self._create_time_expanded_graph()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.evaporation_rate = evaporation_rate
        self.dispersion_rate = dispersion_rate
        self.initial_epsilon = initial_epsilon
        if method == "aco":
            self.initialize_policy = self._initialize_pheromone
            self.update_policy = self.update_pheromone
            self.decision_function = self._aco_decision_function
        elif method == "q-learning":
            self.initialize_policy = self._initialize_q
            self.update_policy = self.update_q
            self.decision_function = self._q_decision_function
        elif method == "simplified-q-learning":
            self.initialize_policy = self._initialize_q
            self.update_policy = self.update_simplified_q
            self.decision_function = self._q_decision_function
        else:
            raise NotImplementedError(f"Method {method} is not implemented")
        self.initialize_policy()
        self.stored_paths = deque(maxlen=max_stored_iterations)
        self.occupancy_matrix = np.zeros_like(np.zeros((len(self.nodelist), self.time_horizon)))
        self.other_occupancy = np.zeros_like(self.occupancy_matrix)
        self.n_iterations = n_iterations
        self.n_episodes = n_episodes
        self.collision_weight = collision_weight
        self.length_weight = 1.0 - collision_weight if length_weight is None else length_weight

    def _create_time_expanded_graph(self):
        G_t = nx.DiGraph()
        for t in range(self.time_horizon):
            for v in self.nodelist:
                G_t.add_node((v, t))
                if t > 0:
                    G_t.add_edge((v, t-1), (v, t))  # Wait action
                    for neighbor in self.G.neighbors(v):
                        G_t.add_edge((v, t-1), (neighbor, t))  # Move action
        return G_t

    def _initialize_pheromone(self):
        nx.set_edge_attributes(self.G_t, 1.0, 'pheromone')

    def _initialize_q(self):
        nx.set_edge_attributes(self.G_t, 1.0, 'Q')
        
    def get_node_pheromones(self):
        node_pheromones = np.zeros((len(self.nodelist), self.time_horizon))
        for (u, v, pheromone) in self.G_t.edges(data='pheromone'):
            node_pheromones[self.nodelist.index(v[0]), v[1]] += pheromone
        return node_pheromones

    def update_q(self, paths):
        for path in paths:
            if not path:
                continue
            for t in range(len(path) - 1):
                state = path[t]
                next_state = path[t + 1]
                if not len(path[t+1:]):
                    break
                quality = self._calculate_path_qualities([path[t+1:]], normalize=False)[0]
                
                # Current Q-value
                current_q = self.G_t[state][next_state]['Q']

                # Update Q-value
                self.G_t[state][next_state]['Q'] += self.alpha * (quality - current_q)
                
    def update_simplified_q(self, paths):
        qualities = self._calculate_path_qualities(paths)
        for path, quality in zip(paths, qualities):
            for t in range(len(path) - 1):
                state = path[t]
                next_state = path[t + 1]
                
                # Current Q-value
                current_q = self.G_t[state][next_state]['Q']

                # Update Q-value
                self.G_t[state][next_state]['Q'] += self.alpha * (quality - current_q)


    def _calculate_collision_probability(self, path):
        collision_prob = 0
        for node, time in path:
            collision_prob += self.other_occupancy[self.nodelist.index(node), time]
        return collision_prob / len(path)
    
    def _calculate_path_length(self, path):
        if path[-1][0] == self.goal_position:
            additional_length = 0
        else:
            additional_length = nx.shortest_path_length(self.G, path[-1][0], self.goal_position)
        return len(path) + additional_length - 1
    
    def _calculate_path_qualities(self, paths, normalize=True):
        if not paths:
            return []
        path_lengths = [ self._calculate_path_length(path) for path in paths ]
        collision_probs = [ self._calculate_collision_probability(path) for path in paths ]
        
        if path_lengths:
            # Normalize path lengths and collision probabilities to [0, 1] range
            # bad: 0, good: 1
            if normalize:
                max_length = max(path_lengths)
                min_length = min(path_lengths)
                max_collision_prob = max(collision_probs)
                min_collision_prob = min(collision_probs)
                path_lengths = [(max_length - length) / (max_length - min_length) if max_length != min_length else 1 for length in path_lengths]
                collision_probs = [(max_collision_prob - prob) / (max_collision_prob - min_collision_prob) if max_collision_prob != min_collision_prob else 1 for prob in collision_probs]
            
            # Combine length and collision probability (you can adjust the weights)
            # bad: 0, good: 1
            combined_scores = [self.length_weight * length + self.collision_weight * (1 - collision_prob) for length, collision_prob in zip(path_lengths, collision_probs)]
            return combined_scores
        return []
            

    def update_pheromone(self, paths):
        # Evaporation
        for u, v in self.G_t.edges():
            self.G_t[u][v]['pheromone'] *= (1 - self.evaporation_rate)

        # Remove duplicate paths
        unique_paths = list(set(tuple(path) for path in paths))
        
        if unique_paths:
            combined_scores = self._calculate_path_qualities(unique_paths) 
            # Update pheromones
            for path, score in zip(unique_paths, combined_scores):
                for i in range(len(path) - 1):
                    if i >= self.time_horizon:
                        break
                    u, v = path[i], path[i+1]
                    self.G_t[u][v]['pheromone'] += score

        # Pheromone dispersion
        self._disperse_pheromone()

    def _disperse_pheromone(self):
        new_pheromone = {(u, v): data['pheromone'] for u, v, data in self.G_t.edges(data=True)}
        
        for (u, v), pheromone_value in new_pheromone.items():
            (u_node, t1), (v_node, t2) = u, v
            
            # Disperse to adjacent time-steps
            adjacent_edges = [
                ((u_node, t1-1), (v_node, t2-1)),  # Previous time-step
                ((u_node, t1+1), (v_node, t2+1))   # Next time-step
            ]
            
            for adj_edge in adjacent_edges:
                if self.G_t.has_edge(*adj_edge):
                    dispersed_value = pheromone_value * self.dispersion_rate
                    new_pheromone[adj_edge] = new_pheromone.get(adj_edge, 0) + dispersed_value
                    new_pheromone[(u, v)] -= dispersed_value

        # Update G_t with new pheromone values
        nx.set_edge_attributes(self.G_t, new_pheromone, 'pheromone')

    def calculate_probabilities(self, current, neighbors, goal):
        probabilities = []
        for neighbor in neighbors:
            edge = (current, neighbor)
            agent_pheromone = self.G_t[current][neighbor]['pheromone']
            other_pheromone = self._calculate_other_pheromones(neighbor[0], neighbor[1])
            heuristic = 1 / (self._heuristic(neighbor[0], goal) + 1)
            probability = (agent_pheromone ** self.alpha) * ((1 / (other_pheromone + 1)) ** self.gamma) * (heuristic ** self.beta)
            probabilities.append(probability)
        return probabilities

    def _calculate_other_pheromones(self, node, t):
        return self.other_pheromones[self.nodelist.index(node), t]

    def _heuristic(self, v, goal):
        return nx.shortest_path_length(self.G, v, goal)
    
    def _aco_decision_function(self, current, neighbors, goal, greedy=False):
        probabilities = self.calculate_probabilities(current, neighbors, goal)
        if greedy:
            return neighbors[np.argmax(probabilities)]
        probabilities = probabilities / np.sum(probabilities)
        return neighbors[np.random.choice(len(probabilities), p=probabilities)]
    
    def _q_decision_function(self, current, neighbors, goal, greedy=False):
        q_values = [self.G_t[current][neighbor]['Q'] for neighbor in neighbors]
        return neighbors[np.argmax(q_values)]

    def epsilon_greedy_decision(self, current, neighbors, goal, epsilon, greedy=False):
        if random.random() < epsilon and not greedy:
            return random.choice(neighbors)
        else:
            return self.decision_function(current, neighbors, goal, greedy=greedy)

    def run_episode(self, iteration, max_iterations, greedy=False):
        current = (self.start_position, 0)
        path = [current]
        
        # Calculate dynamic epsilon
        epsilon = self.initial_epsilon * (1 - iteration / max_iterations)

        while current[0] != self.goal_position and len(path) < self.time_horizon:
            neighbors = list(self.G_t.neighbors(current))
            if not neighbors:
                return None  # No valid path

            next_node = self.epsilon_greedy_decision(current, neighbors, self.goal_position, epsilon, greedy=greedy)
            path.append(next_node)
            current = next_node

        return path
    

    def run_aco_iterations(self):
        all_paths = []
        for iteration in range(self.n_iterations):
            paths = []
            for _ in range(self.n_episodes):
                path = self.run_episode(iteration, self.n_iterations, greedy=False)
                if path is not None:
                    paths.append(path)
            if paths:
                all_paths.extend(paths)
                self.update_policy(paths)
        
        self.stored_paths.append(all_paths)
        self.update_occupancy_matrix()
        return all_paths
    
    def update_occupancy_matrix(self):
        self.occupancy_matrix.fill(0)
        for paths in self.stored_paths:
            for path in paths:
                for node, t in path:
                    self.occupancy_matrix[self.nodelist.index(node), t] += 1
        self.occupancy_matrix /= sum([len(path) for path in self.stored_paths])

class ACOMultiAgentPathfinder:
    def __init__(self,
                 graph,
                 start_positions,
                 goal_positions,
                 n_episodes=10,
                 n_iterations=100,
                 alpha=1.0,
                 beta=0.5,
                 gamma=1.5,
                 evaporation_rate=0.2,
                 dispersion_rate=0.2,
                 communication_interval=10,
                 initial_epsilon=0.2,
                 collision_weight=0.5,
                 length_weight=None,
                 horizon=None,
                    method='aco',
            ):
        
        self.G = graph
        self.nodelist = list(self.G.nodes())
        self.n_agents = len(start_positions)
        self.n_episodes = n_episodes
        self.n_iterations = n_iterations
        self.communication_interval = communication_interval
        self.method = method

        if horizon is None:
            self.time_horizon = int(np.sqrt(len(self.G.nodes())+20)*3)
        else:
            self.time_horizon = horizon

        self.agents = [Agent(i,
                            start_positions[i],
                            goal_positions[i],
                            self.G,
                            self.nodelist,
                            self.time_horizon, 
                            alpha=alpha,
                            beta=beta,
                            gamma=gamma, 
                            evaporation_rate=evaporation_rate, 
                            dispersion_rate=dispersion_rate,
                            initial_epsilon=initial_epsilon, 
                            n_episodes=self.n_episodes,
                            n_iterations=self.communication_interval,
                            collision_weight=collision_weight,
                            length_weight=length_weight,
                            method=method
                        )
                            
                       for i in range(self.n_agents)]
        self._communicate()


    def _check_conflicts(self, paths):
        occupied = {}
        for agent, path in enumerate(paths):
            for t, node in enumerate(path):
                if t >= self.time_horizon:
                    return True
                # Check for same-time conflicts
                if (t, node[0]) in occupied and occupied[(t, node[0])] != agent:
                    return True
                occupied[(t, node[0])] = agent

                # Check for adjacent time-step conflicts (swapping positions)
                if t > 0:
                    prev_node = path[t-1][0]
                    if (t-1, node[0]) in occupied and occupied[(t-1, node[0])] != agent and \
                       (t, prev_node) in occupied and occupied[(t, prev_node)] == occupied[(t-1, node[0])]:
                        return True

        return False

    def _communicate(self):
        # Calculate the sum of all node pheromone matrices
        if self.method=='aco':
            all_node_pheromones = [agent.get_node_pheromones() for agent in self.agents]
            pheromone_sum = sum(all_node_pheromones)
            
            # Update each agent with the sum
            for i, agent in enumerate(self.agents):
                agent.other_pheromones = (pheromone_sum - all_node_pheromones[i]) / (self.n_agents - 1)
            
        occupancy = [agent.occupancy_matrix for agent in self.agents]
        occupancy_sum = sum(occupancy)
        for i, agent in enumerate(self.agents):
            agent.other_occupancy = (occupancy_sum - occupancy[i]) / (self.n_agents - 1)

    def solve(self, use_adjacent=False):
        best_solution = None
        best_solution_length = float('inf')

        for global_iteration in tqdm(range(int(self.n_iterations / self.communication_interval)), desc="Global Iterations"):
            all_agent_paths = [agent.run_aco_iterations() for agent in self.agents]
            
            # Find the best solution among all paths
            for paths in zip(*all_agent_paths):
                if not self._check_conflicts(paths):
                    total_length = sum(path[-1][1] for path in paths)
                    if total_length < best_solution_length:
                        best_solution = paths
                        best_solution_length = total_length

            # Communicate between agents at specified intervals
            self._communicate()

        # Generate final greedy solution
        final_solution = self._generate_greedy_solution()

        return final_solution if final_solution else best_solution

    def _generate_greedy_solution(self):
        greedy_paths = []
        for agent in self.agents:
            path = agent.run_episode(self.n_iterations, self.n_iterations, greedy=True)  # Use the last iteration's epsilon
            if path is None:
                return None
            greedy_paths.append(path)

        if not self._check_conflicts(greedy_paths):
            return greedy_paths
        return None
