import networkx as nx
import random
import numpy as np
from scipy import sparse
from tqdm import tqdm

class Agent:
    def __init__(self, agent_id, start_position, goal_position, G, nodelist, time_horizon, n_agents, alpha=1, beta=2, gamma=4, evaporation_rate=0.1, dispersion_rate=0.1, initial_epsilon=0.8):
        self.agent_id = agent_id
        self.start_position = start_position
        self.goal_position = goal_position
        self.time_horizon = time_horizon
        self.G = G
        self.nodelist = nodelist
        self.G_t = self._create_time_expanded_graph()
        self.n_agents = n_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.evaporation_rate = evaporation_rate
        self.dispersion_rate = dispersion_rate
        self.initial_epsilon = initial_epsilon
        self._initialize_pheromone()
        self.all_node_pheromones_sum = None  # Store sum of all node pheromone matrices

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

    def get_node_pheromones(self):
        node_pheromones = np.zeros((len(self.nodelist), self.time_horizon))
        for (u, v, pheromone) in self.G_t.edges(data='pheromone'):
            node_pheromones[self.nodelist.index(v[0]), v[1]] += pheromone
        return node_pheromones

    def update_pheromone(self, paths):
        # Evaporation
        for u, v in self.G_t.edges():
            self.G_t[u][v]['pheromone'] *= (1 - self.evaporation_rate)

        # Remove duplicate paths
        unique_paths = list(set(tuple(path) for path in paths))
        
        # Calculate path lengths
        path_lengths = [len(path) - 1 for path in unique_paths]
        
        if path_lengths:
            max_length = max(path_lengths)
            min_length = min(path_lengths)
            
            # Normalize path lengths to [0, 1] range, where shorter paths have higher values
            normalized_lengths = [(max_length - length) / (max_length - min_length) if max_length != min_length else 1 for length in path_lengths]
            
            # Update pheromones
            for path, norm_length in zip(unique_paths, normalized_lengths):
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    self.G_t[u][v]['pheromone'] += norm_length

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

    def epsilon_greedy_decision(self, current, neighbors, goal, epsilon, use_adjacent=False):
        if random.random() < epsilon:
            # Explore: choose a random neighbor
            return random.choice(neighbors)
        else:
            # Exploit: choose the best neighbor based on probabilities
            if use_adjacent:
                probabilities = self.calculate_probabilities_with_adjacent(current, neighbors, goal)
            else:
                probabilities = self.calculate_probabilities(current, neighbors, goal)
            return neighbors[np.argmax(probabilities)]

    def ant_tour(self, iteration, max_iterations, use_adjacent=False):
        current = (self.start_position, 0)
        path = [current]
        
        # Calculate dynamic epsilon
        epsilon = self.initial_epsilon * (1 - iteration / max_iterations)

        while current[0] != self.goal_position:
            neighbors = list(self.G_t.neighbors(current))
            if not neighbors:
                return None  # No valid path

            next_node = self.epsilon_greedy_decision(current, neighbors, self.goal_position, epsilon, use_adjacent)
            path.append(next_node)
            current = next_node

        return path

class ACOMultiAgentPathfinder:
    def __init__(self, graph, start_positions, goal_positions, n_ants=10, n_iterations=100, alpha=1, beta=2, gamma=4, evaporation_rate=0.1, dispersion_rate=0.1, communication_interval=10, initial_epsilon=1.0):
        self.G = graph
        self.nodelist = list(self.G.nodes())
        self.n_agents = len(start_positions)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.communication_interval = communication_interval

        self.time_horizon = len(self.G.nodes()) * 2

        self.agents = [Agent(i, start_positions[i], goal_positions[i], self.G, self.nodelist, self.time_horizon, self.n_agents, alpha, beta, gamma, evaporation_rate, dispersion_rate, initial_epsilon)
                       for i in range(self.n_agents)]
        self._communicate()


    def _check_conflicts(self, paths):
        occupied = {}
        for agent, path in enumerate(paths):
            for t, node in enumerate(path):
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
        all_node_pheromones = [agent.get_node_pheromones() for agent in self.agents]
        pheromone_sum = sum(all_node_pheromones)
        
        # Update each agent with the sum
        for i, agent in enumerate(self.agents):
            agent.other_pheromones = (pheromone_sum - all_node_pheromones[i]) / (self.n_agents - 1)


    def solve(self, use_adjacent=False):
        best_solution = None
        best_solution_length = float('inf')

        for iteration in tqdm(range(self.n_iterations), desc="ACO Iterations"):
            # Generate paths for all ants
            all_ant_paths = []
            for _ in range(self.n_ants):
                ant_paths = []
                for agent in self.agents:
                    path = agent.ant_tour(iteration, self.n_iterations, use_adjacent)
                    if path is None:
                        break
                    ant_paths.append(path)
                if len(ant_paths) == self.n_agents:
                    all_ant_paths.append(ant_paths)

            # If no valid solutions found, skip to next iteration
            if not all_ant_paths:
                continue

            # Find the best solution among all ants
            for ant_paths in all_ant_paths:
                if not self._check_conflicts(ant_paths):
                    total_length = sum(len(path) - 1 for path in ant_paths)
                    if total_length < best_solution_length:
                        best_solution = ant_paths
                        best_solution_length = total_length

            # Update pheromones based on all valid paths found in this iteration
            for agent_index, agent in enumerate(self.agents):
                agent_paths = [ant_paths[agent_index] for ant_paths in all_ant_paths]
                agent.update_pheromone(agent_paths)

            # Communicate between agents at specified intervals
            if iteration % self.communication_interval == 0:
                self._communicate()

        # Generate final greedy solution
        final_solution = self._generate_greedy_solution()

        return final_solution if final_solution else best_solution

    def _generate_greedy_solution(self):
        greedy_paths = []
        for agent in self.agents:
            path = agent.ant_tour(self.n_iterations, self.n_iterations, use_adjacent=False)  # Use the last iteration's epsilon
            if path is None:
                return None
            greedy_paths.append(path)

        if not self._check_conflicts(greedy_paths):
            return greedy_paths
        return None