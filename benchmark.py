# benchmark.py

import networkx as nx
from itertools import cycle
from ACOMultiAgentPathfinder import ACOMultiAgentPathfinder
import logging

import networkx as nx
from itertools import cycle
from ACOMultiAgentPathfinder import ACOMultiAgentPathfinder
import logging
import pandas as pd
from tqdm import tqdm

class Benchmark:
    def __init__(self, name, graph_generator, start_goal_generator):
        self.name = name
        self.graph_generator = graph_generator
        self.start_goal_generator = start_goal_generator

    def run(self, benchmark_params: dict, planner_params: dict):
        G = self.graph_generator(**benchmark_params)
        start_positions, goal_positions = self.start_goal_generator(G, **benchmark_params)
        
        logging.info(f"\nRunning {self.name}:")
        logging.info(f"Graph size: {len(G.nodes())}")
        logging.info(f"Number of agents: {len(start_positions)}")
        for i, (start, goal) in enumerate(zip(start_positions, goal_positions)):
            logging.info(f"Agent {i+1}: {start} -> {goal}")

        solver = ACOMultiAgentPathfinder(G, start_positions, goal_positions, **planner_params)
        solution = solver.solve()

        return solution, G

def run_benchmark(benchmark, benchmark_params: dict, planner_params: dict, n_trials=3):
    results = []
    for trial in tqdm(range(n_trials), desc=f"Running {benchmark.name}"):
        solution, G = benchmark.run(benchmark_params, planner_params)
        success = solution is not None
        if success:
            path_lengths = [len(path) - 1 for path in solution]
            avg_path_length = sum(path_lengths) / len(path_lengths)
            max_path_length = max(path_lengths)
        else:
            avg_path_length = max_path_length = float('inf')
        
        results.append({
            'benchmark': benchmark.name,
            'trial': trial,
            'n_agents': len(G.nodes()),
            'success': success,
            'longest_path': max_path_length,
            'mean_path_length': avg_path_length,
            **benchmark_params,
            **planner_params
        })
    
    df = pd.DataFrame(results)
    
    logging.info(f"\nResults for {benchmark.name}:")
    logging.info(f"Success rate: {df['success'].mean():.2%}")
    logging.info(f"Average path length: {df[df['success']]['mean_path_length'].mean():.2f}")
    logging.info(f"Average max path length: {df[df['success']]['longest_path'].mean():.2f}")
    
    return df


def grid_graph_generator(width, height, **kwargs):
    return nx.grid_2d_graph(width, height)

def grid_start_goal_generator(G, n_agents, **kwargs):
    width = max(node[0] for node in G.nodes()) + 1
    height = max(node[1] for node in G.nodes()) + 1
    
    left_positions = [(0, y) for y in range(height)]
    right_positions = [(width - 1, y) for y in range(height)]
    
    max_agents = 2 * height
    n_agents = min(n_agents, max_agents)
    
    left_cycle = cycle(left_positions)
    right_cycle = cycle(right_positions)
    
    start_positions = []
    goal_positions = []
    for i in range(n_agents):
        if i % 2 == 0:
            start_positions.append(next(left_cycle))
            goal_positions.append(next(right_cycle))
        else:
            start_positions.append(next(right_cycle))
            goal_positions.append(next(left_cycle))
    
    return start_positions, goal_positions

def passage_graph_generator(width, height, passage_length, **kwargs):
    G = nx.Graph()
    for x in range(width):
        for y in range(height):
            if x < (width - passage_length) // 2 or x >= (width + passage_length) // 2 or y == height // 2:
                G.add_node((x, y))
    
    for (x1, y1) in G.nodes():
        for (x2, y2) in G.nodes():
            if abs(x1 - x2) + abs(y1 - y2) == 1:
                G.add_edge((x1, y1), (x2, y2))
    
    return G

def passage_start_goal_generator(G, **kwargs):
    width = max(node[0] for node in G.nodes()) + 1
    height = max(node[1] for node in G.nodes()) + 1
    start_positions = [(0, height // 2), (width - 1, height // 2)]
    goal_positions = [(width - 1, height // 2), (0, height // 2)]
    return start_positions, goal_positions

def star_graph_generator(n_branches, branch_length, **kwargs):
    G = nx.Graph()
    G.add_node('center')
    for i in range(n_branches):
        for j in range(1, branch_length + 1):
            G.add_edge(f'branch_{i}_{j-1}', f'branch_{i}_{j}')
        G.add_edge('center', f'branch_{i}_0')
    return G

def star_start_goal_generator(G, n_branches, branch_length, empty_branch=False, **kwargs):
    start_positions = [f'branch_{i}_{branch_length}' for i in range(n_branches-1)]
    if empty_branch:
        # all go to the empty branch which is branch n_branches-1
        goal_positions = [f'branch_{n_branches-1}' for _ in range(n_branches-1)]
    else:
        # go to the next branch, leave the empty branch empty
        goal_positions = [f'branch_{(i+1)%(n_branches-1)}' for i in range(n_branches-1)]
    return start_positions, goal_positions

def linear_graph_generator(length, **kwargs):
    return nx.path_graph(length)

def linear_start_goal_generator(G, n_agents, **kwargs):
    length = len(G)
    start_positions = [ (i + 1) // 2 if i % 2 == 0 else length - i // 2 for i in range(n_agents)]
    goal_positions = [ (i + 1) // 2 if i % 2 != 0 else length - i // 2 for i in range(n_agents)]
    
    return start_positions, goal_positions




# Define benchmark instances
grid_benchmark = Benchmark("Grid", grid_graph_generator, grid_start_goal_generator)
passage_benchmark = Benchmark("Passage", passage_graph_generator, passage_start_goal_generator)
star_benchmark = Benchmark("Star", star_graph_generator, star_start_goal_generator)
linear_benchmark = Benchmark("Linear", linear_graph_generator, linear_start_goal_generator)

# List of all benchmarks
all_benchmarks = [
    (linear_benchmark, {'length': 4, 'n_agents': 2}),
    (linear_benchmark, {'length': 5, 'n_agents': 2}),
    (linear_benchmark, {'length': 10, 'n_agents': 2}),
    (star_benchmark, {'n_branches': 3, 'branch_length': 1, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 4, 'branch_length': 1, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 5, 'branch_length': 1, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 3, 'branch_length': 2, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 4, 'branch_length': 2, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 5, 'branch_length': 2, 'empty_branch': False}),
    (star_benchmark, {'n_branches': 3, 'branch_length': 1, 'empty_branch': True}),
    (star_benchmark, {'n_branches': 4, 'branch_length': 1, 'empty_branch': True}),
    (star_benchmark, {'n_branches': 5, 'branch_length': 1, 'empty_branch': True}),
    (star_benchmark, {'n_branches': 3, 'branch_length': 2, 'empty_branch': True}),
    (star_benchmark, {'n_branches': 4, 'branch_length': 2, 'empty_branch': True}),
    (star_benchmark, {'n_branches': 5, 'branch_length': 2, 'empty_branch': True}),
    (grid_benchmark, {'width': 3, 'height': 3, 'n_agents': 6}),
    (grid_benchmark, {'width': 3, 'height': 6, 'n_agents': 6}),
    (grid_benchmark, {'width': 4, 'height': 4, 'n_agents': 8}),
    (grid_benchmark, {'width': 4, 'height': 3, 'n_agents': 8}),
    (grid_benchmark, {'width': 5, 'height': 5, 'n_agents': 10}),
    (grid_benchmark, {'width': 5, 'height': 3, 'n_agents': 10}),
    (passage_benchmark, {'width': 10, 'height': 5, 'passage_length': 2}),
    (passage_benchmark, {'width': 10, 'height': 5, 'passage_length': 2}),
]