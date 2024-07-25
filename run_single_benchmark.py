import sys
import yaml
from benchmark import all_benchmarks
from ACOMultiAgentPathfinder import ACOMultiAgentPathfinder
from datetime import datetime
import random
import logging

def run_single_benchmark(benchmark_index):
    benchmark, benchmark_params = all_benchmarks[benchmark_index % len(all_benchmarks)]
    print("running benchmark for benchmark index: " + str(benchmark_index))
    print("benchmark name: " + benchmark.name)
    random.seed(benchmark_index)
    
    # Example planner parameters (you may want to load these from a config file)
    planner_params = {
        'n_ants': 20,
        'n_iterations': 200,
        'alpha': 1.5,
        'beta': 0.3,
        'gamma': 2.5,
        'evaporation_rate': 0.5,
        'dispersion_rate': 0.2,
        'communication_interval': 3,
        'collision_weight': 0.7,
        'initial_epsilon': 0.85,
    }
    
    # Run the benchmark
    solution, G = benchmark.run(benchmark_params, planner_params)
    
    # Process and save results
    success = solution is not None
    if success:
        path_lengths = [len(path) - 1 for path in solution]
        avg_path_length = sum(path_lengths) / len(path_lengths)
        max_path_length = max(path_lengths)
    else:
        avg_path_length = max_path_length = float('inf')
    
    results = {
        'benchmark': benchmark.name,
        'n_agents': len(G.nodes()),
        'success': success,
        'longest_path': max_path_length,
        'mean_path_length': avg_path_length,
        'run_number' : benchmark_index // len(all_benchmarks),
        **benchmark_params,
        **planner_params
    }
    
    # Save results to YAML file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/benchmark_{benchmark.name}_{date_str}.yaml'
    
    with open(filename, 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single_benchmark.py <benchmark_index>")
        sys.exit(1)
    
    benchmark_index = int(sys.argv[1])
    run_single_benchmark(benchmark_index)
