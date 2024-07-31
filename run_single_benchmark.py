import sys
import yaml
from benchmark import all_benchmarks
from datetime import datetime
import random
import logging
from pathlib import Path

def run_single_benchmark(benchmark_index):
    benchmark = all_benchmarks[benchmark_index % len(all_benchmarks)]
    print("running benchmark for benchmark index: " + str(benchmark_index))
    print("benchmark: " + str(benchmark))
    random.seed(benchmark_index)
    
    # Example planner parameters (you may want to load these from a config file)
    planner_params = {
        'n_episodes': 20,
        'n_iterations': 200,
        'alpha': 1.0,
        'beta': 0.5,
        'gamma': 1.5,
        'evaporation_rate': 0.2,
        'dispersion_rate': 0.2,
        'communication_interval': 5,
        'collision_weight': 0.5,
        'initial_epsilon': 0.2,
        'method': 'aco',
    }
    
    # Run the benchmark
    solution = benchmark.run(**planner_params)
    
    # Process and save results
    success = solution is not None and all(path[-1] == goal for path, goal in zip(solution, benchmark.goal_positions))
    if success:
        path_lengths = [len(path) - 1 for path in solution]
        avg_path_length = sum(path_lengths) / len(path_lengths)
        max_path_length = max(path_lengths)
    else:
        avg_path_length = max_path_length = float('inf')
    
    results = {
        'benchmark': str(benchmark),
        'benchmark_type': benchmark.name,
        'n_agents': len(benchmark.goal_positions),
        'success': success,
        'longest_path': max_path_length,
        'mean_path_length': avg_path_length,
        'run_number' : benchmark_index // len(all_benchmarks),
        **benchmark.benchmark_params,
        **planner_params
    }
    
    # Save results to YAML file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f'results/benchmark_{benchmark}_{date_str}.yaml'
    Path.mkdir(Path(filename).parent, exist_ok=True, parents=True)
    
    with open(filename, 'w') as f:
        yaml.dump(results, f)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_single_benchmark.py <benchmark_index>")
        sys.exit(1)
    
    benchmark_index = int(sys.argv[1])
    run_single_benchmark(benchmark_index)
