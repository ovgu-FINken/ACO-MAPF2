# main.py

import yaml
from datetime import datetime
import os
import logging
import pandas as pd
import numpy as np
from benchmark import all_benchmarks, run_benchmark

def run_all_benchmarks(planner_params: dict):
    all_results = []
    
    for benchmark, benchmark_params in all_benchmarks:
        df = run_benchmark(benchmark, benchmark_params, planner_params)
        all_results.append(df)
    
    combined_results = pd.concat(all_results, ignore_index=True)
    
    # Calculate overall statistics
    overall_success_rate = combined_results['success'].mean()
    overall_avg_path_length = combined_results[combined_results['success']]['mean_path_length'].mean()
    overall_max_path_length = combined_results[combined_results['success']]['longest_path'].mean()
    
    logging.info("\nOverall Results:")
    logging.info(f"Overall Success Rate: {overall_success_rate:.2%}")
    logging.info(f"Overall Average Path Length: {overall_avg_path_length:.2f}")
    logging.info(f"Overall Average Max Path Length: {overall_max_path_length:.2f}")
    
    # Save results to YAML file
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'results/test_run_{date_str}.yaml'
    
    yaml_data = {
        'planner_params': planner_params,
        'overall_results': {
            'success_rate': overall_success_rate,
            'avg_path_length': overall_avg_path_length,
            'max_path_length': overall_max_path_length,
        },
        'detailed_results': combined_results.to_dict(orient='records')
    }
    
    os.makedirs('results', exist_ok=True)
    with open(filename, 'w') as f:
        yaml.dump(yaml_data, f)
    
    logging.info(f"Results saved to {filename}")
    
    return combined_results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Example planner parameters
    planner_params = {
        'n_episodes': 20,
        'n_iterations': 100,
        'alpha': 1,
        'beta': 2,
        'gamma': 1,
        'evaporation_rate': 0.1,
        'communication_interval': 5,
        'initial_epsilon': 0.8
    }
    
    results_df = run_all_benchmarks(planner_params)
    
    print(results_df.groupby('benchmark')['success'].mean())