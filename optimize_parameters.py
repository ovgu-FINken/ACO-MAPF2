# optimize_parameters.py
import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.acquisition import gaussian_ei
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import Matern
from benchmark import all_benchmarks
from ACOMultiAgentPathfinder import ACOMultiAgentPathfinder
from datetime import datetime
import multiprocessing
import os
import logging
from yaml_utils import save_results, load_results

# Define the parameter space
space = [
    #Real(0.0, 1.0, name='alpha'),
    #Real(0.0, 10.0, name='beta'),
    Real(0.0, 1.0, name='gamma'),
    #Real(0.0, 0.3, name='evaporation_rate'),
    #Real(0.0, 0.3, name='dispersion_rate'),
    Real(0.3, 1.0, name='initial_epsilon'),
    Real(0.0, 1.0, name='collision_weight'),
]

def run_benchmark(args):
    benchmark, solver_params, seed = args
    np.random.seed(seed)
    solution = benchmark.run(**solver_params)
    result = benchmark.evaluate(solution)
    return result['success'], result['mean_path_length']

# Global variables to keep track of the optimization process
best_observed_params = None
best_observed_score = np.inf
iteration = 0
X = []
y = []

@use_named_args(space)
def objective(**params):
    global best_observed_params, best_observed_score, iteration, X, y
    iteration += 1
    
    planner_params = {
        'n_episodes': 20,
        'n_iterations': 200,
        'alpha': 0.5,
        'beta': 2.3,
        'gamma': 1,
        'evaporation_rate': 0.1,
        'communication_interval': 1,
        'initial_epsilon': 0.8,
        'method': 'q-learning',
    }
    all_args = []
    for benchmark in all_benchmarks:
        for _ in range(11):  # 3 runs per benchmark
            seed = np.random.randint(0, 10000)
            all_args.append((benchmark, planner_params | params, seed))
    
    with multiprocessing.Pool() as pool:
        results = pool.map(run_benchmark, all_args)
    
    successes, path_lengths = zip(*results)
    success_rate = np.mean(successes)
    avg_path_length = np.mean([l for l in path_lengths if l < np.inf])
    
    # Compute the objective score (to be minimized)
    score = -success_rate * 100 - (1 / (avg_path_length + 1))
    if np.isnan(score):
        score = 0
    
    # Update best observed results if necessary
    if score < best_observed_score:
        best_observed_score = score
        best_observed_params = params
    
    # Store the parameters and score for model fitting
    X.append(list(params.values()))
    y.append(score)
    
    print(f"\nIteration {iteration}:")
    print(f"Success Rate: {success_rate:.2%}")
    print(f"Average Path Length: {avg_path_length:.2f}")
    print(f"Score: {score:.2f}")
    
    return score

def report_model_predictions(model, space):
    # Generate a grid of points to evaluate
    param_names = [dim.name for dim in space]
    n_samples = 10000
    X_samples = np.array([np.random.uniform(dim.low, dim.high, n_samples) for dim in space]).T
    
    # Predict mean and std for these points
    y_mean, y_std = model.predict(X_samples, return_std=True)
    
    # Find the point with the best predicted mean
    best_idx = np.argmin(y_mean)
    best_predicted_params = dict(zip(param_names, X_samples[best_idx]))
    best_predicted_score = y_mean[best_idx]
    best_predicted_uncertainty = y_std[best_idx]
    
    print("\nModel Predictions:")
    print(f"Best predicted parameters:")
    for param, value in best_predicted_params.items():
        print(f"  {param}: {value:.4f}")
    print(f"Predicted score: {best_predicted_score:.4f} ± {best_predicted_uncertainty:.4f}")
    
    # Compute Expected Improvement
    best_observed = np.min(y)
    ei = gaussian_ei(X_samples, model, y_opt=best_observed)
    best_ei_idx = np.argmax(ei)
    best_ei_params = dict(zip(param_names, X_samples[best_ei_idx]))
    
    #print("\nPoint with highest Expected Improvement:")
    #for param, value in best_ei_params.items():
    #    print(f"  {param}: {value:.4f}")
    #print(f"EI value: {ei[best_ei_idx]:.4f}")
    
    # Model fit quality
    y_pred = model.predict(X)
    r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
    print(f"\nModel R² score: {r2:.4f}")
    
    return {
        'best_predicted_params': best_predicted_params,
        'best_predicted_score': float(best_predicted_score),
        'best_predicted_uncertainty': float(best_predicted_uncertainty),
        'best_ei_params': best_ei_params,
        'best_ei_value': float(ei[best_ei_idx]),
        'model_r2_score': float(r2)
    }

def optimize_parameters(n_calls=50):
    global best_observed_params, best_observed_score, iteration, X, y
    
    # Create a directory for results
    os.makedirs("optimization_results", exist_ok=True)
    kernel = Matern(nu=2.5)    
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=5)
    
    result = gp_minimize(objective,
                         space,
                         n_calls=n_calls,
                         n_initial_points=10,
                         base_estimator=gpr,
                         acq_func="EI")
    
    print("\nOptimization completed.")
    print("Best observed parameters:")
    for param, value in best_observed_params.items():
        print(f"{param}: {value}")
    print(f"Best observed score: {best_observed_score:.4f}")
    
    # Report model predictions
    model_predictions = report_model_predictions(result.models[-1], space)
    
    # Save final results
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'optimization_results/final_results_{date_str}.yaml'
    
    results = {
        'best_observed_params': best_observed_params,
        'best_observed_score': float(best_observed_score),
        'optimization_iterations': iteration,
        'space': space,
        'model_predictions': model_predictions
    }
    
    save_results(filename, results)
    print(f"\nFinal results saved to {filename}")

if __name__ == "__main__":
    optimize_parameters()
