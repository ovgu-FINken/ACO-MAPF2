# collect_results.py
import glob
import yaml
import pandas as pd


def collect_results():
    result_files = glob.glob('results/benchmark_*.yaml')
    all_results = []

    for file in result_files:
        with open(file, 'r') as f:
            results = yaml.safe_load(f)
            all_results.append(results)

    df = pd.DataFrame(all_results)

    # Group by benchmark and calculate statistics
    grouped = df.groupby('benchmark')
    stats = grouped.agg({
        'success': ['mean', 'std'],
        'mean_path_length': ['mean', 'std'],
        'longest_path': ['mean', 'std']
    })

    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns.values]

    print("Benchmark Statistics:")
    print(stats)

    # Calculate overall statistics
    overall_success_rate = df['success'].mean()
    overall_avg_path_length = df[df['success']]['mean_path_length'].mean()
    overall_max_path_length = df[df['success']]['longest_path'].mean()

    print("\nOverall Statistics:")
    print(f"Overall Success Rate: {overall_success_rate:.2%}")
    print(f"Overall Average Path Length: {overall_avg_path_length:.2f}")
    print(f"Overall Average Max Path Length: {overall_max_path_length:.2f}")

    # Save detailed results to CSV
    df.to_csv('benchmark_results_detailed.csv', index=False)
    stats.to_csv('benchmark_results_summary.csv')
    print("Detailed results saved to benchmark_results_detailed.csv")
    print("Summary statistics saved to benchmark_results_summary.csv")

if __name__ == "__main__":
    collect_results()

