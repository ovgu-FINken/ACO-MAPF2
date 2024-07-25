#!/bin/bash
#SBATCH --job-name=aco_benchmark
#SBATCH --output=output/benchmark_%A_%a.out
#SBATCH --error=output/benchmark_%A_%a.err
#SBATCH --array=0-176
#SBATCH --time=02:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1

# Load any necessary modules here
# module load python/3.8

# Activate your virtual environment if you're using one
source venv/bin/activate

# Run the benchmark
python run_single_benchmark.py $SLURM_ARRAY_TASK_ID
