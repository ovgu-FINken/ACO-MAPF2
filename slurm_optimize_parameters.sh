#!/bin/bash
#SBATCH --job-name=aco_optimize
#SBATCH --output=output/optimize_%j.out
#SBATCH --error=output/optimize_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=32

# Load any necessary modules here
# module load python/3.8

# Activate your virtual environment if you're using one
source venv/bin/activate

python optimize_parameters.py
