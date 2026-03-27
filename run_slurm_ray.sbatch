#!/bin/bash
#SBATCH --job-name=flocora-ray
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=52:00:00

set -euo pipefail

mkdir -p logs

module load mamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate dieRoche

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Keep job launcher minimal; defaults are defined in Python.
python main_ray.py

# If you want repeated runs in one allocation, uncomment:
# for i in {1..5}; do
#   echo "starting run $i"
#   python main_ray.py
#   echo "finish run $i"
# done
