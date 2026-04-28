#!/bin/bash
#BSUB -J hyperparameter_tuning
#BSUB -o logs/tune_%J.out
#BSUB -e logs/tune_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 48:00
#BSUB -q normal

# Hyperparameter Tuning LSF Submission Script
# This script runs comprehensive hyperparameter tuning for all methods on LSF cluster

echo "=========================================="
echo "Hyperparameter Tuning Job"
echo "Job ID: $LSB_JOBID"
echo "Started: $(date)"
echo "=========================================="

# Load required modules (adjust for your cluster)
module load python/3.9
module load gcc/9.3.0

# Activate virtual environment
source venv_quvine/bin/activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Run tuning with configuration file
python QuVINE/scripts/tune_by_task_with_config.py \
  --config QuVINE/scripts/tuning_config.yaml \
  --network-type all \
  --n-trials 50 \
  --n-graphs 10 \
  --output-dir tuning_results_lsf

echo "=========================================="
echo "Job completed: $(date)"
echo "=========================================="

