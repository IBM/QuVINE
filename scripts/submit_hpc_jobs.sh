#!/bin/bash
#BSUB -J quvine_exp[0-799]%100
#BSUB -o logs/job_%J_%I.out
#BSUB -e logs/job_%J_%I.err
#BSUB -W 4:00
#BSUB -M 32GB
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#
# Large-Scale QuVINE Experiment - HPC Job Array Submission Script (LSF)
#
# This script submits an LSF job array to process all graphs in parallel.
# Each job processes one graph through the complete pipeline.
#
# Usage:
#   bsub < scripts/submit_hpc_jobs.sh
#
# Or for testing a single job:
#   bsub -J "quvine_exp[0]" < scripts/submit_hpc_jobs.sh
#
# Check job status:
#   bjobs -w
#   bjobs -l <job_id>
#
# Kill jobs:
#   bkill <job_id>
#   bkill -J "quvine_exp"  # Kill all jobs in array
#
# Configuration:
#   - Adjust [0-799] range based on number of graphs (0 to N-1)
#   - Adjust %100 to control max concurrent jobs
#   - Adjust -W (walltime), -M (memory), -n (cores) based on your needs
#   - Add -q <queue_name> if needed for your cluster
#   - Add -P <project_name> if needed for accounting

# Print job information
echo "=========================================="
echo "Job ID: $LSB_JOBID"
echo "Job Index: $LSB_JOBINDEX"
echo "Job Name: $LSB_JOBNAME"
echo "Host: $HOSTNAME"
echo "Queue: $LSB_QUEUE"
echo "Start time: $(date)"
echo "=========================================="

# Configuration
CONFIG_FILE="configs/large_scale_experiment.yaml"
EXPERIMENT_DIR="experiments/large_scale"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

# Load conda/virtual environment if needed
# Uncomment and modify as needed for your setup:
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate quvine
# OR
# source /path/to/venv/bin/activate

# Load modules if needed (common on HPC systems)
# module load python/3.8
# module load gcc/9.3.0

# Change to project directory
cd "$PROJECT_DIR"

# Print environment info
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Working directory: $(pwd)"
echo "Job index: $LSB_JOBINDEX"
echo ""

# Run the pipeline for this job
echo "Running pipeline for job index $LSB_JOBINDEX..."
python scripts/run_single_graph_pipeline.py \
    --job_id $LSB_JOBINDEX \
    --config "$CONFIG_FILE"

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job $LSB_JOBINDEX completed successfully"
else
    echo "Job $LSB_JOBINDEX failed with exit code $EXIT_CODE"
fi

echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE

# Made with Bob
