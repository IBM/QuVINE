#!/bin/bash
#BSUB -J topo_complexity
#BSUB -o logs/topo_complexity_%J.out
#BSUB -e logs/topo_complexity_%J.err
#BSUB -n 32
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 12:00
#BSUB -q normal

# LSF submission script for computing topological complexity metrics
# on all 560 graphs in ppi_disease_v3 and merging into comprehensive results
#
# Usage: bsub < submit_topological_complexity.sh

echo "=========================================================================="
echo "Topological Complexity Computation and Integration"
echo "=========================================================================="
echo "Job started at: $(date)"
echo "Running on host: $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "Number of cores: $LSB_DJOB_NUMPROC"
echo ""

# Set up environment
cd /dccstor/boseukb/Q/NetMed/QuVINE || exit 1

# Activate conda/virtual environment if needed
# source /path/to/conda/etc/profile.d/conda.sh
# conda activate quvine

# Or use module system
# module load python/3.9
# module load gcc/9.3.0

# Install ripser if not already installed
echo "Checking ripser installation..."
pip install ripser --quiet || echo "Ripser already installed"
echo ""

# ============================================================================
# STEP 1: Compute topological complexity metrics
# ============================================================================
echo "=========================================================================="
echo "STEP 1: Computing topological complexity metrics for 560 graphs"
echo "=========================================================================="
echo "Using 32 parallel workers..."
echo ""

python scripts/add_topological_complexity_ppi_complete.py --n-jobs 32

if [ $? -ne 0 ]; then
    echo "ERROR: Topological complexity computation failed!"
    exit 1
fi

echo ""
echo "Topological complexity computation completed successfully!"
echo ""

# ============================================================================
# STEP 2: Merge topological metrics into comprehensive results
# ============================================================================
echo "=========================================================================="
echo "STEP 2: Merging topological metrics into comprehensive results CSV"
echo "=========================================================================="
echo ""

python scripts/merge_topological_to_comprehensive.py

if [ $? -ne 0 ]; then
    echo "ERROR: Merge failed!"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "Job completed successfully at: $(date)"
echo "=========================================================================="
echo ""
echo "Output files:"
echo "  - Checkpoint: /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/topological_checkpoint.csv"
echo "  - Updated complexity files: /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/*/[network_id]_complexity.csv"
echo "  - Comprehensive results: /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/comprehensive_results_ppi3_with_topology.csv"
echo ""
