#!/bin/bash
################################################################################
# Quick Test Job Submission Script
#
# Purpose: Validate the entire tuning pipeline with minimal compute time
# 
# Test Coverage:
#   - Synthetic Networks: 2 networks × 12 methods = 24 jobs
#   - PPI Networks: 2 networks × 1 disease × 12 methods = 24 jobs
#   - Total: 48 tuning jobs + 2 aggregation jobs = 50 jobs
#
# Configuration:
#   - n_trials: 5 (instead of 20-50)
#   - Expected runtime: 15-30 minutes per job
#   - Total time: ~30-60 minutes (parallelized)
#
# Usage:
#   bash scripts/submit_quick_test.sh [--dry-run] [--use-gpu]
#
# What This Tests:
#   ✓ All 12 methods run successfully
#   ✓ Separate output files created per method × network
#   ✓ Aggregation jobs collect results correctly
#   ✓ No crashes or configuration errors
#
################################################################################

set -e

# Parse arguments
DRY_RUN=""
GPU_FLAG=""
for arg in "$@"; do
    case $arg in
        --dry-run) DRY_RUN="--dry-run"; echo "DRY RUN MODE - No jobs will be submitted"; echo "" ;;
        --use-gpu) GPU_FLAG="--use-gpu"; echo "GPU MODE - DNN jobs will request GPU nodes"; echo "" ;;
    esac
done

# Configuration
TEST_CONFIG="scripts/test_tuning_config.yaml"
QUEUE="normal"
WALLTIME="2:00"  # 2 hours should be plenty for n_trials=5
MEMORY="16"      # 16GB should be sufficient

echo "======================================================"
echo " Quick Test Job Submission"
echo "======================================================"
echo " Config: $TEST_CONFIG"
echo " Queue: $QUEUE"
echo " Wall time: $WALLTIME"
echo " Memory: ${MEMORY}GB"
echo " n_trials: 5 (per method)"
echo ""
echo " Test Coverage:"
echo "   - Synthetic: 2 networks × 12 methods = 24 jobs"
echo "   - PPI: 2 networks × 1 disease × 12 methods = 24 jobs"
echo "   - Total: 48 tuning + 2 aggregation = 50 jobs"
echo ""
echo " Expected Runtime: ~30-60 minutes (parallelized)"
echo "======================================================"
echo ""

# Verify config exists
if [ ! -f "$TEST_CONFIG" ]; then
    echo "ERROR: Test config not found: $TEST_CONFIG"
    exit 1
fi

# Submit synthetic network jobs
echo "Submitting synthetic network jobs (24 jobs)..."
bash scripts/submit_tuning_jobs.sh \
    --config "$TEST_CONFIG" \
    --networks erdos_renyi,modular_strong \
    --queue "$QUEUE" \
    --walltime "$WALLTIME" \
    --memory "$MEMORY" \
    --n-graphs 5 \
    $GPU_FLAG \
    $DRY_RUN

echo ""
echo "======================================================"
echo ""

# Submit PPI network jobs
echo "Submitting PPI network jobs (24 jobs)..."
bash scripts/submit_ppi_tuning_jobs.sh \
    --config "$TEST_CONFIG" \
    --networks STRING,BioPlex3 \
    --diseases asthma \
    --queue "$QUEUE" \
    --walltime "$WALLTIME" \
    --memory "$MEMORY" \
    --n-replicates 2 \
    $GPU_FLAG \
    $DRY_RUN

echo ""
echo "======================================================"
echo " Quick Test Submission Complete"
echo "======================================================"
echo ""
echo "Monitor jobs:"
echo "  bjobs"
echo "  bjobs -u \$USER -w"
echo ""
echo "Check logs:"
echo "  tail -f tuning_by_task/logs/tune_quvine_rwr_erdos_renyi.out"
echo "  tail -f ppi_tuning_by_task/logs/ppi_tune_quvine_rwr_STRING_asthma.out"
echo ""
echo "Expected outputs:"
echo "  Synthetic: tuning_by_task/NETWORK_METHOD_tuning_by_task.json"
echo "  PPI: ppi_tuning_by_task/NETWORK_DISEASE_METHOD_tuning_by_task.json"
echo ""
echo "Aggregated results:"
echo "  tuning_by_task/erdos_renyi_tuning_by_task.json"
echo "  tuning_by_task/modular_strong_tuning_by_task.json"
echo "  ppi_tuning_by_task/STRING_asthma_tuning_by_task.json"
echo "  ppi_tuning_by_task/BioPlex3_asthma_tuning_by_task.json"
echo ""
echo "======================================================"

# Made with Bob
