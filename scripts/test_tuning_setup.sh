#!/bin/bash
################################################################################
# Test Hyperparameter Tuning Setup
#
# This script runs a minimal test to verify the tuning infrastructure works
# before running full experiments.
#
# Usage:
#   bash scripts/test_tuning_setup.sh
################################################################################

set -e

echo "============================================================"
echo " Testing Hyperparameter Tuning Setup"
echo "============================================================"
echo ""

# Check if we're in the QuVINE directory
if [ ! -f "scripts/tune_local_test.py" ]; then
    echo "Error: Must run from QuVINE directory"
    exit 1
fi

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python not found"
    exit 1
fi

echo "Using: $($PYTHON --version)"
echo ""

# Check required packages
echo "Checking required packages..."
$PYTHON -c "import networkx; print('✓ networkx')" || { echo "✗ networkx missing"; exit 1; }
$PYTHON -c "import numpy; print('✓ numpy')" || { echo "✗ numpy missing"; exit 1; }
$PYTHON -c "import pandas; print('✓ pandas')" || { echo "✗ pandas missing"; exit 1; }
$PYTHON -c "import sklearn; print('✓ sklearn')" || { echo "✗ sklearn missing"; exit 1; }

# Check optional packages
$PYTHON -c "import optuna; print('✓ optuna (TPE sampler available)')" 2>/dev/null || echo "⚠ optuna not installed (will use random search)"
echo ""

# Create test output directory
TEST_DIR="./tuning_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"
echo "Test output directory: $TEST_DIR"
echo ""

# Run minimal test
echo "============================================================"
echo " Running Minimal Test"
echo "============================================================"
echo "Network: erdos_renyi"
echo "Methods: quvine_walks, node2vec"
echo "Trials: 3"
echo "Graphs: 2"
echo "Nodes: 50"
echo ""

$PYTHON scripts/tune_local_test.py \
    --network-type erdos_renyi \
    --methods quvine_walks node2vec \
    --n-trials 3 \
    --n-graphs 2 \
    --n-nodes 50 \
    --output-dir "$TEST_DIR" \
    2>&1 | tee "$TEST_DIR/test.log"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo " ✓ TEST PASSED"
    echo "============================================================"
    echo ""
    echo "Results saved to: $TEST_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review results: cat $TEST_DIR/erdos_renyi_tuning_results.json"
    echo "  2. Run full test: python scripts/tune_local_test.py --help"
    echo "  3. See README: cat scripts/README_HYPERPARAMETER_TUNING.md"
else
    echo " ✗ TEST FAILED"
    echo "============================================================"
    echo ""
    echo "Check the log: cat $TEST_DIR/test.log"
    echo ""
    echo "Common issues:"
    echo "  - Missing dependencies: pip install -r requirements.txt"
    echo "  - Wrong directory: cd to QuVINE root"
    echo "  - Import errors: check sys.path in script"
fi
echo ""

exit $EXIT_CODE

