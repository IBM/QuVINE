#!/bin/bash
# Quick test runner for QuVINE test suite

set -e

echo "================================"
echo "QuVINE Test Suite Runner"
echo "================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest not found. Install with: pip install pytest"
    exit 1
fi

# Parse arguments
RUN_SLOW=false
COVERAGE=false
VERBOSE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --slow)
            RUN_SLOW=true
            shift
            ;;
        --coverage)
            COVERAGE=true
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--slow] [--coverage] [--quiet]"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest tests/"

if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

if [ "$RUN_SLOW" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not slow'"
    echo "Running fast tests only (use --slow to include neural network tests)"
else
    echo "Running all tests (including slow neural network tests)"
fi

if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=quvine --cov-report=term --cov-report=html"
    echo "Coverage reporting enabled"
fi

echo ""
echo "Command: $PYTEST_CMD"
echo ""
echo "================================"
echo ""

# Run tests
$PYTEST_CMD

# Print summary
echo ""
echo "================================"
echo "Test Summary"
echo "================================"

if [ "$COVERAGE" = true ]; then
    echo "Coverage report generated in htmlcov/index.html"
fi

echo ""
echo "✓ Tests completed successfully!"

# Made with Bob
