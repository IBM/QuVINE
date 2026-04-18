#!/usr/bin/env python3
"""Test if ripser is available in joblib workers."""
from joblib import Parallel, delayed
import sys

def test_ripser():
    try:
        import ripser
        return f"SUCCESS: Ripser {ripser.__version__} available in worker"
    except ImportError as e:
        return f"FAILED: Ripser not available - {e}"

# Test in main process
print("Main process:")
print(test_ripser())

# Test in joblib workers
print("\nJoblib workers:")
results = Parallel(n_jobs=2, backend="loky")(
    delayed(test_ripser)() for _ in range(2)
)
for i, result in enumerate(results):
    print(f"  Worker {i}: {result}")

# Made with Bob
