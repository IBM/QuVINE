"""
quvine

Quantum and classical network embedding framework for
target discovery, ranking, and module recovery.

Core features:
- Multi-view graph construction
- Classical and quantum random walks
- SGNS-based embedding learning
- Iterative, reproducible evaluation pipelines
"""

__version__ = "0.1.0"

# Optional: expose high-level entry points
from quvine.pipeline import Pipeline

__all__ = [
    "Pipeline",
]
