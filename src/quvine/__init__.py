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


def __getattr__(name: str):
    """Lazy-load heavy submodules to avoid importing TensorFlow/Keras at import time."""
    if name == "Pipeline":
        from quvine.pipeline import Pipeline
        return Pipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Pipeline",
]
