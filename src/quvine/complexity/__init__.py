"""
Complexity Module for QuVINE

This module provides complexity metrics for graphs, including:
- Graph-based complexity metrics (spectral, quantum, structural)
- QBioCode complexity metrics (when available)

Usage:
    from quvine.complexity.graph import compute_spectral_gap, compute_quantum_complexity
    from quvine.complexity.qbc import compute_intrinsic_dimension
"""

# Import key functions from graph module
from quvine.complexity.graph import (
    compute_laplacian_spectrum,
    compute_spectral_gap,
    compute_algebraic_connectivity,
    compute_spectral_entropy,
    compute_von_neumann_entropy,
    compute_estrada_index,
    compute_quantum_complexity,
    compute_inverse_participation_ratio,
    compute_participation_ratio,
    compute_quantum_advantage_metrics,
    compute_graph_complexity_metrics,
    compare_graph_complexities,
    rank_graphs_by_complexity,
)

# Import QBC functions (if available)
try:
    from quvine.complexity.qbc import (
        check_qbc_available,
        compute_qbc_complexity_from_laplacian,
        compute_qbc_complexity_multimethod,
        compute_comprehensive_complexity,
        compare_qbc_complexity,
        get_qbc_installation_instructions,
    )
    QBC_AVAILABLE = True
except ImportError:
    QBC_AVAILABLE = False
    check_qbc_available = lambda: False
    compute_qbc_complexity_from_laplacian = None
    compute_qbc_complexity_multimethod = None
    compute_comprehensive_complexity = None
    compare_qbc_complexity = None
    get_qbc_installation_instructions = lambda: "QBioCode not available"

__all__ = [
    # Graph complexity functions
    'compute_laplacian_spectrum',
    'compute_spectral_gap',
    'compute_algebraic_connectivity',
    'compute_spectral_entropy',
    'compute_von_neumann_entropy',
    'compute_estrada_index',
    'compute_quantum_complexity',
    'compute_inverse_participation_ratio',
    'compute_participation_ratio',
    'compute_quantum_advantage_metrics',
    'compute_graph_complexity_metrics',
    'compare_graph_complexities',
    'rank_graphs_by_complexity',
    # QBC functions
    'check_qbc_available',
    'compute_qbc_complexity_from_laplacian',
    'compute_qbc_complexity_multimethod',
    'compute_comprehensive_complexity',
    'compare_qbc_complexity',
    'get_qbc_installation_instructions',
    'QBC_AVAILABLE',
]

