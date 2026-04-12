"""
QuVINE Data Module

This module provides data loading, preparation, random graph generation,
and graph complexity analysis utilities (including QBioCode integration).
"""

from .data_loader import load_graph, load_gwas_data, load_pegasus_results, load_seeds_and_targets
from .prepare import prepare_graph, PrepareGraphConfig, keep_largest_connected_component
from .random_graphs import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_powerlaw_cluster,
    generate_stochastic_block_model,
    generate_random_geometric,
    generate_modular_network,
    generate_hierarchical_network,
    generate_core_periphery,
    generate_bipartite_random,
    add_hub_nodes,
    generate_graph_with_seeds_and_targets,
    get_graph_statistics,
)
# Import from new complexity module location
from quvine.complexity.graph import (
    compute_laplacian_spectrum,
    compute_spectral_gap,
    compute_algebraic_connectivity,
    compute_spectral_entropy,
    compute_von_neumann_entropy,
    compute_estrada_index,
    compute_quantum_complexity,
    fiedler_eigenvalue_sparse,
    compute_laplacian_centrality_complexity,
    compute_graph_complexity_metrics,
    compare_graph_complexities,
    rank_graphs_by_complexity,
)
from quvine.complexity.qbc import (
    compute_qbc_complexity_from_laplacian,
    compute_qbc_complexity_multimethod,
    compute_comprehensive_complexity,
    compare_qbc_complexity,
    check_qbc_available,
    get_qbc_installation_instructions,
)

__all__ = [
    # Data loading
    'load_graph',
    'load_gwas_data',
    'load_pegasus_results',
    'load_seeds_and_targets',
    # Graph preparation
    'prepare_graph',
    'PrepareGraphConfig',
    'keep_largest_connected_component',
    # Random graph generators
    'generate_erdos_renyi',
    'generate_barabasi_albert',
    'generate_watts_strogatz',
    'generate_powerlaw_cluster',
    'generate_stochastic_block_model',
    'generate_random_geometric',
    'generate_modular_network',
    'generate_hierarchical_network',
    'generate_core_periphery',
    'generate_bipartite_random',
    'add_hub_nodes',
    'generate_graph_with_seeds_and_targets',
    'get_graph_statistics',
    # Graph complexity metrics
    'compute_laplacian_spectrum',
    'compute_spectral_gap',
    'compute_algebraic_connectivity',
    'compute_spectral_entropy',
    'compute_von_neumann_entropy',
    'compute_estrada_index',
    'compute_quantum_complexity',
    'fiedler_eigenvalue_sparse',
    'compute_laplacian_centrality_complexity',
    'compute_graph_complexity_metrics',
    'compare_graph_complexities',
    'rank_graphs_by_complexity',
    # QBioCode complexity integration
    'compute_qbc_complexity_from_laplacian',
    'compute_qbc_complexity_multimethod',
    'compute_comprehensive_complexity',
    'compare_qbc_complexity',
    'check_qbc_available',
    'get_qbc_installation_instructions',
]

