"""
Graph Complexity Metrics for QuVINE

This module provides functions to compute various complexity metrics for graphs,
including Laplacian-based measures and quantum complexity metrics inspired by
QBioCode (https://github.com/IBM/QBioCode/).

These metrics help characterize graph structure and can be used to:
- Compare different graph types
- Understand embedding difficulty
- Select appropriate graphs for testing
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, Tuple
from scipy import linalg
from scipy.stats import entropy
from scipy.sparse.linalg import eigsh


def compute_laplacian_spectrum(G: nx.Graph, normalized: bool = True) -> np.ndarray:
    """
    Compute the eigenvalues of the graph Laplacian.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian; otherwise use unnormalized
        
    Returns
    -------
    eigenvalues : np.ndarray
        Sorted eigenvalues of the Laplacian (ascending order)
    """
    if G.number_of_nodes() == 0:
        return np.array([])
    
    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    
    eigenvalues = linalg.eigvalsh(L)
    return np.sort(eigenvalues)


def compute_spectral_gap(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the spectral gap (difference between first and second eigenvalues).
    
    The spectral gap is related to graph connectivity and mixing time.
    Larger gaps indicate better connectivity.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian
        
    Returns
    -------
    float
        Spectral gap (λ₂ - λ₁)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)
    
    if len(eigenvalues) < 2:
        return 0.0
    
    # For Laplacian, smallest eigenvalue is ~0
    return float(eigenvalues[1] - eigenvalues[0])


def fiedler_eigenvalue_sparse(G: nx.Graph, normalized: bool = False) -> Tuple[float, np.ndarray]:
    """
    Compute Fiedler eigenvalue and eigenvector using sparse matrix methods.
    
    This is more efficient for large graphs than computing the full spectrum.
    The Fiedler eigenvalue is the second smallest eigenvalue of the Laplacian,
    and its eigenvector (Fiedler vector) is useful for graph partitioning.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=False
        If True, use normalized Laplacian; otherwise use unnormalized
        
    Returns
    -------
    lambda2 : float
        Fiedler eigenvalue (second smallest eigenvalue)
    fiedler_vec : np.ndarray
        Fiedler eigenvector
    """
    if G.number_of_nodes() < 2:
        return 0.0, np.array([])
    
    if normalized:
        L = nx.normalized_laplacian_matrix(G)
    else:
        L = nx.laplacian_matrix(G)
    
    try:
        # Compute 2 smallest eigenvalues
        eigenvalues, eigenvectors = eigsh(L, k=2, which='SM')
        
        # Sort them
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        lambda2 = float(eigenvalues[1])
        fiedler_vec = eigenvectors[:, 1]
        
        return lambda2, fiedler_vec
    except Exception:
        # Fallback to dense computation for small graphs
        eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)
        if len(eigenvalues) < 2:
            return 0.0, np.array([])
        return float(eigenvalues[1]), np.array([])


def compute_algebraic_connectivity(G: nx.Graph) -> float:
    """
    Compute algebraic connectivity (Fiedler value).
    
    This is the second smallest eigenvalue of the Laplacian matrix.
    Higher values indicate better connectivity.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    float
        Algebraic connectivity (λ₂)
    """
    if not nx.is_connected(G):
        return 0.0
    
    lambda2, _ = fiedler_eigenvalue_sparse(G, normalized=False)
    return lambda2


def compute_spectral_entropy(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute spectral entropy based on Laplacian eigenvalues.
    
    Spectral entropy measures the complexity/randomness of the graph structure.
    Higher entropy indicates more complex/random structure.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian
        
    Returns
    -------
    float
        Spectral entropy
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Remove near-zero eigenvalues and normalize
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Normalize to create probability distribution
    probs = eigenvalues / eigenvalues.sum()
    
    return float(entropy(probs))


def compute_von_neumann_entropy(G: nx.Graph) -> float:
    """
    Compute von Neumann entropy of the graph.
    
    This is the quantum analog of Shannon entropy, computed from the
    normalized Laplacian eigenvalues. It measures graph complexity.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    float
        Von Neumann entropy
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    
    # Normalize eigenvalues to sum to 1 (density matrix)
    n = G.number_of_nodes()
    eigenvalues = eigenvalues / n
    
    # Remove near-zero eigenvalues
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Von Neumann entropy: -Σ λᵢ log(λᵢ)
    vn_entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    
    return float(vn_entropy)


def compute_estrada_index(G: nx.Graph) -> float:
    """
    Compute Estrada index based on Laplacian eigenvalues.
    
    The Estrada index measures the "folding" or complexity of the graph.
    Related to the number of closed walks.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    float
        Estrada index
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=False)
    
    if len(eigenvalues) == 0:
        return 0.0
    
    # Estrada index: Σ exp(λᵢ)
    estrada = np.sum(np.exp(eigenvalues))
    
    return float(estrada)


def compute_quantum_complexity(G: nx.Graph) -> float:
    """
    Compute quantum complexity metric inspired by QBioCode.
    
    This combines spectral properties to measure how "quantum" or complex
    the graph structure is. Higher values indicate more complex structures
    that may benefit from quantum walks.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    float
        Quantum complexity score
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    
    if len(eigenvalues) < 2:
        return 0.0
    
    # Compute various spectral measures
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    spectral_radius = eigenvalues[-1]
    
    # Effective dimension (participation ratio)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        participation_ratio = (eigenvalues_pos.sum() ** 2) / (eigenvalues_pos ** 2).sum()
    else:
        participation_ratio = 1.0
    
    # Von Neumann entropy
    vn_entropy = compute_von_neumann_entropy(G)
    
    # Combine metrics (normalized)
    n = G.number_of_nodes()
    complexity = (
        0.3 * (spectral_gap / spectral_radius if spectral_radius > 0 else 0) +
        0.3 * (participation_ratio / n) +
        0.4 * (vn_entropy / np.log2(n) if n > 1 else 0)
    )
    
    return float(complexity)


def compute_effective_resistance(G: nx.Graph, source: int, target: int) -> float:
    """
    Compute effective resistance between two nodes.
    
    Effective resistance is related to random walk commute time and
    provides a distance metric on the graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    source : int
        Source node
    target : int
        Target node
        
    Returns
    -------
    float
        Effective resistance
    """
    if source not in G.nodes() or target not in G.nodes():
        return float('inf')
    
    if source == target:
        return 0.0
    
    # Compute pseudoinverse of Laplacian
    L = nx.laplacian_matrix(G).toarray()
    
    # Add small regularization for numerical stability
    L_reg = L + 1e-10 * np.eye(L.shape[0])
    
    try:
        L_pinv = linalg.pinv(L_reg)
    except:
        return float('inf')
    
    # Get node indices
    nodes = list(G.nodes())
    i = nodes.index(source)
    j = nodes.index(target)
    
    # Effective resistance: R(i,j) = L⁺ᵢᵢ + L⁺ⱼⱼ - 2L⁺ᵢⱼ
    resistance = float(L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j])
    
    return float(max(0.0, resistance))


def compute_laplacian_centrality_complexity(G: nx.Graph, normalized: bool = True) -> Dict[str, float]:
    """
    Compute centrality-based complexity metrics from the Laplacian.
    
    This function computes complexity measures based on the distribution
    of Laplacian eigenvector centralities, which capture how information
    flows through the network.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian
        
    Returns
    -------
    dict
        Dictionary of centrality complexity metrics including:
        - centrality_entropy: Entropy of eigenvector centrality distribution
        - centrality_variance: Variance of centrality values
        - centrality_gini: Gini coefficient (inequality measure)
        - centrality_range: Range of centrality values
        - dominant_eigenvector_centrality: Centrality from dominant eigenvector
    """
    if G.number_of_nodes() == 0:
        return {
            'centrality_entropy': 0.0,
            'centrality_variance': 0.0,
            'centrality_gini': 0.0,
            'centrality_range': 0.0,
            'dominant_eigenvector_centrality': 0.0,
        }
    
    # Get Laplacian and compute eigenvectors
    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()
    
    eigenvalues, eigenvectors = linalg.eigh(L)
    
    # Sort by eigenvalue (ascending)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Use the Fiedler vector (second smallest eigenvalue's eigenvector)
    # This captures the most important structural information
    if eigenvectors.shape[1] > 1:
        fiedler_vector = np.abs(eigenvectors[:, 1])
    else:
        fiedler_vector = np.abs(eigenvectors[:, 0])
    
    # Normalize to create a probability distribution
    if fiedler_vector.sum() > 0:
        centrality_dist = fiedler_vector / fiedler_vector.sum()
    else:
        centrality_dist = np.ones(len(fiedler_vector)) / len(fiedler_vector)
    
    # Compute entropy of centrality distribution
    centrality_entropy = float(entropy(centrality_dist))
    
    # Compute variance
    centrality_variance = float(np.var(fiedler_vector))
    
    # Compute Gini coefficient (measure of inequality)
    sorted_centrality = np.sort(fiedler_vector)
    n = len(sorted_centrality)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_centrality)) / (n * np.sum(sorted_centrality)) - (n + 1) / n)
    
    # Compute range
    centrality_range = float(np.max(fiedler_vector) - np.min(fiedler_vector))
    
    # Dominant eigenvector centrality (from largest eigenvalue)
    if eigenvectors.shape[1] > 0:
        dominant_vector = np.abs(eigenvectors[:, -1])
        dominant_centrality = float(np.max(dominant_vector))
    else:
        dominant_centrality = 0.0
    
    return {
        'centrality_entropy': centrality_entropy,
        'centrality_variance': centrality_variance,
        'centrality_gini': gini,
        'centrality_range': centrality_range,
        'dominant_eigenvector_centrality': dominant_centrality,
    }


def compute_graph_complexity_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute comprehensive complexity metrics for a graph.
    
    This function computes multiple complexity measures including:
    - Spectral properties (gap, entropy, etc.)
    - Quantum complexity metrics
    - Structural measures
    - Centrality-based complexity from Laplacian
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        Dictionary of complexity metrics
    """
    if G.number_of_nodes() == 0:
        return {
            'spectral_gap': 0.0,
            'algebraic_connectivity': 0.0,
            'spectral_entropy': 0.0,
            'von_neumann_entropy': 0.0,
            'estrada_index': 0.0,
            'quantum_complexity': 0.0,
            'centrality_entropy': 0.0,
            'centrality_variance': 0.0,
            'centrality_gini': 0.0,
            'centrality_range': 0.0,
            'num_nodes': 0,
            'num_edges': 0,
        }
    
    metrics = {
        # Basic properties
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        
        # Spectral properties
        'spectral_gap': compute_spectral_gap(G, normalized=True),
        'algebraic_connectivity': compute_algebraic_connectivity(G),
        'spectral_entropy': compute_spectral_entropy(G, normalized=True),
        
        # Quantum-inspired metrics
        'von_neumann_entropy': compute_von_neumann_entropy(G),
        'estrada_index': compute_estrada_index(G),
        'quantum_complexity': compute_quantum_complexity(G),
    }
    
    # Add centrality complexity metrics
    centrality_metrics = compute_laplacian_centrality_complexity(G, normalized=True)
    metrics.update(centrality_metrics)
    
    # Add eigenvalue statistics
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    if len(eigenvalues) > 0:
        metrics['eigenvalue_mean'] = float(np.mean(eigenvalues))
        metrics['eigenvalue_std'] = float(np.std(eigenvalues))
        metrics['eigenvalue_max'] = float(np.max(eigenvalues))
        metrics['eigenvalue_min'] = float(np.min(eigenvalues))
    
    return metrics


def compare_graph_complexities(graphs: Dict[str, nx.Graph]) -> Dict[str, Dict[str, float]]:
    """
    Compare complexity metrics across multiple graphs.
    
    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
        
    Returns
    -------
    dict
        Dictionary mapping graph names to their complexity metrics
    """
    results = {}
    
    for name, G in graphs.items():
        results[name] = compute_graph_complexity_metrics(G)
    
    return results


def rank_graphs_by_complexity(
    graphs: Dict[str, nx.Graph],
    metric: str = 'quantum_complexity'
) -> list:
    """
    Rank graphs by a specific complexity metric.
    
    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
    metric : str, default='quantum_complexity'
        Metric to use for ranking
        
    Returns
    -------
    list
        List of (name, score) tuples sorted by complexity (descending)
    """
    complexities = compare_graph_complexities(graphs)
    
    rankings = [
        (name, metrics.get(metric, 0.0))
        for name, metrics in complexities.items()
    ]
    
    return sorted(rankings, key=lambda x: x[1], reverse=True)
