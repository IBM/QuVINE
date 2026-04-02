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
    Larger gaps indicate better connectivity and faster mixing.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian

    Returns
    -------
    float
        Spectral gap (lambda_2 - lambda_1)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    if len(eigenvalues) < 2:
        return 0.0

    # For Laplacian, smallest eigenvalue is ~0
    return float(eigenvalues[1] - eigenvalues[0])


def fiedler_eigenvalue_sparse(
    G: nx.Graph, normalized: bool = False
) -> Tuple[float, np.ndarray]:
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

    This is the second smallest eigenvalue of the unnormalized Laplacian matrix.
    Higher values indicate better connectivity and robustness to node removal.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Algebraic connectivity (lambda_2)
    """
    if not nx.is_connected(G):
        return 0.0

    lambda2, _ = fiedler_eigenvalue_sparse(G, normalized=False)
    return lambda2


def compute_spectral_entropy(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute spectral entropy based on Laplacian eigenvalues.

    Spectral entropy measures the complexity/randomness of the graph structure
    by treating the normalized positive eigenvalues as a probability distribution.
    Higher entropy indicates more complex or random structure.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian

    Returns
    -------
    float
        Spectral entropy H = -sum(p_i * log(p_i)) where p_i = lambda_i / sum(lambda)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    if len(eigenvalues) == 0:
        return 0.0

    # Remove near-zero eigenvalues (trivial zero mode of Laplacian)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    # Normalize to create probability distribution
    probs = eigenvalues / eigenvalues.sum()

    return float(entropy(probs))


def compute_von_neumann_entropy(G: nx.Graph) -> float:
    """
    Compute von Neumann entropy of the graph.

    Implements the Passerini-Severini (2008) definition: the graph is
    associated with a density matrix rho = L / Tr(L), where L is the
    combinatorial (unnormalized) Laplacian and Tr(L) = sum of (weighted)
    degrees. The von Neumann entropy is then:

        S = -Tr(rho log2 rho) = -sum_i (lambda_i / Tr(L)) * log2(lambda_i / Tr(L))

    where the sum is over non-zero eigenvalues of L.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Von Neumann entropy S in bits (log base 2)
    """
    if G.number_of_nodes() == 0:
        return 0.0

    # Use unnormalized Laplacian; Tr(L) = sum of weighted degrees
    L = nx.laplacian_matrix(G).toarray()
    trace_L = float(np.trace(L))

    if trace_L == 0:
        return 0.0

    eigenvalues = np.sort(linalg.eigvalsh(L))

    # Normalize eigenvalues to form density matrix spectrum: rho_i = lambda_i / Tr(L)
    rho_eigs = eigenvalues / trace_L

    # Remove near-zero entries (zero eigenvalue of Laplacian gives 0 * log(0) = 0)
    rho_eigs = rho_eigs[rho_eigs > 1e-12]

    if len(rho_eigs) == 0:
        return 0.0

    # Von Neumann entropy: -sum(rho_i * log2(rho_i))
    vn_entropy = -np.sum(rho_eigs * np.log2(rho_eigs))

    return float(vn_entropy)


def compute_estrada_index(G: nx.Graph) -> float:
    """
    Compute the Laplacian Estrada index.

    The Laplacian Estrada Index (LEE) is defined as:

        LEE = sum_i exp(lambda_i)

    where lambda_i are the eigenvalues of the unnormalized Laplacian.
    It is related to the number of closed walks in the graph and captures
    the overall "folding" or connectivity complexity.

    Note: For large dense graphs the exponentials can be very large. This
    implementation uses log-space accumulation when any eigenvalue exceeds
    500 to avoid float64 overflow.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Laplacian Estrada index LEE = sum exp(lambda_i)
    """
    eigenvalues = compute_laplacian_spectrum(G, normalized=False)

    if len(eigenvalues) == 0:
        return 0.0

    # Guard against float64 overflow (exp overflows above ~709)
    if eigenvalues.max() > 500:
        # Use log-sum-exp: log(LEE) = max + log(sum(exp(x - max)))
        max_val = eigenvalues.max()
        log_estrada = max_val + np.log(np.sum(np.exp(eigenvalues - max_val)))
        return float(np.exp(log_estrada))

    return float(np.sum(np.exp(eigenvalues)))


def compute_quantum_complexity(G: nx.Graph) -> float:
    """
    Compute quantum complexity metric inspired by QBioCode.

    This combines spectral properties to measure how "quantum" or complex
    the graph structure is. Higher values indicate more complex structures
    that may benefit from quantum walks.

    The metric is a weighted combination (weights: 0.3, 0.3, 0.4) of:
    - Spectral gap ratio (gap / spectral radius)
    - Spectral participation ratio (fraction of active modes)
    - Normalised von Neumann entropy

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    float
        Quantum complexity score in [0, 1]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    eigenvalues = compute_laplacian_spectrum(G, normalized=True)

    if len(eigenvalues) < 2:
        return 0.0

    # Compute various spectral measures
    spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
    spectral_radius = eigenvalues[-1]

    # Effective dimension (spectral participation ratio)
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


def compute_spectral_concentration(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute spectral concentration from the Laplacian eigenvalue distribution.

    Measures how concentrated the spectral energy is among the eigenvalues:

        SC = sum(lambda_i^4) / (sum(lambda_i^2))^2

    This is analogous to an inverse participation ratio applied to the
    eigenvalue spectrum (not eigenvectors). Values near 1/k (where k is the
    number of non-zero eigenvalues) indicate uniform spectral spread; values
    near 1 indicate extreme spectral concentration in a few modes.

    Note: this metric operates on eigenvalues and measures the shape of the
    spectrum. For eigenvector-based localization, see
    compute_inverse_participation_ratio().

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use normalized Laplacian eigenvalues

    Returns
    -------
    float
        Spectral concentration in [1/k, 1] where k = number of non-zero eigenvalues
    """
    if G.number_of_nodes() == 0:
        return 0.0

    eigenvalues = compute_laplacian_spectrum(G, normalized=normalized)

    # Remove near-zero eigenvalues
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues_pos) == 0:
        return 0.0

    # SC = sum(lambda^4) / (sum(lambda^2))^2
    sum_lambda_squared = np.sum(eigenvalues_pos ** 2)
    sum_lambda_fourth = np.sum(eigenvalues_pos ** 4)

    if sum_lambda_squared == 0:
        return 0.0

    return float(sum_lambda_fourth / (sum_lambda_squared ** 2))


def compute_inverse_participation_ratio(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the mean Inverse Participation Ratio (IPR) over all Laplacian eigenmodes.

    For each normalised eigenvector v of the Laplacian, the IPR is defined as:

        IPR(v) = sum_j v_j^4

    Because the eigenvectors are L2-normalised (sum v_j^2 = 1), IPR(v) lies in
    [1/n, 1].  A value of 1/n corresponds to a perfectly delocalised mode
    (uniform over all n nodes), while IPR = 1 means the mode is entirely
    concentrated on a single node (Anderson localisation limit).

    This function returns the mean IPR averaged over all n eigenmodes.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use the normalised Laplacian; otherwise use the combinatorial
        (unnormalised) Laplacian

    Returns
    -------
    float
        Mean IPR in [1/n, 1]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    # Compute Laplacian matrix
    L = (
        nx.normalized_laplacian_matrix(G).toarray()
        if normalized
        else nx.laplacian_matrix(G).toarray()
    )

    # Eigenvectors as columns of V; eigh guarantees real, orthonormal columns
    _, V = np.linalg.eigh(L)

    # IPR per mode: sum over nodes of (v_j)^4
    ipr_per_mode = np.sum(V ** 4, axis=0)  # shape: (n_nodes,)

    return float(np.mean(ipr_per_mode))


def compute_participation_ratio(G: nx.Graph, normalized: bool = True) -> float:
    """
    Compute the mean Participation Ratio (PR) over all Laplacian eigenmodes.

    The Participation Ratio is the inverse of the IPR for each eigenmode:

        PR(v) = 1 / IPR(v) = 1 / sum_j v_j^4

    PR(v) estimates the effective number of nodes over which eigenmode v is
    spread. It ranges from 1 (fully localised on one node) to n (perfectly
    delocalised across all nodes). The mean over all modes is returned.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, use the normalised Laplacian; otherwise use the combinatorial
        Laplacian

    Returns
    -------
    float
        Mean participation ratio in [1, n]
    """
    if G.number_of_nodes() == 0:
        return 0.0

    L = (
        nx.normalized_laplacian_matrix(G).toarray()
        if normalized
        else nx.laplacian_matrix(G).toarray()
    )

    _, V = np.linalg.eigh(L)

    # PR per mode: 1 / sum(v_j^4); guard against exact zeros (shouldn't occur)
    ipr_per_mode = np.sum(V ** 4, axis=0)
    pr_per_mode = np.where(ipr_per_mode > 0, 1.0 / ipr_per_mode, 0.0)

    return float(np.mean(pr_per_mode))


def compute_effective_resistance(G: nx.Graph, source: int, target: int) -> float:
    """
    Compute effective resistance between two nodes.

    Effective resistance is related to random walk commute time and
    provides a distance metric on the graph.

        R(i, j) = L^+_ii + L^+_jj - 2 L^+_ij

    where L^+ is the Moore-Penrose pseudoinverse of the Laplacian.

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
        Effective resistance (non-negative)
    """
    if source not in G.nodes() or target not in G.nodes():
        return float('inf')

    if source == target:
        return 0.0

    # Compute pseudoinverse of Laplacian
    L = nx.laplacian_matrix(G).toarray()

    try:
        L_pinv = linalg.pinv(L)
    except Exception:
        return float('inf')

    # Get node indices
    nodes = list(G.nodes())
    i = nodes.index(source)
    j = nodes.index(target)

    # Effective resistance: R(i,j) = L+_ii + L+_jj - 2 L+_ij
    resistance = float(L_pinv[i, i] + L_pinv[j, j] - 2 * L_pinv[i, j])

    return float(max(0.0, resistance))


def compute_laplacian_centrality_complexity(
    G: nx.Graph, normalized: bool = True
) -> Dict[str, float]:
    """
    Compute centrality-based complexity metrics from the Laplacian.

    Uses the Fiedler vector (eigenvector of the second-smallest eigenvalue)
    as a node-centrality proxy and characterises its distribution via entropy,
    variance, Gini coefficient, and range.

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
        - centrality_entropy: Shannon entropy of the Fiedler-vector distribution
        - centrality_variance: Variance of absolute Fiedler-vector entries
        - centrality_gini: Gini coefficient of absolute Fiedler-vector entries
        - centrality_range: Range (max - min) of absolute entries
        - dominant_eigenvector_centrality: Max entry of the largest-eigenvalue eigenvector
    """
    if G.number_of_nodes() == 0:
        return {
            'centrality_entropy': 0.0,
            'centrality_variance': 0.0,
            'centrality_gini': 0.0,
            'centrality_range': 0.0,
            'dominant_eigenvector_centrality': 0.0,
        }

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
    if eigenvectors.shape[1] > 1:
        fiedler_vector = np.abs(eigenvectors[:, 1])
    else:
        fiedler_vector = np.abs(eigenvectors[:, 0])

    # Normalize to probability distribution for entropy
    if fiedler_vector.sum() > 0:
        centrality_dist = fiedler_vector / fiedler_vector.sum()
    else:
        centrality_dist = np.ones(len(fiedler_vector)) / len(fiedler_vector)

    centrality_entropy = float(entropy(centrality_dist))
    centrality_variance = float(np.var(fiedler_vector))

    # Gini coefficient
    sorted_centrality = np.sort(fiedler_vector)
    n = len(sorted_centrality)
    index = np.arange(1, n + 1)
    gini = float(
        (2 * np.sum(index * sorted_centrality)) / (n * np.sum(sorted_centrality)) - (n + 1) / n
    )

    centrality_range = float(np.max(fiedler_vector) - np.min(fiedler_vector))

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
            # Topological metrics
            'orc_gJC_mean': 0.0,
            'orc_kLB_mean': 0.0,
            'orc_negative_fraction': 0.0,
            'cyclomatic_number': 0,
            'kirchhoff_index': 0.0,
            'betti_0': 0,
            'betti_1': 0,
            'betti_2': 0,
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

        # Participation metrics
        'inverse_participation_ratio': compute_inverse_participation_ratio(G, normalized=True),
        'participation_ratio': compute_participation_ratio(G, normalized=True),
        'spectral_concentration': compute_spectral_concentration(G, normalized=True),
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

    # Add quantum advantage metrics
    qa_metrics = compute_quantum_advantage_metrics(G)
    metrics.update(qa_metrics)
    
    # Add topological/geometric complexity metrics
    # Note: Betti numbers can be expensive for large graphs, so make it optional
    try:
        topo_metrics = compute_topological_metrics(
            G,
            include_betti=G.number_of_nodes() < 500,  # Only for smaller graphs
            include_persistence_entropy=G.number_of_nodes() < 500,
            maxdim=2,
            filtration_scale=1.0
        )
        metrics.update(topo_metrics)
    except Exception as e:
        # If topological metrics fail (e.g., ripser not installed), continue
        import warnings
        warnings.warn(f"Topological metrics computation failed: {e}")
        # Add placeholder values
        metrics.update({
            'orc_gJC_mean': 0.0, 'orc_kLB_mean': 0.0,
            'cyclomatic_number': 0,
            'kirchhoff_index': 0.0,
            'betti_0': 0, 'betti_1': 0, 'betti_2': 0,
        })

    return metrics


def compare_graph_complexities(
    graphs: Dict[str, nx.Graph]
) -> Dict[str, Dict[str, float]]:
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
    return {name: compute_graph_complexity_metrics(G) for name, G in graphs.items()}


def compute_quantum_advantage_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute metrics that predict quantum advantage in graph algorithms.

    These metrics help identify when quantum walks are likely to outperform
    classical random walks based on graph structure.

    Parameters
    ----------
    G : nx.Graph
        Input graph

    Returns
    -------
    dict
        Dictionary including:
        - spectral_dimension: Effective number of active eigenvalues
        - modularity: Community structure strength (Louvain greedy)
        - path_length_ratio: avg_path_length / diameter
        - clustering_mean/std: Local clustering statistics
        - degree_heterogeneity: Coefficient of variation of degree sequence
        - quantum_advantage_score: Weighted composite prediction score
    """
    if G.number_of_nodes() == 0:
        return {
            'spectral_dimension': 0.0,
            'modularity': 0.0,
            'path_length_ratio': 0.0,
            'clustering_mean': 0.0,
            'clustering_std': 0.0,
            'degree_heterogeneity': 0.0,
            'quantum_advantage_score': 0.0,
            'quantum_advantage_arithmetic': 0.0,
            'quantum_advantage_geometric': 0.0,
            'quantum_advantage_harmonic': 0.0,
        }

    metrics = {}

    # 1. Spectral dimension (effective number of active eigenvalues)
    #    PR_spectral = (sum lambda_i)^2 / sum(lambda_i^2)
    eigenvalues = compute_laplacian_spectrum(G, normalized=True)
    eigenvalues_pos = eigenvalues[eigenvalues > 1e-10]
    if len(eigenvalues_pos) > 0:
        metrics['spectral_dimension'] = float(
            (eigenvalues_pos.sum() ** 2) / (eigenvalues_pos ** 2).sum()
        )
    else:
        metrics['spectral_dimension'] = 1.0

    # 2. Modularity (community structure)
    try:
        communities = nx.community.greedy_modularity_communities(G)
        metrics['modularity'] = float(nx.community.modularity(G, communities))
    except Exception:
        metrics['modularity'] = 0.0

    # 3. Path length ratio (compactness)
    if nx.is_connected(G):
        try:
            avg_path = nx.average_shortest_path_length(G)
            diameter = nx.diameter(G)
            metrics['path_length_ratio'] = float(avg_path / diameter if diameter > 0 else 0.0)
        except Exception:
            metrics['path_length_ratio'] = 0.0
    else:
        metrics['path_length_ratio'] = 0.0

    # 4. Clustering coefficient distribution
    clustering_values = list(nx.clustering(G).values())
    if clustering_values:
        metrics['clustering_mean'] = float(np.mean(clustering_values))
        metrics['clustering_std'] = float(np.std(clustering_values))
    else:
        metrics['clustering_mean'] = 0.0
        metrics['clustering_std'] = 0.0

    # 5. Degree heterogeneity (coefficient of variation)
    degrees = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    metrics['degree_heterogeneity'] = float(
        np.std(degrees) / mean_deg if mean_deg > 0 else 0.0
    )

    # 6. Quantum advantage scores (multiple formulations)
    qc = compute_quantum_complexity(G)
    sg = compute_spectral_gap(G, normalized=True)
    ipr = compute_inverse_participation_ratio(G, normalized=True)

    # Normalized components (all in [0, 1])
    modularity_norm = metrics['modularity']                    # in [0, 1]
    spectral_gap_norm = 1.0 - min(sg, 1.0)                    # low gap → high advantage
    ipr_norm = min(ipr, 1.0)                                   # more localised → more advantage
    clustering_norm = metrics['clustering_mean']               # in [0, 1]
    
    # Add small epsilon to avoid log(0) and division by zero
    eps = 1e-10
    components = np.array([
        modularity_norm + eps,
        spectral_gap_norm + eps,
        ipr_norm + eps,
        clustering_norm + eps
    ])
    weights = np.array([0.30, 0.25, 0.25, 0.20])
    
    # Arithmetic mean (current default - additive contributions)
    qa_arithmetic = float(np.sum(weights * components))
    
    # Geometric mean (synergistic interactions - all features must be present)
    qa_geometric = float(np.prod(components ** weights))
    
    # Harmonic mean (emphasizes minimum - bottleneck-sensitive)
    qa_harmonic = float(1.0 / np.sum(weights / components))
    
    metrics['quantum_advantage_score'] = qa_arithmetic  # Keep as default
    metrics['quantum_advantage_arithmetic'] = qa_arithmetic
    metrics['quantum_advantage_geometric'] = qa_geometric
    metrics['quantum_advantage_harmonic'] = qa_harmonic

    return metrics


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



# ═════════════════════════════════════════════════════════════════════════════
# Topological & Geometric Complexity Metrics
# ═════════════════════════════════════════════════════════════════════════════

def _hop_distance_matrix(G: nx.Graph) -> np.ndarray:
    """
    Compute all-pairs shortest-path distance matrix using hop counts (unweighted).
    
    Disconnected pairs receive distance = max_finite + 1.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    np.ndarray
        Distance matrix with shape (n, n)
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path as _scipy_shortest_path
    
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros((0, 0))
    
    # Build unweighted adjacency matrix
    A_bool = (nx.to_numpy_array(G, weight=None) > 0).astype(np.float64)
    D = _scipy_shortest_path(csr_matrix(A_bool), directed=False)
    
    # Replace inf with sentinel value
    finite_vals = D[np.isfinite(D)]
    sentinel = (finite_vals.max() + 1.0) if len(finite_vals) > 0 else 1.0
    D[~np.isfinite(D)] = sentinel
    
    return D


def _laplacian_nonzero_eigenvalues(G: nx.Graph, tol: float = 1e-10) -> np.ndarray:
    """
    Compute sorted positive eigenvalues of the combinatorial Laplacian.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Threshold for filtering near-zero eigenvalues
        
    Returns
    -------
    np.ndarray
        Sorted positive eigenvalues
    """
    from scipy import linalg
    
    L = nx.laplacian_matrix(G).toarray()
    eigs = np.sort(linalg.eigvalsh(L))
    return eigs[eigs > tol]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Ollivier-Ricci Curvature (ORC)
# ─────────────────────────────────────────────────────────────────────────────

def compute_orc_per_edge(G: nx.Graph) -> Dict[Tuple, Dict[str, float]]:
    """
    Compute Ollivier-Ricci curvature approximations for every edge.
    
    Two approximations are computed:
    
    1. **Generalized Jaccard (gJC)**: Fast O(d) proxy
       gJC(u, v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
       
    2. **Jost-Liu lower bound (κ_LB)**: Tighter spectral bound
       κ_LB(u, v) = Δ / max(dᵤ, d_v) + 1/dᵤ + 1/d_v − 1
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        Mapping (u, v) → {'gJC': float, 'kappa_LB': float, 'triangles': int}
    """
    result = {}
    
    for u, v in G.edges():
        du = G.degree(u)
        dv = G.degree(v)
        
        # Common neighbours (excluding u and v)
        Nu = set(G.neighbors(u))
        Nv = set(G.neighbors(v))
        common = (Nu & Nv) - {u, v}
        Delta = len(common)
        
        # Generalized Jaccard
        union_size = du + dv - Delta
        gJC = Delta / union_size if union_size > 0 else 0.0
        
        # Jost-Liu lower bound
        if du == 0 or dv == 0:
            kappa_LB = -1.0
        else:
            kappa_LB = Delta / max(du, dv) + 1.0 / du + 1.0 / dv - 1.0
        
        result[(u, v)] = {
            "gJC": float(gJC),
            "kappa_LB": float(kappa_LB),
            "triangles": Delta,
        }
    
    return result


def compute_orc_stats(G: nx.Graph) -> Dict[str, float]:
    """
    Aggregate ORC statistics over all edges.
    
    Returns mean, min, max, std for both gJC and κ_LB, plus the fraction
    of edges with negative κ_LB (bottleneck indicator).
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        ORC statistics with keys:
        - orc_gJC_mean, orc_gJC_min, orc_gJC_max, orc_gJC_std
        - orc_kLB_mean, orc_kLB_min, orc_kLB_max, orc_kLB_std
        - orc_negative_fraction (fraction of edges with κ_LB < 0)
        - orc_num_edges
    """
    if G.number_of_edges() == 0:
        return {
            "orc_gJC_mean": 0.0, "orc_gJC_min": 0.0, "orc_gJC_max": 0.0, "orc_gJC_std": 0.0,
            "orc_kLB_mean": 0.0, "orc_kLB_min": 0.0, "orc_kLB_max": 0.0, "orc_kLB_std": 0.0,
            "orc_negative_fraction": 0.0,
            "orc_num_edges": 0.0,
        }
    
    per_edge = compute_orc_per_edge(G)
    gJC_vals = np.array([v["gJC"] for v in per_edge.values()])
    kLB_vals = np.array([v["kappa_LB"] for v in per_edge.values()])
    
    return {
        "orc_gJC_mean": float(gJC_vals.mean()),
        "orc_gJC_min": float(gJC_vals.min()),
        "orc_gJC_max": float(gJC_vals.max()),
        "orc_gJC_std": float(gJC_vals.std()),
        "orc_kLB_mean": float(kLB_vals.mean()),
        "orc_kLB_min": float(kLB_vals.min()),
        "orc_kLB_max": float(kLB_vals.max()),
        "orc_kLB_std": float(kLB_vals.std()),
        "orc_negative_fraction": float(np.mean(kLB_vals < 0)),
        "orc_num_edges": float(G.number_of_edges()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. Cyclomatic Number (Circuit Rank)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cyclomatic_number(G: nx.Graph) -> int:
    """
    Compute cyclomatic number (circuit rank / first Betti number of 1-skeleton).
    
    μ(G) = m − n + c
    
    where m = |E|, n = |V|, c = number of connected components.
    
    Interpretation:
    - μ = 0 iff G is a forest (no cycles)
    - μ counts minimum edges to remove to make G acyclic
    - Dimension of cycle space H₁(G; ℤ₂)
    - For quantum walks: counts interference-generating loops
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    int
        Non-negative cyclomatic number
    """
    m = G.number_of_edges()
    n = G.number_of_nodes()
    c = nx.number_connected_components(G)
    return max(0, m - n + c)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kirchhoff Index (Total Effective Resistance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kirchhoff_index(G: nx.Graph, tol: float = 1e-10) -> float:
    """
    Compute Kirchhoff index (total effective resistance).
    
    R_K = n · Σᵢ 1/λᵢ
    
    summing over all positive eigenvalues λᵢ of the Laplacian.
    
    Interpretation:
    - For disconnected graphs: R_K = ∞
    - Classical random-walk mixing time ∝ R_K
    - Large R_K indicates bottlenecked topology → potential quantum speedup
    - Complete graph K_n: R_K = n−1
    - Path graph P_n: R_K = n(n²−1)/6
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Eigenvalue threshold for filtering zero mode
        
    Returns
    -------
    float
        Kirchhoff index, or np.inf for disconnected graphs
    """
    if G.number_of_nodes() == 0:
        return 0.0
    
    if not nx.is_connected(G):
        return float("inf")
    
    n = G.number_of_nodes()
    eigs_pos = _laplacian_nonzero_eigenvalues(G, tol=tol)
    
    if len(eigs_pos) == 0:
        return float("inf")
    
    return float(n * np.sum(1.0 / eigs_pos))


def compute_kirchhoff_stats(G: nx.Graph, tol: float = 1e-10) -> Dict[str, float]:
    """
    Compute Kirchhoff index and normalized variants.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    tol : float
        Eigenvalue threshold
        
    Returns
    -------
    dict
        - kirchhoff_index: Raw R_K
        - kirchhoff_per_pair: R_K / C(n, 2) (mean effective resistance)
        - kirchhoff_normalised: R_K / R_K(P_n) (fraction of path-graph value)
    """
    n = G.number_of_nodes()
    Rk = compute_kirchhoff_index(G, tol=tol)
    
    num_pairs = n * (n - 1) / 2 if n > 1 else 1.0
    Rk_path = n * (n ** 2 - 1) / 6.0 if n > 1 else 1.0
    
    return {
        "kirchhoff_index": Rk,
        "kirchhoff_per_pair": Rk / num_pairs if np.isfinite(Rk) else float("inf"),
        "kirchhoff_normalised": Rk / Rk_path if np.isfinite(Rk) else float("inf"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Persistent Betti Numbers via Ripser
# ─────────────────────────────────────────────────────────────────────────────

def compute_betti_numbers(
    G: nx.Graph,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, object]:
    """
    Compute persistent Betti numbers β₀, β₁, β₂ using Ripser.
    
    Uses hop-count shortest-path distance matrix for Vietoris-Rips filtration.
    
    At filtration scale ε = 1 (one hop = one edge):
    - β₀ = number of connected components
    - β₁ = number of independent cycles in clique complex
    - β₂ = voids (unfilled tetrahedra) in clique complex
    
    Relationship to cyclomatic number μ:
    - μ counts all cycles in 1-skeleton (graph edges only)
    - β₁ counts cycles not filled by triangles
    - β₁ ≤ μ (triangles reduce cycle count)
    - β₁ = μ iff G is triangle-free
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    maxdim : int, default=2
        Maximum homological dimension
    filtration_scale : float, default=1.0
        ε at which Betti numbers are evaluated
        
    Returns
    -------
    dict
        - betti_0, betti_1, betti_2: Betti numbers at filtration_scale
        - persistence_diagrams: list of numpy arrays
        - betti_sum: β₀ + β₁ + β₂
        - euler_characteristic: β₀ − β₁ + β₂
    """
    try:
        from ripser import ripser as _ripser
    except ImportError:
        import warnings
        warnings.warn("ripser not installed. Install with: pip install ripser")
        return {
            "betti_0": 0, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [],
            "betti_sum": 0, "euler_characteristic": 0,
        }
    
    n = G.number_of_nodes()
    if n == 0:
        return {
            "betti_0": 0, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [],
            "betti_sum": 0, "euler_characteristic": 0,
        }
    
    if n == 1:
        return {
            "betti_0": 1, "betti_1": 0, "betti_2": 0,
            "persistence_diagrams": [np.array([[0.0, np.inf]])],
            "betti_sum": 1, "euler_characteristic": 1,
        }
    
    # Build hop-count distance matrix
    D = _hop_distance_matrix(G)
    
    # Run Ripser
    result = _ripser(D, maxdim=maxdim, distance_matrix=True)
    dgms = result["dgms"]
    
    # Count features alive at filtration_scale ε
    eps = filtration_scale
    betti = []
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            betti.append(0)
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        alive = int(np.sum((births <= eps) & (deaths > eps)))
        betti.append(alive)
    
    # Pad to 3 dimensions
    while len(betti) < 3:
        betti.append(0)
    
    b0, b1, b2 = betti[0], betti[1], betti[2]
    
    return {
        "betti_0": b0,
        "betti_1": b1,
        "betti_2": b2,
        "persistence_diagrams": dgms,
        "betti_sum": b0 + b1 + b2,
        "euler_characteristic": b0 - b1 + b2,
    }


def compute_persistence_entropy(
    G: nx.Graph,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, float]:
    """
    Compute persistence entropy for each homological dimension.
    
    For persistence diagram D_k = {(bᵢ, dᵢ)}, persistence entropy is:
    
    H_k = −Σᵢ (lᵢ / L) · log(lᵢ / L)
    
    where lᵢ = dᵢ − bᵢ is persistence lifetime and L = Σᵢ lᵢ.
    
    Measures complexity of multi-scale topological structure:
    - High H_k: many cycles with diverse lifetimes
    - Low H_k: one dominant topological feature
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    maxdim : int, default=2
        Maximum dimension
    filtration_scale : float
        Unused (entropy computed over all features)
        
    Returns
    -------
    dict
        persistence_entropy_H0, persistence_entropy_H1, persistence_entropy_H2
    """
    betti_result = compute_betti_numbers(G, maxdim=maxdim, filtration_scale=filtration_scale)
    dgms = betti_result["persistence_diagrams"]
    
    entropies = {}
    for dim in range(3):
        key = f"persistence_entropy_H{dim}"
        if dim >= len(dgms) or len(dgms[dim]) == 0:
            entropies[key] = 0.0
            continue
        
        dgm = dgms[dim]
        births = dgm[:, 0]
        deaths = dgm[:, 1].copy()
        
        # Replace inf with max-finite + 1
        finite_mask = np.isfinite(deaths)
        if finite_mask.any():
            max_finite = deaths[finite_mask].max()
        else:
            max_finite = 0.0
        deaths[~finite_mask] = max_finite + 1.0
        
        lifetimes = deaths - births
        lifetimes = lifetimes[lifetimes > 0]
        
        if len(lifetimes) == 0:
            entropies[key] = 0.0
            continue
        
        L = lifetimes.sum()
        probs = lifetimes / L
        entropies[key] = float(-np.sum(probs * np.log(probs + 1e-300)))
    
    return entropies


# ─────────────────────────────────────────────────────────────────────────────
# 5. Combined Topological Metrics Interface
# ─────────────────────────────────────────────────────────────────────────────

def compute_topological_metrics(
    G: nx.Graph,
    include_betti: bool = True,
    include_persistence_entropy: bool = True,
    maxdim: int = 2,
    filtration_scale: float = 1.0,
) -> Dict[str, object]:
    """
    Compute all topological/geometric complexity metrics.
    
    Metrics computed:
    - ORC (Ollivier-Ricci curvature): gJC and κ_LB approximations
    - Cyclomatic number: Circuit rank μ
    - Kirchhoff index: Total effective resistance R_K
    - Betti numbers: β₀, β₁, β₂ (if include_betti=True)
    - Persistence entropy: H₀, H₁, H₂ (if include_persistence_entropy=True)
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    include_betti : bool, default=True
        Compute Betti numbers (expensive for large graphs)
    include_persistence_entropy : bool, default=True
        Compute persistence entropy (requires include_betti=True)
    maxdim : int, default=2
        Maximum homological dimension
    filtration_scale : float, default=1.0
        ε at which Betti numbers are evaluated
        
    Returns
    -------
    dict
        All topological metrics
    """
    metrics = {}
    
    # Basic properties
    metrics["num_nodes"] = G.number_of_nodes()
    metrics["num_edges"] = G.number_of_edges()
    
    # ORC stats
    metrics.update(compute_orc_stats(G))
    
    # Cyclomatic number
    metrics["cyclomatic_number"] = compute_cyclomatic_number(G)
    
    # Kirchhoff index
    metrics.update(compute_kirchhoff_stats(G))
    
    # Betti numbers + persistence entropy
    if include_betti and G.number_of_nodes() > 0:
        betti = compute_betti_numbers(G, maxdim=maxdim, filtration_scale=filtration_scale)
        # Don't store raw diagrams in flat dict
        betti.pop("persistence_diagrams", None)
        metrics.update(betti)
        
        if include_persistence_entropy:
            metrics.update(
                compute_persistence_entropy(G, maxdim=maxdim, filtration_scale=filtration_scale)
            )
    
    return metrics
