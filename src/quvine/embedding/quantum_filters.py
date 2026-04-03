"""
QuVINE Quantum Filters: Quantum-Calibrated Graph Diffusion

This module implements quantum-calibrated graph diffusion for QuVINE, which uses
local quantum walk statistics from sampled subnetworks to calibrate parameters of
global graph diffusion operators.

Key Innovation:
- Run quantum walks on small subgraphs (scalable)
- Fit global diffusion parameters to match quantum behavior
- Apply calibrated classical operator to full graph (efficient)

Reference:
- QuVINE notebook: notebooks/Q-Caliber_Quantum_Calibrated_Graph_Diffusion.ipynb
- Hiperwalk: https://hiperwalk.org/

Author: QuVINE Team
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import expm
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================

def _normalize_laplacian(L: Union[np.ndarray, sp.spmatrix]) -> Union[np.ndarray, sp.spmatrix]:
    """
    Normalize Laplacian: L_norm = D^(-1/2) L D^(-1/2)
    
    Args:
        L: Laplacian matrix (dense or sparse)
    
    Returns:
        Normalized Laplacian
    """
    if sp.issparse(L):
        # Sparse case
        D = sp.diags(L.diagonal())
        D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(L.diagonal(), 1e-12)))
        return D_inv_sqrt @ L @ D_inv_sqrt
    else:
        # Dense case
        D = np.diag(np.diag(L))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.diag(L), 1e-12)))
        return D_inv_sqrt @ L @ D_inv_sqrt


def _expm_multiply_restricted(
    L: Union[np.ndarray, sp.spmatrix],
    t: float,
    x: np.ndarray,
    nodes: List[int],
    node_to_idx: Dict[int, int]
) -> np.ndarray:
    """
    Compute exp(-t*L) @ x and extract values at specified nodes.
    
    Uses scipy's expm_multiply for efficiency with sparse matrices.
    
    Args:
        L: Laplacian matrix
        t: Time parameter
        x: Initial vector
        nodes: List of node IDs to extract
        node_to_idx: Mapping from node IDs to matrix indices
    
    Returns:
        Array of values at specified nodes
    """
    # Compute exp(-t*L) @ x
    if sp.issparse(L):
        y = spla.expm_multiply(-t * L, x)
    else:
        y = expm(-t * L) @ x
    
    # Extract values at specified nodes
    node_indices = [node_to_idx[n] for n in nodes]
    return np.asarray(y)[node_indices]


# ============================================================================
# Calibration Functions
# ============================================================================

def calibrate_heat_kernel(
    L: Union[np.ndarray, sp.spmatrix],
    q_targets: List[Dict],
    t_grid: np.ndarray,
    node_to_idx: Dict[int, int],
    loss: str = 'l2'
) -> Tuple[float, float]:
    """
    Calibrate heat kernel time parameter by matching quantum walk targets.
    
    Fits the parameter t in g_t(L) = exp(-t*L) by minimizing the loss between
    heat kernel diffusion and quantum walk probability distributions on subnetworks.
    
    Args:
        L: Laplacian matrix of the full graph
        q_targets: List of quantum walk target distributions, each containing:
            - 'nodes': List of node IDs in subnetwork
            - 'center': Center node ID
            - 'pQ': Quantum walk probability distribution
        t_grid: Grid of time values to search over
        node_to_idx: Dictionary mapping node IDs to matrix indices
        loss: Loss function ('l2' or 'kl')
    
    Returns:
        Tuple of (best_loss, best_t)
    
    Example:
        >>> L = nx.laplacian_matrix(G).astype(float)
        >>> q_targets = [{'nodes': [0,1,2], 'center': 0, 'pQ': np.array([0.5, 0.3, 0.2])}]
        >>> t_grid = np.linspace(0.1, 5.0, 20)
        >>> node_to_idx = {i: i for i in range(G.number_of_nodes())}
        >>> loss_val, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx)
    """
    best_loss, best_t = np.inf, None
    N = L.shape[0]
    
    logger.info(f"Calibrating heat kernel over {len(t_grid)} time values...")
    
    for t in t_grid:
        tot = 0.0
        for item in q_targets:
            nodes = item['nodes']
            center = item['center']
            pQ = item['pQ']
            
            # Initialize with delta at center
            x = np.zeros(N)
            x[node_to_idx[center]] = 1.0
            
            # Compute heat kernel diffusion restricted to subnetwork
            yS = _expm_multiply_restricted(L, t, x, nodes, node_to_idx)
            yS = np.maximum(yS, 0)  # Ensure non-negative
            pT = yS / yS.sum() if yS.sum() > 0 else yS
            
            # Compute loss
            if loss == 'l2':
                tot += np.sum((pT - pQ) ** 2)
            elif loss == 'kl':
                eps = 1e-12
                tot += np.sum(pQ * (np.log(pQ + eps) - np.log(pT + eps)))
            else:
                raise ValueError(f"Unknown loss function: {loss}")
        
        if tot < best_loss:
            best_loss, best_t = tot, t
    
    if best_t is None:
        raise ValueError("No valid time parameter found in t_grid")
    
    logger.info(f"Best heat kernel time: t={best_t:.4f}, loss={best_loss:.6f}")
    return best_loss, best_t


def calibrate_polynomial_filter(
    L: Union[np.ndarray, sp.spmatrix],
    q_targets: List[Dict],
    node_to_idx: Dict[int, int],
    K: int = 4,
    ridge: float = 1e-6
) -> np.ndarray:
    """
    Calibrate polynomial filter coefficients by least squares.
    
    Fits coefficients {a_k} in g(L) = sum_{k=0}^K a_k * L^k by minimizing
    the squared error between polynomial filter output and quantum walk targets.
    
    Args:
        L: Laplacian matrix of the full graph
        q_targets: List of quantum walk target distributions (same format as calibrate_heat_kernel)
        node_to_idx: Dictionary mapping node IDs to matrix indices
        K: Polynomial degree (number of terms - 1)
        ridge: Ridge regularization parameter
    
    Returns:
        Array of polynomial coefficients [a_0, a_1, ..., a_K]
    
    Example:
        >>> coeffs = calibrate_polynomial_filter(L, q_targets, node_to_idx, K=4)
        >>> print(f"Polynomial coefficients: {coeffs}")
    """
    AtA = np.zeros((K + 1, K + 1))
    Atb = np.zeros(K + 1)
    N = L.shape[0]
    
    logger.info(f"Calibrating polynomial filter with degree K={K}...")
    
    for item in q_targets:
        nodes = item['nodes']
        center = item['center']
        pQ = item['pQ']
        
        # Initialize with delta at center
        x = np.zeros(N)
        x[node_to_idx[center]] = 1.0
        
        # Build basis: [x, L@x, L^2@x, ..., L^K@x]
        basis = []
        v = x.copy()
        node_indices = [node_to_idx[n] for n in nodes]
        basis.append(v[node_indices])
        
        for _ in range(1, K + 1):
            if sp.issparse(L):
                v = L @ v
            else:
                v = L @ v
            basis.append(np.asarray(v)[node_indices])
        
        # Stack basis vectors: Phi is |S| x (K+1)
        Phi = np.stack(basis, axis=1)
        b = pQ
        
        # Accumulate normal equations: A^T A and A^T b
        AtA += Phi.T @ Phi
        Atb += Phi.T @ b
    
    # Add ridge regularization
    AtA += ridge * np.eye(K + 1)
    
    # Solve for coefficients
    coeffs = np.linalg.solve(AtA, Atb)
    
    logger.info(f"Polynomial coefficients: {coeffs}")
    return coeffs


# ============================================================================
# Filter Application Functions
# ============================================================================

def apply_heat_filter(
    L: Union[np.ndarray, sp.spmatrix],
    X: np.ndarray,
    t: float
) -> np.ndarray:
    """
    Apply heat kernel filter: Z = exp(-t*L) @ X
    
    Args:
        L: Laplacian matrix
        X: Node features [N, d]
        t: Time parameter
    
    Returns:
        Filtered features [N, d]
    """
    logger.info(f"Applying heat kernel filter with t={t:.4f}...")
    
    if sp.issparse(L):
        # Use sparse matrix exponential
        Z = spla.expm_multiply(-t * L, X)
    else:
        # Dense case
        Z = expm(-t * L) @ X
    
    return np.asarray(Z)


def apply_polynomial_filter(
    L: Union[np.ndarray, sp.spmatrix],
    X: np.ndarray,
    coeffs: np.ndarray
) -> np.ndarray:
    """
    Apply polynomial filter: Z = sum_{k=0}^K a_k * L^k @ X
    
    Uses Horner's method for efficient computation.
    
    Args:
        L: Laplacian matrix
        X: Node features [N, d]
        coeffs: Polynomial coefficients [a_0, a_1, ..., a_K]
    
    Returns:
        Filtered features [N, d]
    """
    logger.info(f"Applying polynomial filter with {len(coeffs)} coefficients...")
    
    Z = coeffs[0] * X
    V = X.copy()
    
    for k in range(1, len(coeffs)):
        if sp.issparse(L):
            V = L @ V
        else:
            V = L @ V
        Z = Z + coeffs[k] * V
    
    return np.asarray(Z)


# ============================================================================
# Embedding Generation Functions
# ============================================================================

def generate_quvine_heat_embedding(
    G: nx.Graph,
    q_targets: List[Dict],
    t_grid: Optional[np.ndarray] = None,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate QuVINE heat kernel embeddings.
    
    Workflow:
    1. Calibrate heat kernel time parameter using quantum walk targets
    2. Apply calibrated heat kernel to node features
    3. Return filtered features as embeddings
    
    Args:
        G: NetworkX graph
        q_targets: List of quantum walk target distributions
        t_grid: Grid of time values to search (default: np.linspace(0.1, 5.0, 20))
        embedding_dim: Embedding dimension (used if generating random features)
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian
    L = nx.laplacian_matrix(G).astype(float)
    if normalize:
        L = _normalize_laplacian(L)
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Default time grid
    if t_grid is None:
        t_grid = np.linspace(0.1, 5.0, 20)
    
    # Calibrate heat kernel
    _, t_star = calibrate_heat_kernel(L, q_targets, t_grid, node_to_idx, loss='l2')
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        X = np.random.randn(N, embedding_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize rows
    
    # Apply heat kernel filter
    Z = apply_heat_filter(L, X, t_star)
    
    return Z


def generate_quvine_poly_embedding(
    G: nx.Graph,
    q_targets: List[Dict],
    K: int = 4,
    ridge: float = 1e-6,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate QuVINE polynomial filter embeddings.
    
    Workflow:
    1. Calibrate polynomial filter coefficients using quantum walk targets
    2. Apply calibrated polynomial filter to node features
    3. Return filtered features as embeddings
    
    Args:
        G: NetworkX graph
        q_targets: List of quantum walk target distributions
        K: Polynomial degree
        ridge: Ridge regularization parameter
        embedding_dim: Embedding dimension (used if generating random features)
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian
    L = nx.laplacian_matrix(G).astype(float)
    if normalize:
        L = _normalize_laplacian(L)
    
    # Create node to index mapping
    node_to_idx = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Calibrate polynomial filter
    coeffs = calibrate_polynomial_filter(L, q_targets, node_to_idx, K=K, ridge=ridge)
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        X = np.random.randn(N, embedding_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize rows
    
    # Apply polynomial filter
    Z = apply_polynomial_filter(L, X, coeffs)
    
    return Z


def generate_baseline_filter_embedding(
    G: nx.Graph,
    filter_type: str = 'heat',
    t: float = 1.0,
    K: int = 4,
    embedding_dim: int = 128,
    use_features: bool = False,
    features: Optional[np.ndarray] = None,
    normalize: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate baseline graph filter embeddings (without quantum calibration).
    
    This serves as a baseline to compare against QuVINE methods.
    
    Args:
        G: NetworkX graph
        filter_type: Type of filter ('heat' or 'poly')
        t: Time parameter for heat kernel (if filter_type='heat')
        K: Polynomial degree (if filter_type='poly')
        embedding_dim: Embedding dimension
        use_features: Whether to use provided features or generate random ones
        features: Node features [N, d] (optional)
        normalize: Whether to normalize Laplacian
        random_state: Random seed
    
    Returns:
        Node embeddings [N, embedding_dim]
    """
    np.random.seed(random_state)
    N = G.number_of_nodes()
    
    # Get Laplacian
    L = nx.laplacian_matrix(G).astype(float)
    if normalize:
        L = _normalize_laplacian(L)
    
    # Generate or use features
    if use_features and features is not None:
        X = features
    else:
        X = np.random.randn(N, embedding_dim)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)  # Normalize rows
    
    # Apply filter
    if filter_type == 'heat':
        logger.info(f"Generating baseline heat kernel embedding with t={t:.4f}")
        Z = apply_heat_filter(L, X, t)
    elif filter_type == 'poly':
        logger.info(f"Generating baseline polynomial embedding with K={K}")
        # Use simple coefficients: [1, -1, 0.5, -0.25, ...] (alternating signs, decreasing magnitude)
        coeffs = np.array([(-1)**k / (2**k) for k in range(K + 1)])
        Z = apply_polynomial_filter(L, X, coeffs)
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    return Z

# Made with Bob
