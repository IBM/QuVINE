"""
QBioCode Complexity Integration for QuVINE

This module provides integration with IBM's QBioCode complexity metrics,
applying them to graph Laplacian matrices for network complexity analysis.

QBioCode Repository: https://github.com/IBM/QBioCode/
"""

import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, Optional, Union
import warnings

# Try to import QBioCode evaluation functions
try:
    from qbiocode.evaluation import evaluate
    QBC_AVAILABLE = True
except ImportError:
    QBC_AVAILABLE = False
    warnings.warn(
        "QBioCode (qbiocode) not installed. Install with:\n"
        "  git clone https://github.com/IBM/QBioCode.git\n"
        "  cd QBioCode\n"
        "  pip install -e .\n"
        "For more info: https://github.com/IBM/QBioCode/",
        ImportWarning
    )


def get_laplacian_matrix(
    G: nx.Graph,
    normalized: bool = True,
    as_array: bool = True
) -> Union[np.ndarray, 'scipy.sparse.csr_matrix']:
    """
    Get the Laplacian matrix of a graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        If True, return normalized Laplacian
    as_array : bool, default=True
        If True, return dense numpy array; otherwise sparse matrix
        
    Returns
    -------
    L : np.ndarray or scipy.sparse matrix
        Laplacian matrix
    """
    if normalized:
        L = nx.normalized_laplacian_matrix(G)
    else:
        L = nx.laplacian_matrix(G)
    
    if as_array:
        return L.toarray()
    return L


def laplacian_to_dataframe(
    G: nx.Graph,
    normalized: bool = True,
    method: str = 'full',
    mean_center: bool = False
) -> pd.DataFrame:
    """
    Convert graph Laplacian to a pandas DataFrame for QBioCode analysis.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        Use normalized Laplacian
    method : str, default='full'
        Method to convert Laplacian:
        - 'full': Use full Laplacian matrix (nodes as samples, nodes as features)
        - 'eigenvalues': Use eigenvalue spectrum as features
        - 'eigenvectors': Use eigenvector matrix
    mean_center : bool, default=False
        If True, mean-center the Laplacian matrix (subtract column means).
        This can improve QBioCode complexity metrics by ensuring zero-centered features.
        
    Returns
    -------
    df : pd.DataFrame
        DataFrame representation suitable for QBioCode
    """
    L = get_laplacian_matrix(G, normalized=normalized, as_array=True)
    
    # Mean-center if requested
    if mean_center:
        L = L - L.mean(axis=0)
    
    if method == 'full':
        # Treat each row as a sample, columns as features
        df = pd.DataFrame(L)
    elif method == 'eigenvalues':
        # Compute eigenvalues and create feature vectors
        eigenvalues = np.linalg.eigvalsh(L)
        # Create multiple samples by bootstrapping or using sliding windows
        # For simplicity, we'll create samples by taking different subsets
        n = len(eigenvalues)
        samples = []
        for i in range(min(n, 100)):  # Create up to 100 samples
            # Circular shift to create different "views"
            shifted = np.roll(eigenvalues, i)
            samples.append(shifted)
        df = pd.DataFrame(samples)
    elif method == 'eigenvectors':
        # Use eigenvectors as samples
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        df = pd.DataFrame(eigenvectors.T)  # Each eigenvector is a sample
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def create_dummy_labels(n_samples: int, method: str = 'binary') -> np.ndarray:
    """
    Create dummy labels for QBioCode evaluation (which expects supervised labels).
    
    Parameters
    ----------
    n_samples : int
        Number of samples
    method : str, default='binary'
        Method to create labels:
        - 'binary': Split samples into two classes
        - 'random': Random binary labels
        
    Returns
    -------
    labels : np.ndarray
        Binary labels
    """
    if method == 'binary':
        # Split into two equal groups
        labels = np.zeros(n_samples, dtype=int)
        labels[n_samples//2:] = 1
    elif method == 'random':
        labels = np.random.randint(0, 2, size=n_samples)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


def compute_qbc_complexity_from_laplacian(
    G: nx.Graph,
    normalized: bool = True,
    laplacian_method: str = 'eigenvectors',
    label_method: str = 'binary',
    dataset_name: Optional[str] = None,
    mean_center: bool = False
) -> Dict[str, float]:
    """
    Compute QBioCode complexity metrics from graph Laplacian.
    
    This function:
    1. Computes the Laplacian matrix
    2. Converts it to a DataFrame format
    3. Creates dummy labels (QBioCode expects supervised data)
    4. Applies QBioCode's evaluate() function
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        Use normalized Laplacian
    laplacian_method : str, default='eigenvectors'
        Method to convert Laplacian: 'full', 'eigenvalues', 'eigenvectors'
    label_method : str, default='binary'
        Method to create dummy labels: 'binary', 'random'
    dataset_name : str, optional
        Name for the dataset in results
    mean_center : bool, default=False
        If True, mean-center the Laplacian matrix before analysis.
        This can improve complexity metrics by ensuring zero-centered features.
        
    Returns
    -------
    dict
        Dictionary of complexity metrics from QBioCode
        
    Raises
    ------
    ImportError
        If QBioCode is not installed
    """
    if not QBC_AVAILABLE:
        raise ImportError(
            "QBioCode (qbiocode) is required for this function.\n"
            "Install with:\n"
            "  git clone https://github.com/IBM/QBioCode.git\n"
            "  cd QBioCode\n"
            "  pip install -e .\n"
            "For more info: https://github.com/IBM/QBioCode/"
        )
    
    # Convert Laplacian to DataFrame
    df = laplacian_to_dataframe(G, normalized=normalized, method=laplacian_method,
                                mean_center=mean_center)
    
    # Create dummy labels
    labels = create_dummy_labels(len(df), method=label_method)
    
    # Set dataset name
    if dataset_name is None:
        dataset_name = f"graph_n{G.number_of_nodes()}_m{G.number_of_edges()}"
    
    # Call QBioCode evaluate function
    try:
        result_df = evaluate(df, labels, dataset_name)
        
        # Convert DataFrame to dictionary
        # QBioCode returns a transposed DataFrame with metrics
        metrics = result_df.to_dict('records')[0] if len(result_df) > 0 else {}
        
        # Add metadata
        metrics['laplacian_method'] = laplacian_method
        metrics['normalized_laplacian'] = normalized
        metrics['graph_nodes'] = G.number_of_nodes()
        metrics['graph_edges'] = G.number_of_edges()
        
        return metrics
        
    except Exception as e:
        warnings.warn(f"QBioCode evaluation failed: {e}")
        return {
            'error': str(e),
            'laplacian_method': laplacian_method,
            'graph_nodes': G.number_of_nodes(),
            'graph_edges': G.number_of_edges()
        }


def compute_qbc_complexity_multimethod(
    G: nx.Graph,
    normalized: bool = True,
    mean_center: bool = False
) -> Dict[str, Dict[str, float]]:
    """
    Compute QBioCode complexity using multiple Laplacian conversion methods.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    normalized : bool, default=True
        Use normalized Laplacian
    mean_center : bool, default=False
        If True, mean-center the Laplacian matrix before analysis
        
    Returns
    -------
    dict
        Dictionary mapping method names to complexity metrics
    """
    if not QBC_AVAILABLE:
        raise ImportError("QBioCode (qbiocode) is required for this function.")
    
    methods = ['eigenvectors', 'eigenvalues', 'full']
    results = {}
    
    for method in methods:
        try:
            results[method] = compute_qbc_complexity_from_laplacian(
                G, normalized=normalized, laplacian_method=method,
                mean_center=mean_center
            )
        except Exception as e:
            warnings.warn(f"Failed to compute complexity for method {method}: {e}")
            results[method] = {'error': str(e)}
    
    return results


def compute_comprehensive_complexity(
    G: nx.Graph,
    include_qbc: bool = True,
    normalized: bool = True,
    laplacian_method: str = 'eigenvectors',
    mean_center: bool = False
) -> Dict[str, float]:
    """
    Compute comprehensive complexity metrics including both custom and QBioCode measures.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    include_qbc : bool, default=True
        Include QBioCode complexity metrics (requires qbiocode package)
    normalized : bool, default=True
        Use normalized Laplacian
    laplacian_method : str, default='eigenvectors'
        Method for QBioCode analysis
    mean_center : bool, default=False
        If True, mean-center the Laplacian matrix for QBioCode analysis
        
    Returns
    -------
    dict
        Combined complexity metrics
    """
    from .graph_complexity import compute_graph_complexity_metrics
    
    # Get standard complexity metrics
    metrics = compute_graph_complexity_metrics(G)
    
    # Add QBioCode metrics if available and requested
    if include_qbc and QBC_AVAILABLE:
        try:
            qbc_metrics = compute_qbc_complexity_from_laplacian(
                G, normalized=normalized, laplacian_method=laplacian_method,
                mean_center=mean_center
            )
            # Add with qbc_ prefix to avoid name conflicts
            for key, value in qbc_metrics.items():
                if key not in ['laplacian_method', 'normalized_laplacian',
                              'graph_nodes', 'graph_edges']:
                    metrics[f'qbc_{key}'] = value
        except Exception as e:
            warnings.warn(f"Failed to compute QBC metrics: {e}")
    
    return metrics


def compare_qbc_complexity(
    graphs: Dict[str, nx.Graph],
    normalized: bool = True,
    laplacian_method: str = 'eigenvectors',
    mean_center: bool = False
) -> pd.DataFrame:
    """
    Compare QBioCode complexity across multiple graphs.
    
    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
    normalized : bool, default=True
        Use normalized Laplacian
    laplacian_method : str, default='eigenvectors'
        Laplacian conversion method
    mean_center : bool, default=False
        If True, mean-center the Laplacian matrix before analysis
        
    Returns
    -------
    pd.DataFrame
        DataFrame with complexity metrics for all graphs
    """
    if not QBC_AVAILABLE:
        raise ImportError("QBioCode (qbiocode) is required for this function.")
    
    results = []
    for name, G in graphs.items():
        try:
            metrics = compute_qbc_complexity_from_laplacian(
                G, normalized=normalized,
                laplacian_method=laplacian_method,
                dataset_name=name,
                mean_center=mean_center
            )
            metrics['graph_name'] = name
            results.append(metrics)
        except Exception as e:
            warnings.warn(f"Failed to compute complexity for {name}: {e}")
            results.append({
                'graph_name': name,
                'error': str(e)
            })
    
    return pd.DataFrame(results)


def check_qbc_available() -> bool:
    """
    Check if QBioCode is available.
    
    Returns
    -------
    bool
        True if QBioCode is installed and importable
    """
    return QBC_AVAILABLE


def get_qbc_installation_instructions() -> str:
    """
    Get installation instructions for QBioCode.
    
    Returns
    -------
    str
        Installation instructions
    """
    return """
To use QBioCode complexity metrics, install the package:

From source (recommended):
    git clone https://github.com/IBM/QBioCode.git
    cd QBioCode
    pip install -e .

For more information, visit: https://github.com/IBM/QBioCode/

Note: QBioCode requires Python 3.8+ and various scientific computing packages.
"""
