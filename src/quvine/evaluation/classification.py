"""
Node Classification Evaluation Module

This module provides functions for evaluating node embeddings on node classification tasks.
Includes multiple label generation strategies and comprehensive evaluation metrics.

Label Generation Strategies:
1. Community-based: Louvain, Label Propagation, Spectral Clustering
2. Degree-based: Structural role binning
3. Centrality-based: Betweenness, Closeness, Eigenvector, PageRank
4. Core-periphery: K-core decomposition, Rich-club

Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score (macro/micro/weighted)
- Confusion matrix
- Per-class metrics
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings


def generate_community_labels(
    G: nx.Graph,
    method: str = 'louvain',
    min_community_size: int = 5,
    resolution: float = 1.0
) -> Dict[int, int]:
    """
    Generate node labels based on community detection.
    
    Args:
        G: NetworkX graph
        method: Community detection method ('louvain', 'label_propagation', 'spectral')
        min_community_size: Minimum nodes per community (smaller communities merged)
        resolution: Resolution parameter for Louvain (higher = more communities)
    
    Returns:
        Dictionary mapping node IDs to community labels
    """
    if method == 'louvain':
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G, resolution=resolution)
        except ImportError:
            warnings.warn("python-louvain not installed, falling back to label propagation")
            method = 'label_propagation'
    
    if method == 'label_propagation':
        communities_gen = nx.algorithms.community.label_propagation_communities(G)
        communities = {}
        for label, nodes in enumerate(communities_gen):
            for node in nodes:
                communities[node] = label
    
    elif method == 'spectral':
        # Use spectral clustering with k determined by modularity optimization
        from sklearn.cluster import SpectralClustering
        adj_matrix = nx.to_scipy_sparse_array(G)
        
        # Try different k values and pick best modularity
        best_k = 2
        best_modularity = -1
        for k in range(2, min(20, G.number_of_nodes() // 10)):
            clustering = SpectralClustering(n_clusters=k, affinity='precomputed', random_state=42)
            labels = clustering.fit_predict(adj_matrix.toarray())
            temp_communities = {node: labels[i] for i, node in enumerate(G.nodes())}
            mod = nx.algorithms.community.modularity(
                G, 
                [{n for n, l in temp_communities.items() if l == label} for label in set(labels)]
            )
            if mod > best_modularity:
                best_modularity = mod
                best_k = k
        
        clustering = SpectralClustering(n_clusters=best_k, affinity='precomputed', random_state=42)
        labels = clustering.fit_predict(adj_matrix.toarray())
        communities = {node: labels[i] for i, node in enumerate(G.nodes())}
    
    # Merge small communities
    label_counts = {}
    for label in communities.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    
    small_labels = {label for label, count in label_counts.items() if count < min_community_size}
    
    if small_labels:
        # Merge small communities into largest community
        largest_label = max(label_counts.items(), key=lambda x: x[1])[0]
        communities = {
            node: (largest_label if label in small_labels else label)
            for node, label in communities.items()
        }
        
        # Relabel to consecutive integers
        unique_labels = sorted(set(communities.values()))
        label_map = {old: new for new, old in enumerate(unique_labels)}
        communities = {node: label_map[label] for node, label in communities.items()}
    
    return communities


def generate_degree_labels(
    G: nx.Graph,
    n_bins: int = 5,
    method: str = 'quantile'
) -> Dict[int, int]:
    """
    Generate node labels based on degree binning.
    
    Args:
        G: NetworkX graph
        n_bins: Number of degree bins
        method: Binning method ('quantile' or 'uniform')
    
    Returns:
        Dictionary mapping node IDs to degree-based labels
    """
    degrees = dict(G.degree())
    degree_values = np.array(list(degrees.values()))
    
    if method == 'quantile':
        # Equal-sized bins
        bins = np.percentile(degree_values, np.linspace(0, 100, n_bins + 1))
    else:  # uniform
        # Equal-width bins
        bins = np.linspace(degree_values.min(), degree_values.max() + 1, n_bins + 1)
    
    labels = {}
    for node, degree in degrees.items():
        label = np.digitize(degree, bins[1:])  # Returns 0 to n_bins-1
        labels[node] = min(label, n_bins - 1)  # Ensure max label is n_bins-1
    
    return labels


def generate_centrality_labels(
    G: nx.Graph,
    centrality_type: str = 'betweenness',
    n_bins: int = 5
) -> Dict[int, int]:
    """
    Generate node labels based on centrality measures.
    
    Args:
        G: NetworkX graph
        centrality_type: Type of centrality ('betweenness', 'closeness', 'eigenvector', 'pagerank')
        n_bins: Number of centrality bins
    
    Returns:
        Dictionary mapping node IDs to centrality-based labels
    """
    if centrality_type == 'betweenness':
        centrality = nx.betweenness_centrality(G)
    elif centrality_type == 'closeness':
        centrality = nx.closeness_centrality(G)
    elif centrality_type == 'eigenvector':
        try:
            centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            warnings.warn("Eigenvector centrality failed to converge, using PageRank")
            centrality = nx.pagerank(G)
    elif centrality_type == 'pagerank':
        centrality = nx.pagerank(G)
    else:
        raise ValueError(f"Unknown centrality type: {centrality_type}")
    
    centrality_values = np.array(list(centrality.values()))
    bins = np.percentile(centrality_values, np.linspace(0, 100, n_bins + 1))
    
    labels = {}
    for node, cent_value in centrality.items():
        label = np.digitize(cent_value, bins[1:])
        labels[node] = min(label, n_bins - 1)
    
    return labels


def generate_core_periphery_labels(
    G: nx.Graph,
    method: str = 'k_core',
    n_bins: int = 3
) -> Dict[int, int]:
    """
    Generate node labels based on core-periphery structure.
    
    Args:
        G: NetworkX graph
        method: Method ('k_core' or 'rich_club')
        n_bins: Number of bins for rich-club method
    
    Returns:
        Dictionary mapping node IDs to core-periphery labels
    """
    if method == 'k_core':
        # Use k-core decomposition
        core_numbers = nx.core_number(G)
        max_core = max(core_numbers.values())
        
        # Create labels: 0=periphery, 1=intermediate, 2=core
        labels = {}
        for node, k in core_numbers.items():
            if k <= max_core * 0.33:
                labels[node] = 0  # Periphery
            elif k <= max_core * 0.67:
                labels[node] = 1  # Intermediate
            else:
                labels[node] = 2  # Core
    
    elif method == 'rich_club':
        # Use rich-club coefficient
        degrees = dict(G.degree())
        degree_values = np.array(list(degrees.values()))
        bins = np.percentile(degree_values, np.linspace(0, 100, n_bins + 1))
        
        labels = {}
        for node, degree in degrees.items():
            label = np.digitize(degree, bins[1:])
            labels[node] = min(label, n_bins - 1)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return labels


def evaluate_node_classification(
    embeddings: np.ndarray,
    labels: Dict[int, int],
    node_list: List[int],
    test_size: float = 0.3,
    classifier: str = 'logistic',
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate node embeddings on classification task.
    
    Args:
        embeddings: Node embedding matrix (n_nodes x embedding_dim)
        labels: Dictionary mapping node IDs to class labels
        node_list: List of node IDs corresponding to embedding rows
        test_size: Fraction of data for testing
        classifier: Classifier type ('logistic' or 'random_forest')
        n_splits: Number of cross-validation splits
        random_state: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Prepare data
    X = embeddings
    y = np.array([labels[node] for node in node_list])
    
    # Check if we have enough samples per class
    unique_labels, label_counts = np.unique(y, return_counts=True)
    min_samples = label_counts.min()
    
    if min_samples < 2:
        warnings.warn(f"Some classes have only {min_samples} sample(s). Skipping evaluation.")
        return {
            'accuracy': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'f1_macro': 0.0,
            'n_classes': len(unique_labels),
            'error': 'insufficient_samples'
        }
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Initialize classifier
    if classifier == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='ovr')
    elif classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    # Single train-test split evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Compute metrics
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'n_classes': len(unique_labels),
        'n_train': len(y_train),
        'n_test': len(y_test)
    }
    
    # Cross-validation if enough samples
    if min_samples >= n_splits:
        cv_scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train_cv, X_test_cv = X_scaled[train_idx], X_scaled[test_idx]
            y_train_cv, y_test_cv = y[train_idx], y[test_idx]
            
            clf_cv = LogisticRegression(max_iter=1000, random_state=random_state, multi_class='ovr')
            clf_cv.fit(X_train_cv, y_train_cv)
            y_pred_cv = clf_cv.predict(X_test_cv)
            
            cv_scores.append(f1_score(y_test_cv, y_pred_cv, average='macro', zero_division=0))
        
        results['f1_macro_cv_mean'] = np.mean(cv_scores)
        results['f1_macro_cv_std'] = np.std(cv_scores)
    
    return results


def evaluate_all_label_strategies(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_list: List[int],
    test_size: float = 0.3,
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate embeddings using all label generation strategies.
    
    Args:
        G: NetworkX graph
        embeddings: Node embedding matrix
        node_list: List of node IDs
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        Dictionary mapping strategy names to evaluation results
    """
    results = {}
    
    # 1. Community-based labels
    for method in ['louvain', 'label_propagation']:
        try:
            labels = generate_community_labels(G, method=method)
            eval_results = evaluate_node_classification(
                embeddings, labels, node_list, test_size, random_state=random_state
            )
            results[f'community_{method}'] = eval_results
        except Exception as e:
            warnings.warn(f"Community detection ({method}) failed: {e}")
            results[f'community_{method}'] = {'error': str(e)}
    
    # 2. Degree-based labels
    try:
        labels = generate_degree_labels(G, n_bins=5)
        eval_results = evaluate_node_classification(
            embeddings, labels, node_list, test_size, random_state=random_state
        )
        results['degree_based'] = eval_results
    except Exception as e:
        warnings.warn(f"Degree-based labeling failed: {e}")
        results['degree_based'] = {'error': str(e)}
    
    # 3. Centrality-based labels
    for cent_type in ['betweenness', 'pagerank']:
        try:
            labels = generate_centrality_labels(G, centrality_type=cent_type, n_bins=5)
            eval_results = evaluate_node_classification(
                embeddings, labels, node_list, test_size, random_state=random_state
            )
            results[f'centrality_{cent_type}'] = eval_results
        except Exception as e:
            warnings.warn(f"Centrality-based labeling ({cent_type}) failed: {e}")
            results[f'centrality_{cent_type}'] = {'error': str(e)}
    
    # 4. Core-periphery labels
    try:
        labels = generate_core_periphery_labels(G, method='k_core')
        eval_results = evaluate_node_classification(
            embeddings, labels, node_list, test_size, random_state=random_state
        )
        results['core_periphery'] = eval_results
    except Exception as e:
        warnings.warn(f"Core-periphery labeling failed: {e}")
        results['core_periphery'] = {'error': str(e)}
    
    return results


def summarize_classification_results(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Summarize classification results across all label strategies.
    
    Args:
        results: Dictionary of results from evaluate_all_label_strategies
    
    Returns:
        Dictionary of summary statistics
    """
    # Extract F1 scores
    f1_scores = []
    accuracy_scores = []
    
    for strategy, metrics in results.items():
        if 'error' not in metrics and 'f1_macro' in metrics:
            f1_scores.append(metrics['f1_macro'])
            accuracy_scores.append(metrics['accuracy'])
    
    if not f1_scores:
        return {
            'mean_f1_macro': 0.0,
            'std_f1_macro': 0.0,
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'n_successful_strategies': 0
        }
    
    return {
        'mean_f1_macro': np.mean(f1_scores),
        'std_f1_macro': np.std(f1_scores),
        'max_f1_macro': np.max(f1_scores),
        'min_f1_macro': np.min(f1_scores),
        'mean_accuracy': np.mean(accuracy_scores),
        'std_accuracy': np.std(accuracy_scores),
        'max_accuracy': np.max(accuracy_scores),
        'min_accuracy': np.min(accuracy_scores),
        'n_successful_strategies': len(f1_scores)
    }

# Made with Bob
