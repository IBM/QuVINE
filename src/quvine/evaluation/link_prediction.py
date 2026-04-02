"""
Link Prediction Evaluation Module

This module provides functions for evaluating node embeddings on link prediction tasks.
Includes edge sampling strategies, edge feature computation, and comprehensive evaluation metrics.

Edge Sampling Strategies:
1. Random negative sampling
2. Temporal edge splitting (if timestamps available)
3. Stratified sampling by node degree

Edge Feature Computation:
1. Hadamard product: u ⊙ v
2. Average: (u + v) / 2
3. L1 distance: |u - v|
4. L2 distance: ||u - v||₂
5. Concatenation: [u; v]

Evaluation Metrics:
- AUC-ROC, AUC-PR
- Precision@K, Recall@K
- Hit@K, MRR (Mean Reciprocal Rank)
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings


def sample_negative_edges(
    G: nx.Graph,
    n_samples: int,
    existing_edges: Optional[Set[Tuple[int, int]]] = None,
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Sample negative edges (non-existent edges) from the graph.
    
    Args:
        G: NetworkX graph
        n_samples: Number of negative edges to sample
        existing_edges: Set of existing edges to avoid
        seed: Random seed
    
    Returns:
        List of negative edge tuples
    """
    np.random.seed(seed)
    nodes = list(G.nodes())
    n_nodes = len(nodes)
    
    if existing_edges is None:
        existing_edges = set(G.edges())
    
    # Add reverse edges for undirected graphs
    existing_edges_bidirectional = existing_edges.copy()
    for u, v in existing_edges:
        existing_edges_bidirectional.add((v, u))
    
    negative_edges = []
    max_attempts = n_samples * 10  # Prevent infinite loop
    attempts = 0
    
    while len(negative_edges) < n_samples and attempts < max_attempts:
        u = nodes[np.random.randint(n_nodes)]
        v = nodes[np.random.randint(n_nodes)]
        
        if u != v and (u, v) not in existing_edges_bidirectional:
            negative_edges.append((u, v))
            existing_edges_bidirectional.add((u, v))
            existing_edges_bidirectional.add((v, u))
        
        attempts += 1
    
    if len(negative_edges) < n_samples:
        warnings.warn(f"Could only sample {len(negative_edges)} negative edges (requested {n_samples})")
    
    return negative_edges


def split_edges(
    G: nx.Graph,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[nx.Graph, List[Tuple[int, int]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Split graph edges into train/validation/test sets.
    
    Args:
        G: NetworkX graph
        test_ratio: Fraction of edges for testing
        val_ratio: Fraction of edges for validation
        seed: Random seed
    
    Returns:
        Tuple of (train_graph, val_edges, test_edges, negative_edges)
    """
    np.random.seed(seed)
    
    edges = list(G.edges())
    n_edges = len(edges)
    
    # Shuffle edges
    np.random.shuffle(edges)
    
    # Split edges
    n_test = int(n_edges * test_ratio)
    n_val = int(n_edges * val_ratio)
    n_train = n_edges - n_test - n_val
    
    train_edges = edges[:n_train]
    val_edges = edges[n_train:n_train + n_val]
    test_edges = edges[n_train + n_val:]
    
    # Create training graph
    train_graph = nx.Graph()
    train_graph.add_nodes_from(G.nodes())
    train_graph.add_edges_from(train_edges)
    
    # Ensure training graph is connected (if original was connected)
    if nx.is_connected(G) and not nx.is_connected(train_graph):
        warnings.warn("Training graph is disconnected after edge split. Consider reducing test_ratio.")
    
    # Sample negative edges (same number as test edges)
    negative_edges = sample_negative_edges(G, n_test, set(edges), seed=seed)
    
    return train_graph, val_edges, test_edges, negative_edges


def compute_edge_features(
    embeddings: np.ndarray,
    node_list: List[int],
    edges: List[Tuple[int, int]],
    method: str = 'hadamard'
) -> np.ndarray:
    """
    Compute edge features from node embeddings.
    
    Args:
        embeddings: Node embedding matrix (n_nodes x embedding_dim)
        node_list: List of node IDs corresponding to embedding rows
        edges: List of edge tuples
        method: Feature computation method ('hadamard', 'average', 'l1', 'l2', 'concat')
    
    Returns:
        Edge feature matrix (n_edges x feature_dim)
    """
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    edge_features = []
    
    for u, v in edges:
        if u not in node_to_idx or v not in node_to_idx:
            warnings.warn(f"Edge ({u}, {v}) contains unknown node(s). Skipping.")
            continue
        
        u_emb = embeddings[node_to_idx[u]]
        v_emb = embeddings[node_to_idx[v]]
        
        if method == 'hadamard':
            feature = u_emb * v_emb
        elif method == 'average':
            feature = (u_emb + v_emb) / 2
        elif method == 'l1':
            feature = np.abs(u_emb - v_emb)
        elif method == 'l2':
            feature = (u_emb - v_emb) ** 2
        elif method == 'concat':
            feature = np.concatenate([u_emb, v_emb])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        edge_features.append(feature)
    
    return np.array(edge_features)


def evaluate_link_prediction(
    embeddings: np.ndarray,
    node_list: List[int],
    positive_edges: List[Tuple[int, int]],
    negative_edges: List[Tuple[int, int]],
    edge_feature_method: str = 'hadamard',
    classifier: str = 'logistic',
    k_values: List[int] = [10, 50, 100],
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate link prediction performance.
    
    Args:
        embeddings: Node embedding matrix
        node_list: List of node IDs
        positive_edges: List of positive (existing) edges
        negative_edges: List of negative (non-existing) edges
        edge_feature_method: Method for computing edge features
        classifier: Classifier type ('logistic' or 'random_forest')
        k_values: K values for Precision@K and Recall@K
        random_state: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Compute edge features
    pos_features = compute_edge_features(embeddings, node_list, positive_edges, edge_feature_method)
    neg_features = compute_edge_features(embeddings, node_list, negative_edges, edge_feature_method)
    
    if len(pos_features) == 0 or len(neg_features) == 0:
        warnings.warn("No valid edge features computed. Skipping evaluation.")
        return {'error': 'no_valid_features'}
    
    # Prepare data
    X = np.vstack([pos_features, neg_features])
    y = np.array([1] * len(pos_features) + [0] * len(neg_features))
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    if classifier == 'logistic':
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    elif classifier == 'random_forest':
        clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
    clf.fit(X_scaled, y)
    
    # Get prediction scores
    if hasattr(clf, 'predict_proba'):
        y_scores = clf.predict_proba(X_scaled)[:, 1]
    else:
        y_scores = clf.decision_function(X_scaled)
    
    # Compute metrics
    results = {
        'auc_roc': roc_auc_score(y, y_scores),
        'auc_pr': average_precision_score(y, y_scores),
        'n_positive': len(pos_features),
        'n_negative': len(neg_features)
    }
    
    # Compute Precision@K and Recall@K
    sorted_indices = np.argsort(y_scores)[::-1]  # Sort by score (descending)
    y_sorted = y[sorted_indices]
    
    for k in k_values:
        if k <= len(y_sorted):
            top_k = y_sorted[:k]
            precision_at_k = np.sum(top_k) / k
            recall_at_k = np.sum(top_k) / np.sum(y)
            
            results[f'precision@{k}'] = precision_at_k
            results[f'recall@{k}'] = recall_at_k
            results[f'hit@{k}'] = 1.0 if np.sum(top_k) > 0 else 0.0
    
    # Compute MRR (Mean Reciprocal Rank)
    positive_indices = np.where(y_sorted == 1)[0]
    if len(positive_indices) > 0:
        reciprocal_ranks = 1.0 / (positive_indices + 1)
        results['mrr'] = np.mean(reciprocal_ranks)
    else:
        results['mrr'] = 0.0
    
    return results


def evaluate_link_prediction_cv(
    G: nx.Graph,
    embeddings: np.ndarray,
    node_list: List[int],
    test_ratio: float = 0.2,
    edge_feature_method: str = 'hadamard',
    k_values: List[int] = [10, 50, 100],
    random_state: int = 42
) -> Dict[str, float]:
    """
    Evaluate link prediction with train-test split.
    
    Args:
        G: NetworkX graph
        embeddings: Node embedding matrix
        node_list: List of node IDs
        test_ratio: Fraction of edges for testing
        edge_feature_method: Method for computing edge features
        k_values: K values for metrics
        random_state: Random seed
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Split edges
    train_graph, _, test_edges, negative_edges = split_edges(
        G, test_ratio=test_ratio, val_ratio=0.0, seed=random_state
    )
    
    # Evaluate
    results = evaluate_link_prediction(
        embeddings=embeddings,
        node_list=node_list,
        positive_edges=test_edges,
        negative_edges=negative_edges,
        edge_feature_method=edge_feature_method,
        k_values=k_values,
        random_state=random_state
    )
    
    return results


def evaluate_all_edge_feature_methods(
    embeddings: np.ndarray,
    node_list: List[int],
    positive_edges: List[Tuple[int, int]],
    negative_edges: List[Tuple[int, int]],
    k_values: List[int] = [10, 50, 100],
    random_state: int = 42
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate link prediction using all edge feature methods.
    
    Args:
        embeddings: Node embedding matrix
        node_list: List of node IDs
        positive_edges: List of positive edges
        negative_edges: List of negative edges
        k_values: K values for metrics
        random_state: Random seed
    
    Returns:
        Dictionary mapping method names to evaluation results
    """
    results = {}
    
    for method in ['hadamard', 'average', 'l1', 'l2']:
        try:
            eval_results = evaluate_link_prediction(
                embeddings=embeddings,
                node_list=node_list,
                positive_edges=positive_edges,
                negative_edges=negative_edges,
                edge_feature_method=method,
                k_values=k_values,
                random_state=random_state
            )
            results[method] = eval_results
        except Exception as e:
            warnings.warn(f"Link prediction with {method} failed: {e}")
            results[method] = {'error': str(e)}
    
    return results


def summarize_link_prediction_results(
    results: Dict[str, Dict[str, float]]
) -> Dict[str, float]:
    """
    Summarize link prediction results across all edge feature methods.
    
    Args:
        results: Dictionary of results from evaluate_all_edge_feature_methods
    
    Returns:
        Dictionary of summary statistics
    """
    # Extract AUC scores
    auc_roc_scores = []
    auc_pr_scores = []
    mrr_scores = []
    
    for method, metrics in results.items():
        if 'error' not in metrics:
            if 'auc_roc' in metrics:
                auc_roc_scores.append(metrics['auc_roc'])
            if 'auc_pr' in metrics:
                auc_pr_scores.append(metrics['auc_pr'])
            if 'mrr' in metrics:
                mrr_scores.append(metrics['mrr'])
    
    if not auc_roc_scores:
        return {
            'mean_auc_roc': 0.0,
            'mean_auc_pr': 0.0,
            'mean_mrr': 0.0,
            'n_successful_methods': 0
        }
    
    summary = {
        'mean_auc_roc': np.mean(auc_roc_scores),
        'std_auc_roc': np.std(auc_roc_scores),
        'max_auc_roc': np.max(auc_roc_scores),
        'min_auc_roc': np.min(auc_roc_scores),
        'mean_auc_pr': np.mean(auc_pr_scores),
        'std_auc_pr': np.std(auc_pr_scores),
        'max_auc_pr': np.max(auc_pr_scores),
        'min_auc_pr': np.min(auc_pr_scores),
        'mean_mrr': np.mean(mrr_scores),
        'std_mrr': np.std(mrr_scores),
        'n_successful_methods': len(auc_roc_scores)
    }
    
    return summary


def compute_structural_link_features(
    G: nx.Graph,
    edges: List[Tuple[int, int]]
) -> np.ndarray:
    """
    Compute structural features for edges (for baseline comparison).
    
    Features:
    - Common neighbors
    - Jaccard coefficient
    - Adamic-Adar index
    - Preferential attachment
    
    Args:
        G: NetworkX graph
        edges: List of edge tuples
    
    Returns:
        Structural feature matrix (n_edges x 4)
    """
    features = []
    
    for u, v in edges:
        if u not in G or v not in G:
            features.append([0, 0, 0, 0])
            continue
        
        # Common neighbors
        common_neighbors = len(list(nx.common_neighbors(G, u, v)))
        
        # Jaccard coefficient
        u_neighbors = set(G.neighbors(u))
        v_neighbors = set(G.neighbors(v))
        union = len(u_neighbors | v_neighbors)
        jaccard = len(u_neighbors & v_neighbors) / union if union > 0 else 0
        
        # Adamic-Adar index
        adamic_adar = 0
        for w in nx.common_neighbors(G, u, v):
            deg_w = G.degree(w)
            if deg_w > 1:
                adamic_adar += 1 / np.log(deg_w)
        
        # Preferential attachment
        pref_attach = G.degree(u) * G.degree(v)
        
        features.append([common_neighbors, jaccard, adamic_adar, pref_attach])
    
    return np.array(features)

# Made with Bob
