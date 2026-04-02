# Quantum Advantage Formula & Downstream Tasks Analysis

## 1. Quantum Advantage Formula: Arithmetic vs Geometric Mean

### Current Formula (Arithmetic Mean)

The current `quantum_advantage_score` uses a **weighted arithmetic mean**:

```python
quantum_advantage_score = (
    0.3 * spectral_dimension_norm +
    0.25 * modularity +
    0.2 * (1 - path_length_ratio) +
    0.15 * clustering_mean +
    0.1 * degree_heterogeneity
)
```

### Proposed Alternative: Geometric Mean

```python
quantum_advantage_score_geometric = (
    spectral_dimension_norm^0.3 *
    modularity^0.25 *
    (1 - path_length_ratio)^0.2 *
    clustering_mean^0.15 *
    degree_heterogeneity^0.1
)
```

### Comparison & Recommendation

| Aspect | Arithmetic Mean | Geometric Mean |
|--------|----------------|----------------|
| **Sensitivity** | Linear combination | Multiplicative combination |
| **Zero handling** | One zero component doesn't kill score | One zero component → score = 0 |
| **Outliers** | Sensitive to large values | Dampens effect of outliers |
| **Interpretation** | Additive contributions | Synergistic interactions |
| **Range** | [0, 1] (with normalization) | [0, 1] (naturally bounded) |
| **Physical meaning** | Independent feature contributions | Coupled feature interactions |

### **Recommendation: Use BOTH**

**Rationale:**
1. **Arithmetic mean** captures **additive quantum advantage** where features contribute independently
2. **Geometric mean** captures **synergistic quantum advantage** where features must co-occur
3. **Empirical validation** needed to determine which correlates better with actual quantum speedup

**Implementation Strategy:**
```python
# Compute both
qa_arithmetic = weighted_arithmetic_mean(features, weights)
qa_geometric = weighted_geometric_mean(features, weights)

# Also compute harmonic mean for completeness
qa_harmonic = weighted_harmonic_mean(features, weights)

# Return all three for analysis
return {
    'quantum_advantage_arithmetic': qa_arithmetic,
    'quantum_advantage_geometric': qa_geometric,
    'quantum_advantage_harmonic': qa_harmonic,
    'quantum_advantage_score': qa_arithmetic,  # Keep current as default
}
```

**Analysis Plan:**
- Correlate all three with actual embedding performance (precision@K, recall@K)
- Identify which formula best predicts quantum advantage
- Consider ensemble: `qa_combined = α * qa_arith + β * qa_geom + γ * qa_harm`

---

## 2. Downstream Tasks Beyond Node Prioritization

### Current Task: Node Prioritization
- **Task**: Rank nodes by relevance to seed set
- **Metrics**: Precision@K, Recall@K, F1@K, NDCG@K
- **Use case**: Disease gene discovery, drug target identification

### Proposed Additional Tasks

#### 2.1 Node Classification

**Task Description:**
Predict node labels (classes) using learned embeddings.

**Label Generation Strategies:**

1. **Community-Based Labels**
   ```python
   def generate_community_labels(G, method='louvain'):
       """
       Assign labels based on community detection.
       
       Methods:
       - louvain: Louvain community detection
       - label_propagation: Label propagation algorithm
       - spectral: Spectral clustering on Laplacian
       """
       if method == 'louvain':
           import community as community_louvain
           partition = community_louvain.best_partition(G)
           labels = np.array([partition[node] for node in G.nodes()])
       elif method == 'label_propagation':
           communities = nx.community.label_propagation_communities(G)
           labels = np.zeros(G.number_of_nodes(), dtype=int)
           for i, comm in enumerate(communities):
               for node in comm:
                   labels[node] = i
       elif method == 'spectral':
           from sklearn.cluster import SpectralClustering
           A = nx.to_numpy_array(G)
           n_clusters = estimate_n_clusters(G)  # Use eigengap heuristic
           clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
           labels = clustering.fit_predict(A)
       return labels
   ```

2. **Degree-Based Labels** (Structural roles)
   ```python
   def generate_degree_labels(G, n_bins=5):
       """
       Bin nodes by degree into structural role classes.
       
       Classes:
       - Low degree (hubs)
       - Medium degree (connectors)
       - High degree (periphery)
       """
       degrees = np.array([d for _, d in G.degree()])
       labels = np.digitize(degrees, bins=np.percentile(degrees, np.linspace(0, 100, n_bins+1)))
       return labels
   ```

3. **Centrality-Based Labels**
   ```python
   def generate_centrality_labels(G, metric='betweenness', n_bins=5):
       """
       Bin nodes by centrality into importance classes.
       
       Metrics:
       - betweenness: Betweenness centrality
       - closeness: Closeness centrality
       - eigenvector: Eigenvector centrality
       - pagerank: PageRank
       """
       if metric == 'betweenness':
           centrality = nx.betweenness_centrality(G)
       elif metric == 'closeness':
           centrality = nx.closeness_centrality(G)
       elif metric == 'eigenvector':
           centrality = nx.eigenvector_centrality(G, max_iter=1000)
       elif metric == 'pagerank':
           centrality = nx.pagerank(G)
       
       values = np.array([centrality[node] for node in G.nodes()])
       labels = np.digitize(values, bins=np.percentile(values, np.linspace(0, 100, n_bins+1)))
       return labels
   ```

4. **Core-Periphery Labels**
   ```python
   def generate_core_periphery_labels(G, method='k_core'):
       """
       Identify core vs periphery nodes.
       
       Methods:
       - k_core: K-core decomposition
       - rich_club: Rich-club coefficient
       """
       if method == 'k_core':
           core_numbers = nx.core_number(G)
           max_core = max(core_numbers.values())
           # Binary: core (top 20%) vs periphery
           threshold = np.percentile(list(core_numbers.values()), 80)
           labels = np.array([1 if core_numbers[node] >= threshold else 0 
                             for node in G.nodes()])
       return labels
   ```

**Evaluation Metrics:**
- Accuracy, Precision, Recall, F1-score (macro/micro/weighted)
- Confusion matrix
- ROC-AUC (for binary classification)
- Silhouette score (clustering quality)

**Implementation:**
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def evaluate_node_classification(embeddings, labels, test_size=0.3):
    """
    Evaluate node classification performance.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings (n_nodes, embedding_dim)
    labels : np.ndarray
        Node labels (n_nodes,)
    test_size : float
        Fraction of nodes for testing
        
    Returns
    -------
    dict
        Classification metrics
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=test_size, stratify=labels, random_state=42
    )
    
    clf = LogisticRegression(max_iter=1000, multi_class='ovr')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'n_classes': len(np.unique(labels)),
        'n_train': len(X_train),
        'n_test': len(X_test),
    }
```

#### 2.2 Link Prediction

**Task Description:**
Predict missing or future edges in the graph.

**Edge Sampling Strategies:**

1. **Random Negative Sampling**
   ```python
   def sample_edges(G, test_frac=0.1, neg_pos_ratio=1.0):
       """
       Sample positive and negative edges for link prediction.
       
       Parameters
       ----------
       G : nx.Graph
           Input graph
       test_frac : float
           Fraction of edges to hold out for testing
       neg_pos_ratio : float
           Ratio of negative to positive samples
           
       Returns
       -------
       train_edges, test_edges_pos, test_edges_neg
       """
       edges = list(G.edges())
       n_test = int(len(edges) * test_frac)
       
       # Sample positive test edges
       test_edges_pos = random.sample(edges, n_test)
       train_edges = [e for e in edges if e not in test_edges_pos]
       
       # Sample negative edges (non-edges)
       non_edges = list(nx.non_edges(G))
       n_neg = int(n_test * neg_pos_ratio)
       test_edges_neg = random.sample(non_edges, n_neg)
       
       return train_edges, test_edges_pos, test_edges_neg
   ```

2. **Temporal Edge Sampling** (if timestamps available)
   ```python
   def temporal_edge_split(G, timestamp_attr='timestamp', split_time=None):
       """
       Split edges by timestamp for temporal link prediction.
       """
       if split_time is None:
           timestamps = [G.edges[e][timestamp_attr] for e in G.edges()]
           split_time = np.median(timestamps)
       
       train_edges = [e for e in G.edges() 
                     if G.edges[e][timestamp_attr] < split_time]
       test_edges = [e for e in G.edges() 
                    if G.edges[e][timestamp_attr] >= split_time]
       
       return train_edges, test_edges
   ```

**Edge Features from Embeddings:**
```python
def compute_edge_features(embeddings, edge_list, method='hadamard'):
    """
    Compute edge features from node embeddings.
    
    Methods:
    - hadamard: Element-wise product (u ⊙ v)
    - average: Element-wise average ((u + v) / 2)
    - l1: L1 distance (|u - v|)
    - l2: L2 distance (||u - v||)
    - concat: Concatenation ([u; v])
    """
    features = []
    for u, v in edge_list:
        emb_u = embeddings[u]
        emb_v = embeddings[v]
        
        if method == 'hadamard':
            feat = emb_u * emb_v
        elif method == 'average':
            feat = (emb_u + emb_v) / 2
        elif method == 'l1':
            feat = np.abs(emb_u - emb_v)
        elif method == 'l2':
            feat = np.linalg.norm(emb_u - emb_v)
        elif method == 'concat':
            feat = np.concatenate([emb_u, emb_v])
        
        features.append(feat)
    
    return np.array(features)
```

**Evaluation Metrics:**
- AUC-ROC, AUC-PR
- Precision@K, Recall@K for top-K predictions
- Hit@K (fraction of test edges in top-K)
- MRR (Mean Reciprocal Rank)

**Implementation:**
```python
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate_link_prediction(embeddings, test_edges_pos, test_edges_neg, method='hadamard'):
    """
    Evaluate link prediction performance.
    
    Parameters
    ----------
    embeddings : np.ndarray
        Node embeddings
    test_edges_pos : list
        Positive test edges
    test_edges_neg : list
        Negative test edges
    method : str
        Edge feature computation method
        
    Returns
    -------
    dict
        Link prediction metrics
    """
    # Compute edge features
    X_pos = compute_edge_features(embeddings, test_edges_pos, method=method)
    X_neg = compute_edge_features(embeddings, test_edges_neg, method=method)
    
    # Create labels
    y_pos = np.ones(len(X_pos))
    y_neg = np.zeros(len(X_neg))
    
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([y_pos, y_neg])
    
    # Train classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    auc_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    ap_scores = cross_val_score(clf, X, y, cv=5, scoring='average_precision')
    
    return {
        'auc_roc_mean': auc_scores.mean(),
        'auc_roc_std': auc_scores.std(),
        'auc_pr_mean': ap_scores.mean(),
        'auc_pr_std': ap_scores.std(),
        'n_pos': len(test_edges_pos),
        'n_neg': len(test_edges_neg),
    }
```

#### 2.3 Graph Reconstruction

**Task Description:**
Reconstruct the original graph from embeddings.

**Metrics:**
- Adjacency matrix reconstruction error (Frobenius norm)
- Edge overlap (Jaccard similarity)
- Degree distribution KL divergence

```python
def evaluate_graph_reconstruction(G, embeddings, threshold=0.5):
    """
    Evaluate graph reconstruction from embeddings.
    
    Parameters
    ----------
    G : nx.Graph
        Original graph
    embeddings : np.ndarray
        Node embeddings
    threshold : float
        Threshold for edge prediction
        
    Returns
    -------
    dict
        Reconstruction metrics
    """
    # Compute similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Predict edges
    n = len(embeddings)
    pred_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i, j] > threshold:
                pred_edges.add((i, j))
    
    # Compare with original
    true_edges = set(G.edges())
    
    # Jaccard similarity
    intersection = len(pred_edges & true_edges)
    union = len(pred_edges | true_edges)
    jaccard = intersection / union if union > 0 else 0.0
    
    # Precision, Recall
    precision = intersection / len(pred_edges) if len(pred_edges) > 0 else 0.0
    recall = intersection / len(true_edges) if len(true_edges) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'jaccard': jaccard,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_true_edges': len(true_edges),
        'n_pred_edges': len(pred_edges),
    }
```

---

## 3. Implementation Plan

### Phase 1: Quantum Advantage Formula Enhancement
1. Implement geometric and harmonic mean variants
2. Add all three to `compute_quantum_advantage_metrics()`
3. Update `comprehensive_embedding_analysis.py` to compute all variants
4. Correlate with embedding performance

### Phase 2: Node Classification
1. Implement label generation functions (community, degree, centrality, core-periphery)
2. Add `evaluate_node_classification()` to evaluation module
3. Integrate into comprehensive analysis pipeline
4. Generate classification results for all networks and methods

### Phase 3: Link Prediction
1. Implement edge sampling functions
2. Add `evaluate_link_prediction()` to evaluation module
3. Integrate into comprehensive analysis pipeline
4. Generate link prediction results

### Phase 4: Analysis & Visualization
1. Correlate complexity metrics with all downstream tasks
2. Identify which complexity metrics predict performance on each task
3. Create task-specific method recommendations
4. Generate comprehensive visualization dashboard

---

## 4. Expected Insights

### Quantum Advantage Formula
- **Hypothesis**: Geometric mean will better predict quantum advantage when **all** structural features are present (synergistic)
- **Hypothesis**: Arithmetic mean will better predict when **any** feature is sufficient (additive)
- **Validation**: Empirical correlation with actual quantum speedup

### Node Classification
- **Hypothesis**: Quantum methods excel when classes have distinct topological signatures
- **Hypothesis**: Community-based labels → quantum advantage in modular networks
- **Hypothesis**: Centrality-based labels → quantum advantage in scale-free networks

### Link Prediction
- **Hypothesis**: Quantum methods excel in predicting long-range connections (high Kirchhoff index)
- **Hypothesis**: Classical methods excel in predicting local triangles (high clustering)

### Task-Complexity Relationships
- **High cyclomatic number** (many loops) → quantum advantage in node prioritization
- **High Kirchhoff index** (bottlenecks) → quantum advantage in link prediction
- **Negative ORC** (bottleneck edges) → quantum advantage in community detection/classification

---

## 5. Deliverables

1. **Enhanced complexity metrics** with topological features
2. **Multiple quantum advantage formulas** (arithmetic, geometric, harmonic)
3. **Node classification** evaluation pipeline
4. **Link prediction** evaluation pipeline
5. **Comprehensive correlation analysis** between complexity and all tasks
6. **Task-specific method recommendations** based on network characteristics
7. **Visualization dashboard** showing all relationships

This comprehensive approach will provide deep insights into when and why quantum embedding methods outperform classical approaches across different downstream tasks.