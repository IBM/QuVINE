ple# Hyperparameter Reference Guide

This document provides detailed descriptions of all hyperparameters for each method in the tuning configuration.

---

## Table of Contents
1. [Quantum Methods](#quantum-methods)
2. [Filter-Based Methods](#filter-based-methods)
3. [Random Walk Methods](#random-walk-methods)
4. [Graph Neural Networks](#graph-neural-networks)
5. [Common Parameters](#common-parameters)

---

## Quantum Methods

### quvine_walks

QuVINE (Quantum-inspired View-based Network Embedding) uses quantum walks on multiple graph views.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Dimensionality of the output embedding vectors | Higher = more expressive but slower, risk of overfitting |
| `num_views` | int | [2, 3, 4, 5] | Number of different graph views to generate and combine | More views = better coverage of graph structure but slower |
| `walk_length` | int | [10, 20, 40, 80] | Length of each random walk sequence | Longer = captures more distant relationships but slower |
| `num_walks` | int | [10, 20, 40, 80] | Number of walks to generate per node | More walks = better sampling but slower |
| `window_size` | int | [5, 10, 15, 20] | Context window size for Word2Vec training | Larger = considers more distant co-occurrences |
| `negative_samples` | int | [1, 5, 10, 20] | Number of negative samples for Word2Vec | More = better discrimination but slower |
| `epochs` | int | [5, 10, 20] | Number of training epochs for Word2Vec | More = better convergence but slower |
| `restart_prob` | float | [0.1, 0.15, 0.2, 0.3] | Probability of restarting walk at source node | Higher = more local exploration, lower = more global |
| `max_degree` | int | [30, 50, 100] | Maximum node degree to consider in view construction | Lower = focuses on core structure, higher = includes hubs |
| `degree_alpha` | float | [0.3, 0.5, 0.7] | Degree normalization parameter for view weighting | Controls how much to weight high-degree nodes |

#### Recommended Starting Points
- **Small graphs (<500 nodes)**: `embedding_dim=64, num_views=3, walk_length=40, num_walks=10`
- **Medium graphs (500-5000 nodes)**: `embedding_dim=128, num_views=3, walk_length=80, num_walks=20`
- **Large graphs (>5000 nodes)**: `embedding_dim=256, num_views=4, walk_length=80, num_walks=40`

---

## Filter-Based Methods

### baseline_filter_heat

Heat kernel diffusion filter that simulates heat propagation on the graph.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Dimensionality of output embeddings | Higher = more expressive |
| `tau` | float | [0.1, 0.5, 1.0, 2.0, 5.0, 10.0] | Heat diffusion time parameter | Higher = more global diffusion, lower = more local |
| `filter_order` | int | [2, 3, 4, 5, 6, 8, 10] | Order of polynomial approximation | Higher = more accurate but slower |

#### How It Works
- Applies heat kernel: `H(t) = exp(-t * L)` where L is the graph Laplacian
- `tau` controls diffusion time: small τ = local structure, large τ = global structure
- `filter_order` determines approximation accuracy via Chebyshev polynomials

#### Recommended Starting Points
- **Local structure**: `tau=0.5, filter_order=5`
- **Global structure**: `tau=5.0, filter_order=10`
- **Balanced**: `tau=2.0, filter_order=6`

---

### baseline_filter_poly

Polynomial filter that applies polynomial transformations to the graph Laplacian.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Dimensionality of output embeddings | Higher = more expressive |
| `filter_order` | int | [2, 3, 4, 5, 6, 8, 10, 15, 20] | Degree of polynomial filter | Higher = captures longer-range dependencies |
| `alpha` | float | [0.1, 0.3, 0.5, 0.7, 0.9] | Polynomial coefficient weighting | Controls balance between local and global |

#### How It Works
- Applies polynomial filter: `P(L) = Σ α^k L^k`
- `filter_order` = maximum polynomial degree (k-hop neighborhood)
- `alpha` controls decay: small α = emphasize local, large α = emphasize global

#### Recommended Starting Points
- **Local focus**: `filter_order=5, alpha=0.3`
- **Global focus**: `filter_order=15, alpha=0.7`
- **Balanced**: `filter_order=10, alpha=0.5`

---

### baseline_gcnmf

Graph Convolutional Network with Matrix Factorization.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Final embedding dimensionality | Higher = more expressive |
| `n_layers` | int | [2, 3, 4, 5] | Number of GCN layers | More layers = larger receptive field |
| `window_size` | int | [5, 10, 15, 20] | Context window for matrix factorization | Larger = considers more distant neighbors |
| `negative_samples` | int | [1, 5, 10, 20] | Number of negative samples | More = better discrimination |
| `learning_rate` | float | [0.001, 0.005, 0.01, 0.05] | Optimization learning rate | Higher = faster but less stable |
| `epochs` | int | [50, 100, 200] | Number of training epochs | More = better convergence |

#### How It Works
- Combines GCN message passing with matrix factorization
- `n_layers` determines how many hops of neighbors to aggregate
- Matrix factorization learns low-rank approximation of proximity matrix

#### Recommended Starting Points
- **Fast training**: `n_layers=2, epochs=50, learning_rate=0.01`
- **Better quality**: `n_layers=3, epochs=200, learning_rate=0.005`

---

## Random Walk Methods

### node2vec

Biased random walk method with return and in-out parameters.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Dimensionality of embeddings | Higher = more expressive |
| `walk_length` | int | [10, 20, 40, 80] | Length of each random walk | Longer = captures more context |
| `num_walks` | int | [10, 20, 40, 80] | Number of walks per node | More = better sampling |
| `p` | float | [0.25, 0.5, 1.0, 2.0, 4.0] | Return parameter (likelihood of returning to previous node) | Low p = BFS-like, high p = DFS-like |
| `q` | float | [0.25, 0.5, 1.0, 2.0, 4.0] | In-out parameter (likelihood of exploring outward) | Low q = outward exploration, high q = local |
| `window_size` | int | [5, 10, 15, 20] | Context window for Skip-gram | Larger = considers more distant co-occurrences |
| `negative_samples` | int | [1, 5, 10, 20] | Negative samples for Skip-gram | More = better discrimination |
| `epochs` | int | [5, 10, 20] | Training epochs | More = better convergence |

#### How p and q Work
- **p (return parameter)**:
  - `p < 1`: Encourages returning to previous node (local exploration)
  - `p = 1`: Unbiased
  - `p > 1`: Discourages returning (more exploration)

- **q (in-out parameter)**:
  - `q < 1`: Encourages outward exploration (BFS-like)
  - `q = 1`: Unbiased
  - `q > 1`: Encourages staying close (DFS-like)

#### Recommended Combinations
- **Homophily (similar nodes connect)**: `p=1.0, q=0.5` (BFS-like)
- **Structural equivalence**: `p=1.0, q=2.0` (DFS-like)
- **Balanced**: `p=1.0, q=1.0` (unbiased)

---

### netmf

Network Embedding as Matrix Factorization - implicit matrix factorization approach.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Dimensionality of embeddings | Higher = more expressive |
| `window_size` | int | [5, 10, 15, 20] | Context window size | Larger = considers more distant neighbors |
| `rank` | int | [64, 128, 256, 512] | Rank of matrix factorization | Higher = more accurate approximation |
| `negative_samples` | int | [1, 5, 10, 20] | Number of negative samples | More = better discrimination |

#### How It Works
- Factorizes implicit proximity matrix derived from random walks
- `window_size` determines which nodes are considered "close"
- `rank` controls approximation quality (can be > embedding_dim)

#### Recommended Starting Points
- **Fast**: `rank=128, window_size=10`
- **Accurate**: `rank=512, window_size=20`

---

## Graph Neural Networks

### graphsage

Graph Sample and Aggregate - inductive GNN with neighborhood sampling.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Output embedding dimensionality | Higher = more expressive |
| `n_layers` | int | [2, 3, 4] | Number of aggregation layers | More = larger receptive field |
| `hidden_dim` | int | [64, 128, 256, 512] | Hidden layer dimensionality | Higher = more capacity |
| `aggregator` | str | ["mean", "gcn", "pool", "lstm"] | Aggregation function type | Different inductive biases |
| `dropout` | float | [0.0, 0.1, 0.3, 0.5] | Dropout rate for regularization | Higher = more regularization |
| `learning_rate` | float | [0.001, 0.005, 0.01] | Optimization learning rate | Higher = faster but less stable |
| `epochs` | int | [50, 100, 200] | Number of training epochs | More = better convergence |
| `batch_size` | int | [32, 64, 128] | Mini-batch size | Larger = more stable but more memory |

#### Aggregator Types
- **mean**: Simple average of neighbor features (fast, effective)
- **gcn**: Graph Convolutional Network aggregation (normalized)
- **pool**: Max/mean pooling with MLP (more expressive)
- **lstm**: LSTM aggregation (order-sensitive, most expressive)

#### Recommended Starting Points
- **Fast training**: `n_layers=2, aggregator="mean", epochs=50`
- **Better quality**: `n_layers=3, aggregator="lstm", epochs=200`

---

### appnp

Approximate Personalized Propagation of Neural Predictions.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Output embedding dimensionality | Higher = more expressive |
| `hidden_dim` | int | [64, 128, 256, 512] | Hidden layer dimensionality | Higher = more capacity |
| `n_layers` | int | [2, 3, 4] | Number of neural network layers | More = more transformation |
| `alpha` | float | [0.05, 0.1, 0.15, 0.2] | Teleport probability (restart probability) | Higher = more local, lower = more global |
| `k_hops` | int | [5, 10, 15, 20] | Number of propagation steps | More = larger receptive field |
| `dropout` | float | [0.0, 0.1, 0.3, 0.5] | Dropout rate | Higher = more regularization |
| `learning_rate` | float | [0.001, 0.005, 0.01] | Optimization learning rate | Higher = faster but less stable |
| `epochs` | int | [50, 100, 200] | Number of training epochs | More = better convergence |

#### How It Works
- Separates feature transformation from propagation
- `alpha` controls PageRank-like propagation: high α = stay local, low α = propagate far
- `k_hops` determines propagation depth (independent of neural network depth)

#### Recommended Starting Points
- **Local structure**: `alpha=0.15, k_hops=10`
- **Global structure**: `alpha=0.05, k_hops=20`

---

### gat_baseline

Graph Attention Network - learns attention weights for neighbor aggregation.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Output embedding dimensionality | Higher = more expressive |
| `hidden_dim` | int | [64, 128, 256] | Hidden layer dimensionality | Higher = more capacity |
| `n_layers` | int | [2, 3, 4] | Number of GAT layers | More = larger receptive field |
| `n_heads` | int | [1, 2, 4, 8] | Number of attention heads | More = more diverse attention patterns |
| `dropout` | float | [0.0, 0.1, 0.3, 0.5] | Dropout rate | Higher = more regularization |
| `attn_dropout` | float | [0.0, 0.1, 0.3, 0.5] | Attention coefficient dropout | Higher = more robust attention |
| `learning_rate` | float | [0.001, 0.005, 0.01] | Optimization learning rate | Higher = faster but less stable |
| `epochs` | int | [50, 100, 200] | Number of training epochs | More = better convergence |
| `negative_slope` | float | [0.1, 0.2, 0.3] | LeakyReLU negative slope | Controls activation function |

#### How Attention Works
- Learns importance weights for each neighbor
- `n_heads` = multiple attention mechanisms in parallel (multi-head attention)
- `attn_dropout` prevents over-reliance on specific neighbors

#### Recommended Starting Points
- **Small graphs**: `n_heads=4, n_layers=2, hidden_dim=64`
- **Large graphs**: `n_heads=8, n_layers=3, hidden_dim=128`

---

### graphgps_baseline

Graph GPS (Graph Positional and Structural encoding) - combines local MPNN with global attention.

#### Parameters

| Parameter | Type | Range | Description | Impact |
|-----------|------|-------|-------------|--------|
| `embedding_dim` | int | [32, 64, 128, 256] | Output embedding dimensionality | Higher = more expressive |
| `hidden_dim` | int | [64, 128, 256] | Hidden layer dimensionality | Higher = more capacity |
| `n_layers` | int | [2, 3, 4, 6] | Number of GPS layers | More = more transformation |
| `n_heads` | int | [1, 2, 4, 8] | Number of attention heads | More = more diverse attention |
| `dropout` | float | [0.0, 0.1, 0.3, 0.5] | Dropout rate | Higher = more regularization |
| `attn_dropout` | float | [0.0, 0.1, 0.3, 0.5] | Attention dropout | Higher = more robust attention |
| `learning_rate` | float | [0.001, 0.005, 0.01] | Optimization learning rate | Higher = faster but less stable |
| `epochs` | int | [50, 100, 200] | Number of training epochs | More = better convergence |
| `mpnn_type` | str | ["gine", "gcn", "gin"] | Local message passing type | Different local aggregation strategies |
| `global_model_type` | str | ["Transformer", "Performer"] | Global attention type | Transformer = full attention, Performer = linear |

#### MPNN Types
- **gine**: Graph Isomorphism Network with edge features (most expressive)
- **gcn**: Graph Convolutional Network (normalized aggregation)
- **gin**: Graph Isomorphism Network (sum aggregation)

#### Global Model Types
- **Transformer**: Full self-attention (O(n²) complexity, most expressive)
- **Performer**: Linear attention approximation (O(n) complexity, faster)

#### Recommended Starting Points
- **Small graphs**: `mpnn_type="gcn", global_model_type="Transformer", n_layers=3`
- **Large graphs**: `mpnn_type="gine", global_model_type="Performer", n_layers=4`

---

## Common Parameters

### Embedding Dimension
**Range**: [32, 64, 128, 256, 512]

**Guidelines**:
- **32-64**: Small graphs (<500 nodes), fast prototyping
- **128**: Default for most applications, good balance
- **256-512**: Large graphs (>5000 nodes), complex structures

**Trade-offs**:
- Higher dimension = more expressive but slower and more memory
- Risk of overfitting with high dimensions on small graphs

---

### Learning Rate
**Range**: [0.0001, 0.001, 0.005, 0.01, 0.05]

**Guidelines**:
- **0.0001-0.001**: Stable, slow convergence, good for fine-tuning
- **0.005-0.01**: Default range, good balance
- **0.05+**: Fast but unstable, may diverge

**Tips**:
- Start high, reduce if training is unstable
- Use learning rate scheduling for best results

---

### Dropout
**Range**: [0.0, 0.1, 0.2, 0.3, 0.5, 0.6]

**Guidelines**:
- **0.0**: No regularization (risk of overfitting)
- **0.1-0.3**: Light regularization (default)
- **0.5-0.6**: Heavy regularization (small datasets)

**When to use**:
- High dropout for small datasets or complex models
- Low dropout for large datasets or simple models

---

### Number of Epochs
**Range**: [10, 50, 100, 200, 300, 500]

**Guidelines**:
- **10-50**: Fast methods (node2vec, netmf)
- **100-200**: GNN methods (default)
- **300-500**: Complex models or difficult tasks

**Tips**:
- Use early stopping to prevent overfitting
- Monitor validation performance

---

## Hyperparameter Tuning Strategies

### 1. Coarse-to-Fine Search
```yaml
# Stage 1: Coarse search (few values)
embedding_dim: [64, 128, 256]
learning_rate: [0.001, 0.01]

# Stage 2: Fine search (more values around best)
embedding_dim: [96, 128, 160]
learning_rate: [0.003, 0.005, 0.007]
```

### 2. Task-Specific Priorities

**Node Classification**:
- Priority: `embedding_dim`, `n_layers`, `dropout`
- Less important: `learning_rate`, `epochs`

**Link Prediction**:
- Priority: `embedding_dim`, `window_size`, `negative_samples`
- Less important: `dropout`

**Node Ranking**:
- Priority: `embedding_dim`, walk parameters (`p`, `q`, `walk_length`)
- Less important: `epochs`

### 3. Method-Specific Tips

**Random Walk Methods** (node2vec, netmf):
- Tune `p` and `q` together (they interact)
- `window_size` has large impact on performance

**GNN Methods** (GraphSAGE, GAT, GraphGPS):
- Start with 2 layers, increase if needed
- `dropout` is crucial for preventing overfitting
- `learning_rate` needs careful tuning

**Filter Methods** (heat, poly):
- `tau` or `alpha` has the largest impact
- `filter_order` affects accuracy vs speed trade-off

---

## Performance vs Computational Cost

### Fast Methods (< 1 minute for 1000 nodes)
- **netmf**: Low rank, small window
- **baseline_filter_heat**: Low filter order
- **baseline_filter_poly**: Low filter order

### Medium Methods (1-5 minutes for 1000 nodes)
- **node2vec**: Moderate walks and length
- **graphsage**: 2 layers, mean aggregator
- **appnp**: Moderate k_hops

### Slow Methods (> 5 minutes for 1000 nodes)
- **quvine_walks**: Multiple views, long walks
- **gat_baseline**: Many heads, many layers
- **graphgps_baseline**: Transformer, many layers

---

## Summary

This reference provides detailed information about all hyperparameters. Key takeaways:

1. **Start with defaults** from "Recommended Starting Points"
2. **Tune systematically**: One parameter at a time or use Optuna
3. **Consider trade-offs**: Performance vs speed vs memory
4. **Task matters**: Different tasks benefit from different parameters
5. **Graph size matters**: Adjust parameters based on graph size

For automated tuning, use the `tuning_config.yaml` with appropriate ranges for your use case.