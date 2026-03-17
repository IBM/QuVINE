# Random Graph Generator Guide for QuVINE

This guide explains how to use the random graph generators in QuVINE for testing and evaluating embedding algorithms.

## Overview

The `quvine.data.random_graphs` module provides functions to generate various types of random graphs with known structures or specific properties that are suitable for embedding tasks. These graphs can be used for:

- Testing embedding algorithms
- Benchmarking performance
- Evaluating graph neural networks
- Understanding how different graph structures affect embeddings

## Installation

The random graph generators are included in the QuVINE package. No additional dependencies are required beyond the standard QuVINE requirements.

## Quick Start

```python
from quvine.data.random_graphs import (
    generate_barabasi_albert,
    generate_graph_with_seeds_and_targets,
    get_graph_statistics
)

# Generate a scale-free network
G = generate_barabasi_albert(n=100, m=3, seed=42)

# Get statistics
stats = get_graph_statistics(G)
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")

# Generate a graph with seeds and targets for embedding evaluation
G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=100,
    num_seeds=10,
    num_targets=15,
    graph_type='barabasi_albert',
    m=3,
    seed=42
)
```

## Available Graph Types

### 1. Erdős-Rényi Random Graphs

Classic random graphs where edges are added with uniform probability.

```python
from quvine.data.random_graphs import generate_erdos_renyi

# G(n,p) model - probability of each edge
G = generate_erdos_renyi(n=100, p=0.05, seed=42)

# G(n,m) model - exact number of edges
G = generate_erdos_renyi(n=100, m=250, seed=42)
```

**Use cases:**
- Baseline comparisons
- Testing algorithms on random structure
- Understanding null models

### 2. Barabási-Albert Scale-Free Networks

Networks with power-law degree distribution, common in real-world networks.

```python
from quvine.data.random_graphs import generate_barabasi_albert

G = generate_barabasi_albert(n=100, m=3, seed=42)
```

**Parameters:**
- `n`: Number of nodes
- `m`: Number of edges to attach from new node to existing nodes
- `seed`: Random seed

**Use cases:**
- Modeling biological networks (PPI, metabolic)
- Social networks
- Citation networks
- Testing hub detection

### 3. Watts-Strogatz Small-World Networks

Networks with high clustering and short path lengths.

```python
from quvine.data.random_graphs import generate_watts_strogatz

G = generate_watts_strogatz(n=100, k=6, p=0.3, seed=42)
```

**Parameters:**
- `n`: Number of nodes
- `k`: Each node connected to k nearest neighbors
- `p`: Rewiring probability
- `seed`: Random seed

**Use cases:**
- Neural networks
- Social networks
- Testing clustering-aware embeddings

### 4. Powerlaw Cluster Graphs

Graphs with power-law degree distribution and high clustering.

```python
from quvine.data.random_graphs import generate_powerlaw_cluster

G = generate_powerlaw_cluster(n=100, m=3, p=0.3, seed=42)
```

**Use cases:**
- Biological networks with clustering
- Combining scale-free and clustering properties

### 5. Stochastic Block Models

Graphs with explicit community structure.

```python
from quvine.data.random_graphs import generate_stochastic_block_model

sizes = [30, 40, 30]
p_matrix = [
    [0.4, 0.05, 0.01],
    [0.05, 0.3, 0.1],
    [0.01, 0.1, 0.5]
]

G = generate_stochastic_block_model(sizes, p_matrix, seed=42)
```

**Use cases:**
- Testing community detection
- Evaluating embeddings on modular structures
- Controlled community experiments

### 6. Modular Networks

Simplified interface for creating networks with communities.

```python
from quvine.data.random_graphs import generate_modular_network

G, communities = generate_modular_network(
    num_communities=5,
    nodes_per_community=20,
    p_intra=0.3,  # Within-community edge probability
    p_inter=0.01,  # Between-community edge probability
    seed=42
)
```

**Returns:**
- `G`: NetworkX graph with 'community' node attribute
- `communities`: Dict mapping node ID to community ID

**Use cases:**
- Disease module analysis
- Functional module detection
- Community-aware embeddings

### 7. Hierarchical Networks

Tree-like structures with additional random edges.

```python
from quvine.data.random_graphs import generate_hierarchical_network

G, node_levels = generate_hierarchical_network(
    levels=4,
    branching_factor=3,
    p_level=0.1,
    seed=42
)
```

**Returns:**
- `G`: NetworkX graph with 'level' node attribute
- `node_levels`: Dict mapping node ID to hierarchy level

**Use cases:**
- Ontology structures
- Organizational networks
- Taxonomies

### 8. Core-Periphery Networks

Networks with dense core and sparse periphery.

```python
from quvine.data.random_graphs import generate_core_periphery

G, core_nodes, periphery_nodes = generate_core_periphery(
    n_core=20,
    n_periphery=80,
    p_core=0.5,
    p_core_periphery=0.1,
    p_periphery=0.01,
    seed=42
)
```

**Returns:**
- `G`: NetworkX graph with 'type' node attribute
- `core_nodes`: Set of core node IDs
- `periphery_nodes`: Set of periphery node IDs

**Use cases:**
- Biological networks (essential vs. non-essential genes)
- Social networks (influencers vs. regular users)
- Testing centrality-aware embeddings

### 9. Bipartite Networks

Two-mode networks with edges only between different node sets.

```python
from quvine.data.random_graphs import generate_bipartite_random

G, set1, set2 = generate_bipartite_random(
    n1=30,
    n2=50,
    p=0.1,
    seed=42
)
```

**Use cases:**
- Gene-disease associations
- Drug-target interactions
- User-item networks

### 10. Random Geometric Graphs

Nodes in space, edges based on distance.

```python
from quvine.data.random_graphs import generate_random_geometric

G = generate_random_geometric(n=100, radius=0.15, dim=2, seed=42)
```

**Use cases:**
- Spatial networks
- Sensor networks
- Testing distance-based embeddings

## Advanced Features

### Adding Hub Nodes

Add highly connected hub nodes to any existing graph.

```python
from quvine.data.random_graphs import add_hub_nodes

G = generate_erdos_renyi(n=80, p=0.05, seed=42)
G, hub_nodes = add_hub_nodes(G, num_hubs=5, hub_degree=20, seed=42)
```

### Graphs with Seeds and Targets

Generate graphs with designated seed and target nodes for embedding evaluation.

```python
from quvine.data.random_graphs import generate_graph_with_seeds_and_targets

G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=100,
    num_seeds=10,
    num_targets=15,
    graph_type='barabasi_albert',
    m=3,
    seed=42
)
```

**Supported graph types:**
- `'erdos_renyi'`: Additional kwargs: `p`
- `'barabasi_albert'`: Additional kwargs: `m`
- `'watts_strogatz'`: Additional kwargs: `k`, `p`
- `'powerlaw_cluster'`: Additional kwargs: `m`, `p`
- `'modular'`: Additional kwargs: `num_communities`, `p_intra`, `p_inter`

**Returns:**
- `G`: NetworkX graph with 'role' node attribute ('seed', 'target', or 'regular')
- `seeds`: List of seed node IDs
- `targets`: List of target node IDs

### Graph Statistics

Compute comprehensive statistics for any graph.

```python
from quvine.data.random_graphs import get_graph_statistics

stats = get_graph_statistics(G)
print(stats)
```

**Returns dictionary with:**
- `num_nodes`: Number of nodes
- `num_edges`: Number of edges
- `density`: Graph density
- `is_connected`: Whether graph is connected
- `avg_degree`: Average node degree
- `max_degree`: Maximum node degree
- `min_degree`: Minimum node degree
- `diameter`: Graph diameter (if connected)
- `avg_shortest_path`: Average shortest path length (if connected)
- `num_components`: Number of connected components (if not connected)
- `largest_cc_size`: Size of largest connected component (if not connected)
- `avg_clustering`: Average clustering coefficient
- `transitivity`: Graph transitivity

## Integration with QuVINE Pipeline

### Example: Testing Embeddings on Different Graph Types

```python
from quvine.data.random_graphs import (
    generate_barabasi_albert,
    generate_modular_network,
    generate_graph_with_seeds_and_targets
)
from quvine.embedding.word2vec import train_word2vec
from quvine.walks.rwr import random_walk_with_restart

# Generate test graph
G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=500,
    num_seeds=20,
    num_targets=30,
    graph_type='modular',
    num_communities=10,
    p_intra=0.3,
    p_inter=0.01,
    seed=42
)

# Use with QuVINE walks and embeddings
# (integrate with your existing pipeline)
```

### Example: Benchmarking on Multiple Graph Types

```python
from quvine.data.random_graphs import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_modular_network
)

graph_configs = [
    ('Erdős-Rényi', lambda: generate_erdos_renyi(n=100, p=0.05, seed=42)),
    ('Scale-Free', lambda: generate_barabasi_albert(n=100, m=3, seed=42)),
    ('Small-World', lambda: generate_watts_strogatz(n=100, k=6, p=0.3, seed=42)),
    ('Modular', lambda: generate_modular_network(5, 20, 0.3, 0.01, seed=42)[0]),
]

results = {}
for name, graph_fn in graph_configs:
    G = graph_fn()
    # Run your embedding pipeline
    # results[name] = evaluate_embedding(G)
```

## Best Practices

1. **Always use seeds**: Set `seed` parameter for reproducibility
2. **Check connectivity**: Use `get_graph_statistics()` to verify graph properties
3. **Start small**: Test with smaller graphs (n=100-500) before scaling up
4. **Match real data**: Choose graph types that match your real-world data characteristics
5. **Multiple trials**: Run experiments with different seeds to ensure robustness

## Examples

See `examples/random_graph_examples.py` for comprehensive examples of all graph types.

Run examples:
```bash
python examples/random_graph_examples.py
```

## References

- Erdős, P., & Rényi, A. (1960). On the evolution of random graphs.
- Barabási, A. L., & Albert, R. (1999). Emergence of scaling in random networks.
- Watts, D. J., & Strogatz, S. H. (1998). Collective dynamics of 'small-world' networks.
- Holland, P. W., et al. (1983). Stochastic blockmodels.

## Support

For issues or questions, please refer to the main QuVINE documentation or open an issue on GitHub.