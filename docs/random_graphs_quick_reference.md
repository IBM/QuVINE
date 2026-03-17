# Random Graph Generators - Quick Reference

## Import

```python
from quvine.data.random_graphs import *
# or
from quvine.data import generate_barabasi_albert, get_graph_statistics
```

## Basic Generators

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `generate_erdos_renyi()` | Random graph with uniform edge probability | `n`, `p` or `m`, `seed` |
| `generate_barabasi_albert()` | Scale-free network (power-law degree) | `n`, `m`, `seed` |
| `generate_watts_strogatz()` | Small-world network | `n`, `k`, `p`, `seed` |
| `generate_powerlaw_cluster()` | Power-law with clustering | `n`, `m`, `p`, `seed` |
| `generate_random_geometric()` | Spatial network | `n`, `radius`, `dim`, `seed` |

## Structured Generators

| Function | Description | Returns |
|----------|-------------|---------|
| `generate_stochastic_block_model()` | Communities with custom structure | `Graph` |
| `generate_modular_network()` | Simple modular structure | `Graph, communities_dict` |
| `generate_hierarchical_network()` | Tree-like hierarchy | `Graph, levels_dict` |
| `generate_core_periphery()` | Dense core + sparse periphery | `Graph, core_set, periphery_set` |
| `generate_bipartite_random()` | Two-mode network | `Graph, set1, set2` |

## Special Functions

| Function | Description | Use Case |
|----------|-------------|----------|
| `add_hub_nodes()` | Add hubs to existing graph | Testing hub detection |
| `generate_graph_with_seeds_and_targets()` | Graph with labeled nodes | Embedding evaluation |
| `get_graph_statistics()` | Compute graph metrics | Analysis & validation |

## Common Patterns

### Generate for Embedding Testing
```python
G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=100, num_seeds=10, num_targets=15,
    graph_type='barabasi_albert', m=3, seed=42
)
```

### Generate with Communities
```python
G, communities = generate_modular_network(
    num_communities=5, nodes_per_community=20,
    p_intra=0.3, p_inter=0.01, seed=42
)
```

### Generate and Analyze
```python
G = generate_barabasi_albert(n=100, m=3, seed=42)
stats = get_graph_statistics(G)
print(f"Avg degree: {stats['avg_degree']:.2f}")
```

## Graph Type Selection Guide

| Real-World Network | Recommended Generator |
|-------------------|----------------------|
| Protein-Protein Interaction | `generate_barabasi_albert` or `generate_powerlaw_cluster` |
| Gene Regulatory | `generate_hierarchical_network` |
| Disease Modules | `generate_modular_network` |
| Drug-Target | `generate_bipartite_random` |
| Social Network | `generate_watts_strogatz` or `generate_barabasi_albert` |
| Spatial/Sensor | `generate_random_geometric` |

## Parameter Guidelines

### Node Count (`n`)
- Small tests: 50-100
- Medium tests: 100-500
- Large tests: 500-5000
- Production: Match your real data size

### Edge Density
- **Sparse** (biological): `p=0.01-0.05` or `m=2-5`
- **Medium**: `p=0.05-0.15` or `m=5-10`
- **Dense**: `p=0.15-0.5` or `m=10-20`

### Community Structure
- **Strong**: `p_intra=0.3-0.5`, `p_inter=0.01-0.05`
- **Moderate**: `p_intra=0.2-0.3`, `p_inter=0.05-0.1`
- **Weak**: `p_intra=0.1-0.2`, `p_inter=0.05-0.15`

## Best Practices

1. ✅ Always set `seed` for reproducibility
2. ✅ Use `get_graph_statistics()` to verify properties
3. ✅ Start with small graphs for testing
4. ✅ Match graph type to your research question
5. ✅ Run multiple trials with different seeds

## Examples

See:
- `examples/random_graph_examples.py` - Comprehensive examples
- `tests/test_random_graphs.py` - Unit tests
- `docs/random_graphs_guide.md` - Full documentation