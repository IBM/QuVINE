# Graph Complexity Metrics Guide for QuVINE

This guide explains how to compute and interpret graph complexity metrics in QuVINE, including Laplacian-based measures and quantum-inspired complexity metrics.

## Overview

Graph complexity metrics help characterize the structural properties of networks and can be used to:
- **Select appropriate graphs** for testing embedding algorithms
- **Understand embedding difficulty** - more complex graphs may require more sophisticated methods
- **Compare graph structures** - quantify differences between network types
- **Predict quantum advantage** - identify graphs where quantum walks may outperform classical methods

## Installation

The complexity metrics are included in the QuVINE package. They require:
- `numpy`
- `scipy`
- `networkx`

All dependencies are included in the standard QuVINE installation.

## Quick Start

```python
from quvine.data import generate_barabasi_albert, compute_graph_complexity_metrics

# Generate a graph
G = generate_barabasi_albert(n=100, m=3, seed=42)

# Compute all complexity metrics
metrics = compute_graph_complexity_metrics(G)

print(f"Quantum Complexity: {metrics['quantum_complexity']:.4f}")
print(f"Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")
print(f"Spectral Gap: {metrics['spectral_gap']:.4f}")
```

## Available Metrics

### 1. Laplacian Spectrum

The eigenvalues of the graph Laplacian matrix provide fundamental information about graph structure.

```python
from quvine.data import compute_laplacian_spectrum

eigenvalues = compute_laplacian_spectrum(G, normalized=True)
```

**Parameters:**
- `normalized`: If True, use normalized Laplacian (recommended)

**Interpretation:**
- Eigenvalues reveal connectivity, clustering, and community structure
- Distribution shape indicates graph type (regular, random, scale-free, etc.)

### 2. Spectral Gap

The difference between the first and second smallest eigenvalues.

```python
from quvine.data import compute_spectral_gap

gap = compute_spectral_gap(G, normalized=True)
```

**Interpretation:**
- **Larger gap** → Better connectivity, faster mixing time
- **Smaller gap** → Bottlenecks, slower diffusion
- Related to graph expansion and random walk convergence

**Typical values:**
- Regular graphs: 0.1 - 0.5
- Random graphs: 0.05 - 0.2
- Scale-free: 0.01 - 0.1

### 3. Algebraic Connectivity (Fiedler Value)

The second smallest eigenvalue of the unnormalized Laplacian.

```python
from quvine.data import compute_algebraic_connectivity

fiedler = compute_algebraic_connectivity(G)
```

**Interpretation:**
- **Higher values** → More robust connectivity
- **Zero** → Graph is disconnected
- Measures how well-connected the graph is

### 4. Spectral Entropy

Shannon entropy of the normalized Laplacian eigenvalue distribution.

```python
from quvine.data import compute_spectral_entropy

entropy = compute_spectral_entropy(G, normalized=True)
```

**Interpretation:**
- **Higher entropy** → More complex/random structure
- **Lower entropy** → More regular/ordered structure
- Measures structural complexity

**Typical values:**
- Regular graphs: 1.0 - 2.0
- Random graphs: 2.5 - 4.0
- Scale-free: 2.0 - 3.5

### 5. Von Neumann Entropy

Quantum analog of Shannon entropy, computed from normalized Laplacian eigenvalues.

```python
from quvine.data import compute_von_neumann_entropy

vn_entropy = compute_von_neumann_entropy(G)
```

**Interpretation:**
- **Higher entropy** → More quantum complexity
- **Lower entropy** → More classical/regular structure
- Indicates potential for quantum advantage in walks

**Use cases:**
- Predicting quantum walk performance
- Identifying graphs suitable for quantum algorithms
- Measuring information content

### 6. Estrada Index

Sum of exponentials of Laplacian eigenvalues.

```python
from quvine.data import compute_estrada_index

estrada = compute_estrada_index(G)
```

**Interpretation:**
- Related to the number of closed walks
- Measures graph "folding" or complexity
- Higher values indicate more complex structure

### 7. Quantum Complexity

Combined metric measuring how "quantum" or complex the graph structure is.

```python
from quvine.data import compute_quantum_complexity

qc = compute_quantum_complexity(G)
```

**Interpretation:**
- **Higher values** → More complex, potentially benefits from quantum walks
- **Lower values** → Simpler structure, classical methods may suffice
- Combines spectral gap, participation ratio, and von Neumann entropy

**Range:** 0.0 to 1.0 (normalized)

**Use cases:**
- Selecting graphs for quantum walk experiments
- Predicting when quantum methods outperform classical
- Benchmarking algorithm performance

## Comprehensive Analysis

### Compute All Metrics

```python
from quvine.data import compute_graph_complexity_metrics

metrics = compute_graph_complexity_metrics(G)
```

**Returns dictionary with:**
- `num_nodes`, `num_edges`: Basic properties
- `spectral_gap`: Spectral gap
- `algebraic_connectivity`: Fiedler value
- `spectral_entropy`: Spectral entropy
- `von_neumann_entropy`: Von Neumann entropy
- `estrada_index`: Estrada index
- `quantum_complexity`: Quantum complexity score
- `eigenvalue_mean`, `eigenvalue_std`, `eigenvalue_max`, `eigenvalue_min`: Eigenvalue statistics

### Compare Multiple Graphs

```python
from quvine.data import compare_graph_complexities

graphs = {
    'Scale-Free': generate_barabasi_albert(100, 3, seed=42),
    'Small-World': generate_watts_strogatz(100, 6, 0.3, seed=42),
    'Random': generate_erdos_renyi(100, p=0.05, seed=42),
}

complexities = compare_graph_complexities(graphs)
```

### Rank Graphs by Complexity

```python
from quvine.data import rank_graphs_by_complexity

rankings = rank_graphs_by_complexity(graphs, metric='quantum_complexity')

for i, (name, score) in enumerate(rankings, 1):
    print(f"{i}. {name}: {score:.4f}")
```

## Interpretation Guide

### Graph Type Signatures

Different graph types have characteristic complexity profiles:

| Graph Type | Spectral Gap | VN Entropy | Quantum Complexity |
|------------|--------------|------------|-------------------|
| Regular | High (0.3-0.5) | Low (1-2) | Low (0.1-0.3) |
| Random | Medium (0.05-0.2) | High (3-4) | Medium (0.3-0.5) |
| Scale-Free | Low (0.01-0.1) | Medium (2-3) | High (0.5-0.7) |
| Small-World | Medium (0.1-0.3) | Medium (2-3) | Medium (0.4-0.6) |
| Modular | Low-Medium | Medium-High | Medium-High |

### When to Use Quantum Walks

Consider quantum walks when:
- **High quantum complexity** (> 0.5)
- **High von Neumann entropy** (> 3.0)
- **Low spectral gap** (< 0.1) - indicates bottlenecks where quantum tunneling helps
- **Scale-free or modular structure** - quantum walks can exploit long-range correlations

### Embedding Difficulty

Graphs are harder to embed when:
- **High spectral entropy** - more structural diversity
- **Low spectral gap** - poor connectivity
- **High quantum complexity** - requires capturing complex patterns

## Practical Examples

### Example 1: Selecting Test Graphs

```python
from quvine.data import (
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_modular_network,
    rank_graphs_by_complexity
)

# Generate candidate graphs
graphs = {
    'BA-1': generate_barabasi_albert(100, 2, seed=42),
    'BA-2': generate_barabasi_albert(100, 5, seed=42),
    'SW-1': generate_watts_strogatz(100, 4, 0.1, seed=42),
    'SW-2': generate_watts_strogatz(100, 6, 0.5, seed=42),
    'Mod': generate_modular_network(5, 20, 0.3, 0.01, seed=42)[0],
}

# Rank by quantum complexity
rankings = rank_graphs_by_complexity(graphs, metric='quantum_complexity')

print("Top 3 most complex graphs for quantum walk testing:")
for i, (name, score) in enumerate(rankings[:3], 1):
    print(f"{i}. {name}: {score:.4f}")
```

### Example 2: Analyzing Real Network

```python
import networkx as nx
from quvine.data import compute_graph_complexity_metrics

# Load your network
G = nx.read_edgelist("my_network.txt")

# Compute complexity
metrics = compute_graph_complexity_metrics(G)

print(f"Network Complexity Analysis:")
print(f"  Nodes: {metrics['num_nodes']}")
print(f"  Edges: {metrics['num_edges']}")
print(f"  Quantum Complexity: {metrics['quantum_complexity']:.4f}")
print(f"  Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")

# Recommendation
if metrics['quantum_complexity'] > 0.5:
    print("\nRecommendation: Consider quantum walks (CTQW/DTQW)")
else:
    print("\nRecommendation: Classical random walks may suffice")
```

### Example 3: Comparing PPI Networks

```python
from quvine.data import load_graph, compute_graph_complexity_metrics

# Load different PPI networks
networks = {
    'BioPlex': load_graph(cfg_bioplex),
    'STRING': load_graph(cfg_string),
    'HumanNet': load_graph(cfg_humannet),
}

# Compare complexity
complexities = compare_graph_complexities(networks)

# Create comparison table
import pandas as pd
df = pd.DataFrame(complexities).T
print(df[['quantum_complexity', 'von_neumann_entropy', 'spectral_gap']])
```

## Integration with QuVINE Pipeline

### Use in Embedding Workflow

```python
from quvine.data import (
    generate_graph_with_seeds_and_targets,
    compute_graph_complexity_metrics
)

# Generate test graph
G, seeds, targets = generate_graph_with_seeds_and_targets(
    n=500, num_seeds=20, num_targets=30,
    graph_type='barabasi_albert', m=3, seed=42
)

# Analyze complexity
metrics = compute_graph_complexity_metrics(G)

# Select walk type based on complexity
if metrics['quantum_complexity'] > 0.5:
    walk_type = 'ctqw'  # Use quantum walk
else:
    walk_type = 'rwr'   # Use classical walk

print(f"Selected walk type: {walk_type}")
print(f"Quantum complexity: {metrics['quantum_complexity']:.4f}")
```

## Mathematical Background

### Laplacian Matrix

**Unnormalized:** L = D - A
- D: degree matrix
- A: adjacency matrix

**Normalized:** L = I - D^(-1/2) A D^(-1/2)

### Von Neumann Entropy

S = -Tr(ρ log₂ ρ)

where ρ is the density matrix derived from normalized Laplacian eigenvalues.

### Quantum Complexity

Combines:
1. **Spectral gap ratio**: λ₂/λₙ
2. **Participation ratio**: (Σλᵢ)² / Σλᵢ²
3. **Normalized von Neumann entropy**: S / log₂(n)

## References

1. Chung, F. R. (1997). Spectral Graph Theory. American Mathematical Society.
2. Braunstein, S. L., et al. (2006). Laplacian versus adjacency matrix in quantum walk search.
3. Passerini, F., & Severini, S. (2009). Quantifying complexity in networks: the von Neumann entropy.
4. Estrada, E. (2000). Characterization of 3D molecular structure.

## See Also

- `docs/random_graphs_guide.md` - Random graph generation
- `examples/graph_complexity_examples.py` - Comprehensive examples
- QBioCode: https://github.com/IBM/QBioCode/