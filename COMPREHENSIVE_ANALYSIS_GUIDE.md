# Comprehensive Embedding Analysis Guide

This guide explains how to run a comprehensive analysis comparing different embedding methods (QuVINE variants, NetMF, and Node2Vec) across networks with varying complexity characteristics.

## Overview

The analysis pipeline:
1. Generates 40 random networks (20 scale-free + 20 modular)
2. Computes comprehensive complexity metrics including **Inverse Participation Ratio (IPR)**
3. Runs 6 embedding methods on each network
4. Evaluates downstream performance (precision@K, recall@K)
5. Analyzes correlations between complexity and performance
6. Generates recommendations for method selection

## Embedding Methods Compared

1. **QuVINE-fused**: Fusion of multiple quantum walk types (RWR + CTQW + DTQW)
2. **QuVINE-RWR**: Random Walk with Restart (classical baseline within QuVINE)
3. **QuVINE-CTQW**: Continuous-Time Quantum Walk
4. **QuVINE-DTQW**: Discrete-Time Quantum Walk
5. **NetMF**: Network Embedding as Matrix Factorization
6. **Node2Vec**: Classical node embedding baseline

## Complexity Metrics

The analysis computes the following complexity metrics:

### Spectral Metrics
- **Spectral Gap**: Difference between first and second Laplacian eigenvalues
- **Algebraic Connectivity**: Second smallest eigenvalue (Fiedler value)
- **Spectral Entropy**: Entropy of eigenvalue distribution
- **Eigenvalue Statistics**: Mean, std, min, max of eigenvalues

### Quantum-Inspired Metrics
- **Von Neumann Entropy**: Quantum analog of Shannon entropy
- **Quantum Complexity**: Combined metric measuring quantum advantage potential
- **Estrada Index**: Measures graph folding/complexity

### Participation Metrics (NEW)
- **Inverse Participation Ratio (IPR)**: Measures eigenstate localization
- **Participation Ratio (PR)**: Effective spectral dimension

### Centrality-Based Metrics
- **Centrality Entropy**: Entropy of eigenvector centrality distribution
- **Centrality Variance**: Variance of centrality values
- **Centrality Gini**: Inequality measure of centrality distribution
- **Centrality Range**: Range of centrality values

## Installation

Ensure you have the required dependencies:

```bash
cd QuVINE
pip install -r requirements.txt
```

Additional requirements:
```bash
pip install node2vec seaborn scipy scikit-learn
```

## Running the Analysis

### Quick Start

```bash
cd QuVINE
python run_comprehensive_analysis.py
```

This will:
- Generate 40 networks (20 scale-free, 20 modular)
- Run all 6 embedding methods on each network
- Compute complexity metrics and performance metrics
- Generate correlation analysis
- Create visualizations
- Produce method recommendations

### Custom Configuration

You can modify the analysis parameters in `run_comprehensive_analysis.py`:

```python
analysis = ComprehensiveEmbeddingAnalysis(
    output_dir="outputs/comprehensive_analysis",
    n_networks_per_type=20,  # Number of networks per type
    n_nodes=200,              # Network size
    num_seeds=15,             # Number of seed nodes
    num_targets=25,           # Number of target nodes
    embedding_dim=128,        # Embedding dimension
    seed=42                   # Random seed
)
```

### Using the Analysis Module Directly

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Create analysis instance
analysis = ComprehensiveEmbeddingAnalysis(
    output_dir="outputs/my_analysis",
    n_networks_per_type=10,
    n_nodes=150
)

# Run complete pipeline
results = analysis.run_complete_analysis()

# Access results
complexity_df = results['complexity']
performance_df = results['performance']
correlations_df = results['correlations']
recommendations_df = results['recommendations']
```

## Output Files

The analysis generates the following outputs in `outputs/comprehensive_analysis/`:

### Data Files
- **complexity_metrics.csv**: All complexity metrics for each network
- **embedding_performance.csv**: Performance metrics for each method on each network
- **complexity_performance_correlations.csv**: Correlation analysis results
- **method_recommendations.csv**: Recommended methods for different complexity conditions
- **recommendations_report.txt**: Human-readable recommendation guide

### Visualizations (in `visualizations/` subdirectory)
- **complexity_distributions.png**: Distribution of complexity metrics by network type
- **performance_comparison.png**: Bar charts comparing method performance
- **correlation_heatmap_[method].png**: Heatmaps showing complexity-performance correlations
- **significant_correlations.png**: Scatter plots of significant correlations

## Understanding the Results

### Complexity Metrics Interpretation

**High Quantum Complexity** (> 0.5):
- Network has complex structure that may benefit from quantum walks
- Recommended: CTQW, DTQW, or fused methods

**Low Spectral Gap** (< 0.1):
- Network has bottlenecks or weak connectivity
- Quantum tunneling may help
- Recommended: CTQW

**High Inverse Participation Ratio**:
- Eigenvalues are localized
- Network has hierarchical or modular structure
- Recommended: DTQW or modular-aware methods

**Low Inverse Participation Ratio**:
- Eigenvalues are delocalized
- Network is more homogeneous
- Recommended: RWR or NetMF

### Performance Metrics

- **Precision@K**: Fraction of top-K predictions that are true targets
- **Recall@K**: Fraction of true targets found in top-K predictions
- **Centroid**: Uses centroid of seed embeddings for scoring
- **Max**: Uses maximum similarity to any seed for scoring

### Correlation Analysis

The correlation analysis reveals:
- Which complexity metrics predict embedding performance
- Which methods work best under specific complexity conditions
- Statistical significance of relationships (p-values)

## Example Workflow

### 1. Generate and Analyze Networks

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

analysis = ComprehensiveEmbeddingAnalysis()
results = analysis.run_complete_analysis()
```

### 2. Examine Complexity Distribution

```python
import pandas as pd

complexity_df = pd.read_csv("outputs/comprehensive_analysis/complexity_metrics.csv")

# Compare scale-free vs modular
print(complexity_df.groupby('network_type')[
    ['quantum_complexity', 'inverse_participation_ratio', 'spectral_gap']
].mean())
```

### 3. Compare Method Performance

```python
performance_df = pd.read_csv("outputs/comprehensive_analysis/embedding_performance.csv")

# Average recall@50 by method
print(performance_df.groupby('method')['recall@50_centroid'].mean().sort_values(ascending=False))
```

### 4. Find Best Method for Your Network

```python
# Calculate your network's complexity
from quvine.data.graph_complexity import compute_graph_complexity_metrics

my_graph = ...  # Your NetworkX graph
metrics = compute_graph_complexity_metrics(my_graph)

print(f"Quantum Complexity: {metrics['quantum_complexity']:.3f}")
print(f"IPR: {metrics['inverse_participation_ratio']:.3f}")

# Check recommendations
recommendations_df = pd.read_csv("outputs/comprehensive_analysis/method_recommendations.csv")
print(recommendations_df[recommendations_df['complexity_metric'] == 'quantum_complexity'])
```

## Advanced Usage

### Running on Custom Networks

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
import networkx as nx

# Create custom network list
networks = [
    ("my_network_1", my_graph_1, seeds_1, targets_1),
    ("my_network_2", my_graph_2, seeds_2, targets_2),
]

analysis = ComprehensiveEmbeddingAnalysis()

# Compute complexity
complexity_df = analysis.compute_complexity_for_all(networks)

# Run methods
performance_df = analysis.run_all_methods_on_networks(networks)

# Analyze
correlation_df, merged_df = analysis.analyze_correlations(complexity_df, performance_df)
```

### Testing Individual Methods

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

analysis = ComprehensiveEmbeddingAnalysis()

# Test single method
embedding = analysis.run_embedding_method(
    method_name='ctqw',  # or 'rwr', 'dtqw', 'fused', 'netmf', 'node2vec'
    G=my_graph,
    seeds=seed_nodes,
    targets=target_nodes
)

# Evaluate
results = analysis.evaluate_embedding(
    embedding, my_graph, seed_nodes, target_nodes,
    method_name='ctqw', network_id='my_network'
)
```

## Interpreting Recommendations

The analysis generates a recommendations report that suggests which method to use based on complexity characteristics:

**Example recommendations:**

```
QUANTUM_COMPLEXITY
----------------------------------------------------------
  LOW quantum_complexity (<= 0.250):
    → Recommended: NETMF
    → Avg Recall@50: 0.456

  HIGH quantum_complexity (>= 0.650):
    → Recommended: FUSED
    → Avg Recall@50: 0.623

INVERSE_PARTICIPATION_RATIO
----------------------------------------------------------
  LOW inverse_participation_ratio (<= 0.120):
    → Recommended: RWR
    → Avg Recall@50: 0.512

  HIGH inverse_participation_ratio (>= 0.180):
    → Recommended: DTQW
    → Avg Recall@50: 0.587
```

## Troubleshooting

### Memory Issues
If you encounter memory issues with large networks:
- Reduce `n_nodes` parameter
- Reduce `n_networks_per_type`
- Process networks in batches

### Slow Execution
To speed up analysis:
- Reduce `num_walks_per_root` in QuVINE config
- Use fewer views
- Reduce embedding dimensions
- Use parallel processing (already implemented)

### Missing Dependencies
```bash
pip install networkx numpy pandas scipy matplotlib seaborn scikit-learn node2vec gensim omegaconf
```

## Citation

If you use this analysis in your research, please cite:

```bibtex
@software{quvine_comprehensive_analysis,
  title={Comprehensive Embedding Analysis for QuVINE},
  author={QuVINE Team},
  year={2024},
  url={https://github.com/IBM/QuVINE}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.

## License

This code is licensed under the Apache License 2.0. See LICENSE file for details.