# QuVINE: Quantum View-based Network Embeddings

<p align="center">
  <a href="./pyproject.toml"><img alt="pypi package" src="https://img.shields.io/badge/pypi_package-0.1.0-52c41a"></a>
  <a href="./pyproject.toml"><img alt="Python >= 3.10" src="https://img.shields.io/badge/Python-%3E%3D%203.10-0b8ecf"></a>
  <a href="./pyproject.toml"><img alt="Python <= 3.12" src="https://img.shields.io/badge/Python-%3C%3D%203.12-8a5bd6"></a>
  <a href="https://ibm.github.io/QuVINE/"><img alt="docs sphinx" src="https://img.shields.io/badge/docs-sphinx-0b8ecf"></a>
  <a href="./LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
</p>

<p align="center">
  <img src="./images/quvine_framework.png" alt="QuVINE Framework" width="600" height="400">
</p>

**QuVINE** is a comprehensive framework for **Qu**antum and classical **V**iew-based **N**etwork **E**mbeddings. It combines quantum walk-based methods with state-of-the-art classical approaches to generate high-quality graph embeddings for biological networks, social networks, and complex systems analysis.

## 🌟 Key Features

- **39 Embedding Methods**: 16 quantum and 23 classical methods spanning multiple architectures
- **Quantum Walk Embeddings**: Continuous-Time Quantum Walk (CTQW) and Discrete-Time Quantum Walk (DTQW)
- **Modern Architectures**: Graph Attention Networks (GAT) and Graph Transformers (GraphGPS)
- **Classical Baselines**: Random Walk with Restart (RWR), Node2Vec, NetMF, GCN, GraphSAGE, APPNP
- **Multi-View Fusion**: Hierarchical fusion strategy combining quantum and classical perspectives
- **Complexity Analysis**: Comprehensive graph complexity metrics including spectral, quantum-inspired, and participation measures
- **Downstream Tasks**: Node classification, link prediction, and node ranking evaluation
- **Hyperparameter Tuning**: Automated Optuna-based hyperparameter optimization
- **HPC-Ready**: Parallelized workflows optimized for high-performance computing clusters
- **Biological Applications**: Specialized support for protein-protein interaction networks and disease gene analysis

## 📊 Performance Highlights

QuVINE has been extensively evaluated across:
- **270+ synthetic networks** (Erdős-Rényi, Barabási-Albert, Watts-Strogatz, SBM, etc.)
- **Real-world networks** (Karate Club, Les Misérables, Political Books, Deezer, Reddit, Twitch)
- **Biological networks** (BioPlex3, HumanNet, STRING, ProteomeHD)
- **Disease-specific analyses** (Asthma, Autism, Schizophrenia)

Results show quantum methods provide advantages on networks with specific structural properties, particularly those with strong community structure and intermediate spectral complexity.

## 🔬 Complete Method Registry

QuVINE implements **39 methods** organized into 5 categories:

### 1. SGNS (Skip-Gram with Negative Sampling) - 3 methods
- `quvine_rwr` - Random Walk with Restart (classical)
- `quvine_ctqw` - Continuous-Time Quantum Walk (quantum)
- `quvine_dtqw` - Discrete-Time Quantum Walk (quantum)

### 2. Graph Filters - 6 methods
- `quvine_baseline_heat` - Heat kernel filter, no walk (classical)
- `quvine_baseline_poly` - Polynomial filter, no walk (classical)
- `quvine_rwr_heat` - RWR + heat filter (classical)
- `quvine_rwr_poly` - RWR + polynomial filter (classical)
- `quvine_ctqw_heat` - CTQW + heat filter (quantum)
- `quvine_ctqw_poly` - CTQW + polynomial filter (quantum)

### 3. GAT (Graph Attention Networks) - 12 methods

**Baseline:**
- `gat_baseline` - GAT without quantum calibration (classical)

**With filters only:**
- `gat_heat` - GAT + heat filter, no walk (classical)
- `gat_poly` - GAT + polynomial filter, no walk (classical)

**With walks only:**
- `gat_rwr` - GAT + RWR (classical)
- `gat_ctqw` - GAT + CTQW (quantum)
- `gat_dtqw` - GAT + DTQW (quantum)

**With walks + filters:**
- `gat_rwr_heat` - GAT + RWR + heat filter (classical)
- `gat_rwr_poly` - GAT + RWR + polynomial filter (classical)
- `gat_ctqw_heat` - GAT + CTQW + heat filter (quantum)
- `gat_ctqw_poly` - GAT + CTQW + polynomial filter (quantum)
- `gat_dtqw_heat` - GAT + DTQW + heat filter (quantum)
- `gat_dtqw_poly` - GAT + DTQW + polynomial filter (quantum)

### 4. GraphGPS (Graph Transformer) - 12 methods

**Baseline:**
- `graphgps_baseline` - GraphGPS without quantum calibration (classical)

**With filters only:**
- `graphgps_heat` - GraphGPS + heat filter, no walk (classical)
- `graphgps_poly` - GraphGPS + polynomial filter, no walk (classical)

**With walks only:**
- `graphgps_rwr` - GraphGPS + RWR (classical)
- `graphgps_ctqw` - GraphGPS + CTQW (quantum)
- `graphgps_dtqw` - GraphGPS + DTQW (quantum)

**With walks + filters:**
- `graphgps_rwr_heat` - GraphGPS + RWR + heat filter (classical)
- `graphgps_rwr_poly` - GraphGPS + RWR + polynomial filter (classical)
- `graphgps_ctqw_heat` - GraphGPS + CTQW + heat filter (quantum)
- `graphgps_ctqw_poly` - GraphGPS + CTQW + polynomial filter (quantum)
- `graphgps_dtqw_heat` - GraphGPS + DTQW + heat filter (quantum)
- `graphgps_dtqw_poly` - GraphGPS + DTQW + polynomial filter (quantum)

### 5. Classical Baselines - 6 methods
- `node2vec` - Node2Vec (classical)
- `netmf` - Network Embedding as Matrix Factorization (classical)
- `graphsage` - GraphSAGE (classical)
- `appnp` - Approximate Personalized Propagation of Neural Predictions (classical)
- `baseline_filter` - Classical filter baseline (classical)
- `baseline_gcnmf` - Classical GCN-MF baseline (classical)

**Total: 16 quantum methods, 23 classical methods**

For complete details, see [METHOD_REGISTRY.md](./METHOD_REGISTRY.md).

## 🚀 Quick Start

### Installation

```bash
git clone <repository-url>
cd QuVINE
pip install -e .
```

For detailed setup instructions, see [`docs/setup/SETUP_INSTRUCTIONS.md`](./docs/setup/SETUP_INSTRUCTIONS.md).

### Basic Usage

#### Command Line Interface

```bash
# Run QuVINE with default settings
quvine

# Quick test with parallelization (5-15 minutes)
python run_comprehensive_analysis.py --quick

# Full analysis (30-90 minutes with parallelization)
python run_comprehensive_analysis.py
```

#### Python API

```python
from quvine.pipeline import EmbeddingPipeline
import networkx as nx

# Load or create a graph
G = nx.karate_club_graph()

# Initialize pipeline
pipeline = EmbeddingPipeline(
    embedding_dim=128,
    walk_type='ctqw',  # or 'dtqw', 'rwr'
    fusion_method='concatenate'
)

# Generate embeddings
embeddings = pipeline.fit_transform(G)

# Evaluate on downstream task
from quvine.evaluation import evaluate_node_ranking
results = evaluate_node_ranking(G, embeddings)
print(f"Precision@10: {results['precision@10']:.3f}")
```

### Example: Run All 39 Methods

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Initialize analysis with all methods
analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=20,
    n_nodes=200,
    embedding_dim=128,
    output_dir='outputs/all_methods',
    embedding_methods='all'  # Uses all 39 methods
)

# Run complete analysis
results = analysis.run_complete_analysis()

# Results include:
# - Performance metrics for all 39 methods
# - Complexity correlations
# - Method recommendations
# - Fusion results
# - Visualizations
```

### Example: Hyperparameter Tuning

QuVINE uses an efficient **method grouping strategy** for hyperparameter tuning:

- **8 representative methods** are tuned (instead of all 39)
- Parameters are **reused** across method families
- **79% reduction** in tuning time while maintaining quality

```bash
# Run hyperparameter tuning for synthetic networks
# Tunes 7 representative methods, applies to all 39
sbatch scripts/submit_simulated_data_jobs_with_tuning.sh

# Run hyperparameter tuning for PPI networks
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh

# Tuning results are automatically used in subsequent analysis jobs
```

**Method Groups:**
- **Quantum methods** (11 methods) → tune `quvine_walks`, reuse for all
- **GAT variants** (12 methods) → tune `gat_baseline`, reuse for all
- **GraphGPS variants** (12 methods) → tune `graphgps_baseline`, reuse for all
- **Classical baselines** (5 methods) → tune individually (node2vec, netmf, graphsage, appnp, baseline_gcnmf)

See [HYPERPARAMETER_TUNING_STRATEGY.md](./HYPERPARAMETER_TUNING_STRATEGY.md) for details.

## 📁 Project Structure

```text
QuVINE/
├── configs/                 # Experiment configuration files
├── data/                    # Raw and processed datasets
├── docs/                    # Documentation
│   ├── api/                 # API reference
│   ├── guides/              # User guides
│   ├── setup/               # Installation guides
│   └── development/         # Development docs
├── examples/                # Runnable examples
├── notebooks/               # Jupyter notebooks for analysis
├── scripts/                 # Utility and batch execution scripts
│   ├── submit_simulated_data_jobs_with_tuning.sh
│   ├── submit_ppi_comprehensive_with_tuning.sh
│   └── run_hyperparameter_tuning.py
├── src/quvine/              # Main package source code
│   ├── analysis/            # Result analysis and comparison
│   ├── baselines/           # Classical and hybrid baselines
│   │   ├── gat.py           # 12 GAT variants
│   │   ├── graphgps.py      # 12 GraphGPS variants
│   │   └── gcn_mf.py        # Classical baselines
│   ├── complexity/          # Graph complexity metrics
│   ├── data/                # Graph/data preparation utilities
│   ├── embedding/           # Embedding and quantum filters
│   │   ├── quantum_filters.py  # 6 filter methods
│   │   └── registry.py      # Method registry
│   ├── evaluation/          # Downstream task evaluation
│   ├── fusion/              # Embedding fusion methods
│   │   └── fuse.py          # Hierarchical fusion
│   ├── utils/               # Reusable utilities
│   ├── views/               # Graph view generation
│   ├── walks/               # Classical and quantum walks
│   ├── comprehensive_embedding_analysis.py  # Main analysis
│   └── pipeline.py          # End-to-end pipeline
├── tests/                   # Unit and integration tests
├── METHOD_REGISTRY.md       # Complete method documentation
├── IMPLEMENTATION_PLAN.md   # Development roadmap
├── pyproject.toml           # Package configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📚 Documentation

### Quick Links

- **[Method Registry](./METHOD_REGISTRY.md)** - Complete list of all 39 methods
- **[Implementation Plan](./IMPLEMENTATION_PLAN.md)** - Development roadmap and architecture
- **[Quick Start Guide](./docs/guides/QUICK_START.md)** - Get started in minutes
- **[Comprehensive Analysis Guide](./docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)** - Full analysis workflows
- **[Dataset Generation Guide](./docs/guides/COMPREHENSIVE_DATASET_GUIDE.md)** - Generate synthetic networks
- **[HPC Deployment Guide](./docs/setup/HPC_DEPLOYMENT.md)** - Deploy on computing clusters
- **[API Documentation](https://ibm.github.io/QuVINE/)** - Full API reference

### Setup Guides

- [Setup Instructions](./docs/setup/SETUP_INSTRUCTIONS.md)
- [Python Version Guide](./docs/setup/PYTHON_VERSION_GUIDE.md)
- [HPC Deployment](./docs/setup/HPC_DEPLOYMENT.md)
- [QBioCode Installation](./docs/setup/QBIOCODE_INSTALL.md)

### User Guides

- [Quick Start](./docs/guides/QUICK_START.md)
- [Comprehensive Analysis](./docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)
- [Dataset Generation](./docs/guides/COMPREHENSIVE_DATASET_GUIDE.md)
- [Parallelization](./docs/guides/PARALLELIZATION_GUIDE.md)
- [Quantum Advantage & Downstream Tasks](./docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md)

### Development

- [Contributing Guidelines](./docs/development/CONTRIBUTING.md)
- [Implementation Summary](./docs/development/IMPLEMENTATION_SUMMARY.md)
- [Improvements & Fixes](./docs/development/IMPROVEMENTS_AND_FIXES.md)
- [Bug Review & Fixes](./docs/development/BUG_REVIEW_AND_FIXES.md)

## 🔬 Core Capabilities

### 1. Graph Complexity Analysis

Compute comprehensive complexity metrics:

```python
from quvine.complexity.graph import compute_graph_complexity_metrics
import networkx as nx

G = nx.karate_club_graph()
metrics = compute_graph_complexity_metrics(G)

print(f"Spectral Gap: {metrics['spectral_gap']:.3f}")
print(f"Von Neumann Entropy: {metrics['von_neumann_entropy']:.3f}")
print(f"Inverse Participation Ratio: {metrics['inverse_participation_ratio']:.3f}")
```

**Available Metrics:**
- Spectral: gap, algebraic connectivity, entropy
- Quantum-inspired: Von Neumann entropy, quantum complexity, Estrada index
- Participation: IPR, participation ratio
- Centrality-based: entropy, variance, Gini coefficient

### 2. Embedding Methods

QuVINE provides 39 methods across 5 categories:

**Quantum Methods (16):**
- SGNS: quvine_ctqw, quvine_dtqw
- Filters: quvine_ctqw_heat, quvine_ctqw_poly
- GAT: 6 quantum variants
- GraphGPS: 6 quantum variants

**Classical Methods (23):**
- SGNS: quvine_rwr
- Filters: 4 classical variants
- GAT: 6 classical variants
- GraphGPS: 6 classical variants
- Baselines: node2vec, netmf, graphsage, appnp, baseline_filter, baseline_gcnmf

### 3. Multi-View Fusion

Hierarchical fusion strategy combining quantum and classical perspectives:

```python
from quvine.fusion.fuse import fuse_embeddings

# Generate multiple embeddings
embeddings_ctqw = pipeline_ctqw.fit_transform(G)
embeddings_dtqw = pipeline_dtqw.fit_transform(G)
embeddings_rwr = pipeline_rwr.fit_transform(G)

# Fuse embeddings
fused = fuse_embeddings(
    [embeddings_ctqw, embeddings_dtqw, embeddings_rwr],
    method='svd',  # or 'concatenate', 'average', 'weighted'
    target_dim=128
)
```

**Fusion Strategy:**
1. Within-type fusion (e.g., all GAT methods)
2. Quantum vs classical fusion per type
3. Cross-type fusion for best methods
4. Final quantum and classical fused embeddings

### 4. Downstream Evaluation

**Node Ranking:**
```python
from quvine.evaluation import evaluate_node_ranking

results = evaluate_node_ranking(
    G, embeddings,
    seed_nodes=seed_nodes,
    target_nodes=target_nodes,
    k_values=[10, 20, 50, 100]
)
```

**Link Prediction:**
```python
from quvine.evaluation import evaluate_link_prediction

results = evaluate_link_prediction(
    G, embeddings,
    test_ratio=0.2,
    negative_strategy='degree_matched'
)
```

**Node Classification:**
```python
from quvine.evaluation import evaluate_node_classification

results = evaluate_node_classification(
    G, embeddings, labels,
    test_ratio=0.2
)
```

### 5. Hyperparameter Tuning

Automated hyperparameter optimization using Optuna:

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=10,
    n_nodes=200,
    embedding_dim=128,
    enable_tuning=True,
    n_tuning_trials=50
)

# Tuning optimizes:
# - Walk parameters (time, steps, restart probability)
# - Filter parameters (scale, polynomial degree)
# - Neural network hyperparameters (layers, heads, dropout)
# - Training parameters (learning rate, epochs, batch size)
```

## 🧪 Examples

### Example 1: Analyze Graph Complexity

```python
from quvine.data.random_graphs import generate_scale_free_network
from quvine.complexity.graph import compute_graph_complexity_metrics

# Generate a scale-free network
G = generate_scale_free_network(n=200, m=3, seed=42)

# Compute complexity metrics
metrics = compute_graph_complexity_metrics(G)

# Display key metrics
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
```

### Example 2: Compare All 39 Methods

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Run comprehensive comparison
analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=10,
    n_nodes=200,
    embedding_dim=128,
    embedding_methods='all',  # All 39 methods
    output_dir='outputs/method_comparison'
)

results = analysis.run_complete_analysis()

# View method rankings
print(results['method_rankings'])

# View quantum vs classical comparison
print(results['quantum_vs_classical'])
```

### Example 3: Biological Network Analysis

```python
from quvine.data.prepare import load_ppi_network
from quvine.pipeline import EmbeddingPipeline
from quvine.evaluation import evaluate_disease_gene_ranking

# Load protein-protein interaction network
G = load_ppi_network('BioPlex3')

# Generate embeddings with quantum method
pipeline = EmbeddingPipeline(
    embedding_dim=128,
    walk_type='ctqw',
    method='gat_ctqw_heat'  # GAT + CTQW + heat filter
)
embeddings = pipeline.fit_transform(G)

# Evaluate disease gene ranking
results = evaluate_disease_gene_ranking(
    G, embeddings,
    disease='asthma',
    seed_genes=known_disease_genes
)

print(f"Precision@50: {results['precision@50']:.3f}")
print(f"Recall@50: {results['recall@50']:.3f}")
```

### Example 4: Fusion Pipeline

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Run analysis with fusion
analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=20,
    n_nodes=200,
    embedding_dim=128,
    enable_fusion=True,
    fusion_strategy='hierarchical'
)

results = analysis.run_complete_analysis()

# Compare individual methods vs fused embeddings
print("Best individual quantum method:", results['best_quantum_method'])
print("Best individual classical method:", results['best_classical_method'])
print("Fused quantum performance:", results['fused_quantum_performance'])
print("Fused classical performance:", results['fused_classical_performance'])
```

## 🔧 Advanced Usage

### Custom Graph Generators

```python
from quvine.data.random_graphs import (
    generate_modular_network,
    generate_core_periphery_network,
    generate_hierarchical_network
)

# Generate modular network with strong communities
G_modular = generate_modular_network(
    n=200,
    n_communities=5,
    p_in=0.4,
    p_out=0.02,
    seed=42
)

# Generate core-periphery structure
G_core_periphery = generate_core_periphery_network(
    n=200,
    core_size=40,
    p_core=0.6,
    p_periphery=0.05,
    seed=42
)
```

### Parallelized Batch Processing

```bash
# Use all available CPU cores
python run_comprehensive_analysis.py --n-jobs -1

# Specify number of workers
python run_comprehensive_analysis.py --n-jobs 8

# Quick test mode
python run_comprehensive_analysis.py --quick --n-jobs 4
```

### HPC Cluster Deployment

```bash
# Submit hyperparameter tuning jobs
sbatch scripts/submit_simulated_data_jobs_with_tuning.sh

# Submit PPI network analysis
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh

# Monitor progress
squeue -u $USER

# Aggregate results
python scripts/aggregate_results.py --input-dir outputs/hpc_runs
```

### Method Selection

```python
# Run specific methods
analysis = ComprehensiveEmbeddingAnalysis(
    embedding_methods=[
        'quvine_ctqw',
        'gat_ctqw_heat',
        'graphgps_ctqw_poly',
        'node2vec'
    ]
)

# Run all quantum methods
analysis = ComprehensiveEmbeddingAnalysis(
    embedding_methods='quantum'
)

# Run all classical methods
analysis = ComprehensiveEmbeddingAnalysis(
    embedding_methods='classical'
)

# Run all methods (default)
analysis = ComprehensiveEmbeddingAnalysis(
    embedding_methods='all'
)
```

## 📊 Results and Analysis

QuVINE includes comprehensive analysis tools:

### Meta-Analysis

```python
from quvine.analysis import run_meta_analysis

# Analyze results across all datasets
meta_results = run_meta_analysis(
    results_dir='outputs/comprehensive_analysis',
    output_dir='outputs/meta_analysis'
)

# Generates:
# - Performance delta analysis (quantum vs classical)
# - Win/loss summaries by dataset
# - Correlation with complexity metrics
# - Forest plots and boxplots
# - Method recommendations
```

### Visualization

```python
from quvine.analysis import create_performance_plots

create_performance_plots(
    results_df,
    output_dir='outputs/visualizations',
    plot_types=['boxplot', 'forest', 'correlation', 'heatmap']
)
```

### Complexity-Performance Correlation

```python
from quvine.analysis import analyze_complexity_correlation

correlation_results = analyze_complexity_correlation(
    results_df,
    complexity_metrics=['spectral_gap', 'von_neumann_entropy', 'modularity'],
    performance_metrics=['precision@10', 'auc_roc', 'accuracy']
)

# Identifies which complexity metrics predict method performance
```

## 🧬 Biological Applications

QuVINE is designed for biological network analysis:

- **Protein-Protein Interaction Networks**: BioPlex3, HumanNet, STRING, ProteomeHD
- **Disease Gene Prioritization**: Asthma, Autism, Schizophrenia
- **Pathway Analysis**: Functional enrichment and module detection
- **Drug Target Discovery**: Network-based target identification

See [`docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md`](./docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md) for detailed biological use cases.

## 🤝 Contributing

We welcome contributions! Please see [`docs/development/CONTRIBUTING.md`](./docs/development/CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd QuVINE

# Create virtual environment
python -m venv venv_quvine
source venv_quvine/bin/activate  # On Windows: venv_quvine\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 📝 Citation

If you use QuVINE in your research, please cite:

```bibtex
@software{quvine2024,
  title={QuVINE: Quantum View-based Network Embeddings},
  author={[Authors]},
  year={2024},
  url={https://github.com/ibm/QuVINE}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [NetworkX](https://networkx.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- Quantum walk implementations based on [HiperWalk](https://github.com/hiperwalk/hiperwalk)
- Baseline methods from [Node2Vec](https://github.com/aditya-grover/node2vec), [NetMF](https://github.com/xptree/NetMF)
- Graph neural networks powered by [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- Hyperparameter tuning with [Optuna](https://optuna.org/)

## 📧 Contact

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Check the [documentation](https://ibm.github.io/QuVINE/)
- Review existing [guides](./docs/guides/)

## 🗺️ Roadmap

- [x] 39 embedding methods (16 quantum, 23 classical)
- [x] Hierarchical fusion strategy
- [x] Automated hyperparameter tuning
- [x] HPC cluster deployment
- [ ] GPU acceleration for large-scale networks
- [ ] Interactive visualization dashboard
- [ ] Pre-trained embeddings for common biological networks
- [ ] Extended biological pathway databases
- [ ] Real-time embedding updates for dynamic networks
- [ ] Integration with additional graph neural network frameworks

---

**QuVINE** - Bridging quantum computing and network science for next-generation graph embeddings.