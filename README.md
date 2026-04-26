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

**QuVINE** is a comprehensive framework for **Qu**antum and classical **V**iew-based **N**etwork **E**mbeddings. It combines quantum walk-based methods with classical approaches to generate high-quality graph embeddings for biological networks, social networks, and complex systems analysis.

## 🌟 Key Features

- **Quantum Walk Embeddings**: Continuous-Time Quantum Walk (CTQW) and Discrete-Time Quantum Walk (DTQW)
- **Classical Baselines**: Random Walk with Restart (RWR), Node2Vec, NetMF, GCN, GAT, GraphGPS
- **Multi-View Fusion**: Combine multiple embedding perspectives for enhanced performance
- **Complexity Analysis**: Comprehensive graph complexity metrics including spectral, quantum-inspired, and participation measures
- **Downstream Tasks**: Node classification, link prediction, and node ranking evaluation
- **HPC-Ready**: Parallelized workflows optimized for high-performance computing clusters
- **Biological Applications**: Specialized support for protein-protein interaction networks and disease gene analysis

## 📊 Performance Highlights

QuVINE has been extensively evaluated across:
- **270+ synthetic networks** (Erdős-Rényi, Barabási-Albert, Watts-Strogatz, SBM, etc.)
- **Real-world networks** (Karate Club, Les Misérables, Political Books, Deezer, Reddit, Twitch)
- **Biological networks** (BioPlex3, HumanNet, STRING, ProteomeHD)
- **Disease-specific analyses** (Asthma, Autism, Schizophrenia)

Results show quantum methods provide advantages on networks with specific structural properties, particularly those with strong community structure and intermediate spectral complexity.

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

### Example: Comprehensive Analysis

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Initialize analysis
analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=20,
    n_nodes=200,
    embedding_dim=128,
    output_dir='outputs/my_analysis'
)

# Run complete analysis
results = analysis.run_complete_analysis()

# Results include:
# - Embedding performance metrics
# - Complexity correlations
# - Method recommendations
# - Visualizations
```

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
├── src/quvine/              # Main package source code
│   ├── analysis/            # Result analysis and comparison
│   ├── baselines/           # Classical and hybrid baselines
│   ├── complexity/          # Graph complexity metrics
│   ├── data/                # Graph/data preparation utilities
│   ├── embedding/           # Embedding and quantum filters
│   ├── evaluation/          # Downstream task evaluation
│   ├── fusion/              # Embedding fusion methods
│   ├── utils/               # Reusable utilities
│   ├── views/               # Graph view generation
│   └── walks/               # Classical and quantum walks
├── tests/                   # Unit and integration tests
├── pyproject.toml           # Package configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 📚 Documentation

### Quick Links

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

**Quantum Methods:**
- **CTQW**: Continuous-Time Quantum Walk
- **DTQW**: Discrete-Time Quantum Walk

**Classical Methods:**
- **RWR**: Random Walk with Restart
- **Node2Vec**: Skip-gram based embeddings
- **NetMF**: Matrix factorization approach
- **GCN**: Graph Convolutional Networks
- **GAT**: Graph Attention Networks
- **GraphGPS**: Graph transformer architecture

### 3. Multi-View Fusion

Combine embeddings from different methods:

```python
from quvine.fusion.fuse import fuse_embeddings

# Generate multiple embeddings
embeddings_ctqw = pipeline_ctqw.fit_transform(G)
embeddings_dtqw = pipeline_dtqw.fit_transform(G)
embeddings_rwr = pipeline_rwr.fit_transform(G)

# Fuse embeddings
fused = fuse_embeddings(
    [embeddings_ctqw, embeddings_dtqw, embeddings_rwr],
    method='concatenate'  # or 'average', 'weighted'
)
```

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
    test_ratio=0.2
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

### Example 2: Compare Embedding Methods

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Quick comparison (10 networks, ~5-15 minutes)
analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=5,
    n_nodes=100,
    embedding_dim=64
)

results = analysis.run_complete_analysis()

# View recommendations
print(results['recommendations'])
```

### Example 3: Biological Network Analysis

```python
from quvine.data.prepare import load_ppi_network
from quvine.pipeline import EmbeddingPipeline
from quvine.evaluation import evaluate_disease_gene_ranking

# Load protein-protein interaction network
G = load_ppi_network('BioPlex3')

# Generate embeddings
pipeline = EmbeddingPipeline(embedding_dim=128, walk_type='ctqw')
embeddings = pipeline.fit_transform(G)

# Evaluate disease gene ranking
results = evaluate_disease_gene_ranking(
    G, embeddings,
    disease='asthma',
    seed_genes=known_disease_genes
)

print(f"Precision@50: {results['precision@50']:.3f}")
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
# Submit batch job
sbatch scripts/run_hpc_analysis.sh

# Monitor progress
squeue -u $USER

# Aggregate results
python scripts/aggregate_results.py --input-dir outputs/hpc_runs
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
# - Performance delta analysis
# - Win/loss summaries by dataset
# - Correlation with complexity metrics
# - Forest plots and boxplots
```

### Visualization

```python
from quvine.analysis import create_performance_plots

create_performance_plots(
    results_df,
    output_dir='outputs/visualizations',
    plot_types=['boxplot', 'forest', 'correlation']
)
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

## 📧 Contact

For questions, issues, or collaboration opportunities:
- Open an issue on GitHub
- Check the [documentation](https://ibm.github.io/QuVINE/)
- Review existing [guides](./docs/guides/)

## 🗺️ Roadmap

- [ ] Additional quantum walk variants
- [ ] GPU acceleration for large-scale networks
- [ ] Interactive visualization dashboard
- [ ] Pre-trained embeddings for common biological networks
- [ ] Integration with graph neural network frameworks
- [ ] Extended biological pathway databases

---

**QuVINE** - Bridging quantum computing and network science for next-generation graph embeddings.