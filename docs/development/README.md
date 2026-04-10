# QuVINE: Quantum-enabled View-Integrated Network Embeddings

[![QuVINE Framework][quvine]](#)

We introduced QuVINE, a quantum-enhanced multi-view network embedding framework designed to address the inherent complexity and heterogeneity of biological data in precision medicine. By moving beyond the limitations of classical, single-view random walks, QuVINE leverages quantum-inspired dynamics to capture higher-order topological features and long-range dependencies that are frequently lost in standard diffusion-based models.

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv_quvine
source venv_quvine/bin/activate  # On Windows: venv_quvine\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install QuVINE
pip install -e .

# 4. Run QuVINE
python -m quvine.main --config-path configs/ --config-name config.yaml
```

For detailed installation instructions, see [docs/setup/SETUP_INSTRUCTIONS.md](docs/setup/SETUP_INSTRUCTIONS.md).

## 📚 Documentation

Comprehensive documentation is available in the [docs/](docs/) directory:

### 🚀 [Setup & Installation](docs/setup/)
- **[SETUP_INSTRUCTIONS.md](docs/setup/SETUP_INSTRUCTIONS.md)** - Complete installation guide
- **[QBIOCODE_INSTALL.md](docs/setup/QBIOCODE_INSTALL.md)** - QBioCode integration

### 📖 [User Guides](docs/guides/)
- **[QUICK_START.md](docs/guides/QUICK_START.md)** - Get started quickly
- **[COMPREHENSIVE_ANALYSIS_GUIDE.md](docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)** - Full analysis pipeline
- **[COMPREHENSIVE_DATASET_GUIDE.md](docs/guides/COMPREHENSIVE_DATASET_GUIDE.md)** - Dataset generation
- **[PARALLELIZATION_GUIDE.md](docs/guides/PARALLELIZATION_GUIDE.md)** - HPC and parallel execution
- **[QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md](docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md)** - Quantum advantage analysis

### 🛠️ [Development](docs/development/)
- **[CONTRIBUTING.md](docs/development/CONTRIBUTING.md)** - Contribution guidelines
- **[IMPLEMENTATION_SUMMARY.md](docs/development/IMPLEMENTATION_SUMMARY.md)** - Technical details
- **[IMPROVEMENTS_AND_FIXES.md](docs/development/IMPROVEMENTS_AND_FIXES.md)** - Recent updates

### 📊 [API Reference](docs/)
- **[graph_complexity_guide.md](docs/graph_complexity_guide.md)** - Complexity metrics
- **[random_graphs_guide.md](docs/random_graphs_guide.md)** - Random graph generation

## ✨ Key Features

### Embedding Methods
- **Quantum Walks**: CTQW (Continuous-Time), DTQW (Discrete-Time)
- **Classical Walks**: RWR (Random Walk with Restart)
- **Fusion**: Multi-view embedding integration
- **Baselines**: NetMF, Node2Vec

### Complexity Metrics (35+)
- **Spectral**: Eigenvalue-based measures
- **Topological**: Ollivier-Ricci Curvature, Kirchhoff Index, Betti Numbers
- **Quantum**: Quantum advantage formulas (arithmetic, geometric, harmonic)
- **Structural**: Clustering, modularity, assortativity

### Downstream Tasks
- **Node Prioritization**: Disease gene ranking
- **Node Classification**: 6+ label generation strategies
- **Link Prediction**: 7 edge feature methods, 3 negative sampling strategies

### Evaluation Features
- **Hard Negative Sampling**: 2-hop, same-community
- **Inner Product & Cosine Similarity**: Quantum fidelity-based features
- **Data Leakage Prevention**: Proper train/test separation
- **Comprehensive Metrics**: AUC-ROC, AUC-PR, Precision@K, Recall@K, F1, MRR

## 📊 Usage Examples

### Basic Usage
```python
from quvine.main import main

# Run with default config
main()
```

### Generate Random Networks
```python
from quvine.data.random_graphs import generate_random_graph

# Generate scale-free network
G = generate_random_graph(
    graph_type='scale_free',
    n_nodes=500,
    params={'m': 3}
)
```

### Compute Complexity Metrics
```python
from quvine.data.graph_complexity import compute_graph_complexity_metrics

# Compute all metrics
metrics = compute_graph_complexity_metrics(G)
print(f"Spectral gap: {metrics['spectral_gap']}")
print(f"ORC mean: {metrics['orc_mean']}")
```

### Run Comprehensive Analysis
```python
from quvine.comprehensive_embedding_analysis import run_comprehensive_analysis

# Analyze embeddings across multiple tasks
results = run_comprehensive_analysis(
    graph=G,
    embedding_methods=['quvine_fused', 'netmf', 'node2vec'],
    tasks=['ranking', 'classification', 'link_prediction']
)
```

## 📓 Notebooks

Interactive examples are available in [notebooks/](notebooks/):
- **[quvine_embedding.ipynb](notebooks/quvine_embedding.ipynb)** - Basic QuVINE usage
- **[complexity_and_embedding_demo.ipynb](notebooks/complexity_and_embedding_demo.ipynb)** - Complexity analysis
- **[embedding_methods_comparison.ipynb](notebooks/embedding_methods_comparison.ipynb)** - Method comparison

## 🔬 Research Workflow

```
1. Generate Networks → 2. Compute Complexity → 3. Generate Embeddings
         ↓                       ↓                        ↓
4. Evaluate Tasks → 5. Analyze Correlations → 6. Generate Recommendations
```

See [docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md](docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md) for details.

## 📝 Citation

Please cite the following article if you use QuVINE:

*Quantum-enhanced Network Embeddings via Multi-view Integration for Precision Medicine*,
A. Bose, F. Utro and L. Parida, 2026. (Under Review)

## 🤝 Contributing

We welcome contributions! Please read [docs/development/CONTRIBUTING.md](docs/development/CONTRIBUTING.md) for guidelines.

## 📄 License

See [LICENSE](LICENSE) for details.

## 🆕 Recent Updates (April 2026)

- ✅ **Fixed critical data leakage bugs** in classification and link prediction
- ✅ **Added hard negative sampling** (2-hop, same-community)
- ✅ **Added inner product & cosine similarity** edge features
- ✅ **13 new topological complexity metrics**
- ✅ **3 quantum advantage formulas** for empirical comparison
- ✅ **Multi-task evaluation pipeline** (ranking, classification, link prediction)
- ✅ **Organized documentation** into structured folders

See [docs/development/IMPROVEMENTS_AND_FIXES.md](docs/development/IMPROVEMENTS_AND_FIXES.md) for details.

---

**Version**: 2.0 (Multi-task evaluation with bug fixes)
**Last Updated**: April 2, 2026

<!-- MARKDOWN LINKS & IMAGES -->
[quvine]: images/quvine_framework.png