# QuVINE Documentation

Welcome to the QuVINE documentation! This directory contains comprehensive guides, setup instructions, and development documentation.

## 📁 Documentation Structure

### 🚀 [Setup](./setup/)
Installation and configuration guides:
- **[SETUP_INSTRUCTIONS.md](./setup/SETUP_INSTRUCTIONS.md)** - Complete installation guide
- **[QBIOCODE_INSTALL.md](./setup/QBIOCODE_INSTALL.md)** - QBioCode integration setup

### 📖 [User Guides](./guides/)
Comprehensive usage guides and tutorials:
- **[QUICK_START.md](./guides/QUICK_START.md)** - Get started quickly with QuVINE
- **[COMPREHENSIVE_ANALYSIS_GUIDE.md](./guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)** - Full analysis pipeline walkthrough
- **[COMPREHENSIVE_DATASET_GUIDE.md](./guides/COMPREHENSIVE_DATASET_GUIDE.md)** - Dataset generation and management
- **[PARALLELIZATION_GUIDE.md](./guides/PARALLELIZATION_GUIDE.md)** - Parallel execution and HPC usage
- **[QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md](./guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md)** - Quantum advantage analysis and downstream task evaluation

### 🛠️ [Development](./development/)
Developer documentation and implementation details:
- **[CONTRIBUTING.md](./development/CONTRIBUTING.md)** - Contribution guidelines
- **[IMPLEMENTATION_SUMMARY.md](./development/IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[IMPROVEMENTS_AND_FIXES.md](./development/IMPROVEMENTS_AND_FIXES.md)** - Recent improvements and bug fixes

### 📚 [API Documentation](../src/quvine/)
Code-level documentation:
- **[graph_complexity_guide.md](./graph_complexity_guide.md)** - Graph complexity metrics reference
- **[random_graphs_guide.md](./random_graphs_guide.md)** - Random graph generation guide
- **[random_graphs_quick_reference.md](./random_graphs_quick_reference.md)** - Quick reference for random graphs

## 🎯 Quick Navigation

### For New Users
1. Start with [SETUP_INSTRUCTIONS.md](./setup/SETUP_INSTRUCTIONS.md)
2. Follow [QUICK_START.md](./guides/QUICK_START.md)
3. Explore [COMPREHENSIVE_ANALYSIS_GUIDE.md](./guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)

### For Researchers
1. Review [QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md](./guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md)
2. Check [COMPREHENSIVE_DATASET_GUIDE.md](./guides/COMPREHENSIVE_DATASET_GUIDE.md)
3. Use [PARALLELIZATION_GUIDE.md](./guides/PARALLELIZATION_GUIDE.md) for large-scale experiments

### For Developers
1. Read [CONTRIBUTING.md](./development/CONTRIBUTING.md)
2. Review [IMPLEMENTATION_SUMMARY.md](./development/IMPLEMENTATION_SUMMARY.md)
3. Check [IMPROVEMENTS_AND_FIXES.md](./development/IMPROVEMENTS_AND_FIXES.md)

## 📊 Key Features Documented

### Complexity Metrics
- **35+ complexity metrics** including spectral, topological, and quantum measures
- **Ollivier-Ricci Curvature**, **Kirchhoff Index**, **Persistent Betti Numbers**
- **3 Quantum Advantage Formulas**: Arithmetic, Geometric, Harmonic means

### Embedding Methods
- **QuVINE variants**: RWR, CTQW, DTQW, Fused
- **Classical baselines**: NetMF, Node2Vec
- **6 embedding methods** compared across multiple tasks

### Downstream Tasks
- **Node Prioritization** (Ranking)
- **Node Classification** (6+ label generation strategies)
- **Link Prediction** (7 edge feature methods, 3 negative sampling strategies)

### Evaluation Features
- **Hard Negative Sampling**: 2-hop, same-community
- **Inner Product & Cosine Similarity**: Quantum fidelity-based features
- **Data Leakage Prevention**: Proper train/test separation
- **Comprehensive Metrics**: AUC-ROC, AUC-PR, Precision@K, Recall@K, F1, MRR

## 🔬 Research Workflow

```
1. Setup Environment
   └─> docs/setup/SETUP_INSTRUCTIONS.md

2. Generate Networks
   └─> docs/guides/COMPREHENSIVE_DATASET_GUIDE.md

3. Compute Complexity
   └─> docs/graph_complexity_guide.md

4. Run Embeddings
   └─> docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md

5. Evaluate Tasks
   └─> docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md

6. Analyze Results
   └─> docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md

7. Generate Recommendations
   └─> docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md
```

## 📝 Recent Updates

### Latest Features (2026-04-02)
- ✅ **Fixed critical data leakage bugs** in classification and link prediction
- ✅ **Added hard negative sampling** (2-hop, same-community)
- ✅ **Added inner product & cosine similarity** edge features
- ✅ **13 new topological complexity metrics**
- ✅ **3 quantum advantage formulas** for empirical comparison
- ✅ **Multi-task evaluation pipeline** (ranking, classification, link prediction)

See [IMPROVEMENTS_AND_FIXES.md](./development/IMPROVEMENTS_AND_FIXES.md) for details.

## 🤝 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](./development/CONTRIBUTING.md) for:
- Code style guidelines
- Pull request process
- Testing requirements
- Documentation standards

## 📧 Support

For questions or issues:
1. Check the relevant guide in this documentation
2. Review [IMPLEMENTATION_SUMMARY.md](./development/IMPLEMENTATION_SUMMARY.md)
3. Open an issue on GitHub with the `question` label

## 📄 License

See [LICENSE](../LICENSE) in the root directory.

---

**Last Updated**: April 2, 2026  
**Version**: 2.0 (Multi-task evaluation with bug fixes)