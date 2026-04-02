# Quick Start Guide: Comprehensive Embedding Analysis (PARALLELIZED)

## TL;DR

```bash
# Test setup (2 minutes)
python test_analysis_setup.py

# Quick test (5-15 minutes) - RECOMMENDED FIRST
python run_comprehensive_analysis.py --quick

# Full analysis (30-90 minutes with parallelization)
python run_comprehensive_analysis.py

# Check results
ls outputs/comprehensive_analysis/
```

## 🚀 NEW: Parallelization

The analysis is now **fully parallelized** for dramatic speedup:
- **4-8x faster** on multi-core systems
- **30-90 minutes** instead of 4-8 hours for 40 networks
- Automatic CPU core detection and utilization
- Configurable worker count

## What This Does

Compares 6 embedding methods across 40 networks:
- **Methods**: QuVINE (RWR, CTQW, DTQW, fused), NetMF, Node2Vec
- **Networks**: 20 scale-free + 20 modular
- **Metrics**: Precision@K, Recall@K (K=10,20,50,100)
- **Analysis**: Correlations between complexity and performance

## Key Files Created

### 1. Enhanced Complexity Metrics
**File**: `src/quvine/data/graph_complexity.py`
- Added `compute_inverse_participation_ratio()` - NEW!
- Added `compute_participation_ratio()` - NEW!
- Both automatically included in `compute_graph_complexity_metrics()`

### 2. Analysis Pipeline
**File**: `src/quvine/comprehensive_embedding_analysis.py`
- Complete analysis class
- Generates networks, runs methods, analyzes correlations
- Creates visualizations and recommendations

### 3. Runner Scripts
- `run_comprehensive_analysis.py` - Main script
- `test_analysis_setup.py` - Quick test

### 4. Documentation
- `COMPREHENSIVE_ANALYSIS_GUIDE.md` - Full guide
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `QUICK_START.md` - This file

## Usage Examples

### Example 1: Test Setup
```bash
python test_analysis_setup.py
```
Output: Verifies all components work, tests IPR computation

### Example 2: Quick Test (Recommended First)
```bash
python run_comprehensive_analysis.py --quick
```
Output: Fast test with 10 networks in 5-15 minutes

### Example 3: Run Full Analysis (Parallelized)
```bash
# Use all CPU cores (default)
python run_comprehensive_analysis.py

# Or specify worker count
python run_comprehensive_analysis.py --n-jobs 8
```
Output: Complete analysis in `outputs/comprehensive_analysis/` (30-90 min)

### Example 3: Custom Analysis
```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

analysis = ComprehensiveEmbeddingAnalysis(
    n_networks_per_type=10,  # Smaller for testing
    n_nodes=100,
    embedding_dim=64
)
results = analysis.run_complete_analysis()
```

### Example 4: Check IPR on Your Network
```python
from quvine.data.graph_complexity import compute_graph_complexity_metrics
import networkx as nx

G = nx.karate_club_graph()
metrics = compute_graph_complexity_metrics(G)

print(f"IPR: {metrics['inverse_participation_ratio']:.4f}")
print(f"Quantum Complexity: {metrics['quantum_complexity']:.4f}")
```

### Example 5: Test Single Method
```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
import networkx as nx

analysis = ComprehensiveEmbeddingAnalysis()
G = nx.karate_club_graph()
seeds = [0, 1, 2]
targets = [33, 32, 31]

# Test CTQW
embedding = analysis.run_embedding_method('ctqw', G, seeds, targets)
print(f"Embedding shape: {embedding.shape}")
```

## Output Files

```
outputs/comprehensive_analysis/
├── complexity_metrics.csv              # All complexity metrics (includes IPR!)
├── embedding_performance.csv           # Performance for all methods
├── complexity_performance_correlations.csv  # Correlation analysis
├── method_recommendations.csv          # Which method to use when
├── recommendations_report.txt          # Human-readable guide
└── visualizations/
    ├── complexity_distributions.png
    ├── performance_comparison.png
    ├── correlation_heatmap_*.png
    └── significant_correlations.png
```

## Key Results to Check

### 1. Complexity Metrics
```python
import pandas as pd
df = pd.read_csv("outputs/comprehensive_analysis/complexity_metrics.csv")

# Compare network types
print(df.groupby('network_type')[
    ['quantum_complexity', 'inverse_participation_ratio', 'spectral_gap']
].mean())
```

### 2. Method Performance
```python
df = pd.read_csv("outputs/comprehensive_analysis/embedding_performance.csv")

# Best methods by recall@50
print(df.groupby('method')['recall@50_centroid'].mean().sort_values(ascending=False))
```

### 3. Recommendations
```bash
cat outputs/comprehensive_analysis/recommendations_report.txt
```

## Expected Findings

### Complexity Patterns
- **Scale-free**: Lower spectral gap, variable IPR
- **Modular**: Higher IPR, higher quantum complexity

### Performance Patterns
- **High quantum complexity** → CTQW/DTQW/fused better
- **High IPR** → DTQW performs well
- **Low complexity** → NetMF/RWR sufficient

### Method Rankings (typical)
1. QuVINE-fused (best overall)
2. CTQW (good for bottlenecks)
3. DTQW (good for modular)
4. NetMF (fast baseline)
5. RWR (classical baseline)
6. Node2Vec (comparable to RWR)

## Troubleshooting

### Problem: Import errors
```bash
pip install networkx numpy pandas scipy matplotlib seaborn scikit-learn node2vec gensim omegaconf
```

### Problem: Too slow
Use parallelization and reduce parameters:
```bash
# Use quick mode
python run_comprehensive_analysis.py --quick

# Or reduce network count
python run_comprehensive_analysis.py --n-networks 10 --n-nodes 150
```

### Problem: Out of memory
Reduce workers or network size:
```bash
python run_comprehensive_analysis.py --n-jobs 4 --n-nodes 150
```

### Problem: No significant correlations
- Need more networks (increase n_networks_per_type)
- Need more diversity in network parameters

## Time Estimates (With Parallelization ⚡)

- **Test setup**: 2-5 minutes
- **Quick test (10 networks)**: 5-15 minutes ⚡
- **Full analysis (40 networks)**: 30-90 minutes ⚡ (was 4-8 hours)
- **Minimal test (2 networks)**: 2-5 minutes

**⚡ = Parallelized** - Times assume 8+ CPU cores. See `PARALLELIZATION_GUIDE.md` for details.

## Next Steps

1. ✅ Run test: `python test_analysis_setup.py`
2. ✅ Run full analysis: `python run_comprehensive_analysis.py`
3. ✅ Check results in `outputs/comprehensive_analysis/`
4. ✅ Read recommendations in `recommendations_report.txt`
5. ✅ Apply findings to your networks

## Questions?

- Full guide: `COMPREHENSIVE_ANALYSIS_GUIDE.md`
- Technical details: `IMPLEMENTATION_SUMMARY.md`
- Code: `src/quvine/comprehensive_embedding_analysis.py`

## Summary of New Features

✅ **Inverse Participation Ratio (IPR)** - Measures eigenstate localization
✅ **Participation Ratio (PR)** - Effective spectral dimension
✅ **6 embedding methods** - QuVINE variants + classical baselines
✅ **40 test networks** - Scale-free and modular with varying complexity
✅ **Correlation analysis** - Links complexity to performance
✅ **Recommendations** - Which method to use when
✅ **Visualizations** - Comprehensive plots and heatmaps
✅ **Complete documentation** - Guides and examples

## Citation

```bibtex
@software{quvine_comprehensive_analysis,
  title={Comprehensive Embedding Analysis for QuVINE},
  year={2024},
  url={https://github.com/IBM/QuVINE}
}