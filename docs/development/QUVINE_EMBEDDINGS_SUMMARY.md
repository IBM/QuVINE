# QuVINE Embeddings Summary

## Overview
This document lists all available QuVINE embedding methods integrated in `comprehensive_embedding_analysis.py`.

## Complete List of QuVINE Embeddings

### 1. Original Quantum Walk Embeddings (3 methods)
- **`quvine_rwr`** - Random Walk with Restart
- **`quvine_ctqw`** - Continuous-Time Quantum Walk
- **`quvine_dtqw`** - Discrete-Time Quantum Walk

### 2. Q-Caliber Filter Embeddings (2 methods)
- **`quvine_heat`** - Quantum-calibrated Heat Kernel filter
- **`quvine_poly`** - Quantum-calibrated Polynomial filter

### 3. Q-Caliber GCN-MF Embeddings (2 methods)
- **`quvine_hgcnmf`** - Heat kernel + GCN-MF (Graph Convolutional Network with Matrix Factorization)
- **`quvine_pgcnmf`** - Polynomial filter + GCN-MF

### 4. Fusion Embeddings (6 methods)
- **`quvine_fused_svd`** - SVD-based fusion of all 7 QuVINE methods
- **`quvine_fused_graphreg`** - Graph-regularized fusion of all 7 QuVINE methods
- **`quvine_fused_attention`** - Attention-based fusion of all 7 QuVINE methods
- **`quvine_fused_hybrid`** - Hybrid fusion of all 7 QuVINE methods
- **`quvine_fused_svd_shared_priv_heat_poly`** - SVD shared/private decomposition with attention gate (heat + poly)
- **`quvine_fused_svd_shared_priv_moe_heat_poly`** - SVD shared/private decomposition with mixture-of-experts gate (heat + poly)

### 5. Baseline Methods (3 methods)
- **`netmf`** - Network Matrix Factorization
- **`node2vec`** - Node2Vec random walk embeddings
- **`appnp`** - Approximate Personalized Propagation of Neural Predictions

## Total: 16 Methods
- **7 QuVINE variants** (3 quantum walks + 2 Q-Caliber filters + 2 Q-Caliber GCN-MF)
- **6 Fusion methods** (combining multiple QuVINE embeddings)
- **3 Baselines** (for comparison)

## Fusion Method Details

### Standard Fusion Methods
These fuse all 7 QuVINE methods (ctqw, dtqw, rwr, heat, poly, hgcnmf, pgcnmf):
1. **SVD**: Singular Value Decomposition-based fusion
2. **GraphReg**: Graph-regularized fusion using Laplacian
3. **Attention**: Attention-weighted fusion
4. **Hybrid**: Combination of multiple fusion strategies

### Advanced SVD Shared/Private Fusion
These use SVD to decompose embeddings into shared and private components:
1. **Attention Gate**: Uses attention mechanism to combine shared/private components
2. **Mixture of Experts (MoE)**: Uses MLP-based gating to combine components

Default configuration: fuses `heat` and `poly` embeddings with rank-k SVD (k = embedding_dim // 4)

## Custom Fusion Syntax

The fusion system supports flexible method combinations:

```python
# Fuse specific methods with specific fusion type
'quvine_fused_svd_ctqw_heat'              # ctqw + heat, SVD fusion
'quvine_fused_attention_rwr_poly_hgcnmf'  # rwr + poly + hgcnmf, attention fusion

# SVD shared/private with custom methods
'quvine_fused_svd_shared_priv_ctqw_rwr'   # ctqw + rwr, attention gate
'quvine_fused_svd_shared_priv_moe_heat_poly'  # heat + poly, MoE gate
```

## Integration Status

✅ **All methods fully integrated** in `comprehensive_embedding_analysis.py`:
- Line 426-680: `run_embedding_method()` handles all QuVINE variants
- Line 1070-1081: Default methods list in `_process_single_network()`
- Line 1134-1145: Default methods list in `run_all_methods_on_networks()`

## Usage Example

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Initialize analyzer
analyzer = ComprehensiveEmbeddingAnalysis(
    output_dir='results',
    n_jobs=4,
    embedding_dim=128
)

# Run analysis on networks
# This will automatically use all 16 methods
analyzer.run_comprehensive_analysis(
    n_networks=50,
    n_nodes=500
)
```

## Expected Output Format

When running comprehensive analysis, results will include:
- **Ranking performance**: precision@k, recall@k for seed node ranking
- **Classification performance**: accuracy, F1-score for node classification
- **Link prediction performance**: AUC, AP for link prediction
- **Complexity correlations**: relationships between graph complexity and performance

All results saved to CSV files in the output directory.