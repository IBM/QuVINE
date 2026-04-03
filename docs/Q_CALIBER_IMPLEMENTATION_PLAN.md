# Q-Caliber Implementation Plan

## Overview
Implement 8 new embedding methods based on Q-Caliber (Quantum-Calibrated Graph Diffusion) and advanced fusion strategies.

## New Embedding Methods

### 1-2. Q-Caliber Graph Filters
**Module**: `src/quvine/embedding/qcaliber_filters.py`

- **Q-Caliber Heat Filter**: Quantum-calibrated heat kernel diffusion
  - Calibrate `t_star` parameter using quantum walk targets
  - Apply: `Z = exp(-t_star * L) @ X`
  
- **Q-Caliber Poly Filter**: Quantum-calibrated polynomial filter
  - Calibrate polynomial coefficients `{a_k}` using quantum walk targets
  - Apply: `Z = sum_k a_k * L^k @ X`

### 3. Baseline Graph Filter
**Module**: `src/quvine/embedding/baseline_filters.py`

- Standard graph diffusion without quantum calibration
- Fixed parameters (e.g., t=1.0 for heat, degree-4 polynomial)
- Serves as classical baseline

### 4-6. GCN-MF Variants
**Module**: `src/quvine/baselines/gcn_mf.py` (already exists, extend)

- **Baseline GCN-MF**: Standard GCN-MF without quantum calibration
- **Q-Caliber GCN-MF (Heat)**: GCN-MF with heat-diffused features
- **Q-Caliber GCN-MF (Poly)**: GCN-MF with poly-diffused features

### 7-8. Merged Q-Caliber Methods
**Module**: `src/quvine/embedding/qcaliber_merged.py`

- **Merged Graph Filter (Heat + Poly)**: Fuse heat and poly embeddings
- **Merged GCN-MF (Heat + Poly)**: Fuse GCN-MF with heat and poly

## Fusion Strategies

### Strategy 1: SVD Fusion (Existing)
- Concatenate embeddings → SVD → Top-k components
- Fast, simple, strong baseline

### Strategy 2: Attention-Based Fusion (NEW)
```python
# Learn attention weights for each embedding
alpha = softmax(MLP([Z1, Z2, ..., Zn]))
Z_fused = sum(alpha_i * Z_i)
```

### Strategy 3: Gated Residual Fusion (NEW)
```python
# SVD for shared representation
U, S, Vh = SVD(concat([Z_h, Z_p]))
Z_shared = U[:, :k] * S[:k]

# Compute residuals
Z_h_priv = Z_h - Z_shared
Z_p_priv = Z_p - Z_shared

# Gated combination
gate = sigmoid(MLP(concat([Z_shared, Z_h_priv, Z_p_priv])))
Z_final = Z_shared + gate[:, :d]*Z_h_priv + gate[:, d:]*Z_p_priv
```

### Strategy 4: Graph-Regularized Fusion (Existing)
- SVD + Laplacian smoothing
- Solve: `argmin_U ||U - Z_svd||^2 + beta * tr(U^T L U)`

## Implementation Steps

### Phase 1: Q-Caliber Filters (Priority 1)
1. Create `qcaliber_filters.py` with:
   - `calibrate_heat_kernel()`: Fit t_star parameter
   - `calibrate_polynomial_filter()`: Fit polynomial coefficients
   - `apply_qcaliber_heat()`: Generate heat-filtered embeddings
   - `apply_qcaliber_poly()`: Generate poly-filtered embeddings
   - `apply_baseline_filter()`: Generate baseline embeddings

### Phase 2: GCN-MF Extensions (Priority 2)
1. Extend `gcn_mf.py` with:
   - Wrapper functions for easy embedding generation
   - Support for precomputed diffused features
   - Embedding extraction without training

### Phase 3: Merged Methods (Priority 3)
1. Create `qcaliber_merged.py` with:
   - `merge_graph_filters()`: Fuse heat + poly filters
   - `merge_gcnmf()`: Fuse GCN-MF variants

### Phase 4: Advanced Fusion (Priority 4)
1. Update `fusion/fuse.py` with:
   - `fuse_attention()`: Attention-based fusion
   - `fuse_gated_residual()`: Gated residual fusion
   - `fuse_adaptive()`: Adaptive fusion (selects best strategy)

### Phase 5: Integration (Priority 5)
1. Update `embedding/registry.py`:
   - Register all 8 new methods
   - Add fusion strategy selection
2. Update `comprehensive_embedding_analysis.py`:
   - Support new embedding methods
   - Support fusion strategies

## Method Naming Convention

```python
EMBEDDING_METHODS = {
    # Existing
    'quvine_rwr': ...,
    'quvine_ctqw': ...,
    'quvine_dtqw': ...,
    'quvine_fused': ...,  # SVD fusion of RWR+CTQW+DTQW
    'netmf': ...,
    'node2vec': ...,
    
    # New Q-Caliber Filters
    'qcaliber_heat': ...,
    'qcaliber_poly': ...,
    'baseline_filter': ...,
    
    # New GCN-MF
    'gcnmf_baseline': ...,
    'gcnmf_qcaliber_heat': ...,
    'gcnmf_qcaliber_poly': ...,
    
    # New Merged
    'qcaliber_merged_filter': ...,  # Heat + Poly
    'gcnmf_merged': ...,  # GCN-MF Heat + Poly
}

FUSION_STRATEGIES = {
    'svd': ...,  # Existing
    'graphreg': ...,  # Existing
    'attention': ...,  # NEW
    'gated_residual': ...,  # NEW
    'adaptive': ...,  # NEW (auto-select best)
}
```

## Comparison Matrix

| Method | Type | Quantum | Calibrated | Complexity |
|--------|------|---------|------------|------------|
| QuVINE-RWR | Walk | No | No | O(n·d·t) |
| QuVINE-CTQW | Walk | Yes | No | O(n·d·t) |
| QuVINE-DTQW | Walk | Yes | No | O(n·d·t) |
| NetMF | MF | No | No | O(n²·k) |
| Node2Vec | Walk | No | No | O(n·d·t) |
| Q-Caliber Heat | Filter | Yes | Yes | O(n·k) |
| Q-Caliber Poly | Filter | Yes | Yes | O(n·k·K) |
| Baseline Filter | Filter | No | No | O(n·k) |
| GCN-MF Baseline | GNN+MF | No | No | O(n·d·k) |
| GCN-MF Q-Heat | GNN+MF | Yes | Yes | O(n·d·k) |
| GCN-MF Q-Poly | GNN+MF | Yes | Yes | O(n·d·k) |
| Q-Merged Filter | Filter | Yes | Yes | O(n·k) |
| GCN-MF Merged | GNN+MF | Yes | Yes | O(n·d·k) |

## Expected Outcomes

### Research Questions
1. **Does quantum calibration improve performance?**
   - Compare Q-Caliber vs Baseline filters
   - Compare Q-Caliber GCN-MF vs Baseline GCN-MF

2. **Which fusion strategy works best?**
   - SVD vs Attention vs Gated Residual
   - When does each strategy excel?

3. **Do merged methods outperform individual methods?**
   - Heat+Poly vs Heat alone vs Poly alone
   - GCN-MF merged vs individual variants

4. **How do complexity metrics correlate with method performance?**
   - High spectral gap → which method?
   - High modularity → which method?
   - High quantum complexity → quantum advantage?

### Method Recommendations (Goal)
Generate a decision tree/table:
```
IF spectral_gap > 0.5 AND modularity > 0.6:
    USE qcaliber_merged_filter with gated_residual fusion
ELIF quantum_complexity > 0.7:
    USE quvine_ctqw with attention fusion
ELIF ...
```

## Testing Strategy

### Unit Tests
- Test each embedding method independently
- Test each fusion strategy independently
- Test parameter calibration accuracy

### Integration Tests
- Test full pipeline with all methods
- Test on synthetic networks (scale-free, modular)
- Verify output shapes and ranges

### Performance Tests
- Benchmark runtime for each method
- Memory profiling
- Scalability tests (100, 500, 1000, 5000 nodes)

## Documentation

### User Guide
- How to use each embedding method
- How to select fusion strategy
- Parameter tuning guidelines

### API Reference
- Function signatures
- Parameter descriptions
- Return value specifications

### Examples
- Jupyter notebook with all methods
- Comparison visualizations
- Method recommendation demo

## Timeline

- **Week 1**: Phase 1 (Q-Caliber Filters)
- **Week 2**: Phase 2 (GCN-MF Extensions)
- **Week 3**: Phase 3 (Merged Methods) + Phase 4 (Advanced Fusion)
- **Week 4**: Phase 5 (Integration) + Testing + Documentation

## Dependencies

### Required
- numpy, scipy (existing)
- torch (existing, for GCN-MF and fusion)
- networkx (existing)
- scikit-learn (existing)

### Optional
- hiperwalk (existing, for quantum walk calibration)
- python-louvain (existing, for community detection)

## Notes

- All methods should support both numpy and torch tensors
- All methods should handle sparse matrices efficiently
- All methods should have consistent API
- All methods should include proper error handling
- All methods should be well-documented

---

**Status**: Planning Complete
**Next**: Begin Phase 1 Implementation