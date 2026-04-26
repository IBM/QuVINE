# Data Leakage Audit Report

## Executive Summary

**Status:** ⚠️ **CRITICAL LEAKAGE ISSUES FOUND**

This audit identifies train/test data leakage in GAT and quantum filter feature generation that could inflate reported performance metrics.

---

## Issues Identified

### 🔴 CRITICAL: Issue 1 - Global Feature Normalization in GAT
**Location:** `src/quvine/baselines/gat.py:243` in `build_structural_features()`

**Problem:**
```python
def build_structural_features(G, nodelist, normalize=True):
    # ... compute features for ALL nodes ...
    return standardize_columns(X) if normalize else X  # ❌ LEAKAGE
```

The `standardize_columns()` function computes mean/std across **all nodes** in the graph, including test nodes. This leaks test set statistics into training.

**Impact:** High - affects all GAT variants using structural features

**Fix Required:**
```python
def build_structural_features_safe(G, nodelist, train_mask=None, normalize=True):
    """Build structural features with optional train-only normalization."""
    # ... compute features ...
    if normalize and train_mask is not None:
        # Compute stats on train nodes only
        mu = X[train_mask].mean(axis=0, keepdims=True)
        sd = X[train_mask].std(axis=0, keepdims=True)
        X = (X - mu) / np.maximum(sd, 1e-12)
    elif normalize:
        X = standardize_columns(X)  # Fallback for unsupervised
    return X
```

---

### 🔴 CRITICAL: Issue 2 - Quantum Calibration Uses Full Graph
**Location:** `src/quvine/baselines/gat.py:976, 983, 990, 997` and `src/quvine/embedding/quantum_filters.py:87, 160`

**Problem:**
```python
# In build_gat_input_features():
loss, t_star = calibrate_heat_kernel(L, ctqw_targets, node_to_idx, t_grid=t_grid)
X = apply_heat_filter(L, X0, t_star)  # ❌ Applies to ALL nodes
```

The calibration uses quantum targets that may include test nodes, and the filter is applied to the **entire graph** including test nodes. The Laplacian `L` is computed from the full graph structure.

**Impact:** High - affects all quantum-calibrated variants

**Fix Required:**
For **transductive** tasks (node classification), this is actually acceptable because the graph structure is known. However, for **inductive** tasks, we need:
```python
def calibrate_on_train_subgraph(G, train_nodes, ctqw_targets_train, ...):
    """Calibrate using only training subgraph."""
    G_train = G.subgraph(train_nodes).copy()
    L_train = build_normalized_laplacian(G_train, ...)
    # Calibrate on train subgraph only
    t_star = calibrate_heat_kernel(L_train, ctqw_targets_train, ...)
    return t_star
```

---

### 🟡 MODERATE: Issue 3 - Link Prediction Edge Sampling
**Location:** `src/quvine/baselines/gat.py:750-770` in `train_gat_link_reconstruction()`

**Problem:**
```python
def train_gat_link_reconstruction(...):
    # Negative sampling uses full graph
    neg_edges = sample_negative_edges(G, num_neg=num_pos, seed=...)  # ⚠️
```

**Current Status:** Appears safe - negative edges are sampled from non-edges, which is standard practice.

**Recommendation:** Verify that `sample_negative_edges()` doesn't use test positive edges.

---

### 🟢 SAFE: Evaluation Modules
**Location:** `src/quvine/evaluation/classification.py:417-418` and `link_prediction.py:326-327`

**Status:** ✅ **CORRECT**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # ✅ Fit on train
X_test_scaled = scaler.transform(X_test)        # ✅ Transform test
```

The evaluation modules properly separate train/test normalization.

---

## Recommendations

### Priority 1: Fix Feature Normalization (CRITICAL)

**Action:** Modify `build_structural_features()` and `build_gat_input_features()` to accept `train_mask` parameter.

**Implementation:**
1. Add `train_mask` parameter to feature building functions
2. Compute normalization statistics on train nodes only
3. Apply same normalization to test nodes
4. Update all callers to pass train_mask

### Priority 2: Clarify Transductive vs Inductive (HIGH)

**Action:** Document which tasks are transductive (graph structure known) vs inductive (test graph unseen).

**Current Understanding:**
- **Node Classification:** Transductive (full graph known) - current approach is acceptable
- **Link Prediction:** Inductive (test edges hidden) - current approach is acceptable
- **Node Ranking:** Transductive - current approach is acceptable

**Recommendation:** Add explicit `mode='transductive'` parameter to make assumptions clear.

### Priority 3: Add Leakage Tests (MEDIUM)

**Action:** Create unit tests that verify no test data leaks into training.

**Test Cases:**
```python
def test_no_feature_leakage():
    """Verify train/test features use different normalization stats."""
    G = nx.karate_club_graph()
    train_mask = np.zeros(G.number_of_nodes(), dtype=bool)
    train_mask[:20] = True
    
    X_all = build_structural_features(G, normalize=True)
    X_safe = build_structural_features_safe(G, train_mask=train_mask, normalize=True)
    
    # Stats should differ
    assert not np.allclose(X_all.mean(axis=0), X_safe.mean(axis=0))
```

---

## Severity Assessment

| Issue | Severity | Impact | Fix Complexity |
|-------|----------|--------|----------------|
| Global feature normalization | 🔴 Critical | High | Medium |
| Quantum calibration on full graph | 🔴 Critical* | Medium | High |
| Link prediction sampling | 🟡 Moderate | Low | Low |

*Critical for inductive tasks, acceptable for transductive tasks

---

## Conclusion

The current implementation has **data leakage in feature normalization** that could inflate performance metrics. However, for **transductive node classification** (the primary use case), using the full graph structure is theoretically justified.

**Recommended Actions:**
1. ✅ Fix feature normalization to use train-only statistics
2. ✅ Document transductive vs inductive assumptions
3. ✅ Add leakage prevention tests
4. ⚠️ Consider adding `inductive_mode` flag for future use cases

---

## References

- Transductive Learning: Test nodes are in the graph during training, but labels are hidden
- Inductive Learning: Test nodes are completely unseen during training
- Standard practice in GNN literature: Using full graph structure for transductive tasks is acceptable