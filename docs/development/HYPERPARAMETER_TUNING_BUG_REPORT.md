# Hyperparameter Tuning Bug Report

## Critical Bug: GAT and GraphGPS Methods Use Default Hyperparameters

### Summary
All 24 GAT and GraphGPS method variants (12 each) are using **default hyperparameters** instead of tuned hyperparameters because the tuning system has a mapping inconsistency.

### Root Cause

**File:** `QuVINE/scripts/tune_hyperparameters.py`

The `METHOD_TUNING_MAP` (lines 184-196) maps methods to their tuning representatives:

```python
METHOD_TUNING_MAP = {
    # All quvine methods use quvine_walks params
    **{m: "quvine_walks" for m in ALL_39_METHODS if m.startswith("quvine_")},
    # All GAT methods use gat_baseline params (default, not tuned)
    **{m: "gat_baseline" for m in ALL_39_METHODS if m.startswith("gat_")},
    # All GraphGPS methods use graphgps_baseline params (default, not tuned)
    **{m: "graphgps_baseline" for m in ALL_39_METHODS if m.startswith("graphgps_")},
    # Classical methods tune individually
    "node2vec": "node2vec",
    "netmf": "netmf",
    "graphsage": "graphsage",
    "baseline_gcnmf": "baseline_gcnmf",
}
```

**Problem:** `gat_baseline` and `graphgps_baseline` are **NOT** in the `TUNE_METHODS` list, so they are never tuned. The tuning script only tunes 8 methods:
- `quvine_walks`
- `baseline_filter_heat`
- `baseline_filter_poly`
- `baseline_gcnmf`
- `node2vec`
- `netmf`
- `graphsage`
- `appnp`

### Impact

1. **24 methods affected:** All 12 GAT variants + all 12 GraphGPS variants
2. **Performance degradation:** These methods use default hyperparameters which may be suboptimal
3. **Inconsistent results:** QuVINE methods get tuned params, but GAT/GraphGPS don't
4. **Misleading documentation:** Comments say "use gat_baseline params" but those params don't exist in tuned JSON files

### Evidence

**In `comprehensive_embedding_analysis.py`:**

Line 1788 (GAT methods):
```python
hp = (method_hyperparams or {}).get('gat', {})
```

Line 1824 (GraphGPS methods):
```python
hp = (method_hyperparams or {}).get('graphgps', {})
```

Both look for keys that don't exist in the tuned hyperparameter JSON files because `gat_baseline` and `graphgps_baseline` were never tuned.

### Verification

Check any `best_hyperparams.json` file created by the tuning script - it will only contain 8 methods, not including 'gat' or 'graphgps'.

### Recommended Fixes

**Option 1: Add GAT and GraphGPS to tuning (RECOMMENDED)**
- Add `gat_baseline` and `graphgps_baseline` to `TUNE_METHODS`
- Update `METHOD_TUNING_MAP` to map all GAT methods to `gat_baseline` and all GraphGPS to `graphgps_baseline`
- Re-run tuning to generate hyperparameters for these methods

**Option 2: Map to existing tuned methods**
- Map GAT methods to use `graphsage` or `appnp` hyperparameters (similar GNN architectures)
- Map GraphGPS methods to use `graphsage` or `appnp` hyperparameters
- Update `METHOD_TUNING_MAP` accordingly

**Option 3: Document as intentional**
- If using defaults is intentional, update documentation to clarify
- Remove misleading comments about "using gat_baseline params"
- Add explicit logging when methods use defaults vs tuned params

### Current Behavior

When analysis scripts run:
1. Load `best_hyperparams.json` containing 8 tuned methods
2. Pass `method_hyperparams` dict to `run_embedding_method()`
3. For GAT methods: Look for 'gat' key → not found → use empty dict `{}`
4. For GraphGPS methods: Look for 'graphgps' key → not found → use empty dict `{}`
5. Methods fall back to hardcoded defaults in the code

### Files Affected

1. `QuVINE/scripts/tune_hyperparameters.py` - Tuning configuration
2. `QuVINE/src/quvine/comprehensive_embedding_analysis.py` - Hyperparameter application
3. `QuVINE/scripts/run_hard_negative_network.py` - Loads and passes hyperparameters
4. `QuVINE/scripts/run_ppi_network.py` - Loads and passes hyperparameters
5. All `best_hyperparams.json` files - Missing GAT/GraphGPS entries

### Testing Recommendations

1. Add test to verify all 39 methods have hyperparameter entries in tuning output
2. Add test to verify `METHOD_TUNING_MAP` keys match tuned method names
3. Add logging to track when methods use tuned vs default hyperparameters
4. Create integration test that runs all 39 methods and verifies hyperparameter usage

---

**Date:** 2026-04-27  
**Severity:** High  
**Status:** Identified, awaiting fix