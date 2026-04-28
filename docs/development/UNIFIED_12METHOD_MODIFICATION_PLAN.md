# Unified 12-Method Configuration - Detailed Modification Plan

## Overview
This document outlines all changes needed to migrate from separate PPI/synthetic configs to a unified 12-method configuration.

---

## PHASE 2: Update `tune_by_task_with_config.py`

### File: `QuVINE/scripts/tune_by_task_with_config.py`

### Change 2.1: Update Default Config Path
**Location:** Line 811
**Current:**
```python
default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuning_config.yaml')
```
**New:**
```python
default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unified_tuning_config.yaml')
```
**Reason:** Use new unified config as default

---

### Change 2.2: Modify `run_quvine_walks()` to Support Walk Types
**Location:** Lines 344-401
**Current:** Function generates walks using default 'rwr' kind
**New:** Add `walk_type` parameter to specify rwr/ctqw/dtqw

**Detailed Changes:**
```python
# Line 344: Add walk_type parameter
def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any], walk_type: str = 'rwr') -> np.ndarray:
    """Run QuVINE walks method with specified walk type."""
    
    # Line 323: Update walks config to use specified walk_type
    "walks": {
        "kinds": [walk_type],  # Changed from ["rwr"] to [walk_type]
        "num_walks": params.get("num_walks", 10),
        # ... rest stays same
    }
```

**Impact:** Allows single function to handle all 3 quantum walk variants

---

### Change 2.3: Update `generate_embedding()` for 12 Methods
**Location:** Lines 586-690
**Current:** Handles 10 methods (quvine_walks, baseline_filter_heat, etc.)
**New:** Handle 12 methods with proper splits

**Method Mapping:**
```
OLD METHOD              → NEW METHOD(S)
─────────────────────────────────────────────
quvine_walks           → quvine_rwr, quvine_ctqw, quvine_dtqw
baseline_filter_heat   → baseline_filter_heat (keep)
baseline_filter_poly   → baseline_filter_poly (keep)
baseline_gcnmf         → baseline_gcnmf (keep)
node2vec               → node2vec (keep)
netmf                  → netmf (keep)
graphsage              → graphsage (keep)
appnp                  → appnp (keep)
gat_baseline           → gat_baseline (keep)
graphgps_baseline      → graphgps_baseline (keep)
```

**Detailed Changes:**
```python
def generate_embedding(method: str, G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Generate embedding for a method with given parameters."""
    try:
        # NEW: Split quvine_walks into 3 variants
        if method == "quvine_rwr":
            return run_quvine_walks(G, seeds, params, walk_type='rwr')
        elif method == "quvine_ctqw":
            return run_quvine_walks(G, seeds, params, walk_type='ctqw')
        elif method == "quvine_dtqw":
            return run_quvine_walks(G, seeds, params, walk_type='dtqw')
        
        # KEEP: Filter baselines (already split)
        elif method == "baseline_filter_heat":
            return run_filter_embedding(G, seeds, params, "heat")
        elif method == "baseline_filter_poly":
            return run_filter_embedding(G, seeds, params, "poly")
        
        # KEEP: GCN-MF baseline
        elif method == "baseline_gcnmf":
            return run_gcnmf_embedding(G, seeds, params)
        
        # KEEP: Classical random walk methods
        elif method == "node2vec":
            # ... existing code ...
        elif method == "netmf":
            # ... existing code ...
        
        # KEEP: GNN methods
        elif method == "graphsage":
            # ... existing code ...
        elif method == "appnp":
            # ... existing code ...
        elif method == "gat_baseline":
            # ... existing code ...
        elif method == "graphgps_baseline":
            # ... existing code ...
        
        else:
            logger.warning(f"Unknown method: {method}")
            return None
```

**Lines to modify:** 586-690 (entire function)
**New total methods:** 12 (was 10)

---

## PHASE 3: Update `tune_ppi_by_task.py`

### File: `QuVINE/scripts/tune_ppi_by_task.py`

### Change 3.1: Update Default Config Path
**Location:** Line 611
**Current:**
```python
parser.add_argument('--config', type=str, default='scripts/ppi_tuning_config.yaml',
```
**New:**
```python
parser.add_argument('--config', type=str, default='scripts/unified_tuning_config.yaml',
```

---

### Change 3.2: Modify `run_quvine_walks()` to Support Walk Types
**Location:** Lines 225-272
**Current:** Uses ViewBuilder with multiple views
**New:** Add `walk_type` parameter

**Detailed Changes:**
```python
# Line 225: Add walk_type parameter
def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any], walk_type: str = 'rwr') -> np.ndarray:
    """Run QuVINE with specified quantum walk type."""
    
    # Line 251: Update CorpusBuilder to use specific walk type
    corpus_builder = CorpusBuilder(
        G=view,
        walk_length=walk_length,
        num_walks=num_walks,
        restart_prob=restart_prob,
        walk_type=walk_type,  # NEW: specify walk type
        seed=42
    )
```

**Note:** May need to check if CorpusBuilder supports walk_type parameter

---

### Change 3.3: Update `generate_embedding()` for 12 Methods
**Location:** Lines 321-380
**Current:** Handles 13 PPI-specific methods
**New:** Handle unified 12 methods

**Method Mapping:**
```
OLD METHOD              → NEW METHOD(S)              ACTION
──────────────────────────────────────────────────────────────
quvine_fused           → quvine_rwr/ctqw/dtqw      REMOVE (split)
quvine_ctqw            → quvine_ctqw                KEEP (rename from fused)
quvine_dtqw            → quvine_dtqw                KEEP (rename from fused)
quvine_rwr             → quvine_rwr                 KEEP (rename from fused)
quvine_heat            → (use baseline_filter_heat) REMOVE
quvine_poly            → (use baseline_filter_poly) REMOVE
quvine_hgcnmf          → (use baseline_gcnmf)       REMOVE
quvine_pgcnmf          → (use baseline_gcnmf)       REMOVE
netmf                  → netmf                       KEEP
node2vec               → node2vec                    KEEP
baseline_gcnmf         → baseline_gcnmf              KEEP
baseline_filter        → baseline_filter_heat/poly  SPLIT
graphsage              → graphsage                   KEEP
(none)                 → gat_baseline                ADD
(none)                 → graphgps_baseline           ADD
(none)                 → appnp                       ADD
```

**Detailed Changes:**
```python
def generate_embedding(method: str, G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Generate embedding for a method with given parameters."""
    try:
        # NEW: Quantum walk variants
        if method == "quvine_rwr":
            return run_quvine_walks(G, seeds, params, walk_type='rwr')
        elif method == "quvine_ctqw":
            return run_quvine_walks(G, seeds, params, walk_type='ctqw')
        elif method == "quvine_dtqw":
            return run_quvine_walks(G, seeds, params, walk_type='dtqw')
        
        # NEW: Split baseline_filter into heat and poly
        elif method == "baseline_filter_heat":
            return generate_baseline_filter_embedding_wrapper(
                G, filter_type="heat",
                t=params.get('tau', 2.0),
                K=params.get('filter_order', 5),
                embedding_dim=params.get('embedding_dim', 128),
            )
        elif method == "baseline_filter_poly":
            return generate_baseline_filter_embedding_wrapper(
                G, filter_type="poly",
                K=params.get('filter_order', 5),
                alpha=params.get('alpha', 0.5),
                embedding_dim=params.get('embedding_dim', 128),
            )
        
        # KEEP: GCN-MF baseline
        elif method == "baseline_gcnmf":
            # ... existing code ...
        
        # KEEP: Classical methods
        elif method == "node2vec":
            # ... existing code ...
        elif method == "netmf":
            # ... existing code ...
        elif method == "graphsage":
            # ... existing code ...
        
        # NEW: Add GNN baselines
        elif method == "gat_baseline":
            if not GAT_AVAILABLE:
                return None
            # ... implement GAT baseline ...
        elif method == "graphgps_baseline":
            if not GRAPHGPS_AVAILABLE:
                return None
            # ... implement GraphGPS baseline ...
        elif method == "appnp":
            # ... implement APPNP ...
        
        else:
            logger.warning(f"Unknown method: {method}")
            return None
```

**Lines to modify:** 321-380 (entire function)
**Methods removed:** 5 (quvine_fused, quvine_heat, quvine_poly, quvine_hgcnmf, quvine_pgcnmf)
**Methods added:** 3 (gat_baseline, graphgps_baseline, appnp)
**Net change:** 13 → 12 methods

---

## PHASE 4: Update `submit_tuning_jobs.sh`

### File: `QuVINE/scripts/submit_tuning_jobs.sh`

### Change 4.1: Update Header Documentation
**Location:** Lines 1-42
**Current:** References 10 methods, 20 jobs
**New:** Reference 12 methods, 24 jobs

**Changes:**
```bash
# Line 9: Update job count
# OLD: Total jobs: N_METHODS × N_NETWORK_TYPES (default: 10 × 2 = 20 jobs)
# NEW: Total jobs: N_METHODS × N_NETWORK_TYPES (default: 12 × 2 = 24 jobs)

# Line 16: Update example
# OLD: # Parallel mode (default) - 20 jobs (10 methods × 2 networks)
# NEW: # Parallel mode (default) - 24 jobs (12 methods × 2 networks)

# Line 23: Update example
# OLD: # Tune only on erdos_renyi network - 10 jobs
# NEW: # Tune only on erdos_renyi network - 12 jobs

# Line 26: Update example
# OLD: # Tune on specific networks - 30 jobs (10 methods × 3 networks)
# NEW: # Tune on specific networks - 36 jobs (12 methods × 3 networks)
```

---

### Change 4.2: Update Default Config Path
**Location:** Line 60
**Current:**
```bash
CONFIG_FILE="scripts/tuning_config.yaml"
```
**New:**
```bash
CONFIG_FILE="scripts/unified_tuning_config.yaml"
```

---

### Change 4.3: Update METHODS Array
**Location:** Lines 80-100 (approximate)
**Current:**
```bash
METHODS=(
    "quvine_walks"
    "baseline_filter_heat"
    "baseline_filter_poly"
    "baseline_gcnmf"
    "node2vec"
    "netmf"
    "graphsage"
    "appnp"
    "gat_baseline"
    "graphgps_baseline"
)
```
**New:**
```bash
METHODS=(
    "quvine_rwr"
    "quvine_ctqw"
    "quvine_dtqw"
    "baseline_filter_heat"
    "baseline_filter_poly"
    "baseline_gcnmf"
    "gat_baseline"
    "graphgps_baseline"
    "node2vec"
    "netmf"
    "graphsage"
    "appnp"
)
```

**Changes:**
- Split `quvine_walks` → `quvine_rwr`, `quvine_ctqw`, `quvine_dtqw`
- Reorder for clarity (quantum walks first, then filters, then GNNs, then classical)
- Total: 10 → 12 methods

---

## PHASE 5: Update `submit_ppi_tuning_jobs.sh`

### File: `QuVINE/scripts/submit_ppi_tuning_jobs.sh`

### Change 5.1: Update Header Documentation
**Location:** Lines 1-49
**Current:** References 13 methods, 195 jobs
**New:** Reference 12 methods, 180 jobs

**Changes:**
```bash
# Line 9: Update job count
# OLD: Total jobs: N_METHODS × N_NETWORKS × N_DISEASES (default: 13 × 5 × 3 = 195 jobs)
# NEW: Total jobs: N_METHODS × N_NETWORKS × N_DISEASES (default: 12 × 5 × 3 = 180 jobs)

# Line 20: Update example
# OLD: # Parallel mode (default) - 195 jobs (13 methods × 5 networks × 3 diseases)
# NEW: # Parallel mode (default) - 180 jobs (12 methods × 5 networks × 3 diseases)

# Line 26: Update example
# OLD: # Tune only on STRING network - 39 jobs (13 methods × 3 diseases)
# NEW: # Tune only on STRING network - 36 jobs (12 methods × 3 diseases)

# Line 29: Update example
# OLD: # Tune on specific network-disease pairs - 13 jobs
# NEW: # Tune on specific network-disease pairs - 12 jobs
```

---

### Change 5.2: Update Default Config Path
**Location:** Line 70 (approximate)
**Current:**
```bash
CONFIG_FILE="scripts/ppi_tuning_config.yaml"
```
**New:**
```bash
CONFIG_FILE="scripts/unified_tuning_config.yaml"
```

---

### Change 5.3: Update METHODS Array
**Location:** Lines 100-120 (approximate)
**Current:**
```bash
METHODS=(
    "quvine_fused"
    "quvine_ctqw"
    "quvine_dtqw"
    "quvine_rwr"
    "quvine_heat"
    "quvine_poly"
    "quvine_hgcnmf"
    "quvine_pgcnmf"
    "netmf"
    "node2vec"
    "baseline_gcnmf"
    "baseline_filter"
    "graphsage"
)
```
**New:**
```bash
METHODS=(
    "quvine_rwr"
    "quvine_ctqw"
    "quvine_dtqw"
    "baseline_filter_heat"
    "baseline_filter_poly"
    "baseline_gcnmf"
    "gat_baseline"
    "graphgps_baseline"
    "node2vec"
    "netmf"
    "graphsage"
    "appnp"
)
```

**Changes:**
- Remove: quvine_fused, quvine_heat, quvine_poly, quvine_hgcnmf, quvine_pgcnmf
- Split: baseline_filter → baseline_filter_heat, baseline_filter_poly
- Add: gat_baseline, graphgps_baseline, appnp
- Total: 13 → 12 methods

---

## PHASE 6: Verification & Documentation

### Step 6.1: Compile All Python Scripts
```bash
python -m py_compile scripts/tune_by_task_with_config.py
python -m py_compile scripts/tune_ppi_by_task.py
```

### Step 6.2: Validate YAML Configs
```bash
python -c "import yaml; yaml.safe_load(open('scripts/unified_tuning_config.yaml'))"
```

### Step 6.3: Check Method Name Consistency
Verify all 12 method names match across:
- unified_tuning_config.yaml
- tune_by_task_with_config.py
- tune_ppi_by_task.py
- submit_tuning_jobs.sh
- submit_ppi_tuning_jobs.sh

### Step 6.4: Create Migration Guide
Document for users on how to:
- Use new unified config
- Access legacy configs if needed
- Understand method name changes

---

## Summary of Changes

### Files Modified: 5
1. ✅ unified_tuning_config.yaml (created)
2. ⏳ tune_by_task_with_config.py (3 changes)
3. ⏳ tune_ppi_by_task.py (3 changes)
4. ⏳ submit_tuning_jobs.sh (3 changes)
5. ⏳ submit_ppi_tuning_jobs.sh (3 changes)

### Total Changes: 12 modifications
- Config paths: 4 changes
- Method arrays: 2 changes
- Function updates: 4 changes
- Documentation: 2 changes

### Method Count Changes:
- Synthetic: 10 → 12 methods (+2)
- PPI: 13 → 12 methods (-1)
- **Unified: 12 methods across all networks**

### Job Count Changes:
- Synthetic (2 networks): 20 → 24 jobs
- Synthetic (16 networks): 160 → 192 jobs
- PPI: 195 → 180 jobs

---

## Risk Assessment

### Low Risk Changes:
- Config file renames (already backed up)
- Documentation updates
- Method array updates in bash scripts

### Medium Risk Changes:
- Default config path changes (easy to revert)
- Method name changes in generate_embedding()

### High Risk Changes:
- run_quvine_walks() modification (affects core functionality)
- Need to verify CorpusBuilder/BaseWalker support walk_type parameter

---

## Testing Strategy

After each phase:
1. Compile Python scripts
2. Check for syntax errors
3. Verify imports work
4. Test with dry-run mode
5. Run single method test before full deployment

---

## Rollback Plan

If issues arise:
1. Restore legacy configs:
   ```bash
   mv scripts/ppi_tuning_config_legacy.yaml scripts/ppi_tuning_config.yaml
   mv scripts/tuning_config_legacy.yaml scripts/tuning_config.yaml
   ```
2. Revert code changes using git
3. Use legacy configs until issues resolved

---

## Next Steps

1. **Review this plan** - Confirm all changes make sense
2. **Approve Phase 2** - Start with tune_by_task_with_config.py
3. **Implement incrementally** - One phase at a time with verification
4. **Test thoroughly** - After each phase before proceeding

---

**Created:** 2026-04-28
**Status:** AWAITING REVIEW
**Estimated Implementation Time:** 2-3 hours with testing