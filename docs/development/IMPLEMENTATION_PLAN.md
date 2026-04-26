# QuVINE 39-Method Implementation Plan

## Overview

This document provides a detailed implementation plan for supporting all 39 embedding methods in QuVINE, including hyperparameter tuning and fusion logic.

**Total Methods**: 39 (16 quantum, 23 classical)
**Files to Modify**: 6 core files + 2 job submission scripts
**Estimated Effort**: 20-30 hours of development + testing

---

## Phase 1: Core Infrastructure Updates

### 1.1 Update `quantum_filters.py`

**File**: `QuVINE/src/quvine/embedding/quantum_filters.py`

**Current State**: Supports heat and poly filters with quantum walks

**Required Changes**:

#### Add baseline filter functions (no walk):
```python
def generate_baseline_heat_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    scale: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Generate heat kernel filter embedding without any walk.
    Pure spectral filtering approach.
    """
    # 1. Compute graph Laplacian
    # 2. Compute eigendecomposition
    # 3. Apply heat kernel: exp(-scale * eigenvalues)
    # 4. Project to embedding_dim dimensions
    # 5. Return node embeddings
    pass

def generate_baseline_poly_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    order: int = 4,
    **kwargs
) -> np.ndarray:
    """
    Generate polynomial filter embedding without any walk.
    Pure spectral filtering approach.
    """
    # 1. Compute graph Laplacian
    # 2. Compute eigendecomposition
    # 3. Apply polynomial filter: sum(coeff[k] * eigenvalues^k)
    # 4. Project to embedding_dim dimensions
    # 5. Return node embeddings
    pass
```

#### Add RWR + filter combinations:
```python
def generate_rwr_heat_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    restart_prob: float = 0.15,
    scale: float = 1.0,
    **kwargs
) -> np.ndarray:
    """
    Generate RWR walk + heat filter embedding.
    """
    # 1. Generate RWR transition matrix
    # 2. Apply heat kernel filter to RWR matrix
    # 3. Extract embeddings via SVD or other dimensionality reduction
    # 4. Return node embeddings
    pass

def generate_rwr_poly_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    restart_prob: float = 0.15,
    order: int = 4,
    **kwargs
) -> np.ndarray:
    """
    Generate RWR walk + polynomial filter embedding.
    """
    # Similar to rwr_heat but with polynomial filter
    pass
```

**Hyperparameters to tune**:
- `scale`: [0.1, 0.5, 1.0, 2.0, 5.0]
- `order`: [2, 4, 8, 16]
- `restart_prob`: [0.1, 0.15, 0.2, 0.3]

---

### 1.2 Update `gat.py`

**File**: `QuVINE/src/quvine/baselines/gat.py`

**Current State**: Basic GAT implementation

**Required Changes**:

#### Refactor to support 12 variants:

```python
def generate_gat_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    walk_type: str = None,  # None, 'rwr', 'ctqw', 'dtqw'
    filter_type: str = None,  # None, 'heat', 'poly'
    num_heads: int = 4,
    num_layers: int = 2,
    dropout: float = 0.1,
    # Walk parameters
    restart_prob: float = 0.15,
    walk_length: int = 40,
    num_walks: int = 10,
    time_steps: float = 1.0,
    # Filter parameters
    scale: float = 1.0,
    order: int = 4,
    **kwargs
) -> np.ndarray:
    """
    Unified GAT embedding generator supporting all 12 variants.
    
    Variants:
    1. gat_baseline: walk_type=None, filter_type=None
    2. gat_heat: walk_type=None, filter_type='heat'
    3. gat_poly: walk_type=None, filter_type='poly'
    4. gat_rwr: walk_type='rwr', filter_type=None
    5. gat_ctqw: walk_type='ctqw', filter_type=None
    6. gat_dtqw: walk_type='dtqw', filter_type=None
    7. gat_rwr_heat: walk_type='rwr', filter_type='heat'
    8. gat_rwr_poly: walk_type='rwr', filter_type='poly'
    9. gat_ctqw_heat: walk_type='ctqw', filter_type='heat'
    10. gat_ctqw_poly: walk_type='ctqw', filter_type='poly'
    11. gat_dtqw_heat: walk_type='dtqw', filter_type='heat'
    12. gat_dtqw_poly: walk_type='dtqw', filter_type='poly'
    """
    
    # Step 1: Prepare graph features
    if walk_type is None and filter_type is None:
        # Baseline: use node degrees or identity
        features = _prepare_baseline_features(G, embedding_dim)
    
    elif walk_type is not None and filter_type is None:
        # Walk only: generate walk-based features
        features = _generate_walk_features(
            G, walk_type, embedding_dim,
            restart_prob=restart_prob,
            walk_length=walk_length,
            num_walks=num_walks,
            time_steps=time_steps
        )
    
    elif walk_type is None and filter_type is not None:
        # Filter only: apply spectral filter
        features = _apply_spectral_filter(
            G, filter_type, embedding_dim,
            scale=scale, order=order
        )
    
    else:
        # Walk + Filter: combine both
        walk_features = _generate_walk_features(...)
        filtered_features = _apply_spectral_filter(...)
        features = _combine_features(walk_features, filtered_features)
    
    # Step 2: Apply GAT layers
    embeddings = _apply_gat_layers(
        G, features,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        embedding_dim=embedding_dim
    )
    
    return embeddings


def _prepare_baseline_features(G, dim):
    """Prepare baseline features (degree, identity, etc.)"""
    pass

def _generate_walk_features(G, walk_type, dim, **kwargs):
    """Generate features from walks"""
    if walk_type == 'rwr':
        # Use RWR transition matrix
        pass
    elif walk_type == 'ctqw':
        # Use CTQW evolution
        pass
    elif walk_type == 'dtqw':
        # Use DTQW evolution
        pass

def _apply_spectral_filter(G, filter_type, dim, **kwargs):
    """Apply spectral filter"""
    if filter_type == 'heat':
        # Apply heat kernel
        pass
    elif filter_type == 'poly':
        # Apply polynomial filter
        pass

def _combine_features(feat1, feat2):
    """Combine walk and filter features"""
    # Options: concatenate, average, learned combination
    pass

def _apply_gat_layers(G, features, **kwargs):
    """Apply GAT layers using PyTorch Geometric or DGL"""
    pass
```

**Hyperparameters to tune per variant**:
- GAT-specific: `num_heads` [2, 4, 8], `num_layers` [2, 3, 4], `dropout` [0.0, 0.1, 0.3]
- Walk-specific: `restart_prob`, `walk_length`, `num_walks`, `time_steps`
- Filter-specific: `scale`, `order`

---

### 1.3 Update `graphgps.py`

**File**: `QuVINE/src/quvine/baselines/graphgps.py`

**Current State**: Basic GraphGPS implementation

**Required Changes**:

#### Implement same structure as GAT but with GraphGPS architecture:

```python
def generate_graphgps_embedding(
    G: nx.Graph,
    embedding_dim: int = 128,
    walk_type: str = None,  # None, 'rwr', 'ctqw', 'dtqw'
    filter_type: str = None,  # None, 'heat', 'poly'
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,
    use_global_attention: bool = True,
    # Walk parameters
    restart_prob: float = 0.15,
    walk_length: int = 40,
    num_walks: int = 10,
    time_steps: float = 1.0,
    # Filter parameters
    scale: float = 1.0,
    order: int = 4,
    **kwargs
) -> np.ndarray:
    """
    Unified GraphGPS embedding generator supporting all 12 variants.
    
    Same variant structure as GAT but using Graph Transformer architecture.
    """
    
    # Step 1: Prepare features (same as GAT)
    features = _prepare_features_with_walk_and_filter(
        G, walk_type, filter_type, embedding_dim, **kwargs
    )
    
    # Step 2: Apply GraphGPS layers (transformer-based)
    embeddings = _apply_graphgps_layers(
        G, features,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        use_global_attention=use_global_attention,
        embedding_dim=embedding_dim
    )
    
    return embeddings

def _apply_graphgps_layers(G, features, **kwargs):
    """
    Apply GraphGPS layers:
    - Local message passing (MPNN)
    - Global attention (Transformer)
    - Combination of both
    """
    pass
```

**Hyperparameters to tune per variant**:
- GraphGPS-specific: `num_layers` [2, 4, 6], `num_heads` [4, 8], `dropout` [0.0, 0.1, 0.3]
- Walk-specific: same as GAT
- Filter-specific: same as GAT

---

## Phase 2: Method Dispatch Logic

### 2.1 Update `comprehensive_embedding_analysis.py`

**File**: `QuVINE/src/quvine/comprehensive_embedding_analysis.py`

**Required Changes**:

#### Update `_generate_single_embedding` method:

```python
def _generate_single_embedding(
    self,
    G: nx.Graph,
    method: str,
    embedding_dim: int = 128,
    method_hyperparams: Dict = None,
    **kwargs
) -> np.ndarray:
    """
    Generate embedding for a single method.
    Now supports all 39 methods.
    """
    
    if method_hyperparams is None:
        method_hyperparams = {}
    
    # Extract method-specific hyperparameters
    hp = method_hyperparams.get(method, {})
    
    # ========== SGNS Methods (3) ==========
    if method == 'quvine_rwr':
        return self._generate_quvine_rwr_embedding(G, embedding_dim, **hp)
    elif method == 'quvine_ctqw':
        return self._generate_quvine_ctqw_embedding(G, embedding_dim, **hp)
    elif method == 'quvine_dtqw':
        return self._generate_quvine_dtqw_embedding(G, embedding_dim, **hp)
    
    # ========== Filter Methods (6) ==========
    elif method == 'quvine_baseline_heat':
        from quvine.embedding.quantum_filters import generate_baseline_heat_embedding
        return generate_baseline_heat_embedding(G, embedding_dim, **hp)
    
    elif method == 'quvine_baseline_poly':
        from quvine.embedding.quantum_filters import generate_baseline_poly_embedding
        return generate_baseline_poly_embedding(G, embedding_dim, **hp)
    
    elif method == 'quvine_rwr_heat':
        from quvine.embedding.quantum_filters import generate_rwr_heat_embedding
        return generate_rwr_heat_embedding(G, embedding_dim, **hp)
    
    elif method == 'quvine_rwr_poly':
        from quvine.embedding.quantum_filters import generate_rwr_poly_embedding
        return generate_rwr_poly_embedding(G, embedding_dim, **hp)
    
    elif method == 'quvine_ctqw_heat':
        from quvine.embedding.quantum_filters import generate_quvine_heat_embedding
        return generate_quvine_heat_embedding(G, embedding_dim, walk_type='ctqw', **hp)
    
    elif method == 'quvine_ctqw_poly':
        from quvine.embedding.quantum_filters import generate_quvine_poly_embedding
        return generate_quvine_poly_embedding(G, embedding_dim, walk_type='ctqw', **hp)
    
    # ========== GAT Methods (12) ==========
    elif method.startswith('gat_'):
        from quvine.baselines.gat import generate_gat_embedding
        
        # Parse method name to extract walk_type and filter_type
        walk_type, filter_type = _parse_method_name(method, 'gat_')
        
        return generate_gat_embedding(
            G, embedding_dim,
            walk_type=walk_type,
            filter_type=filter_type,
            **hp
        )
    
    # ========== GraphGPS Methods (12) ==========
    elif method.startswith('graphgps_'):
        from quvine.baselines.graphgps import generate_graphgps_embedding
        
        # Parse method name to extract walk_type and filter_type
        walk_type, filter_type = _parse_method_name(method, 'graphgps_')
        
        return generate_graphgps_embedding(
            G, embedding_dim,
            walk_type=walk_type,
            filter_type=filter_type,
            **hp
        )
    
    # ========== Classical Baselines (6) ==========
    elif method == 'node2vec':
        return run_node2vec(G, embedding_dim, **hp)
    elif method == 'netmf':
        return run_netmf(G, embedding_dim, **hp)
    elif method == 'graphsage':
        from quvine.baselines import run_graphsage
        return run_graphsage(G, embedding_dim, **hp)
    elif method == 'appnp':
        return run_appnp(G, embedding_dim, **hp)
    elif method == 'baseline_filter':
        from quvine.embedding.quantum_filters import generate_baseline_filter_embedding
        return generate_baseline_filter_embedding(G, embedding_dim, **hp)
    elif method == 'baseline_gcnmf':
        from quvine.baselines.gcn_mf import generate_baseline_gcnmf_embedding
        return generate_baseline_gcnmf_embedding(G, embedding_dim, **hp)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _parse_method_name(method: str, prefix: str) -> Tuple[str, str]:
    """
    Parse method name to extract walk_type and filter_type.
    
    Examples:
        gat_baseline -> (None, None)
        gat_heat -> (None, 'heat')
        gat_rwr -> ('rwr', None)
        gat_ctqw_heat -> ('ctqw', 'heat')
        gat_dtqw_poly -> ('dtqw', 'poly')
    """
    suffix = method[len(prefix):]
    
    if suffix == 'baseline':
        return None, None
    
    # Check for filter-only
    if suffix in ['heat', 'poly']:
        return None, suffix
    
    # Check for walk-only
    if suffix in ['rwr', 'ctqw', 'dtqw']:
        return suffix, None
    
    # Check for walk + filter
    parts = suffix.split('_')
    if len(parts) == 2:
        walk_type, filter_type = parts
        return walk_type, filter_type
    
    raise ValueError(f"Cannot parse method name: {method}")
```

---

## Phase 3: Hyperparameter Tuning

### 3.1 Create Hyperparameter Tuning Script

**File**: `QuVINE/scripts/run_hyperparameter_tuning.py`

**Purpose**: Tune hyperparameters for each of the 39 methods on a given network

**Structure**:

```python
#!/usr/bin/env python3
"""
Hyperparameter tuning for QuVINE methods.

Usage:
    python run_hyperparameter_tuning.py \
        --network-type ppi \
        --ppi-network BioPlex3 \
        --disease asthma \
        --task ranking \
        --output-file hparam_tuning/BioPlex3_asthma_ranking.json \
        --methods all \
        --n-trials 100
"""

import argparse
import json
import optuna
from optuna.samplers import TPESampler

def tune_method_hyperparameters(
    G, method, task, n_trials=100, seed=42
):
    """
    Tune hyperparameters for a single method.
    
    Returns:
        best_params: dict of best hyperparameters
        best_score: best performance score
    """
    
    def objective(trial):
        # Define hyperparameter search space based on method
        if method.startswith('gat_') or method.startswith('graphgps_'):
            # GNN hyperparameters
            num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
            num_layers = trial.suggest_int('num_layers', 2, 4)
            dropout = trial.suggest_float('dropout', 0.0, 0.5)
            
            # Walk hyperparameters (if applicable)
            if 'rwr' in method or 'ctqw' in method or 'dtqw' in method:
                restart_prob = trial.suggest_float('restart_prob', 0.1, 0.3)
                walk_length = trial.suggest_categorical('walk_length', [20, 40, 80])
                num_walks = trial.suggest_categorical('num_walks', [10, 20, 40])
                
                if 'ctqw' in method:
                    time_steps = trial.suggest_float('time_steps', 0.1, 5.0)
            
            # Filter hyperparameters (if applicable)
            if 'heat' in method:
                scale = trial.suggest_float('scale', 0.1, 5.0)
            elif 'poly' in method:
                order = trial.suggest_categorical('order', [2, 4, 8, 16])
        
        elif method.startswith('quvine_'):
            # SGNS or filter hyperparameters
            if 'rwr' in method:
                restart_prob = trial.suggest_float('restart_prob', 0.1, 0.3)
            if 'ctqw' in method:
                time_steps = trial.suggest_float('time_steps', 0.1, 5.0)
            if 'heat' in method:
                scale = trial.suggest_float('scale', 0.1, 5.0)
            if 'poly' in method:
                order = trial.suggest_categorical('order', [2, 4, 8, 16])
        
        # Generate embedding with trial hyperparameters
        embedding = generate_embedding(G, method, trial.params)
        
        # Evaluate on task
        score = evaluate_embedding(G, embedding, task)
        
        return score
    
    # Run optimization
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler
    )
    study.optimize(objective, n_trials=n_trials)
    
    return study.best_params, study.best_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network-type', required=True)
    parser.add_argument('--ppi-network', default=None)
    parser.add_argument('--disease', default=None)
    parser.add_argument('--task', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--methods', default='all')
    parser.add_argument('--n-trials', type=int, default=100)
    args = parser.parse_args()
    
    # Load network
    G = load_network(args.network_type, args.ppi_network, args.disease)
    
    # Get methods to tune
    if args.methods == 'all':
        methods = ALL_39_METHODS
    else:
        methods = args.methods.split(',')
    
    # Tune each method
    results = {}
    for method in methods:
        print(f"Tuning {method}...")
        best_params, best_score = tune_method_hyperparameters(
            G, method, args.task, args.n_trials
        )
        results[method] = {
            'best_params': best_params,
            'best_score': best_score
        }
    
    # Save results
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved hyperparameters to {args.output_file}")


if __name__ == '__main__':
    main()
```

---

## Phase 4: Fusion Logic

### 4.1 Update `fusion/fuse.py`

**File**: `QuVINE/src/quvine/fusion/fuse.py`

**Required Changes**:

#### Add method-type-aware fusion:

```python
def fuse_by_method_type(
    embeddings_dict: Dict[str, np.ndarray],
    method_type: str,  # 'sgns', 'filter', 'gat', 'graphgps'
    quantum_only: bool = False,
    classical_only: bool = False,
    fusion_method: str = 'svd'
) -> np.ndarray:
    """
    Fuse embeddings within a method type.
    
    Args:
        embeddings_dict: {method_name: embedding_array}
        method_type: Type of methods to fuse
        quantum_only: Only fuse quantum methods
        classical_only: Only fuse classical methods
        fusion_method: 'svd', 'concatenate', 'average', 'weighted'
    
    Returns:
        Fused embedding array
    """
    
    # Filter methods by type and quantum/classical
    filtered_methods = _filter_methods_by_type(
        embeddings_dict.keys(),
        method_type,
        quantum_only,
        classical_only
    )
    
    # Extract embeddings to fuse
    embeddings_to_fuse = [
        embeddings_dict[m] for m in filtered_methods
        if m in embeddings_dict
    ]
    
    if len(embeddings_to_fuse) == 0:
        raise ValueError(f"No embeddings found for type={method_type}, "
                        f"quantum={quantum_only}, classical={classical_only}")
    
    # Perform fusion
    if fusion_method == 'svd':
        return _fuse_via_svd(embeddings_to_fuse)
    elif fusion_method == 'concatenate':
        return np.concatenate(embeddings_to_fuse, axis=1)
    elif fusion_method == 'average':
        return np.mean(embeddings_to_fuse, axis=0)
    elif fusion_method == 'weighted':
        # Weight by performance (requires performance scores)
        return _fuse_weighted(embeddings_to_fuse)
    else:
        raise ValueError(f"Unknown fusion method: {fusion_method}")


def _fuse_via_svd(embeddings_list: List[np.ndarray]) -> np.ndarray:
    """
    Fuse embeddings using SVD.
    
    Steps:
    1. Stack embeddings horizontally
    2. Perform SVD
    3. Keep top-k singular vectors
    4. Return as fused embedding
    """
    # Stack embeddings
    stacked = np.hstack(embeddings_list)
    
    # SVD
    U, S, Vt = np.linalg.svd(stacked, full_matrices=False)
    
    # Keep dimensions equal to first embedding
    target_dim = embeddings_list[0].shape[1]
    fused = U[:, :target_dim] @ np.diag(S[:target_dim])
    
    return fused


def _filter_methods_by_type(
    method_names: List[str],
    method_type: str,
    quantum_only: bool,
    classical_only: bool
) -> List[str]:
    """Filter methods by type and quantum/classical."""
    
    # Define method type patterns
    type_patterns = {
        'sgns': ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw'],
        'filter': [
            'quvine_baseline_heat', 'quvine_baseline_poly',
            'quvine_rwr_heat', 'quvine_rwr_poly',
            'quvine_ctqw_heat', 'quvine_ctqw_poly'
        ],
        'gat': [m for m in ALL_39_METHODS if m.startswith('gat_')],
        'graphgps': [m for m in ALL_39_METHODS if m.startswith('graphgps_')]
    }
    
    # Define quantum methods
    quantum_methods = {
        'quvine_ctqw', 'quvine_dtqw',
        'quvine_ctqw_heat', 'quvine_ctqw_poly',
        'gat_ctqw', 'gat_dtqw',
        'gat_ctqw_heat', 'gat_ctqw_poly',
        'gat_dtqw_heat', 'gat_dtqw_poly',
        'graphgps_ctqw', 'graphgps_dtqw',
        'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
        'graphgps_dtqw_heat', 'graphgps_dtqw_poly'
    }
    
    # Filter by type
    candidates = [m for m in method_names if m in type_patterns[method_type]]
    
    # Filter by quantum/classical
    if quantum_only:
        candidates = [m for m in candidates if m in quantum_methods]
    elif classical_only:
        candidates = [m for m in candidates if m not in quantum_methods]
    
    return candidates
```

#### Add cross-type fusion:

```python
def fuse_best_across_types(
    embeddings_dict: Dict[str, np.ndarray],
    performance_scores: Dict[str, float],
    quantum_only: bool = False,
    classical_only: bool = False
) -> np.ndarray:
    """
    Fuse best-performing method from each type.
    
    Steps:
    1. For each method type (SGNS, Filter, GAT, GraphGPS):
       - Select best-performing method based on scores
    2. Fuse the 4 best methods using SVD
    
    Args:
        embeddings_dict: {method_name: embedding}
        performance_scores: {method_name: score} (mean across replicates)
        quantum_only: Only consider quantum methods
        classical_only: Only consider classical methods
    
    Returns:
        Fused embedding from best methods across types
    """
    
    method_types = ['sgns', 'filter', 'gat', 'graphgps']
    best_methods = []
    
    for mtype in method_types:
        # Get methods of this type
        type_methods = _filter_methods_by_type(
            embeddings_dict.keys(),
            mtype,
            quantum_only,
            classical_only
        )
        
        # Find best-performing method
        if len(type_methods) > 0:
            best_method = max(
                type_methods,
                key=lambda m: performance_scores.get(m, 0.0)
            )
            best_methods.append(best_method)
    
    # Fuse best methods
    best_embeddings = [embeddings_dict[m] for m in best_methods]
    return _fuse_via_svd(best_embeddings)
```

---

## Phase 5: Pipeline Integration

### 5.1 Update `pipeline.py`

**File**: `QuVINE/src/quvine/pipeline.py`

**Required Changes**:

#### Add fusion pipeline:

```python
class FusionPipeline:
    """
    Pipeline for generating and fusing embeddings across method types.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        fusion_method: str = 'svd'
    ):
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
    
    def generate_all_embeddings(
        self,
        G: nx.Graph,
        methods: List[str],
        method_hyperparams: Dict = None
    ) -> Dict[str, np.ndarray]:
        """Generate embeddings for all methods."""
        
        embeddings = {}
        for method in methods:
            try:
                emb = self._generate_single_embedding(
                    G, method, self.embedding_dim, method_hyperparams
                )
                embeddings[method] = emb
            except Exception as e:
                print(f"Error generating {method}: {e}")
        
        return embeddings
    
    def fuse_within_types(
        self,
        embeddings_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Fuse embeddings within each method type.
        
        Returns:
            {
                'fused_quantum_sgns': ...,
                'fused_classical_sgns': ...,
                'fused_quantum_filter': ...,
                'fused_classical_filter': ...,
                'fused_quantum_gat': ...,
                'fused_classical_gat': ...,
                'fused_quantum_graphgps': ...,
                'fused_classical_graphgps': ...
            }
        """
        
        fused = {}
        
        for mtype in ['sgns', 'filter', 'gat', 'graphgps']:
            # Quantum fusion
            try:
                fused[f'fused_quantum_{mtype}'] = fuse_by_method_type(
                    embeddings_dict, mtype,
                    quantum_only=True,
                    fusion_method=self.fusion_method
                )
            except:
                pass
            
            # Classical fusion
            try:
                fused[f'fused_classical_{mtype}'] = fuse_by_method_type(
                    embeddings_dict, mtype,
                    classical_only=True,
                    fusion_method=self.fusion_method
                )
            except:
                pass
        
        return fused
    
    def fuse_across_types(
        self,
        embeddings_dict: Dict[str, np.ndarray],
        performance_scores: Dict[str, float]
    ) -> Dict[str, np.ndarray]:
        """
        Fuse best methods across types.
        
        Returns:
            {
                'fused_q': quantum fusion across types,
                'fused_c': classical fusion across types
            }
        """
        
        return {
            'fused_q': fuse_best_across_types(
                embeddings_dict, performance_scores,
                quantum_only=True
            ),
            'fused_c': fuse_best_across_types(
                embeddings_dict, performance_scores,
                classical_only=True
            )
        }
```

---

## Phase 6: Testing & Validation

### 6.1 Unit Tests

Create tests for each component:

```python
# tests/test_filters.py
def test_baseline_heat_embedding():
    G = nx.karate_club_graph()
    emb = generate_baseline_heat_embedding(G, embedding_dim=64)
    assert emb.shape == (G.number_of_nodes(), 64)

# tests/test_gat_variants.py
def test_gat_all_variants():
    G = nx.karate_club_graph()
    variants = [
        ('gat_baseline', None, None),
        ('gat_heat', None, 'heat'),
        ('gat_rwr', 'rwr', None),
        ('gat_ctqw_heat', 'ctqw', 'heat'),
        # ... all 12 variants
    ]
    for name, walk, filt in variants:
        emb = generate_gat_embedding(
            G, 64, walk_type=walk, filter_type=filt
        )
        assert emb.shape == (G.number_of_nodes(), 64)

# tests/test_fusion.py
def test_fusion_by_type():
    # Create dummy embeddings
    embeddings = {
        'quvine_ctqw': np.random.rand(34, 64),
        'quvine_dtqw': np.random.rand(34, 64),
    }
    fused = fuse_by_method_type(
        embeddings, 'sgns', quantum_only=True
    )
    assert fused.shape == (34, 64)
```

### 6.2 Integration Tests

Test full pipeline:

```python
def test_full_pipeline():
    G = nx.karate_club_graph()
    
    # Generate all embeddings
    pipeline = FusionPipeline(embedding_dim=64)
    embeddings = pipeline.generate_all_embeddings(G, ALL_39_METHODS)
    
    # Fuse within types
    fused_within = pipeline.fuse_within_types(embeddings)
    
    # Evaluate to get scores
    scores = evaluate_all_methods(G, embeddings)
    
    # Fuse across types
    fused_across = pipeline.fuse_across_types(embeddings, scores)
    
    # Verify outputs
    assert 'fused_q' in fused_across
    assert 'fused_c' in fused_across
```

---

## Implementation Checklist

### Core Files
- [ ] `quantum_filters.py` - Add 4 new filter functions
- [ ] `gat.py` - Refactor to support 12 variants
- [ ] `graphgps.py` - Refactor to support 12 variants
- [ ] `comprehensive_embedding_analysis.py` - Update method dispatch
- [ ] `fusion/fuse.py` - Add type-aware fusion
- [ ] `pipeline.py` - Add fusion pipeline

### Scripts
- [ ] `run_hyperparameter_tuning.py` - Create new script
- [ ] `submit_ppi_comprehensive_with_tuning.sh` - Already updated ✓
- [ ] `submit_simulated_data_jobs_with_tuning.sh` - Already updated ✓

### Documentation
- [ ] `METHOD_REGISTRY.md` - Already created ✓
- [ ] `IMPLEMENTATION_PLAN.md` - This document ✓
- [ ] Update main README with new methods

### Testing
- [ ] Unit tests for filters
- [ ] Unit tests for GAT variants
- [ ] Unit tests for GraphGPS variants
- [ ] Unit tests for fusion
- [ ] Integration tests for full pipeline
- [ ] End-to-end test on small network

---

## Estimated Timeline

| Phase | Task | Effort | Dependencies |
|-------|------|--------|--------------|
| 1.1 | Update quantum_filters.py | 4 hours | None |
| 1.2 | Update gat.py | 6 hours | 1.1 |
| 1.3 | Update graphgps.py | 6 hours | 1.1 |
| 2.1 | Update comprehensive_embedding_analysis.py | 4 hours | 1.1-1.3 |
| 3.1 | Create hyperparameter tuning script | 4 hours | 2.1 |
| 4.1 | Update fusion/fuse.py | 3 hours | None |
| 5.1 | Update pipeline.py | 3 hours | 4.1 |
| 6.1 | Unit tests | 4 hours | All |
| 6.2 | Integration tests | 2 hours | All |

**Total**: ~36 hours of development + testing

---

## Notes

1. **Quantum Calibration**: For GAT and GraphGPS, "quantum calibration" means using quantum walk features (CTQW/DTQW) as input to the GNN layers.

2. **Filter Application**: Filters can be applied:
   - Alone (baseline_heat, baseline_poly)
   - After walks (rwr_heat, ctqw_heat, etc.)
   - As preprocessing for GNNs

3. **Hyperparameter Spaces**: Each method type has different hyperparameters to tune. The tuning script should handle this automatically based on method name parsing.

4. **Performance Evaluation**: For fusion across types, we need to run all methods first, evaluate them, then select best performers. This requires a two-pass approach.

5. **Memory Considerations**: Generating 39 embeddings per network can be memory-intensive. Consider:
   - Generating embeddings on-demand
   - Saving embeddings to disk
   - Using memory-mapped arrays

6. **Parallelization**: The hyperparameter tuning can be parallelized across methods since each method is independent.

---

## Contact

For questions about this implementation plan, refer to:
- `METHOD_REGISTRY.md` for method definitions
- `README.md` for general project documentation
- Individual file docstrings for API details