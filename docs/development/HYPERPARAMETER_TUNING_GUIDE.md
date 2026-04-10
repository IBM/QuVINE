# QuVINE Hyperparameter Tuning Guide

## Overview

This guide explains how to use Bayesian optimization for tuning hyperparameters across all QuVINE embedding methods. The tuning system uses **Optuna** with the TPE (Tree-structured Parzen Estimator) sampler for efficient hyperparameter search.

## Supported Methods

The following methods support hyperparameter tuning:

1. **Baseline Methods**
   - GCN-MF (baseline_gcnmf)
   - Node2Vec
   - NetMF

2. **Q-Caliber Methods**
   - Heat GCN-MF (hgcnmf)
   - Polynomial GCN-MF (pgcnmf)

3. **Quantum Walk Methods**
   - Random Walk with Restart (RWR)
   - Continuous-Time Quantum Walk (CTQW)
   - Discrete-Time Quantum Walk (DTQW)

## Installation

```bash
# Install optuna for hyperparameter tuning
pip install optuna>=4.0.0

# Or install all requirements
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
import networkx as nx

# Create analyzer
analyzer = ComprehensiveEmbeddingAnalysis(
    output_dir='results',
    embedding_dim=128
)

# Load your graph
G = nx.karate_club_graph()
seeds = [0, 1, 2, 3, 4]
targets = list(range(5, 15))

# Tune hyperparameters for different methods

# 1. Baseline GCN-MF
result = analyzer.tune_gcnmf_hyperparameters(
    G=G, seeds=seeds, targets=targets,
    n_trials=50, timeout=3600
)
analyzer.tuned_hyperparameters['baseline_gcnmf'] = result['best_params']

# 2. Node2Vec
result = analyzer.tune_node2vec_hyperparameters(
    G=G, seeds=seeds, targets=targets,
    n_trials=50, timeout=1800
)
analyzer.tuned_hyperparameters['node2vec'] = result['best_params']

# 3. NetMF
result = analyzer.tune_netmf_hyperparameters(
    G=G, seeds=seeds, targets=targets,
    n_trials=30, timeout=1200
)
analyzer.tuned_hyperparameters['netmf'] = result['best_params']

# 4. Q-Caliber Heat GCN-MF
result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G=G, seeds=seeds, targets=targets,
    diffusion_type='heat', n_trials=50, timeout=3600
)
analyzer.tuned_hyperparameters['hgcnmf'] = result['best_params']

# 5. Q-Caliber Poly GCN-MF
result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G=G, seeds=seeds, targets=targets,
    diffusion_type='poly', n_trials=50, timeout=3600
)
analyzer.tuned_hyperparameters['pgcnmf'] = result['best_params']

# 6. Quantum Walks (RWR, CTQW, DTQW)
for walk_type in ['rwr', 'ctqw', 'dtqw']:
    result = analyzer.tune_quantum_walk_hyperparameters(
        G=G, seeds=seeds, targets=targets,
        walk_type=walk_type, n_trials=30, timeout=1200
    )
    analyzer.tuned_hyperparameters[walk_type] = result['best_params']

# View results
print(f"Best validation score: {result['best_value']:.4f}")
print(f"Best parameters: {result['best_params']}")

# Generate embeddings with tuned parameters
embedding = analyzer.run_embedding_method('quvine_hgcnmf', G, seeds, targets)
```

## Hyperparameters Tuned

### 1. Baseline GCN-MF

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_layers` | int | [1, 3] | Number of GCN layers |
| `hidden_dim` | categorical | [64, 128, 256] | Hidden layer dimension |
| `mf_dim` | categorical | [32, 64, 128] | Matrix factorization dimension |
| `epochs` | int | [100, 500] (step=100) | Training epochs |
| `lr` | float | [1e-3, 1e-1] (log scale) | Learning rate |
| `weight_decay` | float | [1e-5, 1e-3] (log scale) | L2 regularization |

### 2. Node2Vec

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `walk_length` | int | [10, 80] | Length of each random walk |
| `num_walks` | int | [10, 100] | Number of walks per node |
| `p` | float | [0.25, 4.0] | Return parameter |
| `q` | float | [0.25, 4.0] | In-out parameter |
| `window_size` | int | [5, 10] | Context window size |
| `epochs` | int | [5, 20] | Training epochs |

### 3. NetMF

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `window_size` | int | [5, 10] | Context window size |
| `rank` | categorical | [64, 128, 256] | Matrix rank |
| `negative` | int | [1, 5] | Negative sampling rate |

### 4. Q-Caliber Heat GCN-MF (hgcnmf)

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `n_layers` | int | [1, 3] | Number of GCN layers |
| `hidden_dim` | categorical | [64, 128, 256] | Hidden layer dimension |
| `mf_dim` | categorical | [32, 64, 128] | Matrix factorization dimension |
| `epochs` | int | [100, 500] (step=100) | Training epochs |
| `lr` | float | [1e-3, 1e-1] (log scale) | Learning rate |
| `weight_decay` | float | [1e-5, 1e-3] (log scale) | L2 regularization |

### 5. Q-Caliber Polynomial GCN-MF (pgcnmf)

All parameters from Heat GCN-MF, plus:

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `K` | int | [2, 6] | Polynomial degree |
| `ridge` | float | [1e-7, 1e-5] (log scale) | Ridge regularization |

### 6. Quantum Walk Methods (RWR, CTQW, DTQW)

**Common parameters for all quantum walks:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `num_walks` | int | [5, 20] | Number of walks per node |
| `walk_length` | int | [5, 15] | Length of each walk |
| `num_views` | int | [2, 5] | Number of graph views |

**RWR-specific parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `restart_prob` | float | [0.1, 0.3] | Restart probability |
| `max_iter` | int | [500, 1500] (step=250) | Maximum iterations |

**CTQW-specific parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `time` | float | [0.5, 2.0] | Evolution time |

**DTQW-specific parameters:**

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| `steps` | int | [5, 25] | Number of walk steps |
| `coin` | categorical | ['grover', 'hadamard'] | Coin operator type |

## Tuning Strategy

### 1. Train/Validation Split

The tuning function automatically creates an 80/20 train/validation split:
- **80% of seeds** used for training
- **20% of seeds** used for validation (hyperparameter selection)

### 2. Optimization Objective

Maximizes **validation recall@50**:
- Measures how many validation seeds are recovered in top-50 ranked nodes
- Uses centroid-based ranking (average of seed embeddings)

### 3. Bayesian Optimization

Uses Optuna's TPE sampler:
- Builds probabilistic model of objective function
- Balances exploration vs exploitation
- More efficient than grid search or random search

## Advanced Usage

### Parallel Tuning

```python
# Use multiple cores for faster tuning
result = analyzer.tune_gcnmf_hyperparameters(
    G=G,
    seeds=seeds,
    targets=targets,
    diffusion_type='heat',
    n_trials=100,
    n_jobs_optuna=-1  # Use all available cores
)
```

### Custom Timeout

```python
# Set maximum time for tuning
result = analyzer.tune_gcnmf_hyperparameters(
    G=G,
    seeds=seeds,
    targets=targets,
    diffusion_type='poly',
    n_trials=200,
    timeout=7200  # 2 hours maximum
)
```

### Analyzing Tuning Results

```python
# Get trials dataframe
trials_df = result['trials_df']

# Save to CSV
trials_df.to_csv('tuning_results.csv', index=False)

# Plot optimization history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(trials_df['number'], trials_df['value'])
plt.xlabel('Trial')
plt.ylabel('Validation Recall@50')
plt.title('Hyperparameter Tuning Progress')
plt.savefig('tuning_progress.png')

# Get best trial info
best_trial = trials_df.loc[trials_df['value'].idxmax()]
print(f"Best trial #{best_trial['number']}")
print(f"Best score: {best_trial['value']:.4f}")
```

### Using Optuna Study Object

```python
# Access full Optuna study
study = result['study']

# Plot parameter importances
from optuna.visualization import plot_param_importances
fig = plot_param_importances(study)
fig.show()

# Plot optimization history
from optuna.visualization import plot_optimization_history
fig = plot_optimization_history(study)
fig.show()

# Plot parallel coordinate plot
from optuna.visualization import plot_parallel_coordinate
fig = plot_parallel_coordinate(study)
fig.show()
```

## Integration with Comprehensive Analysis

### Option 1: Tune All Methods Before Analysis

```python
# 1. Tune hyperparameters on a representative graph
analyzer = ComprehensiveEmbeddingAnalysis()
G_representative = nx.karate_club_graph()
seeds, targets = analyzer._select_seeds_targets(G_representative)

# Tune baseline methods
gcnmf_result = analyzer.tune_gcnmf_hyperparameters(
    G_representative, seeds, targets, n_trials=50
)
node2vec_result = analyzer.tune_node2vec_hyperparameters(
    G_representative, seeds, targets, n_trials=50
)
netmf_result = analyzer.tune_netmf_hyperparameters(
    G_representative, seeds, targets, n_trials=30
)

# Tune Q-Caliber methods
heat_result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G_representative, seeds, targets, 'heat', n_trials=50
)
poly_result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G_representative, seeds, targets, 'poly', n_trials=50
)

# Tune quantum walks
rwr_result = analyzer.tune_quantum_walk_hyperparameters(
    G_representative, seeds, targets, 'rwr', n_trials=30
)
ctqw_result = analyzer.tune_quantum_walk_hyperparameters(
    G_representative, seeds, targets, 'ctqw', n_trials=30
)
dtqw_result = analyzer.tune_quantum_walk_hyperparameters(
    G_representative, seeds, targets, 'dtqw', n_trials=30
)

# Store all tuned parameters
analyzer.tuned_hyperparameters.update({
    'baseline_gcnmf': gcnmf_result['best_params'],
    'node2vec': node2vec_result['best_params'],
    'netmf': netmf_result['best_params'],
    'hgcnmf': heat_result['best_params'],
    'pgcnmf': poly_result['best_params'],
    'rwr': rwr_result['best_params'],
    'ctqw': ctqw_result['best_params'],
    'dtqw': dtqw_result['best_params']
})

# 2. Run comprehensive analysis with tuned parameters
analyzer.run_comprehensive_analysis(n_networks=50)
```

### Option 2: Tune Per Network (Slower but More Accurate)

```python
# Create custom analysis loop with per-network tuning
networks = analyzer.generate_networks()

for network_id, G, seeds, targets in networks:
    # Tune for this specific network
    heat_result = analyzer.tune_gcnmf_hyperparameters(
        G, seeds, targets, 'heat', n_trials=20
    )
    analyzer.tuned_hyperparameters['hgcnmf'] = heat_result['best_params']
    
    # Generate embedding with tuned params
    embedding = analyzer.run_embedding_method('quvine_hgcnmf', G, seeds, targets)
    
    # Evaluate...
```

## Performance Considerations

### Computational Cost

- **Per trial**: ~5-30 seconds (depends on graph size and epochs)
- **50 trials**: ~5-25 minutes
- **100 trials**: ~10-50 minutes

### Recommendations

| Graph Size | Recommended Trials | Expected Time |
|------------|-------------------|---------------|
| Small (<100 nodes) | 30-50 | 5-15 min |
| Medium (100-500 nodes) | 50-100 | 15-45 min |
| Large (>500 nodes) | 100-200 | 45-120 min |

### Speed vs Quality Trade-off

```python
# Quick tuning (less accurate)
result = analyzer.tune_gcnmf_hyperparameters(
    G, seeds, targets, 'heat',
    n_trials=20,
    timeout=600  # 10 minutes
)

# Thorough tuning (more accurate)
result = analyzer.tune_gcnmf_hyperparameters(
    G, seeds, targets, 'heat',
    n_trials=200,
    timeout=7200  # 2 hours
)
```

## Best Practices

### 1. Use Representative Graphs

Tune on graphs similar to your target dataset:
```python
# If analyzing scale-free networks, tune on scale-free graph
from quvine.data.random_graphs import generate_barabasi_albert
G_tune = generate_barabasi_albert(n=200, m=3)
```

### 2. Save Tuning Results

```python
import json

# Save best parameters
with open('best_params_heat.json', 'w') as f:
    json.dump(result['best_params'], f, indent=2)

# Load later
with open('best_params_heat.json', 'r') as f:
    analyzer.tuned_hyperparameters['hgcnmf'] = json.load(f)
```

### 3. Monitor Progress

```python
# Optuna shows progress bar by default
# For custom monitoring, use callbacks
def callback(study, trial):
    if trial.number % 10 == 0:
        print(f"Trial {trial.number}: best value = {study.best_value:.4f}")

study.optimize(objective, n_trials=100, callbacks=[callback])
```

### 4. Handle Failed Trials

The tuning function automatically handles failures:
- Failed trials return score of 0.0
- Optimization continues with remaining trials
- Check logs for failure reasons

## Troubleshooting

### Issue: Optuna Not Installed

```
ImportError: Optuna is required for hyperparameter tuning
```

**Solution**: Install optuna
```bash
pip install optuna>=4.0.0
```

### Issue: All Trials Failing

**Possible causes**:
1. Graph too small (< 10 nodes)
2. Not enough seeds for train/val split
3. Memory issues with large graphs

**Solutions**:
- Use larger graphs (>20 nodes)
- Provide more seeds (>5)
- Reduce `hidden_dim` or `mf_dim` for large graphs

### Issue: Tuning Takes Too Long

**Solutions**:
- Reduce `n_trials`
- Set shorter `timeout`
- Use `n_jobs_optuna=-1` for parallelization
- Reduce `epochs` range in search space

## Method-Specific Tuning Functions

### Baseline GCN-MF
```python
result = analyzer.tune_gcnmf_hyperparameters(
    G, seeds, targets,
    n_trials=50,
    timeout=3600,
    n_jobs_optuna=1
)
```

### Node2Vec
```python
result = analyzer.tune_node2vec_hyperparameters(
    G, seeds, targets,
    n_trials=50,
    timeout=1800,
    n_jobs_optuna=1
)
```

### NetMF
```python
result = analyzer.tune_netmf_hyperparameters(
    G, seeds, targets,
    n_trials=30,
    timeout=1200,
    n_jobs_optuna=1
)
```

### Q-Caliber GCN-MF (Heat or Poly)
```python
# Heat diffusion
result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G, seeds, targets,
    diffusion_type='heat',
    n_trials=50,
    timeout=3600,
    n_jobs_optuna=1
)

# Polynomial diffusion
result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
    G, seeds, targets,
    diffusion_type='poly',
    n_trials=50,
    timeout=3600,
    n_jobs_optuna=1
)
```

### Quantum Walks (RWR, CTQW, DTQW)
```python
# Random Walk with Restart
result = analyzer.tune_quantum_walk_hyperparameters(
    G, seeds, targets,
    walk_type='rwr',
    n_trials=30,
    timeout=1200,
    n_jobs_optuna=1
)

# Continuous-Time Quantum Walk
result = analyzer.tune_quantum_walk_hyperparameters(
    G, seeds, targets,
    walk_type='ctqw',
    n_trials=30,
    timeout=1200,
    n_jobs_optuna=1
)

# Discrete-Time Quantum Walk
result = analyzer.tune_quantum_walk_hyperparameters(
    G, seeds, targets,
    walk_type='dtqw',
    n_trials=30,
    timeout=1200,
    n_jobs_optuna=1
)
```

## Example: Complete Workflow

```python
#!/usr/bin/env python3
"""Complete hyperparameter tuning workflow for all methods."""

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
import networkx as nx
import json
import os

# 1. Setup
analyzer = ComprehensiveEmbeddingAnalysis(
    output_dir='tuning_results',
    embedding_dim=128
)
os.makedirs('tuning_results', exist_ok=True)

# 2. Load graph
G = nx.karate_club_graph()
seeds = list(range(5))
targets = list(range(5, 15))

# 3. Tune all methods
methods_to_tune = {
    'baseline_gcnmf': ('tune_gcnmf_hyperparameters', {}),
    'node2vec': ('tune_node2vec_hyperparameters', {}),
    'netmf': ('tune_netmf_hyperparameters', {}),
    'hgcnmf': ('tune_qcaliber_gcnmf_hyperparameters', {'diffusion_type': 'heat'}),
    'pgcnmf': ('tune_qcaliber_gcnmf_hyperparameters', {'diffusion_type': 'poly'}),
    'rwr': ('tune_quantum_walk_hyperparameters', {'walk_type': 'rwr'}),
    'ctqw': ('tune_quantum_walk_hyperparameters', {'walk_type': 'ctqw'}),
    'dtqw': ('tune_quantum_walk_hyperparameters', {'walk_type': 'dtqw'}),
}

results = {}
for method_name, (tune_func, kwargs) in methods_to_tune.items():
    print(f"\nTuning {method_name}...")
    
    # Get tuning function
    tune_method = getattr(analyzer, tune_func)
    
    # Run tuning
    result = tune_method(
        G, seeds, targets,
        n_trials=30,
        timeout=1200,
        **kwargs
    )
    
    # Store results
    results[method_name] = result
    analyzer.tuned_hyperparameters[method_name] = result['best_params']
    
    # Save to files
    with open(f'tuning_results/{method_name}_params.json', 'w') as f:
        json.dump(result['best_params'], f, indent=2)
    
    result['trials_df'].to_csv(f'tuning_results/{method_name}_trials.csv', index=False)
    
    print(f"  Best score: {result['best_value']:.4f}")
    print(f"  Best params: {result['best_params']}")

# 4. Generate embeddings with tuned parameters
print("\nGenerating embeddings with tuned parameters...")
embeddings = {}
for method_name in methods_to_tune.keys():
    if method_name in ['baseline_gcnmf', 'node2vec', 'netmf']:
        embedding_method = method_name
    elif method_name in ['hgcnmf', 'pgcnmf']:
        embedding_method = f'quvine_{method_name}'
    else:  # quantum walks
        embedding_method = f'quvine_{method_name}'
    
    embeddings[method_name] = analyzer.run_embedding_method(
        embedding_method, G, seeds, targets
    )
    print(f"  {method_name}: {embeddings[method_name].shape}")

print("\nDone! All methods tuned and embeddings generated.")
```

## References

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **TPE Sampler**: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (2011)
- **GCN-MF**: Qiu et al., "Network Embedding as Matrix Factorization" (2018)

## Summary

The hyperparameter tuning system provides:

### Supported Methods (8 total)
- ✅ **Baseline GCN-MF** - Graph convolutional network with matrix factorization
- ✅ **Node2Vec** - Random walk-based node embeddings
- ✅ **NetMF** - Network embedding as matrix factorization
- ✅ **Q-Caliber Heat GCN-MF** - Quantum-calibrated heat diffusion with GCN-MF
- ✅ **Q-Caliber Poly GCN-MF** - Quantum-calibrated polynomial diffusion with GCN-MF
- ✅ **RWR** - Random Walk with Restart (quantum walk)
- ✅ **CTQW** - Continuous-Time Quantum Walk
- ✅ **DTQW** - Discrete-Time Quantum Walk

### Key Features
- ✅ Automated Bayesian optimization with Optuna
- ✅ Efficient TPE (Tree-structured Parzen Estimator) sampler
- ✅ Automatic train/validation split (80/20)
- ✅ Parallel execution support (multi-core)
- ✅ Comprehensive result tracking and visualization
- ✅ Easy integration with existing QuVINE pipeline
- ✅ Method-specific hyperparameter ranges
- ✅ Recall@50 optimization objective

### Quick Reference

| Method | Function | Typical Trials | Typical Time |
|--------|----------|----------------|--------------|
| Baseline GCN-MF | `tune_gcnmf_hyperparameters()` | 50 | 15-30 min |
| Node2Vec | `tune_node2vec_hyperparameters()` | 50 | 10-20 min |
| NetMF | `tune_netmf_hyperparameters()` | 30 | 5-10 min |
| Q-Caliber Heat | `tune_qcaliber_gcnmf_hyperparameters(diffusion_type='heat')` | 50 | 15-30 min |
| Q-Caliber Poly | `tune_qcaliber_gcnmf_hyperparameters(diffusion_type='poly')` | 50 | 15-30 min |
| RWR | `tune_quantum_walk_hyperparameters(walk_type='rwr')` | 30 | 5-15 min |
| CTQW | `tune_quantum_walk_hyperparameters(walk_type='ctqw')` | 30 | 5-15 min |
| DTQW | `tune_quantum_walk_hyperparameters(walk_type='dtqw')` | 30 | 5-15 min |

For questions or issues, refer to the main QuVINE documentation or open an issue on GitHub.