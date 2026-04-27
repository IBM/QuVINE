# Hyperparameter Tuning with Configuration File

This guide explains how to use the YAML configuration file for comprehensive hyperparameter tuning.

## Overview

The configuration-based tuning system provides:
- **Centralized hyperparameter search spaces** for all 10 methods
- **Fixed evaluation parameters** (graph size, metrics, etc.)
- **Easy customization** without modifying code
- **Reproducible experiments** with version-controlled configs

## Files

1. **`tuning_config.yaml`** - Main configuration file with:
   - Fixed parameters (graph generation, evaluation settings)
   - Hyperparameter search spaces for each method
   - Optuna settings
   - Experiment configuration

2. **`tune_by_task_with_config.py`** - Script that reads config and runs tuning

3. **`tune_by_task.py`** - Original script (still works, hardcoded parameters)

## Quick Start

### 1. Test Run (3 methods, 5 trials each)
```bash
python QuVINE/scripts/tune_by_task_with_config.py \
  --methods baseline_filter_heat node2vec gat_baseline \
  --n-trials 5 \
  --n-graphs 2 \
  --network-type erdos_renyi
```

### 2. Full Production Run (All methods, both networks)
```bash
python QuVINE/scripts/tune_by_task_with_config.py \
  --config QuVINE/scripts/tuning_config.yaml \
  --network-type all \
  --n-trials 50 \
  --n-graphs 10
```

### 3. Custom Config File
```bash
python QuVINE/scripts/tune_by_task_with_config.py \
  --config my_custom_config.yaml \
  --n-trials 100
```

## Configuration File Structure

### 1. Fixed Parameters

```yaml
fixed_params:
  graph:
    n_nodes: 200  # Graph size
    erdos_renyi:
      p: 0.1  # Edge probability
    modular:
      n_communities: 4
      p_in: 0.3
      p_out: 0.05
  
  evaluation:
    node_classification:
      test_size: 0.3  # 70/30 train/test split
      n_label_strategies: 5
      metric: "f1_macro"
    
    link_prediction:
      test_ratio: 0.2  # Hold out 20% edges
      edge_feature_method: "hadamard"
      metric: "auc_roc"
    
    node_ranking:
      k_hops: 2  # K-hop neighbors as targets
      top_k: 50
      metric: "f1"
```

**These parameters are FIXED during tuning** - they define the evaluation protocol.

### 2. Hyperparameter Search Spaces

```yaml
hyperparameters:
  quvine_walks:
    embedding_dim: [32, 64, 128, 256]  # Categorical choices
    walk_length: [10, 20, 40, 80]
    tau: [0.1, 0.5, 1.0, 2.0, 5.0]
    # ... more parameters
  
  node2vec:
    embedding_dim: [32, 64, 128, 256]
    p: [0.25, 0.5, 1.0, 2.0, 4.0]  # Return parameter
    q: [0.25, 0.5, 1.0, 2.0, 4.0]  # In-out parameter
    # ... more parameters
```

**These parameters are TUNED** - Optuna searches over these values.

### 3. Optuna Settings

```yaml
optuna:
  n_trials: 50  # Trials per method per task
  n_startup_trials: 10  # Random trials before TPE
  sampler: "TPE"  # Tree-structured Parzen Estimator
  pruner: "MedianPruner"
```

### 4. Experiment Settings

```yaml
experiment:
  n_graphs: 10  # Graphs to average over
  network_types: ["erdos_renyi", "modular"]
  output_dir: "tuning_by_task"
  log_level: "INFO"
```

## Customizing the Configuration

### Example 1: Increase Graph Size

Edit `tuning_config.yaml`:
```yaml
fixed_params:
  graph:
    n_nodes: 500  # Changed from 200
```

### Example 2: Add More Hyperparameter Values

```yaml
hyperparameters:
  node2vec:
    embedding_dim: [32, 64, 128, 256, 512]  # Added 512
    p: [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]  # More granular
```

### Example 3: Change Evaluation Metric

```yaml
fixed_params:
  evaluation:
    link_prediction:
      edge_feature_method: "cosine"  # Changed from hadamard
```

### Example 4: More Trials

```yaml
optuna:
  n_trials: 100  # Changed from 50
  n_startup_trials: 20  # More random exploration
```

## Command-Line Overrides

You can override config settings from the command line:

```bash
# Override number of trials
python tune_by_task_with_config.py --n-trials 100

# Override network type
python tune_by_task_with_config.py --network-type modular

# Override methods
python tune_by_task_with_config.py --methods quvine_walks node2vec

# Override output directory
python tune_by_task_with_config.py --output-dir my_results

# Override random seed
python tune_by_task_with_config.py --seed 123
```

## Output Format

Results are saved as JSON files:

```
tuning_by_task/
├── erdos_renyi_tuning_by_task.json
└── modular_tuning_by_task.json
```

Each file contains:
```json
{
  "method_name": {
    "node_classification": {
      "best_params": {
        "embedding_dim": 128,
        "walk_length": 40,
        ...
      },
      "best_score": 0.4244
    },
    "link_prediction": {
      "best_params": {...},
      "best_score": 0.9960
    },
    "node_ranking": {
      "best_params": {...},
      "best_score": 0.6143
    }
  }
}
```

## Hyperparameter Search Spaces by Method

### Quantum Methods

**quvine_walks**
- `embedding_dim`: [32, 64, 128, 256]
- `walk_length`: [10, 20, 40, 80]
- `num_walks`: [10, 20, 40, 80]
- `p`, `q`: [0.25, 0.5, 1.0, 2.0, 4.0]
- `window_size`: [5, 10, 15, 20]
- `quantum_filter`: ["heat", "poly", "wave"]
- `filter_order`: [2, 3, 4, 5, 6]
- `tau`: [0.1, 0.5, 1.0, 2.0, 5.0]

### Filter-Based Methods

**baseline_filter_heat**
- `embedding_dim`: [32, 64, 128, 256]
- `tau`: [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
- `filter_order`: [2, 3, 4, 5, 6, 8, 10]

**baseline_filter_poly**
- `embedding_dim`: [32, 64, 128, 256]
- `filter_order`: [2, 3, 4, 5, 6, 8, 10, 15, 20]
- `alpha`: [0.1, 0.3, 0.5, 0.7, 0.9]

**baseline_gcnmf**
- `embedding_dim`: [32, 64, 128, 256]
- `n_layers`: [2, 3, 4, 5]
- `window_size`: [5, 10, 15, 20]
- `negative_samples`: [1, 5, 10, 20]
- `learning_rate`: [0.001, 0.005, 0.01, 0.05]
- `epochs`: [50, 100, 200]

### Random Walk Methods

**node2vec**
- `embedding_dim`: [32, 64, 128, 256]
- `walk_length`: [10, 20, 40, 80]
- `num_walks`: [10, 20, 40, 80]
- `p`, `q`: [0.25, 0.5, 1.0, 2.0, 4.0]
- `window_size`: [5, 10, 15, 20]
- `negative_samples`: [1, 5, 10, 20]
- `epochs`: [5, 10, 20]

**netmf**
- `embedding_dim`: [32, 64, 128, 256]
- `window_size`: [5, 10, 15, 20]
- `rank`: [64, 128, 256, 512]
- `negative_samples`: [1, 5, 10, 20]

### GNN Methods

**graphsage**
- `embedding_dim`: [32, 64, 128, 256]
- `n_layers`: [2, 3, 4]
- `hidden_dim`: [64, 128, 256, 512]
- `aggregator`: ["mean", "gcn", "pool", "lstm"]
- `dropout`: [0.0, 0.1, 0.3, 0.5]
- `learning_rate`: [0.001, 0.005, 0.01]
- `epochs`: [50, 100, 200]
- `batch_size`: [32, 64, 128]

**appnp**
- `embedding_dim`: [32, 64, 128, 256]
- `hidden_dim`: [64, 128, 256, 512]
- `n_layers`: [2, 3, 4]
- `alpha`: [0.05, 0.1, 0.15, 0.2]
- `k_hops`: [5, 10, 15, 20]
- `dropout`: [0.0, 0.1, 0.3, 0.5]
- `learning_rate`: [0.001, 0.005, 0.01]
- `epochs`: [50, 100, 200]

**gat_baseline**
- `embedding_dim`: [32, 64, 128, 256]
- `hidden_dim`: [64, 128, 256]
- `n_layers`: [2, 3, 4]
- `n_heads`: [1, 2, 4, 8]
- `dropout`, `attn_dropout`: [0.0, 0.1, 0.3, 0.5]
- `learning_rate`: [0.001, 0.005, 0.01]
- `epochs`: [50, 100, 200]
- `negative_slope`: [0.1, 0.2, 0.3]

**graphgps_baseline**
- `embedding_dim`: [32, 64, 128, 256]
- `hidden_dim`: [64, 128, 256]
- `n_layers`: [2, 3, 4, 6]
- `n_heads`: [1, 2, 4, 8]
- `dropout`, `attn_dropout`: [0.0, 0.1, 0.3, 0.5]
- `learning_rate`: [0.001, 0.005, 0.01]
- `epochs`: [50, 100, 200]
- `mpnn_type`: ["gine", "gcn", "gin"]
- `global_model_type`: ["Transformer", "Performer"]

## LSF Deployment

For production runs on LSF cluster:

```bash
# Create LSF submission script
cat > submit_tuning.sh << 'EOF'
#!/bin/bash
#BSUB -J tune_hyperparams
#BSUB -o tune_%J.out
#BSUB -e tune_%J.err
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -W 48:00

module load python/3.9
source venv_quvine/bin/activate

python QuVINE/scripts/tune_by_task_with_config.py \
  --config QuVINE/scripts/tuning_config.yaml \
  --network-type all \
  --n-trials 50 \
  --n-graphs 10
EOF

# Submit job
bsub < submit_tuning.sh
```

## Tips for Effective Tuning

### 1. Start Small
```bash
# Test with 2 trials, 1 graph
python tune_by_task_with_config.py --n-trials 2 --n-graphs 1 --methods node2vec
```

### 2. Increase Gradually
```bash
# Medium run: 10 trials, 3 graphs
python tune_by_task_with_config.py --n-trials 10 --n-graphs 3

# Full run: 50 trials, 10 graphs
python tune_by_task_with_config.py --n-trials 50 --n-graphs 10
```

### 3. Monitor Progress
```bash
# Watch output in real-time
python tune_by_task_with_config.py --n-trials 20 2>&1 | tee tuning.log
```

### 4. Parallel Runs
```bash
# Run different networks in parallel
python tune_by_task_with_config.py --network-type erdos_renyi &
python tune_by_task_with_config.py --network-type modular &
```

## Troubleshooting

### Issue: YAML parsing error
**Solution**: Check YAML syntax, ensure proper indentation (2 spaces)

### Issue: Method not found in config
**Solution**: Add method to `hyperparameters` section in config file

### Issue: Out of memory
**Solution**: Reduce `n_nodes` in fixed_params or use smaller `embedding_dim`

### Issue: Slow tuning
**Solution**: Reduce `n_trials`, `n_graphs`, or `epochs` in hyperparameters

## Comparison: Config vs Hardcoded

| Feature | tune_by_task.py | tune_by_task_with_config.py |
|---------|----------------|----------------------------|
| Hyperparameter spaces | Hardcoded | YAML config |
| Fixed parameters | Hardcoded | YAML config |
| Easy to modify | ❌ Edit code | ✅ Edit YAML |
| Version control | ❌ Code changes | ✅ Config files |
| Reproducibility | ⚠️ Manual | ✅ Automatic |
| Flexibility | Limited | High |

## Best Practices

1. **Version control your configs**: Commit `tuning_config.yaml` to git
2. **Document changes**: Add comments in YAML for modifications
3. **Start conservative**: Use smaller search spaces initially
4. **Validate results**: Check that best_score values are reasonable
5. **Save intermediate results**: Use `--output-dir` for different experiments

## Example Workflow

```bash
# 1. Quick test (5 minutes)
python tune_by_task_with_config.py \
  --methods baseline_filter_heat \
  --n-trials 3 --n-graphs 1 \
  --network-type erdos_renyi

# 2. Medium test (30 minutes)
python tune_by_task_with_config.py \
  --methods baseline_filter_heat node2vec gat_baseline \
  --n-trials 10 --n-graphs 3 \
  --network-type erdos_renyi

# 3. Full production (hours)
python tune_by_task_with_config.py \
  --config tuning_config.yaml \
  --network-type all \
  --n-trials 50 --n-graphs 10

# 4. Analyze results
python analyze_tuning_results.py tuning_by_task/
```

## Summary

The configuration-based tuning system provides:
- ✅ **Comprehensive search spaces** for all 10 methods
- ✅ **Fixed evaluation protocol** for fair comparison
- ✅ **Easy customization** without code changes
- ✅ **Reproducible experiments** with version control
- ✅ **LSF-ready** for production deployment

Ready to deploy! 🚀