# Hyperparameter Tuning Guide

This guide explains how to run hyperparameter tuning for QuVINE methods on erdos_renyi and modular networks.

## Overview

The hyperparameter tuning process:
1. Generates pilot graphs (small test networks)
2. Tests different hyperparameter combinations for each method
3. Evaluates performance on node classification and link prediction
4. Saves the best hyperparameters for each network type and method

## Local Testing (This System)

### Quick Test (Recommended First)

Test with minimal trials to verify everything works:

```bash
cd QuVINE
python scripts/tune_local_test.py \
    --network-type erdos_renyi \
    --methods quvine_walks node2vec \
    --n-trials 5 \
    --n-graphs 2 \
    --output-dir ./tuning_test
```

### Full Local Run

Run all methods on both network types:

```bash
python scripts/tune_local_test.py \
    --network-type all \
    --n-trials 20 \
    --n-graphs 3 \
    --n-nodes 100 \
    --output-dir ./tuning_results_local
```

### Single Network Type

Test only erdos_renyi or modular:

```bash
# Erdos-Renyi only
python scripts/tune_local_test.py \
    --network-type erdos_renyi \
    --n-trials 30 \
    --output-dir ./tuning_erdos_renyi

# Modular only
python scripts/tune_local_test.py \
    --network-type modular \
    --n-trials 30 \
    --output-dir ./tuning_modular
```

### Specific Methods

Tune only specific methods:

```bash
python scripts/tune_local_test.py \
    --methods quvine_walks node2vec netmf \
    --n-trials 20 \
    --output-dir ./tuning_subset
```

## Command Line Options

- `--network-type`: Network type to tune (`erdos_renyi`, `modular`, or `all`)
- `--methods`: List of methods to tune (default: all methods)
- `--n-trials`: Number of hyperparameter trials per method (default: 10)
- `--n-graphs`: Number of pilot graphs per network type (default: 3)
- `--n-nodes`: Number of nodes in pilot graphs (default: 100)
- `--output-dir`: Output directory for results (default: ./tuning_test)
- `--seed`: Base random seed (default: 42)

## Available Methods

The script tunes the following 10 representative methods (which cover all 39 methods):

1. **quvine_walks** - QuVINE quantum walk-based methods (representative for 11 quvine_* methods)
2. **baseline_filter_heat** - Heat kernel filter
3. **baseline_filter_poly** - Polynomial filter
4. **baseline_gcnmf** - GCN Matrix Factorization
5. **node2vec** - Node2Vec baseline
6. **netmf** - NetMF baseline
7. **graphsage** - GraphSAGE baseline
8. **appnp** - APPNP baseline
9. **gat_baseline** - GAT baseline (representative for 12 GAT methods)
10. **graphgps_baseline** - GraphGPS baseline (representative for 12 GraphGPS methods)

**Note:** GAT and GraphGPS methods require PyTorch. If not available, they will be skipped with a warning.

## Output Files

The script generates:

- `{network_type}_tuning_results.json` - Results for each network type
- `all_tuning_results.json` - Combined results for all network types

Each result file contains:
- Best hyperparameters for each method
- Best performance score (combined NC F1 + LP AUC)
- Any errors encountered

## Example Output Structure

```json
{
  "erdos_renyi": {
    "quvine_walks": {
      "best_params": {
        "num_views": 3,
        "max_degree": 50,
        "num_walks": 10,
        "walk_length": 80,
        "embedding_dim": 128,
        ...
      },
      "best_score": 0.7234
    },
    "node2vec": {
      "best_params": {
        "embedding_dim": 128,
        "walk_length": 80,
        "p": 1.0,
        "q": 1.0,
        ...
      },
      "best_score": 0.6891
    }
  }
}
```

## Deployment to LSF

Once local testing confirms the script works:

1. The tuned hyperparameters can be used in the main experiment scripts
2. For large-scale tuning on LSF, use the existing `tune_hyperparameters.py` script
3. The local test script validates the approach before expensive HPC runs

## Troubleshooting

### Cannot Import 'triu' from scipy.linalg

**Error:** `ImportError: cannot import name 'triu' from 'scipy.linalg'`

**Cause:** Compatibility issue between gensim and newer scipy versions (scipy removed `triu` in favor of numpy's version).

**Solution:**
```bash
# Option 1: Upgrade gensim (recommended)
pip install --upgrade gensim

# Option 2: Install specific compatible version
pip install gensim==4.3.2

# Option 3: Downgrade scipy (not recommended)
pip install scipy==1.10.1
```

### Optuna Not Installed

If you see "optuna not installed", the script will fall back to random search. To use the more efficient TPE sampler:

```bash
pip install optuna
```

### Import Errors

Make sure you're running from the QuVINE directory and the virtual environment is activated:

```bash
cd QuVINE
source venv_quvine/bin/activate  # or your venv path
python scripts/tune_local_test.py --help
```

### Memory Issues

If you run out of memory:
- Reduce `--n-nodes` (e.g., to 50)
- Reduce `--n-graphs` (e.g., to 2)
- Reduce `--n-trials` (e.g., to 10)

### Method Failures

Some methods may fail on very small graphs. This is normal - the script will log warnings and continue with other methods.

## Performance Tips

1. **Start small**: Use `--n-trials 5` first to verify everything works
2. **Parallel runs**: Run different network types in parallel:
   ```bash
   python scripts/tune_local_test.py --network-type erdos_renyi &
   python scripts/tune_local_test.py --network-type modular &
   ```
3. **Method subsets**: Tune fast methods first (node2vec, netmf) before slow ones (graphsage, appnp)

## Next Steps

After successful local tuning:

1. Review the results in the output directory
2. Use the best hyperparameters in your main experiments
3. For production runs, deploy to LSF using the full tuning infrastructure