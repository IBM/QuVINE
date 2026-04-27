# Hyperparameter Tuning - Quick Start Guide

## What Was Created

Three new files for local hyperparameter tuning:

1. **`tune_local_test.py`** - Main tuning script for local testing
2. **`test_tuning_setup.sh`** - Quick test to verify setup
3. **`README_HYPERPARAMETER_TUNING.md`** - Detailed documentation

## Quick Test (Recommended First Step)

Run this to verify everything works:

```bash
cd QuVINE
bash scripts/test_tuning_setup.sh
```

This will:
- Check Python and dependencies
- Run a minimal tuning test (3 trials, 2 graphs, 50 nodes)
- Save results to a timestamped directory
- Report success or failure

**Expected runtime:** 2-5 minutes

## What It Does

The tuning script:

1. **Generates pilot graphs** for erdos_renyi and modular networks
2. **Tests hyperparameters** for 10 representative methods (covering all 39):
   - quvine_walks (representative for 11 quantum walk-based methods)
   - baseline_filter_heat, baseline_filter_poly
   - baseline_gcnmf
   - node2vec, netmf, graphsage, appnp
   - gat_baseline (representative for 12 GAT methods)
   - graphgps_baseline (representative for 12 GraphGPS methods)
3. **Evaluates performance** on:
   - Node classification (F1-macro)
   - Link prediction (AUC-ROC)
4. **Saves best parameters** for each network type and method

**Note:** GAT and GraphGPS require PyTorch. If unavailable, they'll be skipped.

## Usage Examples

### Test Single Network Type

```bash
# Test erdos_renyi only
python scripts/tune_local_test.py \
    --network-type erdos_renyi \
    --n-trials 10 \
    --output-dir ./tuning_erdos

# Test modular only
python scripts/tune_local_test.py \
    --network-type modular \
    --n-trials 10 \
    --output-dir ./tuning_modular
```

### Test Both Network Types

```bash
python scripts/tune_local_test.py \
    --network-type all \
    --n-trials 20 \
    --output-dir ./tuning_both
```

### Test Specific Methods

```bash
# Test only fast methods
python scripts/tune_local_test.py \
    --methods quvine_walks node2vec netmf \
    --n-trials 15 \
    --output-dir ./tuning_fast

# Test only quantum methods
python scripts/tune_local_test.py \
    --methods quvine_walks \
    --n-trials 30 \
    --output-dir ./tuning_quantum
```

### Full Local Run

```bash
# All methods, both networks, 30 trials each
python scripts/tune_local_test.py \
    --network-type all \
    --n-trials 30 \
    --n-graphs 3 \
    --n-nodes 100 \
    --output-dir ./tuning_full
```

**Expected runtime:** 1-3 hours depending on system

## Output Files

Results are saved in JSON format:

```
output_dir/
├── erdos_renyi_tuning_results.json
├── modular_tuning_results.json
└── all_tuning_results.json
```

Each file contains:
```json
{
  "method_name": {
    "best_params": { ... },
    "best_score": 0.7234
  }
}
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--network-type` | `all` | `erdos_renyi`, `modular`, or `all` |
| `--methods` | all methods | List of methods to tune |
| `--n-trials` | 10 | Trials per method |
| `--n-graphs` | 3 | Pilot graphs per network type |
| `--n-nodes` | 100 | Nodes in pilot graphs |
| `--output-dir` | `./tuning_test` | Output directory |
| `--seed` | 42 | Random seed |

## Performance Tips

1. **Start small**: Use `--n-trials 5` first
2. **Test one network**: Use `--network-type erdos_renyi`
3. **Test fast methods**: Use `--methods node2vec netmf`
4. **Reduce graph size**: Use `--n-nodes 50` for faster testing

## Troubleshooting

### "cannot import name 'triu' from 'scipy.linalg'"

This is a compatibility issue between gensim and newer scipy versions. Fix:

```bash
# Option 1: Upgrade gensim (recommended)
pip install --upgrade gensim

# Option 2: If that doesn't work, install specific compatible version
pip install gensim==4.3.2

# Option 3: Downgrade scipy (not recommended)
pip install scipy==1.10.1
```

### "optuna not installed"
```bash
pip install optuna
```
Script will work without it (uses random search instead of TPE).

### Import errors
Make sure you're in the QuVINE directory:
```bash
cd QuVINE
python scripts/tune_local_test.py --help
```

### Memory issues
Reduce parameters:
```bash
python scripts/tune_local_test.py \
    --n-nodes 50 \
    --n-graphs 2 \
    --n-trials 5
```

## Next Steps After Local Testing

1. **Review results**: Check the JSON files in output directory
2. **Verify best parameters**: Make sure they look reasonable
3. **Deploy to LSF**: Use the existing `tune_hyperparameters.py` for full-scale tuning
4. **Use in experiments**: Apply best parameters to main experiment scripts

## Integration with LSF

Once local testing confirms the approach works:

1. The tuned hyperparameters can be saved to a JSON file
2. Main experiment scripts can load these parameters
3. For large-scale tuning across all networks, use LSF with the existing infrastructure

## Getting Help

- **Detailed docs**: `cat scripts/README_HYPERPARAMETER_TUNING.md`
- **Script help**: `python scripts/tune_local_test.py --help`
- **Test script**: `bash scripts/test_tuning_setup.sh`

## Summary

```bash
# 1. Quick test (verify setup)
bash scripts/test_tuning_setup.sh

# 2. Test one network type
python scripts/tune_local_test.py --network-type erdos_renyi --n-trials 10

# 3. Full local run
python scripts/tune_local_test.py --network-type all --n-trials 30

# 4. Review results
cat tuning_test/all_tuning_results.json

# 5. Deploy to LSF (when ready)
# Use existing tune_hyperparameters.py with LSF job submission