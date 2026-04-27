# Hyperparameter Tuning Time Estimates

## Configuration

### Pilot Graph Settings
- **Graph size**: 200 nodes (balanced between speed and quality)
- **Number of graphs**: 3 (for stochasticity)
- **Trials per method**: 30 (default, configurable)

### Methods to Tune
8 representative methods (reused for all 39):
1. `quvine_walks` - Quantum walk embeddings
2. `baseline_filter_heat` - Heat kernel filter
3. `baseline_filter_poly` - Polynomial filter
4. `baseline_gcnmf` - GCN-MF baseline
5. `node2vec` - Node2Vec
6. `netmf` - NetMF
7. `graphsage` - GraphSAGE
8. `appnp` - APPNP

## Time Estimates per Method

### Fast Methods (~5-10 min per trial)
- **node2vec**: ~5 min/trial → ~2.5 hours for 30 trials
- **netmf**: ~3 min/trial → ~1.5 hours for 30 trials
- **baseline_filter_heat**: ~2 min/trial → ~1 hour for 30 trials
- **baseline_filter_poly**: ~2 min/trial → ~1 hour for 30 trials

### Medium Methods (~10-20 min per trial)
- **quvine_walks**: ~15 min/trial → ~7.5 hours for 30 trials
- **baseline_gcnmf**: ~10 min/trial → ~5 hours for 30 trials

### Slower Methods (~20-40 min per trial)
- **graphsage**: ~25 min/trial → ~12.5 hours for 30 trials
- **appnp**: ~20 min/trial → ~10 hours for 30 trials

## Total Time Estimates

### Sequential (worst case)
Sum of all methods: **~41 hours** for one network type

### Parallel (HPC cluster)
All 8 methods run simultaneously: **~12.5 hours** (limited by slowest method: graphsage)

### Per Network Type
- **Synthetic networks**: 13 types × 12.5 hours = ~163 hours (parallel) or ~533 hours (sequential)
- **PPI networks**: 5 networks × 12.5 hours = ~62.5 hours (parallel) or ~205 hours (sequential)

## Optimization Strategies

### 1. Reduce Trials (Recommended)
- **20 trials** instead of 30: ~33% time reduction
- **15 trials** instead of 30: ~50% time reduction
- Still provides good hyperparameter coverage

### 2. Reduce Graph Size (Not Recommended)
- 100 nodes: ~40% faster but less representative
- 200 nodes: **Current setting** - good balance
- 300 nodes: ~50% slower but more accurate

### 3. Reduce Number of Graphs
- 2 graphs instead of 3: ~33% time reduction
- May increase variance in results

### 4. Early Stopping
- Optuna can stop unpromising trials early
- Already implemented in the tuning script

## Recommended Configuration

For **fast initial tuning** (testing):
```bash
--n-trials 15 --pilot-nodes 200 --pilot-seeds 2
```
**Time**: ~6 hours per network type (parallel)

For **production tuning** (current default):
```bash
--n-trials 30 --pilot-nodes 200 --pilot-seeds 3
```
**Time**: ~12.5 hours per network type (parallel)

For **thorough tuning** (research):
```bash
--n-trials 50 --pilot-nodes 300 --pilot-seeds 3
```
**Time**: ~30 hours per network type (parallel)

## HPC Deployment

### Simulated Data
```bash
# Fast mode (15 trials)
sbatch scripts/submit_simulated_data_jobs_with_tuning.sh --n-trials 15

# Default mode (30 trials)
sbatch scripts/submit_simulated_data_jobs_with_tuning.sh

# Thorough mode (50 trials)
sbatch scripts/submit_simulated_data_jobs_with_tuning.sh --n-trials 50
```

### PPI Networks
```bash
# Fast mode
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh --n-trials 15 --pilot-nodes 200

# Default mode
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh

# Thorough mode
sbatch scripts/submit_ppi_comprehensive_with_tuning.sh --n-trials 50 --pilot-nodes 300
```

## Memory Requirements

- **Fast methods**: 4-8 GB RAM
- **Medium methods**: 8-16 GB RAM
- **Neural methods** (GraphSAGE, APPNP, GCN-MF): 16-32 GB RAM

Current allocation: **32 GB** per tuning job (safe for all methods)

## Monitoring Progress

Check tuning progress:
```bash
# View running jobs
bjobs -u $USER

# Check tuning output
tail -f results/*/hparam_tuning/*/logs/tune_*.out

# View best parameters so far
cat results/*/hparam_tuning/*/best_hyperparams.json
```

## Summary

**Current Configuration (200 nodes, 30 trials, 3 seeds):**
- ✅ Good balance between speed and quality
- ✅ Completes in ~12.5 hours per network type (parallel)
- ✅ Provides robust hyperparameter estimates
- ✅ Suitable for production use

**For faster testing:** Use `--n-trials 15` to cut time in half
**For research:** Use `--n-trials 50 --pilot-nodes 300` for thorough exploration