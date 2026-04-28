# Quick Test Guide - Validate Tuning Pipeline

## Overview

This quick test validates the entire hyperparameter tuning pipeline with minimal compute time before running the full production jobs.

## What Gets Tested

### Coverage
- **12 methods**: All quantum and classical baselines
- **2 synthetic networks**: erdos_renyi, modular_strong
- **2 PPI networks**: STRING, BioPlex3
- **1 disease**: asthma
- **3 tasks per method**: node_classification, link_prediction, node_ranking

### Job Count
- Synthetic: 2 networks × 12 methods = **24 jobs**
- PPI: 2 networks × 1 disease × 12 methods = **24 jobs**
- Aggregation: 2 jobs (1 synthetic, 1 PPI)
- **Total: 50 jobs**

### Runtime
- **Per job**: 15-30 minutes (n_trials=5 instead of 20-50)
- **Total**: ~30-60 minutes (parallelized)
- **Compute**: ~25-50 CPU-hours total

## Quick Start

### 1. Submit Test Jobs

```bash
# From QuVINE directory
bash scripts/submit_quick_test.sh
```

### 2. Monitor Progress

```bash
# Check all jobs
bjobs

# Watch specific job
tail -f tuning_by_task/logs/tune_quvine_rwr_erdos_renyi.out

# Check for errors
grep -i error tuning_by_task/logs/*.err
grep -i error ppi_tuning_by_task/logs/*.err
```

### 3. Verify Outputs

After jobs complete, check that files were created:

```bash
# Synthetic network outputs (24 files)
ls -lh tuning_by_task/*_tuning_by_task.json | wc -l
# Should show: 24

# PPI network outputs (24 files)
ls -lh ppi_tuning_by_task/*_tuning_by_task.json | wc -l
# Should show: 24

# Aggregated results (4 files)
ls -lh tuning_by_task/erdos_renyi_tuning_by_task.json
ls -lh tuning_by_task/modular_strong_tuning_by_task.json
ls -lh ppi_tuning_by_task/STRING_asthma_tuning_by_task.json
ls -lh ppi_tuning_by_task/BioPlex3_asthma_tuning_by_task.json
```

## Expected Output Structure

### Individual Method Files

**Synthetic:**
```
tuning_by_task/
├── erdos_renyi_quvine_rwr_tuning_by_task.json
├── erdos_renyi_quvine_ctqw_tuning_by_task.json
├── erdos_renyi_quvine_dtqw_tuning_by_task.json
├── erdos_renyi_baseline_filter_heat_tuning_by_task.json
├── erdos_renyi_baseline_filter_poly_tuning_by_task.json
├── erdos_renyi_baseline_gcnmf_tuning_by_task.json
├── erdos_renyi_gat_baseline_tuning_by_task.json
├── erdos_renyi_graphgps_baseline_tuning_by_task.json
├── erdos_renyi_node2vec_tuning_by_task.json
├── erdos_renyi_netmf_tuning_by_task.json
├── erdos_renyi_graphsage_tuning_by_task.json
├── erdos_renyi_appnp_tuning_by_task.json
├── modular_strong_quvine_rwr_tuning_by_task.json
├── ... (12 more for modular_strong)
```

**PPI:**
```
ppi_tuning_by_task/
├── STRING_asthma_quvine_rwr_tuning_by_task.json
├── STRING_asthma_quvine_ctqw_tuning_by_task.json
├── ... (10 more for STRING_asthma)
├── BioPlex3_asthma_quvine_rwr_tuning_by_task.json
├── ... (11 more for BioPlex3_asthma)
```

### Aggregated Files

Each aggregated file contains all 12 methods for that network:

```json
{
  "quvine_rwr": {
    "node_classification": {"best_params": {...}, "best_score": 0.85, ...},
    "link_prediction": {"best_params": {...}, "best_score": 0.92, ...},
    "node_ranking": {"best_params": {...}, "best_score": 0.78, ...}
  },
  "quvine_ctqw": {...},
  ...
}
```

## Validation Checklist

After test completes, verify:

- [ ] All 48 tuning jobs completed successfully (check `bjobs`)
- [ ] 24 synthetic network JSON files created
- [ ] 24 PPI network JSON files created
- [ ] 4 aggregated JSON files created
- [ ] No error messages in log files
- [ ] Each JSON file contains 3 tasks (node_classification, link_prediction, node_ranking)
- [ ] Each task has `best_params`, `best_score`, and `trials` data

## Troubleshooting

### Jobs Failed

```bash
# Check which jobs failed
bjobs -a | grep EXIT

# Check error logs
tail -100 tuning_by_task/logs/tune_METHOD_NETWORK.err
```

### Missing Output Files

```bash
# Check if job completed
bjobs -a | grep tune_METHOD_NETWORK

# Check output log for errors
tail -100 tuning_by_task/logs/tune_METHOD_NETWORK.out
```

### Aggregation Failed

```bash
# Check aggregation job status
bjobs -a | grep aggregate

# Check aggregation log
tail -100 tuning_by_task/logs/tune_aggregate.out
tail -100 ppi_tuning_by_task/logs/ppi_tune_aggregate.out
```

## Next Steps

### If Test Passes ✅

You're ready for production! Submit full jobs:

```bash
# Full synthetic networks (192 jobs, ~4-5 hours per job)
bash scripts/submit_tuning_jobs.sh

# Full PPI networks (180 jobs, ~4-5 hours per job)
bash scripts/submit_ppi_tuning_jobs.sh
```

### If Test Fails ❌

1. Check error logs for specific issues
2. Fix configuration or code issues
3. Re-run quick test
4. Don't proceed to production until test passes

## Configuration Details

### Test Config: `scripts/test_tuning_config.yaml`

- **n_trials**: 5 (vs 20-50 in production)
- **n_graphs**: 5 (vs 10 in production)
- **n_replicates**: 2 (vs 3 in production)
- **All other parameters**: Same as production

### Production Config: `scripts/unified_tuning_config.yaml`

- **n_trials**: 20-50 (method-dependent)
- **n_graphs**: 10
- **n_replicates**: 3
- **Runtime**: 4-5 hours per job

## Dry Run Mode

Test the submission without actually submitting jobs:

```bash
bash scripts/submit_quick_test.sh --dry-run
```

This shows what would be submitted without using compute resources.

## Summary

**Quick test validates:**
- ✓ All 12 methods work correctly
- ✓ Separate files created per method × network
- ✓ Aggregation collects results properly
- ✓ No configuration errors
- ✓ Pipeline is production-ready

**Total cost:** ~25-50 CPU-hours (~30-60 minutes wall time)

**After success:** Proceed to full production run (372 jobs, ~1500-1800 CPU-hours)