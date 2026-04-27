# Parallel Hyperparameter Tuning Guide

## Overview

The parallel tuning system submits **one LSF job per method × network type combination**, allowing all hyperparameter tuning to run in parallel on the cluster.

## Architecture

```
submit_tuning_jobs.sh
├── Submits N_METHODS × N_NETWORK_TYPES jobs
│   ├── Each job runs: tune_by_task_with_config.py
│   │   ├── Tunes 1 method on 1 network type
│   │   └── Tunes for all 3 tasks (classification, link_pred, ranking)
│   └── Uses adaptive n_trials from config (30-120 trials per method)
└── Submits 1 aggregation job (depends on all tuning jobs)
    └── Combines results into final JSON files
```

## Default Configuration

- **Methods**: 10 methods (quvine_walks, baseline_filter_heat, baseline_filter_poly, baseline_gcnmf, node2vec, netmf, graphsage, appnp, gat_baseline, graphgps_baseline)
- **Network Types**: 2 types (erdos_renyi, modular)
- **Total Jobs**: 10 × 2 = **20 parallel jobs**
- **Trials per Method**: Adaptive (30-120 based on complexity)
- **Graphs per Trial**: 10 (for averaging)

## Usage

### 1. Dry Run (Test Without Submitting)

```bash
cd QuVINE
bash scripts/submit_tuning_jobs.sh --dry-run
```

### 2. Submit to LSF

```bash
cd QuVINE
bash scripts/submit_tuning_jobs.sh \
  --queue normal \
  --walltime 48:00 \
  --memory 32 \
  --n-graphs 10
```

### 3. Custom Configuration

```bash
# Use different queue and resources
bash scripts/submit_tuning_jobs.sh \
  --queue gpu \
  --walltime 72:00 \
  --memory 64 \
  --n-graphs 20

# Use custom config file
bash scripts/submit_tuning_jobs.sh \
  --config scripts/custom_tuning_config.yaml
```

## Monitoring

### Check Job Status

```bash
# All jobs
bjobs -u $USER

# Specific tuning jobs
bjobs -u $USER | grep tune_

# Count running/pending
bjobs -u $USER | grep tune_ | wc -l
```

### Monitor Individual Job

```bash
# Watch output in real-time
tail -f tuning_by_task/logs/tune_quvine_walks_erdos_renyi.out

# Check for errors
grep -i error tuning_by_task/logs/tune_quvine_walks_erdos_renyi.err

# Check progress
grep "Trial\|Best\|✓" tuning_by_task/logs/tune_quvine_walks_erdos_renyi.out
```

### Monitor All Jobs

```bash
# Check completion status
for log in tuning_by_task/logs/tune_*.out; do
    echo "=== $(basename $log) ==="
    tail -5 "$log"
done

# Count completed jobs
grep -l "Tuning complete" tuning_by_task/logs/tune_*.out | wc -l

# Find failed jobs
for log in tuning_by_task/logs/tune_*.err; do
    if [ -s "$log" ]; then
        echo "ERROR in $(basename $log)"
        tail -10 "$log"
    fi
done
```

## Output Structure

```
tuning_by_task/
├── logs/
│   ├── tune_quvine_walks_erdos_renyi.out
│   ├── tune_quvine_walks_erdos_renyi.err
│   ├── tune_quvine_walks_erdos_renyi.sh
│   ├── tune_quvine_walks_modular.out
│   ├── ... (20 jobs × 3 files each)
│   ├── tune_aggregate.out
│   └── tune_aggregate.sh
├── erdos_renyi_quvine_walks_tuning_by_task.json
├── erdos_renyi_baseline_filter_heat_tuning_by_task.json
├── ... (20 individual result files)
├── erdos_renyi_tuning_by_task.json  # Aggregated
└── modular_tuning_by_task.json      # Aggregated
```

## Result Format

Each JSON file contains:

```json
{
  "method_name": {
    "node_classification": {
      "best_params": {...},
      "best_score": 0.85
    },
    "link_prediction": {
      "best_params": {...},
      "best_score": 0.92
    },
    "node_ranking": {
      "best_params": {...},
      "best_score": 0.78
    }
  }
}
```

## Adaptive Trial Counts

From `tuning_config.yaml`:

| Method | Trials | Reason |
|--------|--------|--------|
| quvine_walks | 100 | High complexity (10 params) |
| gat_baseline | 100 | High complexity (9 params) |
| graphgps_baseline | 120 | Highest complexity (10 params) |
| node2vec | 80 | Medium complexity (8 params) |
| graphsage | 80 | Medium complexity (8 params) |
| appnp | 80 | Medium complexity (8 params) |
| baseline_gcnmf | 60 | Medium complexity (6 params) |
| baseline_filter_poly | 40 | Simple (3 params) |
| netmf | 40 | Simple (4 params) |
| baseline_filter_heat | 30 | Simplest (3 params) |

## Estimated Runtime

Per job (single method × network):
- Simple methods (30-40 trials): ~2-4 hours
- Medium methods (60-80 trials): ~6-12 hours
- Complex methods (100-120 trials): ~12-24 hours

**Total wall time with parallelization**: ~24-48 hours (longest job)

**Total wall time without parallelization**: ~200-400 hours (sequential)

**Speedup**: ~10-20x faster with parallel execution

## Troubleshooting

### Job Failed to Submit

```bash
# Check LSF queue status
bqueues

# Check your quota
bhosts

# Verify paths in script
head -50 scripts/submit_tuning_jobs.sh
```

### Job Running Too Long

```bash
# Check if stuck
bjobs -l <JOB_ID>

# Kill and resubmit
bkill <JOB_ID>
bash scripts/submit_tuning_jobs.sh --methods quvine_walks --network-type erdos_renyi
```

### Out of Memory

```bash
# Increase memory and resubmit
bash scripts/submit_tuning_jobs.sh --memory 64
```

### Missing Results

```bash
# Check if job completed
grep "Tuning complete" tuning_by_task/logs/tune_*.out

# Check for errors
grep -i "error\|exception" tuning_by_task/logs/tune_*.err

# Resubmit specific job
# Edit submit_tuning_jobs.sh to only include failed method/network
```

## Advanced Usage

### Tune Subset of Methods

Edit `submit_tuning_jobs.sh` and modify the `METHODS` array:

```bash
METHODS=(
    "quvine_walks"
    "node2vec"
    "graphsage"
)
```

### Add More Network Types

Edit `submit_tuning_jobs.sh` and modify the `NETWORK_TYPES` array:

```bash
NETWORK_TYPES=("erdos_renyi" "modular" "scale_free")
```

Then update `tuning_config.yaml` to include the new network type configuration.

### Custom Hyperparameter Ranges

Edit `tuning_config.yaml` to modify search spaces:

```yaml
hyperparameters:
  quvine_walks:
    embedding_dim: [64, 128, 256]  # Reduce search space
    num_walks: [10, 20]            # Fewer options
```

## Integration with Main Pipeline

After tuning completes, use the best hyperparameters:

```python
import json

# Load tuned hyperparameters
with open('tuning_by_task/erdos_renyi_tuning_by_task.json') as f:
    tuned_params = json.load(f)

# Get best params for a method and task
method = 'quvine_walks'
task = 'node_classification'
best_params = tuned_params[method][task]['best_params']
best_score = tuned_params[method][task]['best_score']

print(f"Best {task} params for {method}:")
print(f"  Score: {best_score:.4f}")
print(f"  Params: {best_params}")
```

## Notes

- Each job is independent and can be resubmitted individually if it fails
- The aggregation job waits for all tuning jobs to complete
- Results are saved incrementally (each job saves its own JSON)
- The system uses adaptive trial counts based on method complexity
- Memory requirements vary by method (GNNs need more memory)