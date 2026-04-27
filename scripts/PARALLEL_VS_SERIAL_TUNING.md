# Parallel vs Serial Tuning Guide

## Overview

The `submit_tuning_jobs.sh` script now supports two modes:

1. **Parallel Mode (DEFAULT)**: Maximum parallelism - one job per method × network type
2. **Serial Mode**: All methods in one job per network type

## Usage Examples

### Parallel Mode (Default) - Recommended

Submit 20 parallel jobs (10 methods × 2 networks):

```bash
# Basic usage - submits 20 jobs
bash scripts/submit_tuning_jobs.sh

# With dry-run to preview
bash scripts/submit_tuning_jobs.sh --dry-run

# With custom settings
bash scripts/submit_tuning_jobs.sh \
    --queue normal \
    --walltime 48:00 \
    --memory 32 \
    --n-graphs 10
```

**Output:**
```
======================================================
 Hyperparameter Tuning Job Submission
======================================================
 Methods      : 10 (quvine_walks baseline_filter_heat ...)
 Networks     : 2 (erdos_renyi modular)
 Mode         : Parallel (one method per job)
 Total jobs   : 20
======================================================

  Submitted 12345: tune_quvine_walks_erdos_renyi
  Submitted 12346: tune_quvine_walks_modular
  Submitted 12347: tune_baseline_filter_heat_erdos_renyi
  ...
```

### Serial Mode - For Testing or Resource Constraints

Submit 2 serial jobs (1 per network, all methods sequential):

```bash
# Serial mode - submits 2 jobs
bash scripts/submit_tuning_jobs.sh --serial

# Serial with dry-run
bash scripts/submit_tuning_jobs.sh --serial --dry-run

# Serial with custom settings
bash scripts/submit_tuning_jobs.sh \
    --serial \
    --queue normal \
    --walltime 96:00 \
    --memory 64
```

**Output:**
```
======================================================
 Hyperparameter Tuning Job Submission
======================================================
 Methods      : 10 (quvine_walks baseline_filter_heat ...)
 Networks     : 2 (erdos_renyi modular)
 Mode         : Serial (all methods per job)
 Total jobs   : 2
======================================================

  Submitted 12345: tune_all_methods_erdos_renyi
  Submitted 12346: tune_all_methods_modular
```

## Comparison

| Aspect | Parallel Mode (Default) | Serial Mode (--serial) |
|--------|------------------------|------------------------|
| **Jobs** | 20 (10 methods × 2 networks) | 2 (1 per network) |
| **Execution** | All methods run simultaneously | Methods run sequentially |
| **Wall Time** | ~48 hours per method | ~480 hours total (10× longer) |
| **Fault Tolerance** | High - one method failure doesn't affect others | Low - one failure stops all |
| **Resource Usage** | 20 jobs × 32GB = 640GB total | 2 jobs × 32GB = 64GB total |
| **Completion Time** | Fast - all finish together | Slow - sequential execution |
| **Use Case** | Production runs | Testing, debugging, resource limits |

## When to Use Each Mode

### Use Parallel Mode (Default) When:
- ✅ Running production experiments
- ✅ You have access to multiple compute nodes
- ✅ You want results quickly
- ✅ You want fault tolerance (one method fails, others continue)
- ✅ You want to monitor progress per method

### Use Serial Mode (--serial) When:
- ✅ Testing the pipeline with limited resources
- ✅ Debugging a specific issue
- ✅ You have strict resource quotas
- ✅ You want to minimize the number of jobs in the queue
- ✅ Running on a single node with limited parallelism

## Monitoring Jobs

### Parallel Mode
```bash
# Check all jobs
bjobs -u $USER

# Check specific method
bjobs -u $USER | grep tune_quvine_walks

# Check by network
bjobs -u $USER | grep erdos_renyi
```

### Serial Mode
```bash
# Check all jobs (only 2)
bjobs -u $USER

# Check specific network
bjobs -u $USER | grep tune_all_methods_erdos_renyi
```

## Results Location

Both modes save results to the same location:
```
tuning_by_task/
├── erdos_renyi_quvine_walks_tuning_by_task.json
├── erdos_renyi_baseline_filter_heat_tuning_by_task.json
├── ...
├── modular_quvine_walks_tuning_by_task.json
├── ...
├── erdos_renyi_tuning_by_task.json  (aggregated)
└── modular_tuning_by_task.json      (aggregated)
```

## Recommendations

**For most users:** Use the default parallel mode. It's faster, more fault-tolerant, and provides better visibility into progress.

**For testing:** Use `--dry-run` first to preview what will be submitted:
```bash
bash scripts/submit_tuning_jobs.sh --dry-run
bash scripts/submit_tuning_jobs.sh --serial --dry-run
```

## All Available Options

```bash
bash scripts/submit_tuning_jobs.sh [OPTIONS]

Options:
  --serial        Run all methods in one job per network (default: parallel)
  --queue QUEUE   LSF queue name (default: normal)
  --walltime TIME Wall time limit (default: 48:00)
  --memory MEM    Memory in GB (default: 32)
  --n-graphs N    Number of graphs per trial (default: 10)
  --config FILE   Config file path (default: scripts/tuning_config.yaml)
  --python-env    Path to Python venv activate script
  --dry-run       Show what would be submitted without submitting
```

## Examples

```bash
# Quick test with fewer graphs
bash scripts/submit_tuning_jobs.sh --n-graphs 5 --dry-run

# Production run with more resources
bash scripts/submit_tuning_jobs.sh --memory 64 --walltime 72:00

# Serial mode for debugging
bash scripts/submit_tuning_jobs.sh --serial --n-graphs 3

# Custom config file
bash scripts/submit_tuning_jobs.sh --config my_custom_config.yaml