# APPNP Baseline Integration

This directory contains scripts for integrating APPNP (Approximate Personalized Propagation of Neural Predictions) as a baseline method into the QuVINE experimental results.

## Overview

APPNP is a graph neural network method that decouples feature transformation from propagation, using personalized PageRank for efficient message passing. These scripts run APPNP on all existing experimental graphs and integrate the results into the comprehensive_results.csv files.

## Files

### Core Scripts

1. **`run_appnp_single_network.py`** - Runs APPNP on a single network
   - Loads a graphml file
   - Runs APPNP baseline with full evaluation pipeline
   - Saves results to network-specific directory
   - Used by LSF job submission scripts

2. **`run_appnp_baseline.py`** - Sequential batch processor
   - Processes all graphml files in a directory sequentially
   - Useful for local execution or small batches
   - Includes resume mode to skip completed networks
   - Auto-regenerates comprehensive_results.csv

3. **`check_appnp_progress.py`** - Progress monitoring utility
   - Shows completion percentage
   - Lists completed/pending networks
   - Checks comprehensive_results.csv status

### LSF Submission Scripts

4. **`submit_appnp_hard_negatives.sh`** - LSF jobs for hard_negatives_v4
   - Submits one job per network (480 total)
   - Parallel execution on HPC cluster
   - Aggregation job regenerates comprehensive_results.csv

5. **`submit_appnp_ppi_disease.sh`** - LSF jobs for ppi_disease_v3
   - Submits one job per network (600 total)
   - Parallel execution on HPC cluster
   - Aggregation job regenerates comprehensive_results.csv

## Usage

### Option 1: Local/Sequential Execution

For local execution or when you don't have access to LSF:

```bash
# Run on hard_negatives_v4
cd QuVINE
../venv_quvine/bin/python scripts/run_appnp_baseline.py \
    --input-dir "/path/to/hard_negatives_v4" \
    --resume --verbose

# Run on ppi_disease_v3
../venv_quvine/bin/python scripts/run_appnp_baseline.py \
    --input-dir "/path/to/ppi_disease_v3" \
    --resume --verbose
```

**Pros:** Simple, no cluster setup needed
**Cons:** Sequential execution, takes ~20-30 minutes per directory

### Option 2: Parallel LSF Execution (Recommended)

For HPC cluster with LSF:

```bash
# Submit jobs for hard_negatives_v4
cd QuVINE
bash scripts/submit_appnp_hard_negatives.sh --resume

# Submit jobs for ppi_disease_v3
bash scripts/submit_appnp_ppi_disease.sh --resume

# Monitor progress
bjobs -u $USER

# Check completion status
../venv_quvine/bin/python scripts/check_appnp_progress.py \
    --input-dir "/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v4"
```

**Pros:** Parallel execution, much faster (~1-2 hours total)
**Cons:** Requires HPC cluster access

### LSF Submission Options

Both submission scripts support the following options:

```bash
--queue QUEUE        # LSF queue (default: normal)
--walltime TIME      # Wall time per job (default: 1:00 for HN, 2:00 for PPI)
--memory MEM         # Memory in GB (default: 4 for HN, 8 for PPI)
--python-env PATH    # Path to venv activate script
--resume             # Skip networks with APPNP already done
--dry-run            # Print scripts without submitting
```

Example with custom settings:

```bash
bash scripts/submit_appnp_hard_negatives.sh \
    --queue gpu \
    --walltime 2:00 \
    --memory 8 \
    --resume
```

## Monitoring Progress

### Check Completion Status

```bash
# Check hard_negatives_v4 progress
cd QuVINE
../venv_quvine/bin/python scripts/check_appnp_progress.py \
    --input-dir "/path/to/hard_negatives_v4"

# Check ppi_disease_v3 progress
../venv_quvine/bin/python scripts/check_appnp_progress.py \
    --input-dir "/path/to/ppi_disease_v3"

# Show pending networks
../venv_quvine/bin/python scripts/check_appnp_progress.py \
    --input-dir "/path/to/hard_negatives_v4" \
    --show-pending
```

### Monitor LSF Jobs

```bash
# View all your jobs
bjobs -u $USER

# View specific job details
bjobs -l JOB_ID

# View job output (while running)
bpeek JOB_ID

# Check aggregation job status
bjobs -J appnp_hn_aggregate
bjobs -J appnp_ppi_aggregate
```

## Workflow

### 1. Analysis Jobs (Parallel)

Each network gets its own LSF job that:
- Loads the existing graphml file
- Runs APPNP baseline with default hyperparameters
- Evaluates on all tasks (ranking, classification, link prediction)
- Saves results to network-specific directory

### 2. Aggregation Job (Sequential)

After all analysis jobs complete, one aggregation job:
- Collects results from all network directories
- Regenerates comprehensive_results.csv with APPNP included
- Runs automatically via LSF dependency (`ended()`)

## Output Structure

```
results/
├── NETWORK_ID_1/
│   ├── NETWORK_ID_1.graphml
│   ├── NETWORK_ID_1_appnp_embedding.npy
│   ├── NETWORK_ID_1_ranking_results.csv      (updated with APPNP)
│   ├── NETWORK_ID_1_classification_results.csv (updated with APPNP)
│   ├── NETWORK_ID_1_link_prediction_results.csv (updated with APPNP)
│   └── ...
├── NETWORK_ID_2/
│   └── ...
└── comprehensive_results.csv  (regenerated with APPNP rows)
```

## Expected Results

### hard_negatives_v4
- **Networks:** 480
- **APPNP rows added:** 480
- **Total rows in CSV:** ~7,530 (7,050 existing + 480 APPNP)
- **Estimated time:** 
  - Sequential: ~20 minutes
  - Parallel (LSF): ~1-2 hours (wall time)

### ppi_disease_v3
- **Networks:** 600
- **APPNP rows added:** 600
- **Total rows in CSV:** ~6,360 (5,760 existing + 600 APPNP)
- **Estimated time:**
  - Sequential: ~25 minutes
  - Parallel (LSF): ~1-2 hours (wall time)

## Verification

After completion, verify APPNP integration:

```bash
# Check that APPNP appears in comprehensive_results.csv
cd /path/to/results
head -1 comprehensive_results.csv | grep method
grep "^.*,appnp," comprehensive_results.csv | wc -l

# Should show 480 for hard_negatives_v4
# Should show 600 for ppi_disease_v3
```

## Troubleshooting

### Job Failures

If jobs fail, check error logs:
```bash
# View error log
cat /path/to/logs_appnp/appnp_NETWORK_ID.err

# Common issues:
# - Out of memory: increase --memory
# - Timeout: increase --walltime
# - Missing dependencies: check python environment
```

### Resume Mode

If jobs are interrupted, use `--resume` to skip completed networks:
```bash
bash scripts/submit_appnp_hard_negatives.sh --resume
```

The script checks for existing APPNP results in ranking_results.csv files.

### Manual Aggregation

If the aggregation job fails, run manually:
```bash
cd QuVINE
../venv_quvine/bin/python - << 'EOF'
import sys
sys.path.insert(0, 'src')
from quvine.comprehensive_embedding_analysis import collect_and_aggregate_results

df = collect_and_aggregate_results(
    results_dir='/path/to/results',
    output_file='comprehensive_results.csv',
    verbose=True
)
print(f"Regenerated with {len(df)} rows")
EOF
```

## Notes

- **Resume mode:** Always use `--resume` to avoid re-running completed networks
- **Memory:** APPNP is memory-efficient; 4-8GB is sufficient for most graphs
- **Time:** Each network takes ~1-3 seconds for embedding + evaluation
- **Dependencies:** Requires PyTorch, NetworkX, pandas, numpy, scikit-learn
- **Hyperparameters:** Uses default APPNP hyperparameters (can be tuned if needed)

## Contact

For issues or questions, refer to the main QuVINE documentation or contact the development team.