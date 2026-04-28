# Quick Start Guide - After Unified 12-Method Migration

## Overview

After the migration to unified 12-method configuration, here's how to submit hyperparameter tuning jobs.

---

## Prerequisites

1. **Verify you're on the HPC cluster** (with LSF job scheduler)
2. **Activate Python environment:**
   ```bash
   source ../Python-3.12.2/venv_quvine/bin/activate
   ```
3. **Navigate to project directory:**
   ```bash
   cd /path/to/QuVINE
   ```

---

## Option 1: Synthetic Networks (Recommended to Start)

### Quick Test - Single Method (1 job)
```bash
# Test single method on single network
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --methods quvine_rwr
```

**What this does:**
- Submits 1 job (1 method × 1 network)
- Tunes quvine_rwr on erdos_renyi for all 3 tasks
- Runtime: ~2-3 hours
- Output: `tuning_by_task/erdos_renyi_quvine_rwr_tuning_by_task.json`

### Quick Test - All Methods (12 jobs)
```bash
# Test all methods on single network
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi
```

**What this does:**
- Submits 12 jobs (12 methods × 1 network)
- Each job tunes 1 method on erdos_renyi for all 3 tasks
- Runtime: ~4-5 hours per job (longest: graphgps_baseline)
- Output: `tuning_by_task/erdos_renyi_METHOD_tuning_by_task.json`

### Default - Two Networks (24 jobs)
```bash
# Default: erdos_renyi and modular networks
bash scripts/submit_tuning_jobs.sh
```

**What this does:**
- Submits 24 jobs (12 methods × 2 networks)
- Runtime: ~4-5 hours per job
- Output: `tuning_by_task/NETWORK_METHOD_tuning_by_task.json`

### Full Synthetic Networks (16 networks, 192 jobs)
```bash
# All 16 network types
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular_strong,modular_medium,watts_strogatz_high_p,watts_strogatz_low_p,random_geometric,modular_many_communities,core_periphery,scale_free,powerlaw_cluster,stochastic_block_model,random_regular,heterophilic_sbm,degree_corrected_sbm,grid_torus,configuration_model
```

### Serial Mode (Fewer Jobs, Longer Runtime)
```bash
# 2 jobs instead of 24 (all methods sequential per network)
bash scripts/submit_tuning_jobs.sh --serial

# Or single network, all methods sequential (1 job)
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --serial
```

---

## Option 2: PPI Networks (Production)

### Quick Test (1 network, 1 disease, 12 jobs)
```bash
# Test on STRING network with asthma
# Test single method on STRING-asthma (1 job)
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma --methods quvine_rwr
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma
```

**What this does:**
- Submits 12 jobs (12 methods × 1 network × 1 disease)
- Each job tunes 1 method on STRING-asthma for all 3 tasks
- Runtime: ~4-5 hours per job
- Output: `tuning_by_task/STRING_asthma_METHOD_tuning_by_task.json`

### Full PPI Networks (5 networks, 3 diseases, 180 jobs)
```bash
# All 5 PPI networks × 3 diseases
bash scripts/submit_ppi_tuning_jobs.sh
```

**What this does:**
- Submits 180 jobs (12 methods × 5 networks × 3 diseases)
- Runtime: ~4-5 hours per job
- Total compute time: ~720-900 hours (parallelized)
- Output: `tuning_by_task/NETWORK_DISEASE_METHOD_tuning_by_task.json`

### Serial Mode (Fewer Jobs)
```bash
# 15 jobs instead of 180 (all methods sequential per network-disease pair)
bash scripts/submit_ppi_tuning_jobs.sh --serial
```

---

## Job Submission Options

### Common Options for Both Scripts

```bash
# Custom queue
bash scripts/submit_tuning_jobs.sh --queue gpu

# Custom walltime (hours:minutes)
bash scripts/submit_tuning_jobs.sh --walltime 96:00

# Custom memory (GB)
bash scripts/submit_tuning_jobs.sh --memory 64

# Dry run (see what would be submitted)
bash scripts/submit_tuning_jobs.sh --dry-run

# Custom config file
bash scripts/submit_tuning_jobs.sh --config scripts/my_custom_config.yaml
```

### Synthetic-Specific Options

```bash
# Custom number of graphs per trial
bash scripts/submit_tuning_jobs.sh --n-graphs 20

# Specific networks only
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular_strong
```

### PPI-Specific Options

```bash
# Custom number of subsampling replicates
bash scripts/submit_ppi_tuning_jobs.sh --n-replicates 5

# Specific networks only
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING,BioPlex3

# Specific diseases only
bash scripts/submit_ppi_tuning_jobs.sh --diseases asthma,autism

# Specific methods only
bash scripts/submit_ppi_tuning_jobs.sh --methods quvine_rwr,gat_baseline,node2vec
```

---

## Monitoring Jobs

### Check Job Status
```bash
# All your jobs
bjobs

# Specific job
bjobs JOB_ID

# Job details
bjobs -l JOB_ID

# Jobs by name pattern
bjobs -J "tune_*"
```

### Check Logs
```bash
# Logs are in tuning_by_task/logs/
ls -lh tuning_by_task/logs/

# View specific job output
tail -f tuning_by_task/logs/tune_quvine_rwr_erdos_renyi.out

# Check for errors
grep -i error tuning_by_task/logs/*.err
```

### Kill Jobs
```bash
# Kill specific job
bkill JOB_ID

# Kill all your tuning jobs
bkill -J "tune_*"
```

---

## Expected Output

### File Structure
```
tuning_by_task/
├── logs/
│   ├── tune_quvine_rwr_erdos_renyi.out
│   ├── tune_quvine_rwr_erdos_renyi.err
│   └── ...
├── erdos_renyi_quvine_rwr_tuning_by_task.json
├── erdos_renyi_gat_baseline_tuning_by_task.json
├── STRING_asthma_quvine_rwr_tuning_by_task.json
└── ...
```

### JSON Output Format
```json
{
  "quvine_rwr": {
    "node_classification": {
      "best_params": {
        "embedding_dim": 128,
        "num_walks": 40,
        "walk_length": 80,
        ...
      },
      "best_score": 0.85
    },
    "link_prediction": {...},
    "node_ranking": {...}
  }
}
```

---

## Recommended Workflow

### Step 1: Test on Small Scale
```bash
# Test 1 method on 1 network
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --methods quvine_rwr --dry-run

# If dry-run looks good, submit
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --methods quvine_rwr
```

### Step 2: Monitor First Job
```bash
# Wait for job to start
bjobs

# Check output
tail -f tuning_by_task/logs/tune_quvine_rwr_erdos_renyi.out

# Verify JSON output after completion
cat tuning_by_task/erdos_renyi_quvine_rwr_tuning_by_task.json | python -m json.tool
```

### Step 3: Scale Up Gradually
```bash
# If test successful, try 2 networks
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular_strong

# Then try all 12 methods on 2 networks (24 jobs)
bash scripts/submit_tuning_jobs.sh

# Finally, full scale (192 jobs for synthetic, 180 for PPI)
bash scripts/submit_tuning_jobs.sh --networks <all_16_networks>
bash scripts/submit_ppi_tuning_jobs.sh
```

---

## Troubleshooting

### Job Fails Immediately
```bash
# Check error log
cat tuning_by_task/logs/tune_METHOD_NETWORK.err

# Common issues:
# 1. Python environment not activated
# 2. Config file not found
# 3. Missing dependencies
```

### Job Runs But No Output
```bash
# Check if job is still running
bjobs JOB_ID

# Check output log for progress
tail -f tuning_by_task/logs/tune_METHOD_NETWORK.out

# Look for "Tuning complete" message
```

### Out of Memory
```bash
# Increase memory allocation
bash scripts/submit_tuning_jobs.sh --memory 64

# Or reduce n-graphs/n-replicates
bash scripts/submit_tuning_jobs.sh --n-graphs 5
```

### Jobs Taking Too Long
```bash
# Check trial counts in config
cat scripts/unified_tuning_config.yaml | grep -A 20 "method_trials"

# Reduce trials for testing
# Edit unified_tuning_config.yaml and reduce trial counts
```

---

## After Jobs Complete

### Aggregate Results
The aggregation job runs automatically after all tuning jobs complete.

### Verify Results
```bash
# Check all output files exist
ls -lh tuning_by_task/*_tuning_by_task.json

# Count methods in each file
for f in tuning_by_task/*_tuning_by_task.json; do
    echo "$f: $(python -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo "error") methods"
done
```

### Use Tuned Hyperparameters
The tuned hyperparameters are automatically used by:
- `scripts/tune_hyperparameters.py` (via METHOD_TUNING_MAP)
- Main analysis pipeline
- All 39 methods receive tuned params

---

## Quick Reference

### Synthetic Networks (Default)
```bash
# Quick test (24 jobs, ~2 hours)
bash scripts/submit_tuning_jobs.sh

# Full scale (192 jobs, ~4-5 hours)
bash scripts/submit_tuning_jobs.sh --networks <all_16>
```

### PPI Networks
```bash
# Quick test (12 jobs, ~4-5 hours)
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma

# Full scale (180 jobs, ~4-5 hours)
bash scripts/submit_ppi_tuning_jobs.sh
```

### Monitor
```bash
bjobs                    # Check status
tail -f tuning_by_task/logs/*.out  # Watch progress
```

### Results
```bash
ls tuning_by_task/*_tuning_by_task.json  # List outputs
```

---

## Need Help?

1. **Check logs:** `tuning_by_task/logs/`
2. **Verify config:** `cat scripts/unified_tuning_config.yaml`
3. **Test locally:** `python scripts/tune_by_task_with_config.py --help`
4. **Dry run:** Add `--dry-run` flag to any submission command

---

**Ready to submit? Start with the quick test!**

```bash
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --methods quvine_rwr