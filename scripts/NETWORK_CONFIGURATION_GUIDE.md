# Network Configuration Guide for Hyperparameter Tuning

## Overview

You can configure which network types to use for hyperparameter tuning in three ways:

1. **Command-line flag** (highest priority) - `--networks`
2. **Submission script** (medium priority) - Edit `NETWORK_TYPES` array
3. **Config file** (lowest priority) - Edit `tuning_config.yaml`

## Method 1: Command-Line Flag (Recommended)

The easiest way to specify networks is using the `--networks` flag:

```bash
# Single network
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi

# Multiple networks (comma-separated, no spaces)
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular

# Three networks
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular,scale_free
```

### Examples

```bash
# Test on just erdos_renyi - 10 jobs (10 methods × 1 network)
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi --dry-run

# Production run on both default networks - 20 jobs
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular

# Extended testing on 3 networks - 30 jobs
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular,scale_free

# Serial mode with custom networks - 3 jobs
bash scripts/submit_tuning_jobs.sh --serial --networks erdos_renyi,modular,watts_strogatz
```

## Method 2: Edit Submission Script

For persistent changes, edit the `NETWORK_TYPES` array in `submit_tuning_jobs.sh`:

```bash
# Open the script
vim scripts/submit_tuning_jobs.sh

# Find line ~66 and modify:
NETWORK_TYPES=("erdos_renyi" "modular")

# Change to your desired networks:
NETWORK_TYPES=("erdos_renyi" "modular" "scale_free" "watts_strogatz")
```

Then run without the `--networks` flag:
```bash
bash scripts/submit_tuning_jobs.sh
```

## Method 3: Edit Config File

For project-wide defaults, edit `tuning_config.yaml`:

```bash
# Open the config
vim scripts/tuning_config.yaml

# Find the experiment section (line ~181) and modify:
experiment:
  n_graphs: 10
  network_types: ["erdos_renyi", "modular"]  # ← Change this
  output_dir: "tuning_by_task"
```

Change to:
```yaml
experiment:
  network_types: ["erdos_renyi", "modular", "scale_free"]
```

**Note:** The Python script reads from the config file, but the submission script overrides it with its own `NETWORK_TYPES` array. So editing the config alone won't change the submission script's behavior unless you also modify the script.

## Available Network Types

The following network types are supported (must match the Python script's network generation functions):

### Currently Implemented
- `erdos_renyi` - Random graph with uniform edge probability
- `modular` - Community structure with intra/inter-community edges

### Potentially Available (check `quvine/data/random_graphs.py`)
- `scale_free` - Barabási-Albert preferential attachment
- `watts_strogatz` - Small-world networks
- `powerlaw_cluster` - Power-law with clustering
- `random_geometric` - Geometric random graph
- `sbm_assortative` - Stochastic block model

**Important:** Make sure the network type you specify has a corresponding generation function in the codebase!

## Job Count Calculation

Total jobs = N_METHODS × N_NETWORKS (in parallel mode)

Examples:
- 10 methods × 1 network = **10 jobs**
- 10 methods × 2 networks = **20 jobs** (default)
- 10 methods × 3 networks = **30 jobs**
- 10 methods × 5 networks = **50 jobs**

In serial mode:
- Total jobs = N_NETWORKS (all methods run sequentially per network)

## Complete Examples

### Quick Test on Single Network
```bash
# Test with just erdos_renyi, fewer graphs
bash scripts/submit_tuning_jobs.sh \
    --networks erdos_renyi \
    --n-graphs 5 \
    --dry-run
```

### Production Run on Default Networks
```bash
# 20 jobs: 10 methods × 2 networks
bash scripts/submit_tuning_jobs.sh \
    --networks erdos_renyi,modular \
    --n-graphs 10 \
    --memory 32 \
    --walltime 48:00
```

### Extended Testing on Multiple Networks
```bash
# 30 jobs: 10 methods × 3 networks
bash scripts/submit_tuning_jobs.sh \
    --networks erdos_renyi,modular,scale_free \
    --n-graphs 15 \
    --memory 64 \
    --walltime 72:00
```

### Serial Mode with Custom Networks
```bash
# 3 jobs: 1 per network, all methods sequential
bash scripts/submit_tuning_jobs.sh \
    --serial \
    --networks erdos_renyi,modular,watts_strogatz \
    --walltime 96:00
```

## Verification

Always use `--dry-run` first to verify the configuration:

```bash
bash scripts/submit_tuning_jobs.sh --networks erdos_renyi,modular --dry-run
```

Output will show:
```
======================================================
 Hyperparameter Tuning Job Submission
======================================================
 Networks     : 2 (erdos_renyi modular)
 Mode         : Parallel (one method per job)
 Total jobs   : 20
======================================================
```

## Troubleshooting

### Network Type Not Found
If you get an error about a network type not being found:
1. Check the spelling matches exactly (case-sensitive)
2. Verify the network generation function exists in `quvine/data/random_graphs.py`
3. Make sure there are no spaces in the comma-separated list

### Wrong Number of Jobs
If you're getting unexpected job counts:
1. Check `--dry-run` output to see what's being submitted
2. Verify the `NETWORK_TYPES` array in the script matches your expectation
3. Remember: parallel mode = methods × networks, serial mode = networks only

### Config File Not Being Used
The submission script has its own `NETWORK_TYPES` array that overrides the config file. To use the config file's networks, you would need to modify the submission script to read from the config instead of using the hardcoded array.

## Recommendations

**For most users:**
- Use `--networks` flag for flexibility
- Start with `--dry-run` to verify
- Use default networks (erdos_renyi, modular) for initial testing

**For production:**
- Test on 1-2 networks first
- Scale up to more networks once validated
- Monitor resource usage and adjust memory/walltime accordingly

**For development:**
- Use `--networks erdos_renyi` for quick iteration
- Add `--n-graphs 3` to speed up testing
- Always use `--dry-run` before submitting