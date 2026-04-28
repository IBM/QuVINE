# Hugging Face Graph Dataset Job Submission

This script submits LSF jobs to process graph datasets generated from the Hugging Face notebook preprocessing pipeline.

## Overview

The `submit_huggingface_jobs.sh` script:
- Auto-discovers all CSV/JSON pairs in the data directory
- Generates random seeds and targets (10% each by default)
- Runs comprehensive embedding analysis with all QuVINE methods
- Uses tuned hyperparameters when available
- Submits an aggregation job to collect all results

## Data Location

**Data Directory:** `/dccstor/cgq4hls/Q/hugging-graph-processed-data/huggingface_graph_samples/`

Expected file structure:
```
huggingface_graph_samples/
├── dataset1_n5000_rep0.csv      # Edge list
├── dataset1_n5000_rep0.json     # Metadata
├── dataset1_n10000_rep0.csv
├── dataset1_n10000_rep0.json
└── ...
```

## Usage

### Basic Usage

```bash
# Submit all jobs with default settings
bash scripts/submit_huggingface_jobs.sh
```

### Dry Run (Recommended First)

```bash
# Preview what will be submitted without actually submitting
bash scripts/submit_huggingface_jobs.sh --dry-run
```

### Custom Parameters

```bash
# Custom queue and resources
bash scripts/submit_huggingface_jobs.sh \
    --queue normal \
    --walltime 72:00 \
    --memory 12

# Custom seed/target percentages
bash scripts/submit_huggingface_jobs.sh \
    --seed-pct 15 \
    --target-pct 15

# Custom data and output directories
bash scripts/submit_huggingface_jobs.sh \
    --data-dir /path/to/graphs \
    --output-dir /path/to/results
```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--queue` | `normal` | LSF queue name |
| `--walltime` | `72:00` | Wall time (HH:MM format) |
| `--memory` | `12` | Memory in GB |
| `--seed-pct` | `10` | Percentage of nodes to use as seeds |
| `--target-pct` | `10` | Percentage of nodes to use as targets |
| `--data-dir` | `/dccstor/cgq4hls/Q/hugging-graph-processed-data/huggingface_graph_samples/` | Input data directory |
| `--output-dir` | `/dccstor/cgq4hls/Q/hugging-graph-processed-data/results` | Output results directory |
| `--python-env` | `/u/futro/envs/py311/bin/activate` | Python virtual environment path |
| `--dry-run` | `false` | Preview without submitting |

## Embedding Methods

The script runs all available embedding methods:
- **QuVINE methods:**
  - `quvine_fused` - Fused quantum walks
  - `quvine_ctqw` - Continuous-time quantum walk
  - `quvine_dtqw` - Discrete-time quantum walk
  - `quvine_rwr` - Random walk with restart
  - `quvine_heat` - Heat kernel
  - `quvine_poly` - Polynomial filter
  - `quvine_hgcnmf` - Hadamard GCN matrix factorization
  - `quvine_pgcnmf` - Parameterized GCN matrix factorization
- **Baseline methods:**
  - `netmf` - NetMF
  - `node2vec` - Node2Vec
  - `baseline_gcnmf` - GCN matrix factorization
  - `baseline_filter` - Filter baseline
  - `graphsage` - GraphSAGE

## Output Structure

Results are saved to:
```
/dccstor/cgq4hls/Q/hugging-graph-processed-data/results/
├── logs/                           # Job logs
│   ├── hf_dataset1_n5000_rep0.out
│   ├── hf_dataset1_n5000_rep0.err
│   └── ...
├── results/                        # Analysis results
│   ├── dataset1_n5000_rep0/
│   │   ├── embeddings/
│   │   ├── evaluation/
│   │   └── metadata.json
│   ├── dataset1_n10000_rep0/
│   └── comprehensive_results.csv   # Aggregated results
└── visualizations/                 # Plots and figures
```

## Monitoring Jobs

```bash
# Check job status
bjobs -u $USER

# Check specific job
bjobs <job_id>

# View job output (while running)
bpeek <job_id>

# View completed job logs
cat /dccstor/cgq4hls/Q/hugging-graph-processed-data/results/logs/hf_*.out
```

## Seeds and Targets

The script automatically generates:
- **Seeds:** Random 10% of nodes (used as starting points for walks)
- **Targets:** Random 10% of nodes (used for evaluation, non-overlapping with seeds)
- **Negatives:** Remaining nodes (used for negative sampling in evaluation)

You can adjust these percentages with `--seed-pct` and `--target-pct`.

## Hyperparameters

The script attempts to use tuned hyperparameters from:
1. `/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/real_STRING/best_hyperparams.json`
2. `/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/real_BioPlex3/best_hyperparams.json`

If not found, it uses default parameters from the QuVINE configuration.

**View constraints** are automatically set to:
- `max_nodes: 250`
- `max_edges: 5000`

This ensures quantum walks have sufficient neighbors in the views.

## Aggregation Job

After all analysis jobs complete, an aggregation job automatically:
1. Collects results from all graphs
2. Generates comprehensive CSV with all metrics
3. Creates visualizations comparing methods
4. Saves to `results/comprehensive_results.csv`

## Troubleshooting

### No CSV files found
- Check that the data directory path is correct
- Verify files exist: `ls /dccstor/cgq4hls/Q/hugging-graph-processed-data/huggingface_graph_samples/*.csv`

### Job fails immediately
- Check logs in `results/logs/hf_*.err`
- Verify Python environment is activated correctly
- Ensure all dependencies are installed

### Out of memory
- Increase memory: `--memory 16` or `--memory 24`
- Reduce graph size in the preprocessing notebook

### Jobs pending too long
- Try different queue: `--queue short` or `--queue long`
- Check queue status: `bqueues`

## Example Workflow

```bash
# 1. Preview what will be submitted
bash scripts/submit_huggingface_jobs.sh --dry-run

# 2. Submit with custom parameters
bash scripts/submit_huggingface_jobs.sh \
    --queue normal \
    --walltime 48:00 \
    --memory 12 \
    --seed-pct 10 \
    --target-pct 10

# 3. Monitor progress
watch -n 60 bjobs -u $USER

# 4. Check results when complete
ls /dccstor/cgq4hls/Q/hugging-graph-processed-data/results/results/

# 5. View aggregated results
cat /dccstor/cgq4hls/Q/hugging-graph-processed-data/results/results/comprehensive_results.csv
```

## Notes

- Each graph becomes one LSF job
- Jobs run independently and can be parallelized
- The aggregation job waits for all analysis jobs to complete
- Results include embeddings, evaluation metrics, and metadata
- Compatible with graphs of various sizes (5k-20k nodes typical)