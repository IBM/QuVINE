# Topological Complexity Computation for PPI Networks

This directory contains scripts to compute topological complexity metrics (Betti numbers and persistence entropy) for all 560 graphs in the ppi_disease_v3 dataset and integrate them into the comprehensive results CSV.

## Files

- `add_topological_complexity_ppi_complete.py` - Main computation script
- `merge_topological_to_comprehensive.py` - Merges topological metrics into comprehensive results
- `submit_topological_complexity.sh` - LSF submission script (runs both scripts)
- `check_topological_stats.py` - Check progress and statistics
- `test_topology_debug.py` - Debug/test script

## Metrics Computed

For each graph, the following 8 topological metrics are computed:

1. **betti_0** - Number of connected components at filtration scale ε=1.0
2. **betti_1** - Number of independent cycles (1-dimensional holes)
3. **betti_2** - Number of voids (2-dimensional holes)
4. **betti_sum** - Sum of all Betti numbers (β₀ + β₁ + β₂)
5. **euler_characteristic** - Topological invariant (β₀ - β₁ + β₂)
6. **persistence_entropy_H0** - Entropy of H0 persistence diagram
7. **persistence_entropy_H1** - Entropy of H1 persistence diagram
8. **persistence_entropy_H2** - Entropy of H2 persistence diagram

## Requirements

- Python 3.9+
- ripser package: `pip install ripser`
- Other dependencies: networkx, pandas, joblib, numpy, scipy

## Running on LSF Cluster

### Step 1: Prepare the environment

```bash
cd /dccstor/boseukb/Q/NetMed/QuVINE

# Make sure ripser is installed in your environment
pip install ripser

# Create logs directory if it doesn't exist
mkdir -p logs
```

### Step 2: Submit the job

```bash
bsub < scripts/submit_topological_complexity.sh
```

The job will automatically:
1. Compute topological metrics for all 560 graphs (using 32 cores)
2. Update individual `*_complexity.csv` files with new columns
3. Merge topological metrics into comprehensive results CSV
4. Generate `comprehensive_results_ppi3_with_topology.csv`

### Step 3: Monitor progress

```bash
# Check job status
bjobs

# View live output
tail -f logs/topo_complexity_<JOBID>.out

# Check how many graphs processed
wc -l /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/topological_checkpoint.csv
```

### Step 4: Check results

After completion, the output file will be at:
```
/dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/comprehensive_results_ppi3_with_topology.csv
```

## LSF Job Configuration

- **Cores**: 32 (adjustable with --n-jobs parameter)
- **Memory**: 8GB per core (256GB total)
- **Time limit**: 12 hours
- **Queue**: normal

## Output Files

1. **Checkpoint file**: `ppi_disease_v3/results/topological_checkpoint.csv`
   - Contains topological metrics for all processed graphs
   - Used for resuming if job is interrupted

2. **Updated complexity files**: `ppi_disease_v3/results/[network_id]/[network_id]_complexity.csv`
   - Each file now has 8 additional columns with topological metrics

3. **Comprehensive results with topology**: `ppi_disease_v3/results/comprehensive_results_ppi3_with_topology.csv`
   - Original comprehensive results + 8 new topological metric columns
   - Ready for analysis and visualization

## Workflow

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Compute Topological Metrics (32 cores, ~1-2 hours) │
│  - Processes 560 graphs in parallel                         │
│  - Saves checkpoint every 50 graphs                         │
│  - Updates individual *_complexity.csv files                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Merge into Comprehensive Results (~1 minute)        │
│  - Reads topological metrics from complexity files          │
│  - Merges with comprehensive_results_ppi3.csv               │
│  - Creates comprehensive_results_ppi3_with_topology.csv     │
└─────────────────────────────────────────────────────────────┘
```

## Resuming from Checkpoint

The script automatically resumes from the last checkpoint if interrupted. To start fresh:

```bash
rm /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/topological_checkpoint.csv
```

## Manual Execution

If you want to run the steps separately:

```bash
# Step 1: Compute topological metrics
python scripts/add_topological_complexity_ppi_complete.py --n-jobs 32

# Step 2: Merge into comprehensive results
python scripts/merge_topological_to_comprehensive.py
```

## Troubleshooting

### If all values are zero

This means ripser is not installed or not accessible. Install it:

```bash
pip install ripser
python -c "import ripser; print(ripser.__version__)"
```

### If job runs out of memory

Reduce the number of workers:

```bash
# Edit submit_topological_complexity.sh
python scripts/add_topological_complexity_ppi_complete.py --n-jobs 16
```

### If job times out

The script saves checkpoints every 50 graphs. Simply resubmit the job and it will resume.

### If merge fails

Check that topological metrics were computed:

```bash
# Check a sample complexity file
head -2 /dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/BioPlex3_asthma_rep00/BioPlex3_asthma_rep00_complexity.csv
```

Should show columns: `betti_0`, `betti_1`, `betti_2`, etc.

## Performance

- **Per graph**: ~10-60 seconds (depends on size and complexity)
- **Total computation time**: ~1-2 hours with 32 cores
- **Merge time**: ~1 minute
- **Memory usage**: ~5-8 GB total

## Expected Results

After successful completion:
- 560 graphs processed
- 8 new columns in comprehensive results
- Non-zero Betti numbers for most graphs
- Summary statistics displayed in job output

## Notes

- The computation uses persistent homology via Ripser
- Disconnected graphs are handled correctly
- The filtration scale is set to ε=1.0 (one hop = one edge)
- Threading backend is used to preserve Python environment
- All metrics are computed on the full graph (not just largest component)
