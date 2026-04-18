# Topological Complexity Computation for PPI Networks

This directory contains scripts for computing topological complexity metrics (Betti numbers and persistence entropy) for all 560 graphs in the `ppi_disease_v3` dataset using parallel LSF cluster jobs.

## Overview

The workflow computes 8 topological complexity measures for each network:
- **Betti numbers**: β₀ (connected components), β₁ (cycles), β₂ (voids)
- **Derived metrics**: Betti sum, Euler characteristic
- **Persistence entropy**: H₀, H₁, H₂ (entropy of persistence diagrams)

These metrics are computed using persistent homology via the Ripser library on Vietoris-Rips filtrations at scale ε=1.0.

## Files

### Core Scripts
- **`compute_single_network_topology.py`**: Processes one network at a time
  - Loads graphml file
  - Computes 8 topological metrics
  - Updates the network's complexity CSV file
  
- **`merge_topological_to_comprehensive.py`**: Aggregates results
  - Collects topological metrics from all complexity files
  - Merges with `comprehensive_results_ppi3.csv`
  - Creates `comprehensive_results_ppi3_with_topology.csv`

### Submission Scripts
- **`submit_topological_complexity_jobs.sh`**: Main parallel submission script
  - Submits 560 independent LSF jobs (one per network)
  - Each job runs `compute_single_network_topology.py`
  - Final aggregation job depends on all computation jobs
  - Follows the pattern from `submit_ppi_disease_jobs_v3.sh`

### Legacy Scripts (for reference)
- **`add_topological_complexity_ppi_complete.py`**: Original sequential script
  - Processes all networks in a single job with joblib parallelization
  - Useful for local testing with `--n-jobs 4`
  
- **`submit_topological_complexity.sh`**: Original single-job submission
  - Replaced by parallel approach for better cluster utilization

## Usage

### On LSF Cluster (Recommended)

```bash
# Navigate to project directory
cd /dccstor/boseukb/Q/NetMed/quvine

# Submit 560 parallel jobs + aggregation job
bash scripts/submit_topological_complexity_jobs.sh

# With custom options
bash scripts/submit_topological_complexity_jobs.sh \
    --queue normal \
    --walltime 1:00 \
    --memory 8 \
    --python-env ../Python-3.12.2/venv_quvine/bin/activate

# Dry run (preview without submitting)
bash scripts/submit_topological_complexity_jobs.sh --dry-run

# Monitor jobs
bjobs -u $USER
bjobs -u $USER | grep topo_

# Check aggregation job
bjobs -u $USER | grep topo_aggregate
```

### Local Testing

```bash
# Test on a single network
cd QuVINE
source ../venv_quvine/bin/activate
pip install ripser

python scripts/compute_single_network_topology.py \
    --graphml /path/to/network.graphml \
    --network-id BioPlex3_asthma_rep_0 \
    --output-csv /path/to/BioPlex3_asthma_rep_0_complexity.csv

# Process all networks sequentially (slow, for testing only)
python scripts/add_topological_complexity_ppi_complete.py \
    --n-jobs 4 \
    --reset-checkpoint
```

## Workflow Details

### Step 1: Parallel Computation (560 jobs)
Each job:
1. Loads one graphml file
2. Computes hop-count distance matrix
3. Runs Ripser for persistent homology
4. Computes Betti numbers and persistence entropy
5. Updates the network's `*_complexity.csv` file

**Resources per job:**
- Queue: `normal`
- Wall time: 1 hour
- Memory: 8GB
- Expected runtime: 5-30 minutes per network

### Step 2: Aggregation (1 job)
After all 560 jobs complete:
1. Reads topological metrics from all complexity files
2. Merges with `comprehensive_results_ppi3.csv`
3. Creates `comprehensive_results_ppi3_with_topology.csv`
4. Prints summary statistics

**Resources:**
- Queue: `normal`
- Wall time: 30 minutes
- Memory: 16GB

## Output

### Individual Network Files
Each network's complexity CSV is updated with 8 new columns:
```
/dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/
  ├── BioPlex3_asthma_rep_0/
  │   └── BioPlex3_asthma_rep_0_complexity.csv  (updated)
  ├── BioPlex3_asthma_rep_1/
  │   └── BioPlex3_asthma_rep_1_complexity.csv  (updated)
  ...
```

### Comprehensive Results
Final merged file with all metrics:
```
/dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/
  └── comprehensive_results_ppi3_with_topology.csv
```

### Logs
Job logs stored in:
```
/dccstor/boseukb/Q/NetMed/quvine/ppi_disease_v3/results/logs_topology/
  ├── topo_BioPlex3_asthma_rep_0.out
  ├── topo_BioPlex3_asthma_rep_0.err
  ├── topo_BioPlex3_asthma_rep_0.sh
  ...
  ├── topo_aggregate.out
  ├── topo_aggregate.err
  └── topo_aggregate.sh
```

## Troubleshooting

### Issue: Jobs fail with "ripser not found"
**Solution**: The submission script automatically installs ripser in each job. If this fails, manually install:
```bash
source ../Python-3.12.2/venv_quvine/bin/activate
pip install ripser
```

### Issue: All Betti numbers are 0
**Causes**:
1. Wrong Python environment (not using venv_quvine)
2. Ripser not installed
3. Graph is empty or invalid

**Solution**: Check job logs in `logs_topology/*.err` for errors

### Issue: Aggregation job fails
**Causes**:
1. Some computation jobs failed
2. Complexity CSV files missing topological columns

**Solution**: 
```bash
# Check which jobs failed
bjobs -u $USER | grep EXIT

# Resubmit failed jobs manually
bsub < /dccstor/.../logs_topology/topo_NETWORK_ID.sh

# Or rerun entire workflow
bash scripts/submit_topological_complexity_jobs.sh
```

### Issue: Memory errors
**Solution**: Increase memory allocation:
```bash
bash scripts/submit_topological_complexity_jobs.sh --memory 16
```

## Technical Details

### Persistent Homology
- **Library**: Ripser (fast C++ implementation)
- **Filtration**: Vietoris-Rips complex
- **Distance metric**: Hop-count (shortest path) distance
- **Scale**: ε = 1.0 (captures local topology)
- **Dimensions**: H₀, H₁, H₂ (up to 2-dimensional holes)

### Betti Numbers
- **β₀**: Number of connected components
- **β₁**: Number of 1-dimensional cycles (loops)
- **β₂**: Number of 2-dimensional voids (cavities)
- **Betti sum**: β₀ + β₁ + β₂
- **Euler characteristic**: β₀ - β₁ + β₂

### Persistence Entropy
Measures the "complexity" of the persistence diagram:
```
H = -Σ (pᵢ/L) log(pᵢ/L)
```
where pᵢ is the persistence (death - birth) of feature i, and L is the total persistence.

Higher entropy → more diverse topological features
Lower entropy → dominated by few features

## Performance

### Expected Runtimes
- **Small networks** (<1000 nodes): 5-10 minutes
- **Medium networks** (1000-5000 nodes): 10-20 minutes
- **Large networks** (>5000 nodes): 20-30 minutes

### Cluster Utilization
- **560 parallel jobs**: ~1 hour total (vs. 3-4 hours sequential)
- **Aggregation**: ~5 minutes
- **Total workflow**: ~1.5 hours including queue time

## References

1. **Ripser**: Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. Journal of Applied and Computational Topology, 5(3), 391-423.

2. **Persistent Homology**: Edelsbrunner, H., & Harer, J. (2010). Computational topology: an introduction. American Mathematical Society.

3. **Persistence Entropy**: Rucco, M., et al. (2016). Characterisation of the idiotypic immune network through persistent entropy. In Complex Systems (pp. 117-128).

## Contact

For questions or issues, contact the QuVINE development team.
