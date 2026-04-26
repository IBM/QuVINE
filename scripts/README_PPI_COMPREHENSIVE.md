# PPI Comprehensive Job Submission Infrastructure

## Overview

This infrastructure submits and aggregates results for comprehensive PPI network analysis across:
- **5 PPI networks**: BioPlex3, HumanNet, ProteomeHD, STRING, PCNet
- **3 diseases**: asthma, autism, schizophrenia
- **30 replicates** per (network, disease) combination
- **Total**: 450 analysis jobs

Each job runs the full QuVINE pipeline with all 33 embedding methods and all tasks (ranking, classification, link prediction).

## Files

### 1. `submit_ppi_comprehensive_jobs.sh`
Main LSF job submission script that:
- Submits 450 analysis jobs (5 networks × 3 diseases × 30 reps)
- Loads network-specific hyperparameters from tuning results
- Handles disease seeds and GWAS targets
- Subsamples networks to max_nodes while preserving seeds/targets
- Runs `run_single_network_analysis` for each replicate
- Submits aggregation job with dependency on all analysis jobs
- Submits embedding packaging job with dependency on aggregation

### 2. `aggregate_ppi_comprehensive.py`
Python script that:
- Collects results from all network directories
- Aggregates by (network, disease) combination
- Merges complexity metrics from individual network CSVs
- Creates per-(network, disease) aggregated CSVs
- Creates comprehensive CSV with all results combined
- Prints summary statistics

### 3. `package_embeddings_to_npz.py`
Python script that:
- Packages individual `.npy` embedding files into `.npz` archives
- One `.npz` file per network containing all methods
- Enables easy reloading: `data = np.load(file); emb = data['method']`

## Usage

### Basic Usage (30 replicates, default settings)

```bash
cd QuVINE
bash scripts/submit_ppi_comprehensive_jobs.sh
```

This submits 450 jobs with default settings:
- Queue: `normal`
- Wall time: `72:00` (72 hours)
- Memory: `12GB`
- Max nodes: `4000`
- Replicates: `30`
- All 33 embedding methods

### Custom Settings

```bash
bash scripts/submit_ppi_comprehensive_jobs.sh \
    --queue priority \
    --walltime 48:00 \
    --memory 16 \
    --max-nodes 5000 \
    --n-replicates 50 \
    --methods "quvine_fused,quvine_ctqw,quvine_dtqw,node2vec,netmf"
```

### Dry Run (test without submitting)

```bash
bash scripts/submit_ppi_comprehensive_jobs.sh --dry-run --n-replicates 2
```

### Available Options

- `--queue QUEUE`: LSF queue (default: `normal`)
- `--walltime TIME`: Wall time per job (default: `72:00`)
- `--memory MEM`: Memory in GB (default: `12`)
- `--max-nodes N`: Max nodes per subgraph (default: `4000`)
- `--n-replicates N`: Number of replicates (default: `30`)
- `--methods METHODS`: Comma-separated method list (default: all 33)
- `--python-env PATH`: Path to venv activate script
- `--dry-run`: Print jobs without submitting

## Output Structure

```
/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_comprehensive/
├── logs/
│   ├── ppi_BioPlex3_asthma_rep00.sh
│   ├── ppi_BioPlex3_asthma_rep00.out
│   ├── ppi_BioPlex3_asthma_rep00.err
│   ├── ... (450 job scripts + logs)
│   ├── ppi_aggregate.sh
│   ├── ppi_aggregate.out
│   └── ppi_package_embeddings.out
├── results/
│   ├── BioPlex3_asthma_rep00/
│   │   ├── BioPlex3_asthma_rep00.graphml
│   │   ├── BioPlex3_asthma_rep00_complexity.csv
│   │   ├── BioPlex3_asthma_rep00_ranking_results.csv
│   │   ├── BioPlex3_asthma_rep00_classification_results.csv
│   │   ├── BioPlex3_asthma_rep00_link_prediction_results.csv
│   │   ├── BioPlex3_asthma_rep00_ranking_detailed.csv
│   │   ├── BioPlex3_asthma_rep00_classification_detailed.csv
│   │   ├── BioPlex3_asthma_rep00_link_prediction_detailed.csv
│   │   ├── BioPlex3_asthma_rep00_quvine_fused_embedding.npy
│   │   ├── BioPlex3_asthma_rep00_quvine_ctqw_embedding.npy
│   │   ├── ... (33 embedding .npy files)
│   │   └── BioPlex3_asthma_rep00_embeddings.npz
│   ├── BioPlex3_asthma_rep01/
│   ├── ... (450 network directories)
│   ├── BioPlex3_asthma_aggregated.csv
│   ├── BioPlex3_autism_aggregated.csv
│   ├── ... (15 per-(network,disease) aggregated CSVs)
│   └── ppi_comprehensive_results.csv
```

## Data Persistence

Each network analysis saves:

1. **GraphML file**: Network structure with metadata
2. **Complexity CSV**: Graph complexity metrics (from `graph_enhanced.py` and `qbc.py`)
3. **Task result CSVs**: 6 CSV files (3 tasks × 2 formats: summary + detailed)
4. **Individual embeddings**: 33 `.npy` files (one per method)
5. **Packaged embeddings**: Single `.npz` archive with all methods

## Aggregation

After all jobs complete, the aggregation job:

1. Collects results from all 450 network directories
2. Groups by (network, disease) combination
3. Merges complexity metrics from first replicate
4. Creates 15 per-(network, disease) aggregated CSVs
5. Creates comprehensive CSV with all results
6. Prints summary statistics

## Embedding Packaging

After aggregation, the packaging job:

1. Finds all individual `.npy` embedding files
2. Packages them into `.npz` archives per network
3. Enables easy reloading without recomputation

Example usage:
```python
import numpy as np

# Load all embeddings for a network
data = np.load('BioPlex3_asthma_rep00_embeddings.npz')

# Access specific method
quvine_emb = data['quvine_fused']
node2vec_emb = data['node2vec']

# List all methods
print(data.files)
```

## Monitoring

```bash
# Check job status
bjobs -w

# Check specific job
bjobs -l JOB_ID

# Check aggregation job
bjobs -J ppi_aggregate

# Check packaging job
bjobs -J ppi_package_embeddings

# View job output
tail -f /dccstor/.../ppi_comprehensive/logs/ppi_BioPlex3_asthma_rep00.out

# View aggregation output
tail -f /dccstor/.../ppi_comprehensive/logs/ppi_aggregate.out
```

## Comparison with Extended Generators

This PPI infrastructure mirrors the extended generators infrastructure:

| Feature | Extended Generators | PPI Comprehensive |
|---------|-------------------|-------------------|
| Networks | 7 synthetic types | 5 PPI × 3 diseases |
| Replicates | 30 per type | 30 per (net, disease) |
| Total jobs | 450 (7×30×3 sizes) | 450 (5×3×30) |
| Methods | All 33 | All 33 |
| Tasks | All 3 | All 3 |
| Aggregation | By network type | By (network, disease) |
| Packaging | Yes (.npz) | Yes (.npz) |

## Key Differences from v4 Script

The comprehensive script improves on `submit_ppi_disease_jobs_v4.sh`:

1. **All 5 networks**: Includes HumanNet, ProteomeHD, PCNet (v4 only had STRING, BioPlex3)
2. **All 33 methods**: Includes all baseline methods (v4 had subset)
3. **Automatic aggregation**: Dependency-based aggregation job (v4 used separate script)
4. **Embedding packaging**: Creates `.npz` archives (v4 didn't package)
5. **Consistent structure**: Matches extended generators infrastructure
6. **Better documentation**: Comprehensive README and inline comments

## Hyperparameter Handling

The script loads network-specific hyperparameters from:
```
/dccstor/boseukb/Q/NetMed/QuVINE/results/hparam_tuning/real_{NETWORK}/best_hyperparams.json
```

If hyperparameters are not found, it uses defaults. The script also:
- Overrides `quvine_walks` view constraints to prevent sparse views
- Sets `max_nodes=250, max_edges=5000` for quantum walk methods
- This ensures root nodes always have neighbors in dense PPI subgraphs

## Disease Seeds and Targets

For each disease, the script loads:
- **Seeds**: `/dccstor/.../gene_seeds/{DISEASE}_ncbi_seeds.json`
- **Targets**: `/dccstor/.../gwas_catalog_targets/{DISEASE}_targets_ncbi_gwas_catalog.json`

Seeds and targets are always preserved during subsampling, even if the network exceeds `max_nodes`.

## Troubleshooting

### Job fails with "no seeds or targets"
- Check that disease seed/target files exist
- Verify NCBI IDs in seed/target files match network node IDs
- Check network edge list format (should be NCBI IDs)

### Quantum walk methods fail
- Check that view constraints are applied: `max_nodes=250, max_edges=5000`
- Verify hyperparameter file exists or defaults are used
- Check that subgraph is not too sparse

### Aggregation job fails
- Check that all analysis jobs completed successfully
- Verify result directories exist and contain expected CSVs
- Check aggregation script output for specific errors

### Out of memory
- Increase `--memory` parameter
- Reduce `--max-nodes` parameter
- Reduce number of methods with `--methods`

## Contact

For questions or issues, contact the QuVINE development team.