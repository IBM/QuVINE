# PPI Network Hyperparameter Tuning Guide

## Overview

This guide describes the hyperparameter tuning system for **real PPI (Protein-Protein Interaction) networks** with **GWAS (Genome-Wide Association Studies) data**. The system optimizes embedding methods for three tasks across five PPI networks and three diseases.

## System Components

### 1. Configuration File
**File**: `scripts/ppi_tuning_config.yaml`

Defines:
- **5 PPI Networks**: STRING, BioPlex3, HumanNet, PCNet, ProteomeHD
- **3 Diseases**: asthma, autism, schizophrenia
- **13 Methods**: 8 quantum + 5 classical embedding methods
- **3 Tasks**: node_ranking (GWAS), node_classification (7 strategies), link_prediction (hadamard)
- **Hyperparameter search spaces** for each method
- **Fixed parameters**: graph subsampling (200 nodes), evaluation metrics

### 2. Main Tuning Script
**File**: `scripts/tune_ppi_by_task.py`

**Purpose**: Tunes hyperparameters for a single network-disease-method combination

**Key Features**:
- Loads PPI network from edge list CSV
- Loads GWAS seeds and targets from JSON files
- Subsamples network to 200 nodes using degree-matched sampling
- Generates multiple replicates with different subsampling seeds
- Optimizes hyperparameters separately for each task using Optuna (TPE sampler)
- Saves results as JSON: `{network}_{disease}_tuning_by_task.json`

**Usage**:
```bash
# Tune all methods on STRING network with asthma
python scripts/tune_ppi_by_task.py --network STRING --disease asthma

# Tune specific method
python scripts/tune_ppi_by_task.py --network STRING --disease asthma --methods quvine_fused

# Override number of trials
python scripts/tune_ppi_by_task.py --network STRING --disease asthma --n-trials 20

# Use more replicates
python scripts/tune_ppi_by_task.py --network STRING --disease asthma --n-replicates 5
```

### 3. Job Submission Script
**File**: `scripts/submit_ppi_tuning_jobs.sh`

**Purpose**: Submits LSF jobs for parallel hyperparameter tuning on HPC

**Modes**:

#### Parallel Mode (Default)
- **One job per method × network × disease**
- Total jobs: 13 methods × 5 networks × 3 diseases = **195 jobs**
- Each job tunes one method on one network-disease pair for all 3 tasks
- **Recommended** for fastest completion

#### Serial Mode (`--serial` flag)
- **One job per network × disease**, processing all methods sequentially
- Total jobs: 5 networks × 3 diseases = **15 jobs**
- Each job tunes all 13 methods on one network-disease pair
- Use when job limit is a concern

**Usage**:
```bash
# Parallel mode (default) - 195 jobs
bash scripts/submit_ppi_tuning_jobs.sh

# Serial mode - 15 jobs
bash scripts/submit_ppi_tuning_jobs.sh --serial

# Tune only STRING network - 39 jobs (13 methods × 3 diseases)
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING

# Tune specific network-disease pair - 13 jobs
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma

# Tune specific methods - 15 jobs (3 methods × 5 networks × 1 disease)
bash scripts/submit_ppi_tuning_jobs.sh --methods quvine_fused,node2vec,netmf --diseases asthma

# Custom resources
bash scripts/submit_ppi_tuning_jobs.sh --queue normal --walltime 72:00 --memory 32

# Dry run (show what would be submitted)
bash scripts/submit_ppi_tuning_jobs.sh --dry-run
```

## Network and Disease Details

### PPI Networks

| Network | Full Size | Description |
|---------|-----------|-------------|
| **STRING** | ~19,354 nodes | Protein-protein association network |
| **BioPlex3** | ~15,435 nodes | Human protein interaction network |
| **HumanNet** | ~21,238 nodes | Integrated functional gene network |
| **PCNet** | ~19,781 nodes | Protein correlation network |
| **ProteomeHD** | ~9,792 nodes | Proteome-wide co-regulation network |

### Diseases

| Disease | Seeds | Targets | Source |
|---------|-------|---------|--------|
| **Asthma** | Known disease genes | GWAS catalog targets | GWAS studies |
| **Autism** | Known disease genes | GWAS catalog targets | GWAS studies |
| **Schizophrenia** | Known disease genes | GWAS catalog targets | GWAS studies |

## Methods

### Quantum Methods (8)
1. **quvine_fused**: Fusion of multiple quantum walk views
2. **quvine_ctqw**: Continuous-Time Quantum Walk
3. **quvine_dtqw**: Discrete-Time Quantum Walk
4. **quvine_rwr**: Random Walk with Restart (quantum-inspired)
5. **quvine_heat**: Heat kernel diffusion
6. **quvine_poly**: Polynomial filter diffusion
7. **quvine_hgcnmf**: Heat-based GCN Matrix Factorization
8. **quvine_pgcnmf**: Polynomial-based GCN Matrix Factorization

### Classical Methods (5)
1. **netmf**: Network Embedding as Matrix Factorization
2. **node2vec**: Biased random walk embeddings
3. **baseline_gcnmf**: GCN Matrix Factorization baseline
4. **baseline_filter**: Filter-based diffusion baseline
5. **graphsage**: GraphSAGE inductive learning

## Tasks

### 1. Node Ranking
- **Objective**: Rank nodes by relevance to disease seeds
- **Evaluation**: Recall@50 on GWAS targets
- **Use Case**: Disease gene prioritization

### 2. Node Classification
- **Objective**: Classify nodes into categories
- **Strategies**: 7 label generation strategies (degree, clustering, community, etc.)
- **Evaluation**: F1-macro averaged across strategies
- **Use Case**: Functional annotation

### 3. Link Prediction
- **Objective**: Predict missing protein interactions
- **Edge Features**: Hadamard product of node embeddings
- **Evaluation**: AUC-ROC
- **Use Case**: Interaction discovery

## Workflow

### Step 1: Configure
Edit `scripts/ppi_tuning_config.yaml` if needed:
- Adjust hyperparameter search spaces
- Modify number of trials per method
- Change evaluation parameters

### Step 2: Submit Jobs
```bash
# Full tuning (195 jobs)
bash scripts/submit_ppi_tuning_jobs.sh

# Or start with a subset
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma
```

### Step 3: Monitor
```bash
# Check job status
bjobs -u $USER

# Check specific job output
tail -f ppi_tuning_by_task/logs/ppi_tune_quvine_fused_STRING_asthma.out
```

### Step 4: Collect Results
Results are automatically aggregated when all jobs complete:
- Individual results: `ppi_tuning_by_task/{network}_{disease}_{method}_tuning_by_task.json`
- Aggregated results: `ppi_tuning_by_task/{network}_{disease}_tuning_by_task.json`
- Summary: `ppi_tuning_by_task/tuning_summary.json`

## Output Format

### Individual Method Result
```json
{
  "quvine_fused": {
    "node_classification": {
      "best_params": {
        "embedding_dim": 128,
        "num_views": 4,
        "walk_length": 40,
        ...
      },
      "best_score": 0.7234
    },
    "link_prediction": {
      "best_params": {...},
      "best_score": 0.8456
    },
    "node_ranking": {
      "best_params": {...},
      "best_score": 0.6789
    }
  }
}
```

### Aggregated Network-Disease Result
```json
{
  "quvine_fused": {...},
  "quvine_ctqw": {...},
  "node2vec": {...},
  ...
}
```

## Resource Requirements

### Per Job (Parallel Mode)
- **Memory**: 32 GB (default)
- **Wall Time**: 72 hours (default)
- **CPUs**: 1 (Optuna sequential)

### Total Resources (195 jobs)
- **Peak Memory**: ~6.2 TB (if all run simultaneously)
- **Total CPU-hours**: ~14,040 hours (195 jobs × 72 hours)
- **Actual Time**: Depends on queue availability

## Comparison with Synthetic Network Tuning

| Aspect | Synthetic Networks | PPI Networks |
|--------|-------------------|--------------|
| **Networks** | 16 graph types | 5 real PPI networks |
| **Graph Size** | 200 nodes (generated) | 200 nodes (subsampled) |
| **Tasks** | Same 3 tasks | Same 3 tasks |
| **Node Ranking** | K-hop neighbors | GWAS targets |
| **Replicates** | 10 graphs per type | 3 subsampling seeds |
| **Total Jobs** | 160 (10 methods × 16 types) | 195 (13 methods × 5 nets × 3 diseases) |
| **Purpose** | Method development | Real-world validation |

## Best Practices

### 1. Start Small
```bash
# Test with one network-disease pair first
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma --methods quvine_fused --dry-run
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma --methods quvine_fused
```

### 2. Use Serial Mode for Testing
```bash
# Fewer jobs, easier to monitor
bash scripts/submit_ppi_tuning_jobs.sh --serial --networks STRING --diseases asthma
```

### 3. Adjust Resources Based on Method
- **Fast methods** (filters): 24-48 hours, 16 GB
- **Medium methods** (walks): 48-72 hours, 32 GB
- **Slow methods** (GNNs): 72-96 hours, 32-64 GB

### 4. Monitor Early Jobs
Check first few jobs to ensure:
- Network loading works correctly
- GWAS data maps to network nodes
- Subsampling preserves seeds/targets
- Optuna optimization runs smoothly

### 5. Handle Failures
```bash
# Resubmit failed jobs
bash scripts/submit_ppi_tuning_jobs.sh --networks STRING --diseases asthma --methods quvine_fused
```

## Troubleshooting

### Issue: "No seeds or targets mapped to network"
**Cause**: GWAS gene IDs don't match network node IDs  
**Solution**: Check NCBI ID mapping in network and GWAS files

### Issue: "No seeds or targets in subgraph"
**Cause**: Subsampling removed all seeds/targets  
**Solution**: Increase `n_nodes` in config or adjust `radius` parameter

### Issue: Job runs out of memory
**Cause**: Large network or complex method  
**Solution**: Increase `--memory` flag or reduce `n_nodes`

### Issue: Job times out
**Cause**: Too many trials or slow method  
**Solution**: Reduce `--n-trials` or increase `--walltime`

### Issue: Optuna not available
**Cause**: Missing dependency  
**Solution**: `pip install optuna` in virtual environment

## Advanced Usage

### Custom Hyperparameter Spaces
Edit `scripts/ppi_tuning_config.yaml`:
```yaml
hyperparameters:
  quvine_fused:
    embedding_dim: [64, 128, 256, 512]  # Add 512
    num_views: [3, 4, 5, 6, 7]  # Add 7
    # ... other parameters
```

### Custom Evaluation Metrics
Edit `fixed_params.evaluation` in config:
```yaml
fixed_params:
  evaluation:
    node_ranking:
      top_k: 100  # Change from 50 to 100
```

### Use Tuned Parameters
After tuning, use best parameters in main experiments:
```python
import json

# Load tuned parameters
with open('ppi_tuning_by_task/STRING_asthma_tuning_by_task.json') as f:
    tuned_params = json.load(f)

# Get best params for quvine_fused on node_ranking task
best_params = tuned_params['quvine_fused']['node_ranking']['best_params']

# Use in your experiment
embedding = run_quvine_fused(G, seeds, **best_params)
```

## Next Steps

After tuning completes:

1. **Analyze Results**: Compare method performance across networks and diseases
2. **Select Best Methods**: Identify top performers for each task
3. **Run Full Experiments**: Use tuned parameters on full-scale networks
4. **Validate**: Test on held-out diseases or networks
5. **Publish**: Report findings with optimized hyperparameters

## Support

For issues or questions:
- Check logs in `ppi_tuning_by_task/logs/`
- Review configuration in `scripts/ppi_tuning_config.yaml`
- Consult main tuning guide: `scripts/README_HYPERPARAMETER_TUNING.md`

---

**Made with Bob** - PPI Network Hyperparameter Tuning System