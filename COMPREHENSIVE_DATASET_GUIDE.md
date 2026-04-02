# Comprehensive Dataset Generation and HPC Analysis Guide

## Overview

This guide describes how to generate a comprehensive dataset of random graphs and run large-scale embedding analysis on HPC clusters.

## Table of Contents

1. [Dataset Generation](#dataset-generation)
2. [Dataset Structure](#dataset-structure)
3. [Parameter Variation Strategy](#parameter-variation-strategy)
4. [HPC Job Submission](#hpc-job-submission)
5. [Results Aggregation](#results-aggregation)
6. [Analysis and Visualization](#analysis-and-visualization)

---

## Dataset Generation

### Quick Start

Generate 30 instances of each graph type (270 total networks):

```python
from quvine.data.random_graphs import generate_comprehensive_dataset

# Generate dataset
dataset = generate_comprehensive_dataset(
    n_instances=30,
    base_seed=42,
    n_nodes=200,
    save_dir='data/comprehensive_dataset'
)

# Summary
print(f"Generated {sum(len(v) for v in dataset.values())} networks")
print(f"Graph types: {list(dataset.keys())}")
```

### Graph Types Included

The comprehensive dataset includes 9 graph types:

1. **Erdős-Rényi** (Random)
   - Classic random graph model
   - Parameters: `n` (nodes), `p` (edge probability)
   - Variation: Edge density from 0.05 to 0.15

2. **Barabási-Albert** (Scale-Free)
   - Preferential attachment model
   - Parameters: `n` (nodes), `m` (edges per new node)
   - Variation: `m` from 2 to 6

3. **Watts-Strogatz** (Small-World)
   - Small-world network with clustering
   - Parameters: `n` (nodes), `k` (neighbors), `p` (rewiring probability)
   - Variation: `k` from 4 to 14, `p` from 0.1 to 0.5

4. **Powerlaw Cluster** (Scale-Free with Clustering)
   - Scale-free with triangle formation
   - Parameters: `n` (nodes), `m` (edges), `p` (triangle probability)
   - Variation: `m` from 2 to 5, `p` from 0.1 to 0.4

5. **Stochastic Block Model** (Modular/Community)
   - Community structure with inter/intra-community edges
   - Parameters: `n_communities`, `p_in` (intra), `p_out` (inter)
   - Variation: 3-7 communities, `p_in` 0.3-0.6, `p_out` 0.01-0.05

6. **Random Geometric** (Spatial)
   - Nodes in 2D space, edges based on distance
   - Parameters: `n` (nodes), `radius`, `dim` (dimensions)
   - Variation: Radius from 0.1 to 0.25

7. **Hierarchical** (Tree-like)
   - Tree structure with cross-level edges
   - Parameters: `levels`, `branching_factor`, `p_level`
   - Variation: 4-6 levels, branching 2-4, `p_level` 0.01-0.1

8. **Core-Periphery** (Hub-Spoke)
   - Dense core with sparse periphery
   - Parameters: `core_size`, `p_core`, `p_periphery`, `p_core_periphery`
   - Variation: Core 10-30%, densities varied

9. **Bipartite Random** (Two-Mode)
   - Two disjoint node sets with edges between
   - Parameters: `n1`, `n2`, `p` (edge probability)
   - Variation: Split around 50/50, `p` from 0.05 to 0.2

### Custom Dataset Generation

```python
# Generate with custom parameters
dataset = generate_comprehensive_dataset(
    n_instances=50,        # More instances per type
    base_seed=123,         # Different seed
    n_nodes=500,           # Larger networks
    save_dir='data/large_dataset'
)
```

---

## Dataset Structure

### Directory Layout

```
data/comprehensive_dataset/
├── erdos_renyi/
│   ├── erdos_renyi_000.graphml
│   ├── erdos_renyi_000_metadata.json
│   ├── erdos_renyi_001.graphml
│   ├── erdos_renyi_001_metadata.json
│   └── ...
├── barabasi_albert/
│   ├── barabasi_albert_000.graphml
│   ├── barabasi_albert_000_metadata.json
│   └── ...
├── watts_strogatz/
├── powerlaw_cluster/
├── stochastic_block_model/
├── random_geometric/
├── hierarchical/
├── core_periphery/
└── bipartite_random/
```

### Metadata Format

Each network has an associated JSON metadata file:

```json
{
  "type": "barabasi_albert",
  "instance": 0,
  "seed": 42,
  "n_nodes": 200,
  "m": 3,
  "params": {
    "n": 200,
    "m": 3
  }
}
```

### Loading Dataset

```python
from quvine.data.random_graphs import load_comprehensive_dataset

# Load previously saved dataset
dataset = load_comprehensive_dataset('data/comprehensive_dataset')

# Access specific graph type
ba_graphs = dataset['barabasi_albert']
for G, metadata in ba_graphs:
    print(f"Network {metadata['instance']}: {G.number_of_nodes()} nodes")
```

---

## Parameter Variation Strategy

### Design Philosophy

The dataset uses **implicit parameter variation** through:

1. **Random Seeds**: Each instance uses a different seed (base_seed + instance_id)
2. **Systematic Parameter Sweeps**: Parameters vary across instances
3. **Natural Variation**: Random graph generation captures stochastic variation

### Parameter Ranges

| Graph Type | Parameter | Range | Variation |
|------------|-----------|-------|-----------|
| Erdős-Rényi | `p` (density) | 0.05 - 0.15 | Linear |
| Barabási-Albert | `m` (attachment) | 2 - 6 | Cyclic |
| Watts-Strogatz | `k` (neighbors) | 4 - 14 | Cyclic |
| Watts-Strogatz | `p` (rewiring) | 0.1 - 0.5 | Linear |
| Powerlaw Cluster | `m` (edges) | 2 - 5 | Cyclic |
| Powerlaw Cluster | `p` (triangles) | 0.1 - 0.4 | Linear |
| SBM | `n_communities` | 3 - 7 | Cyclic |
| SBM | `p_in` (intra) | 0.3 - 0.6 | Linear |
| SBM | `p_out` (inter) | 0.01 - 0.05 | Linear |
| Random Geometric | `radius` | 0.1 - 0.25 | Linear |
| Hierarchical | `levels` | 4 - 6 | Cyclic |
| Hierarchical | `branching` | 2 - 4 | Cyclic |
| Core-Periphery | `core_size` | 10-30% | Linear |
| Core-Periphery | `p_core` | 0.5 - 0.8 | Linear |
| Bipartite | `n1/n2` split | 45-55% | Linear |
| Bipartite | `p` (edges) | 0.05 - 0.2 | Linear |

### Why This Approach?

**Advantages:**
- ✅ Captures natural variation in graph properties
- ✅ Large sample size (30 per type = 270 total)
- ✅ Systematic parameter coverage
- ✅ Computationally efficient

**Alternative (Explicit Grid Search):**
- Would require: nodes × density × clustering × ... = 1000s of networks
- More comprehensive but computationally expensive
- Can be added if needed for specific research questions

---

## HPC Job Submission

### Prerequisites

1. **HPC Cluster with LSF**
   - IBM Load Sharing Facility (LSF) scheduler
   - Access to compute nodes

2. **Python Environment**
   - QuVINE installed
   - All dependencies available

3. **Generated Dataset**
   - Run `generate_comprehensive_dataset()` first

### Basic Usage

```bash
# Submit all networks to HPC cluster
bash scripts/submit_hpc_jobs.sh \
    --dataset-dir data/comprehensive_dataset \
    --output-dir outputs/hpc_results \
    --queue normal \
    --walltime 4:00 \
    --memory 16

# Dry run (test without submitting)
bash scripts/submit_hpc_jobs.sh --dry-run
```

### Advanced Options

```bash
# Custom configuration
bash scripts/submit_hpc_jobs.sh \
    --dataset-dir data/large_dataset \
    --output-dir outputs/large_results \
    --queue gpu \
    --walltime 8:00 \
    --memory 32 \
    --python-env /path/to/venv/bin/python
```

### Monitoring Jobs

```bash
# Check all your jobs
bjobs -u $USER

# Check specific job
bjobs 12345

# View job output in real-time
bpeek 12345

# Check job history
bhist -l 12345

# Kill all jobs
bkill $(bjobs -u $USER -o "JOBID" | tail -n +2)
```

### Job Configuration

Each job runs:
- **Single network** analysis
- **All 6 embedding methods**: QuVINE-fused, RWR, CTQW, DTQW, NetMF, Node2Vec
- **Complexity metrics** computation
- **Downstream evaluation** (precision@K, recall@K, etc.)

**Resource Requirements:**
- Memory: 16GB (default, adjust for larger networks)
- Time: 4 hours (default, adjust based on network size)
- CPU: 1 core per job

**Parallelization:**
- One job per network = 270 parallel jobs
- Each job is independent
- Results saved to separate directories

---

## Results Aggregation

### Output Structure

```
outputs/hpc_results/
├── logs/
│   ├── emb_barabasi_albert_barabasi_albert_000.out
│   ├── emb_barabasi_albert_barabasi_albert_000.err
│   └── ...
├── results/
│   ├── barabasi_albert/
│   │   ├── barabasi_albert_000/
│   │   │   ├── complexity_metrics.json
│   │   │   ├── embedding_results.json
│   │   │   └── evaluation_metrics.json
│   │   └── ...
│   └── ...
└── embeddings/
    └── ...
```

### Aggregating Results

```python
import json
import pandas as pd
from pathlib import Path

def aggregate_hpc_results(results_dir='outputs/hpc_results/results'):
    """Aggregate all HPC job results into a single DataFrame."""
    
    all_results = []
    results_path = Path(results_dir)
    
    for graph_type_dir in results_path.iterdir():
        if not graph_type_dir.is_dir():
            continue
        
        graph_type = graph_type_dir.name
        
        for network_dir in graph_type_dir.iterdir():
            if not network_dir.is_dir():
                continue
            
            network_name = network_dir.name
            
            # Load complexity metrics
            complexity_file = network_dir / 'complexity_metrics.json'
            if complexity_file.exists():
                with open(complexity_file) as f:
                    complexity = json.load(f)
            else:
                complexity = {}
            
            # Load embedding results
            embedding_file = network_dir / 'embedding_results.json'
            if embedding_file.exists():
                with open(embedding_file) as f:
                    embeddings = json.load(f)
            else:
                embeddings = {}
            
            # Load evaluation metrics
            eval_file = network_dir / 'evaluation_metrics.json'
            if eval_file.exists():
                with open(eval_file) as f:
                    evaluation = json.load(f)
            else:
                evaluation = {}
            
            # Combine all metrics
            result = {
                'graph_type': graph_type,
                'network_name': network_name,
                **complexity,
                **embeddings,
                **evaluation
            }
            
            all_results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    print(f"Aggregated {len(df)} network results")
    print(f"Columns: {list(df.columns)}")
    
    return df

# Usage
df = aggregate_hpc_results()
df.to_csv('outputs/aggregated_results.csv', index=False)
```

---

## Analysis and Visualization

### Correlation Analysis

```python
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt

# Load aggregated results
df = pd.read_csv('outputs/aggregated_results.csv')

# Complexity metrics
complexity_cols = [
    'spectral_gap', 'von_neumann_entropy', 'quantum_complexity',
    'inverse_participation_ratio', 'participation_ratio'
]

# Performance metrics (for each method)
methods = ['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec']
performance_cols = [f'{method}_precision@10' for method in methods]

# Compute correlations
correlations = {}
for complexity_metric in complexity_cols:
    correlations[complexity_metric] = {}
    for perf_col in performance_cols:
        if complexity_metric in df.columns and perf_col in df.columns:
            corr, pval = spearmanr(df[complexity_metric], df[perf_col], nan_policy='omit')
            correlations[complexity_metric][perf_col] = {
                'correlation': corr,
                'p_value': pval
            }

# Visualize
corr_matrix = pd.DataFrame({
    complexity: {perf: correlations[complexity][perf]['correlation'] 
                 for perf in performance_cols if perf in correlations[complexity]}
    for complexity in complexity_cols
})

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1)
plt.title('Correlation: Complexity Metrics vs Embedding Performance')
plt.tight_layout()
plt.savefig('outputs/complexity_performance_correlation.png', dpi=300)
```

### Method Recommendations

```python
def recommend_method(complexity_metrics):
    """
    Recommend best embedding method based on network complexity.
    
    Parameters
    ----------
    complexity_metrics : dict
        Dictionary of complexity metrics
        
    Returns
    -------
    str
        Recommended method name
    """
    qc = complexity_metrics.get('quantum_complexity', 0)
    sg = complexity_metrics.get('spectral_gap', 0)
    ipr = complexity_metrics.get('inverse_participation_ratio', 0)
    
    # Decision rules (based on empirical analysis)
    if qc > 0.7 and sg < 0.3:
        return 'quvine_ctqw'  # High quantum complexity, low spectral gap
    elif qc > 0.5:
        return 'quvine_fused'  # Moderate-high quantum complexity
    elif ipr > 0.8:
        return 'quvine_dtqw'  # High localization
    elif sg > 0.5:
        return 'netmf'  # High spectral gap, classical methods work well
    else:
        return 'quvine_rwr'  # Default quantum method

# Apply to all networks
df['recommended_method'] = df.apply(
    lambda row: recommend_method(row[complexity_cols].to_dict()),
    axis=1
)

# Evaluate recommendations
for method in methods:
    mask = df['recommended_method'] == method
    if mask.sum() > 0:
        avg_perf = df.loc[mask, f'{method}_precision@10'].mean()
        print(f"{method}: {mask.sum()} networks, avg precision@10 = {avg_perf:.3f}")
```

---

## Summary

This comprehensive dataset and HPC workflow enables:

1. **Large-Scale Analysis**: 270+ networks across 9 graph types
2. **Parallel Execution**: HPC cluster with one job per network
3. **Systematic Comparison**: 6 embedding methods on each network
4. **Complexity-Performance Correlation**: Understand what drives quantum advantage
5. **Method Recommendations**: Data-driven guidance for method selection

**Next Steps:**
1. Generate dataset: `generate_comprehensive_dataset()`
2. Submit HPC jobs: `bash scripts/submit_hpc_jobs.sh`
3. Monitor progress: `bjobs -u $USER`
4. Aggregate results: `aggregate_hpc_results()`
5. Analyze and visualize: Correlation analysis, method recommendations

For questions or issues, see the main README or open an issue on GitHub.