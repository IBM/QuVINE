# Comprehensive Job Submission System with Hyperparameter Tuning

## Overview

This document describes the complete job submission system for running large-scale experiments with hyperparameter tuning across synthetic networks and PPI datasets.

## System Architecture

### Three-Phase Pipeline

```
Phase 1: Hyperparameter Tuning (46 jobs)
    ↓
Phase 2: Main Analysis (2640 jobs) 
    ↓
Phase 3: Aggregation & Packaging (2 jobs)
```

**Total: 2688 jobs**

## Phase 1: Hyperparameter Tuning (46 jobs)

Hyperparameter tuning runs once per network type/configuration to find optimal parameters for all 15 embedding methods.

### 1A. Synthetic Networks - Hard Negatives (16 jobs)

One tuning job per case:
- QW1-9: Quantum walk advantage scenarios (9 cases)
- NC1-4: Null/control cases (4 cases)
- RN1-3: Real network benchmarks (3 cases)

**Job specs**: 72 hours, 16GB RAM per job

**Output**: `hparam_tuning/{case_name}_best_hyperparams.json`

### 1B. Synthetic Networks - Extended Generators (5 jobs)

One tuning job per generator type:
- random_regular
- heterophilic_sbm
- degree_corrected_sbm
- grid_torus
- configuration_model

**Job specs**: 72 hours, 16GB RAM per job

**Output**: `hparam_tuning/{network_type}_best_hyperparams.json`

### 1C. PPI Networks - Ranking Task (15 jobs)

One tuning job per (network, disease) combination:
- **Networks**: BioPlex3, HumanNet, STRING, ProteomeHD, IntAct (5)
- **Diseases**: asthma, autism, schizophrenia (3)
- **Total**: 5 × 3 = 15 jobs

**Job specs**: 72 hours, 32GB RAM per job

**Output**: `hparam_tuning/{ppi_network}_{disease}_ranking_best_hyperparams.json`

### 1D. PPI Networks - Classification & Link Prediction (10 jobs)

One tuning job per (network, task) combination:
- **Networks**: BioPlex3, HumanNet, STRING, ProteomeHD, IntAct (5)
- **Tasks**: classification, link_prediction (2)
- **Total**: 5 × 2 = 10 jobs

**Job specs**: 72 hours, 32GB RAM per job

**Output**: `hparam_tuning/{ppi_network}_{task}_best_hyperparams.json`

## Phase 2: Main Analysis (2640 jobs)

All analysis jobs wait for corresponding tuning jobs to complete using LSF dependencies.

### 2A. Hard Negatives Analysis (1440 jobs)

- **Cases**: 16
- **Replicates**: 30 per (case, size)
- **Node sizes**: 500, 2000, 5000 (3)
- **Total**: 16 × 30 × 3 = 1440 jobs

**Job specs**: 48 hours, 4GB RAM per job

**Dependencies**: Waits for corresponding tuning job (1A)

**Hyperparameters**: Uses `hparam_tuning/{case_name}_best_hyperparams.json`

### 2B. Extended Generators Analysis (450 jobs)

- **Types**: 5
- **Replicates**: 30 per (type, size)
- **Node sizes**: 500, 2000, 5000 (3)
- **Total**: 5 × 30 × 3 = 450 jobs

**Job specs**: 48 hours, 4GB RAM per job

**Dependencies**: Waits for corresponding tuning job (1B)

**Hyperparameters**: Uses `hparam_tuning/{network_type}_best_hyperparams.json`

### 2C. PPI Ranking Analysis (450 jobs)

- **Networks**: 5
- **Diseases**: 3
- **Replicates**: 30 per (network, disease)
- **Total**: 5 × 3 × 30 = 450 jobs

**Job specs**: 48 hours, 8GB RAM per job

**Dependencies**: Waits for corresponding tuning job (1C)

**Hyperparameters**: Uses `hparam_tuning/{ppi_network}_{disease}_ranking_best_hyperparams.json`

### 2D. PPI Classification & Link Prediction (300 jobs)

- **Networks**: 5
- **Tasks**: 2
- **Replicates**: 30 per (network, task)
- **Total**: 5 × 2 × 30 = 300 jobs

**Job specs**: 48 hours, 8GB RAM per job

**Dependencies**: Waits for corresponding tuning job (1D)

**Hyperparameters**: Uses `hparam_tuning/{ppi_network}_{task}_best_hyperparams.json`

## Phase 3: Aggregation & Packaging (2 jobs)

### 3A. Aggregation Job (1 job)

Combines all results into comprehensive CSV files.

**Dependencies**: Waits for ALL 2640 analysis jobs

**Job specs**: 2 hours, 32GB RAM

**Output**: 
- `comprehensive_results.csv`
- `results_by_network_type.csv`
- `results_by_method.csv`

### 3B. Packaging Job (1 job)

Packages all embeddings into compressed NPZ archives.

**Dependencies**: Waits for aggregation job (3A)

**Job specs**: 1 hour, 16GB RAM

**Output**: 
- `embeddings_synthetic.npz`
- `embeddings_ppi.npz`

## Usage

### Basic Usage

```bash
# Submit all jobs with hyperparameter tuning
bash scripts/submit_simulated_data_jobs_with_tuning.sh

# Dry run to preview
bash scripts/submit_simulated_data_jobs_with_tuning.sh --dry-run
```

### Skip Tuning (Use Existing Hyperparameters)

```bash
# Skip tuning phase if hyperparameters already exist
bash scripts/submit_simulated_data_jobs_with_tuning.sh --skip-tuning
```

### Selective Execution

```bash
# Run only synthetic networks
bash scripts/submit_simulated_data_jobs_with_tuning.sh --skip-ppi

# Run only PPI networks
bash scripts/submit_simulated_data_jobs_with_tuning.sh \
    --skip-hard-negatives \
    --skip-extended-gens

# Run only hard negatives
bash scripts/submit_simulated_data_jobs_with_tuning.sh \
    --skip-extended-gens \
    --skip-ppi
```

### Custom Configuration

```bash
# Custom replicates and node sizes
bash scripts/submit_simulated_data_jobs_with_tuning.sh \
    --n-replicates 50 \
    --n-nodes 1000,3000,7000

# Custom queue and resources
bash scripts/submit_simulated_data_jobs_with_tuning.sh \
    --queue priority \
    --walltime 72:00 \
    --memory 8

# Custom output directory
bash scripts/submit_simulated_data_jobs_with_tuning.sh \
    --output-dir /path/to/custom/output
```

## Monitoring

### Check Job Status

```bash
# View all your jobs
bjobs -u $USER

# View specific job details
bjobs -l <job_id>

# View job dependencies
bjobs -l <job_id> | grep -A 5 "DEPENDENCY"
```

### Check Tuning Progress

```bash
# List completed tuning jobs
ls -lh results/simulated_data/hparam_tuning/*.json

# View tuning results
cat results/simulated_data/hparam_tuning/QW1_modular_strong_best_hyperparams.json
```

### Check Analysis Progress

```bash
# Count completed analysis jobs
find results/simulated_data/results -name "results.json" | wc -l

# Check specific network results
ls -lh results/simulated_data/results/QW1_modular_strong_n500_rep00/
```

## File Structure

```
results/simulated_data/
├── hparam_tuning/                    # Phase 1 outputs
│   ├── QW1_modular_strong_best_hyperparams.json
│   ├── random_regular_best_hyperparams.json
│   ├── BioPlex3_asthma_ranking_best_hyperparams.json
│   └── ...
├── results/                          # Phase 2 outputs
│   ├── QW1_modular_strong_n500_rep00/
│   │   ├── embeddings/
│   │   ├── results.json
│   │   └── complexity_metrics.json
│   ├── random_regular_n500_rep00/
│   ├── ppi_ranking/
│   │   └── BioPlex3_asthma_rep00/
│   └── ppi_classification/
│       └── BioPlex3_classification_rep00/
├── logs/                             # Job logs
│   ├── tune_hn_QW1_modular_strong.out
│   ├── sim_hn_QW1_modular_strong_500_00.out
│   └── ...
├── comprehensive_results.csv         # Phase 3 outputs
├── embeddings_synthetic.npz
└── embeddings_ppi.npz
```

## Hyperparameter Search Space

Each tuning job searches over:

### Quantum Walk Parameters
- **walk_length**: [10, 20, 40, 80]
- **num_walks**: [10, 20, 40]
- **time_steps** (CTQW): [0.1, 0.5, 1.0, 2.0, 5.0]
- **coin_type** (DTQW): ['hadamard', 'grover', 'fourier']

### Classical Parameters
- **restart_prob** (RWR): [0.1, 0.15, 0.2, 0.3]
- **window_size** (Node2Vec): [5, 10, 15]
- **p, q** (Node2Vec): [0.5, 1.0, 2.0]

### Filter Parameters
- **filter_type**: ['heat', 'polynomial', 'chebyshev']
- **filter_order**: [2, 4, 8, 16]
- **scale**: [0.5, 1.0, 2.0, 5.0]

### GCN-MF Parameters
- **num_layers**: [2, 3, 4]
- **hidden_dim**: [64, 128, 256]
- **dropout**: [0.0, 0.1, 0.3, 0.5]

**Total combinations per method**: ~100-500 depending on method

**Evaluation metric**: Precision@50 for ranking, F1 for classification, AUC for link prediction

## Expected Runtime

### With Default Settings (30 replicates, 3 sizes)

| Phase | Jobs | Time per Job | Total Time* |
|-------|------|--------------|-------------|
| Tuning | 46 | 24-72 hours | 72 hours |
| Analysis | 2640 | 2-48 hours | 48 hours |
| Aggregation | 1 | 1-2 hours | 2 hours |
| Packaging | 1 | 0.5-1 hour | 1 hour |

*Total time assumes parallel execution on HPC cluster

**End-to-end**: ~5 days (with tuning) or ~2 days (skip tuning)

## Troubleshooting

### Tuning Job Failed

```bash
# Check error log
cat results/simulated_data/logs/tune_hn_QW1_modular_strong.err

# Resubmit individual tuning job
bsub < results/simulated_data/logs/tune_hn_QW1_modular_strong.sh
```

### Analysis Job Failed

```bash
# Check if hyperparameters exist
ls results/simulated_data/hparam_tuning/QW1_modular_strong_best_hyperparams.json

# Check error log
cat results/simulated_data/logs/sim_hn_QW1_modular_strong_500_00.err

# Resubmit with --resume flag
# (edit job script to add --resume flag, then resubmit)
```

### Missing Dependencies

```bash
# Check if tuning jobs completed
bjobs -u $USER | grep tune_

# If tuning incomplete, analysis jobs will wait
# Check dependency status
bjobs -l <analysis_job_id> | grep DEPENDENCY
```

## Best Practices

1. **Always run dry-run first**: `--dry-run` to preview job submission
2. **Start with small test**: Use `--n-replicates 3` for initial testing
3. **Monitor tuning progress**: Check tuning logs regularly
4. **Use skip-tuning for reruns**: If hyperparameters exist, use `--skip-tuning`
5. **Check disk space**: Ensure sufficient space for embeddings (~100GB per 1000 networks)
6. **Save job IDs**: Keep track of submitted job IDs for monitoring

## Advanced: Manual Job Submission

If you need to submit individual jobs manually:

### Submit Single Tuning Job

```bash
bsub -J tune_test \
     -o logs/tune_test.out \
     -e logs/tune_test.err \
     -q normal \
     -W 72:00 \
     -M 16GB \
     -R "rusage[mem=16GB]" \
     python scripts/run_hyperparameter_tuning.py \
         --network-type hard_negative \
         --case-name QW1_modular_strong \
         --output-file hparam_tuning/QW1_test.json \
         --methods all \
         --verbose
```

### Submit Single Analysis Job (with dependency)

```bash
bsub -J analysis_test \
     -o logs/analysis_test.out \
     -e logs/analysis_test.err \
     -q normal \
     -W 48:00 \
     -M 4GB \
     -R "rusage[mem=4GB]" \
     -w "ended(<tuning_job_id>)" \
     python scripts/run_hard_negative_network.py \
         --case-name QW1_modular_strong \
         --network-id QW1_test \
         --output-dir results/QW1_test \
         --hparam-file hparam_tuning/QW1_test.json \
         --methods all \
         --verbose
```

## Contact

For issues or questions:
- Check logs in `results/simulated_data/logs/`
- Review this documentation
- Contact the development team

---

**Last Updated**: 2026-04-26
**Version**: 2.0 (with hyperparameter tuning)