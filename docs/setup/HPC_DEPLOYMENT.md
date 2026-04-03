# HPC Deployment Guide for QuVINE

Complete guide for deploying QuVINE on HPC clusters with LSF job scheduler.

## ⚠️ IMPORTANT: Python Version Requirements

**QuVINE requires Python 3.10, 3.11, or 3.12 (NOT 3.13+)**

- ✅ **Compatible**: Python 3.10.x, 3.11.x, 3.12.x (3.12 recommended)
- ❌ **Not compatible**: Python 3.9.x (too old), Python 3.13+ (too new for QBioCode dependency)

If you encounter: `ERROR: Package 'qbiocode' requires a different Python: 3.13.9 not in '<3.13,>=3.9'`

**Solution**: See [Python Version Compatibility Guide](PYTHON_VERSION_GUIDE.md) for detailed instructions.

## Table of Contents
1. [Python Version Setup](#python-version-setup)
2. [Environment Setup](#environment-setup)
3. [Dependency Installation](#dependency-installation)
4. [Code Transfer](#code-transfer)
5. [Running the Pipeline](#running-the-pipeline)
6. [Troubleshooting](#troubleshooting)

---

## Python Version Setup

### Check Your Python Version

```bash
python --version
# or
python3 --version
```

### If You Have Python 3.13+

You **must** use Python 3.10, 3.11, or 3.12. See [PYTHON_VERSION_GUIDE.md](PYTHON_VERSION_GUIDE.md) for detailed instructions.

**Quick Fix**:
```bash
# macOS (using Homebrew)
brew install python@3.12
python3.12 -m venv venv_quvine

# Linux/HPC (using module system)
module load python/3.12
python3 -m venv venv_quvine

# Using conda
conda create -n quvine python=3.12
conda activate quvine
```

---

## Environment Setup

### Option 1: Using Frozen Requirements (Recommended)

This ensures exact version matching with the development environment.

```bash
# 1. SSH to HPC cluster
ssh your_username@hpc_cluster

# 2. Navigate to your project directory
cd /path/to/your/project

# 3. Clone or transfer the QuVINE repository
# (See Code Transfer section below)

# 4. Load Python module (if using module system)
module load python/3.12  # Adjust version as needed

# 5. Create virtual environment
python3 -m venv venv_quvine

# 6. Activate virtual environment
source venv_quvine/bin/activate

# 7. Upgrade pip
pip install --upgrade pip setuptools wheel

# 8. Install from frozen requirements (exact versions)
pip install -r requirements_frozen.txt

# Note: This may take 10-20 minutes due to large packages (torch, qiskit, etc.)
```

### Option 2: Using Minimal Requirements

If frozen requirements fail due to platform differences, use minimal requirements:

```bash
# Install from minimal requirements
pip install -r requirements.txt

# This installs latest compatible versions
```

### Option 3: Manual Installation (If Both Fail)

```bash
# Core dependencies
pip install numpy scipy networkx pandas matplotlib seaborn
pip install scikit-learn xgboost
pip install gensim node2vec
pip install torch
pip install qiskit qiskit-aer qiskit-algorithms
pip install hiperwalk

# Install QuVINE in development mode
pip install -e .
```

---

## Dependency Installation

### Key Dependencies and Their Purposes

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | 2.4.4 | Numerical computations |
| **scipy** | 1.16.3 | Scientific computing, sparse matrices |
| **networkx** | 3.6.1 | Graph data structures |
| **pandas** | 2.3.3 | Data analysis, results aggregation |
| **matplotlib** | 3.10.8 | Visualization |
| **seaborn** | 0.13.2 | Statistical visualization |
| **scikit-learn** | 1.8.0 | Machine learning (classification, link prediction) |
| **xgboost** | 3.2.0 | Gradient boosting |
| **gensim** | 4.4.0 | Word2Vec for embeddings |
| **node2vec** | 0.5.0 | Classical baseline method |
| **torch** | 2.11.0 | Deep learning (NetMF) |
| **qiskit** | 2.2.0 | Quantum computing framework |
| **qiskit-aer** | 0.17.0 | Quantum simulators |
| **hiperwalk** | 2.0b18 | Quantum walks (CTQW, DTQW) |

### Optional Dependencies

```bash
# For same-community negative sampling in link prediction
pip install python-louvain

# For additional complexity metrics
pip install scikit-dimension
```

---

## Code Transfer

### Method 1: Git Clone (Recommended)

```bash
# On HPC cluster
cd /path/to/your/project
git clone https://github.com/IBM/QuVINE.git
cd QuVINE

# If using private repository
git clone https://github.ibm.com/your-username/QuVINE.git
```

### Method 2: rsync Transfer

```bash
# From local machine
rsync -avz --exclude='venv_quvine' --exclude='*.pyc' --exclude='__pycache__' \
    /path/to/local/QuVINE/ \
    your_username@hpc_cluster:/path/to/remote/QuVINE/

# This excludes virtual environment and Python cache files
```

### Method 3: scp Transfer

```bash
# From local machine
cd /path/to/local
tar -czf QuVINE.tar.gz QuVINE/ --exclude='venv_quvine' --exclude='*.pyc'
scp QuVINE.tar.gz your_username@hpc_cluster:/path/to/remote/

# On HPC cluster
cd /path/to/remote
tar -xzf QuVINE.tar.gz
```

---

## Running the Pipeline

### Step 1: Verify Installation

```bash
# Activate environment
source venv_quvine/bin/activate

# Test imports
python -c "import quvine; print('QuVINE imported successfully')"
python -c "import networkx, numpy, scipy, sklearn; print('Core dependencies OK')"
python -c "import qiskit, hiperwalk; print('Quantum dependencies OK')"
```

### Step 2: Configure HPC Script

Edit `scripts/submit_hpc_jobs_complete.sh`:

```bash
# Update these variables at the top of the script:

# 1. Python environment path
PYTHON_ENV="/path/to/your/venv_quvine/bin/python"

# 2. Project directory
PROJECT_DIR="/path/to/your/QuVINE"

# 3. Output directory
OUTPUT_DIR="${PROJECT_DIR}/outputs/hpc_results"

# 4. LSF queue (check available queues with 'bqueues')
QUEUE="normal"  # or "short", "long", etc.

# 5. Resource requirements
MEMORY="16GB"   # Adjust based on network size
CORES="4"       # Adjust based on available resources
WALLTIME="4:00" # Hours:Minutes
```

### Step 3: Submit Jobs

```bash
# Navigate to project directory
cd /path/to/your/QuVINE

# Make script executable
chmod +x scripts/submit_hpc_jobs_complete.sh

# Submit jobs (dry run first)
bash scripts/submit_hpc_jobs_complete.sh --dry-run

# If dry run looks good, submit for real
bash scripts/submit_hpc_jobs_complete.sh

# Monitor jobs
bjobs -u $USER
bjobs -u $USER -w  # Wide format with more details
```

### Step 4: Monitor Progress

```bash
# Check job status
bjobs -u $USER

# Check specific job details
bjobs -l <job_id>

# Check job output (while running)
bpeek <job_id>

# Check completed job output
cat outputs/hpc_results/logs/scale_free_00.out
cat outputs/hpc_results/logs/scale_free_00.err
```

### Step 5: Collect Results

Results are automatically aggregated when all jobs complete. To manually aggregate:

```bash
# Wait for all jobs to finish
bjobs -u $USER  # Should show no jobs

# Manually aggregate if needed
python scripts/collect_hpc_results.py --results-dir outputs/hpc_results/results

# Check results
ls -lh outputs/hpc_results/results/comprehensive_results.csv
head outputs/hpc_results/results/comprehensive_results.csv
```

---

## Troubleshooting

### Issue 1: Jobs Stuck in PEND Status

**Symptom**: Jobs remain in PEND status indefinitely

**Causes**:
1. Incorrect dependency syntax
2. Resource requirements too high
3. Queue limits reached

**Solutions**:
```bash
# Check job details
bjobs -l <job_id> | grep -A 10 "PENDING REASONS"

# Check dependency syntax (should be: done(123) && done(456))
bjobs -l <job_id> | grep -A 5 "Dependency"

# Reduce resource requirements in submit script
MEMORY="8GB"  # Instead of 16GB
CORES="2"     # Instead of 4

# Try different queue
QUEUE="short"  # For quick jobs
```

### Issue 2: Import Errors

**Symptom**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
```bash
# Verify virtual environment is activated
which python  # Should point to venv_quvine/bin/python

# Reinstall problematic package
pip install --force-reinstall <package_name>

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# Install QuVINE in development mode
cd /path/to/QuVINE
pip install -e .
```

### Issue 3: Memory Errors

**Symptom**: Jobs killed with "TERM_MEMLIMIT" or OOM errors

**Solutions**:
```bash
# Increase memory in submit script
MEMORY="32GB"  # Or higher

# Reduce network size in dataset generation
# Edit src/quvine/data/random_graphs.py
n_nodes_range = (100, 300)  # Instead of (200, 500)

# Process fewer networks per job
# Split dataset into smaller batches
```

### Issue 4: Quantum Simulation Errors

**Symptom**: Qiskit or Hiperwalk errors

**Solutions**:
```bash
# Update quantum packages
pip install --upgrade qiskit qiskit-aer hiperwalk

# Use classical methods only (if quantum fails)
# Edit embedding_methods in submit script:
embedding_methods = ['quvine_rwr', 'netmf', 'node2vec']  # No quantum

# Reduce quantum walk parameters
# Edit src/quvine/walks/ctqw.py and dtqw.py
max_time = 5  # Instead of 10
```

### Issue 5: Disk Space Issues

**Symptom**: "No space left on device"

**Solutions**:
```bash
# Check disk usage
df -h /path/to/your/project
du -sh outputs/hpc_results/*

# Clean up old results
rm -rf outputs/hpc_results/old_run_*

# Compress embeddings (optional)
cd outputs/hpc_results/results
for dir in */; do
    tar -czf "${dir%/}.tar.gz" "$dir"
    rm -rf "$dir"
done

# Use scratch space if available
OUTPUT_DIR="/scratch/$USER/quvine_results"
```

---

## Performance Optimization

### Parallel Execution

The pipeline is designed for job-level parallelization:

```bash
# Each network runs as independent job
# No internal parallelization needed

# Adjust number of concurrent jobs based on cluster limits
# Check with: bqueues -l <queue_name>
```

### Resource Allocation

Recommended resources per network:

| Network Size | Memory | Cores | Walltime |
|--------------|--------|-------|----------|
| 100-200 nodes | 8GB | 2 | 2:00 |
| 200-500 nodes | 16GB | 4 | 4:00 |
| 500-1000 nodes | 32GB | 8 | 8:00 |
| 1000+ nodes | 64GB | 16 | 12:00 |

### Batch Processing

For large datasets (100+ networks):

```bash
# Process in batches of 50 networks
# Edit submit_hpc_jobs_complete.sh:
MAX_CONCURRENT_JOBS=50

# Submit first batch
bash scripts/submit_hpc_jobs_complete.sh --batch 1

# Wait for completion, then submit next batch
bash scripts/submit_hpc_jobs_complete.sh --batch 2
```

---

## Best Practices

1. **Always test with small dataset first**
   ```bash
   # Generate 2-3 networks for testing
   python -c "from quvine.data.random_graphs import generate_comprehensive_dataset; \
              generate_comprehensive_dataset(n_scale_free=2, n_modular=1)"
   ```

2. **Use dry-run mode**
   ```bash
   bash scripts/submit_hpc_jobs_complete.sh --dry-run
   ```

3. **Monitor resource usage**
   ```bash
   # Check job efficiency after completion
   bjobs -l <job_id> | grep -A 20 "RESOURCE REQUIREMENT"
   ```

4. **Keep logs organized**
   ```bash
   # Logs are automatically saved to:
   outputs/hpc_results/logs/<network_id>.out
   outputs/hpc_results/logs/<network_id>.err
   ```

5. **Backup results regularly**
   ```bash
   # Compress and backup results
   tar -czf quvine_results_$(date +%Y%m%d).tar.gz outputs/hpc_results/results/
   ```

---

## Support

For issues or questions:
- Check documentation: `docs/README.md`
- Review examples: `examples/`
- Open issue on GitHub: https://github.com/IBM/QuVINE/issues

---

## Quick Reference

```bash
# Complete workflow
cd /path/to/QuVINE
source venv_quvine/bin/activate
bash scripts/submit_hpc_jobs_complete.sh
bjobs -u $USER  # Monitor
# Wait for completion
ls outputs/hpc_results/results/comprehensive_results.csv  # Check results