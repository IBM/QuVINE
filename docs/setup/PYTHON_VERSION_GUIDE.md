# Python Version Compatibility Guide

## Issue: Python 3.13 Incompatibility

If you encounter this error:
```
ERROR: Package 'qbiocode' requires a different Python: 3.13.9 not in '<3.13,>=3.9'
```

This means you're using Python 3.13, which is **not compatible** with QBioCode (a dependency).

## Compatible Python Versions

### QuVINE Requirements:
- **Python**: >=3.10, <3.13
- **Recommended**: Python 3.12.x
- **Also works**: Python 3.10.x, 3.11.x

### QBioCode Requirements (dependency):
- **Python**: >=3.9, <3.13

### Combined Requirements:
- **Compatible**: Python 3.10, 3.11, or 3.12
- **Not compatible**: Python 3.9 (too old for QuVINE), Python 3.13+ (too new for QBioCode)

---

## Solution 1: Use Python 3.12 (Recommended)

### On macOS (using Homebrew):
```bash
# Install Python 3.12
brew install python@3.12

# Create virtual environment with Python 3.12
python3.12 -m venv venv_quvine

# Activate
source venv_quvine/bin/activate

# Verify version
python --version  # Should show Python 3.12.x

# Install dependencies
pip install --upgrade pip
pip install -r requirements_frozen.txt
```

### On Linux/HPC:
```bash
# Check available Python versions
module avail python

# Load Python 3.12 (adjust module name as needed)
module load python/3.12

# Or use pyenv
pyenv install 3.12.0
pyenv local 3.12.0

# Create virtual environment
python3 -m venv venv_quvine
source venv_quvine/bin/activate

# Verify version
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements_frozen.txt
```

### Using conda/mamba:
```bash
# Create environment with Python 3.12
conda create -n quvine python=3.12
conda activate quvine

# Install dependencies
pip install -r requirements_frozen.txt
```

---

## Solution 2: Use Python 3.11

If Python 3.12 is not available:

```bash
# macOS
brew install python@3.11
python3.11 -m venv venv_quvine

# Linux/HPC
module load python/3.11
python3 -m venv venv_quvine

# conda
conda create -n quvine python=3.11
```

---

## Solution 3: Use Python 3.10

Minimum supported version:

```bash
# macOS
brew install python@3.10
python3.10 -m venv venv_quvine

# Linux/HPC
module load python/3.10
python3 -m venv venv_quvine

# conda
conda create -n quvine python=3.10
```

---

## Verification Steps

After setting up the correct Python version:

```bash
# 1. Activate environment
source venv_quvine/bin/activate  # or conda activate quvine

# 2. Check Python version
python --version
# Should show: Python 3.10.x, 3.11.x, or 3.12.x (NOT 3.13.x)

# 3. Install dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements_frozen.txt

# 4. Verify QuVINE installation
python -c "import quvine; print('QuVINE OK')"

# 5. Verify QBioCode installation (if needed)
python -c "import qbiocode; print('QBioCode OK')"

# 6. Verify quantum packages
python -c "import qiskit, hiperwalk; print('Quantum packages OK')"
```

---

## Common Issues and Solutions

### Issue 1: "python3.12: command not found"

**Solution**: Python 3.12 is not installed. Use one of the installation methods above.

### Issue 2: Virtual environment still uses Python 3.13

**Solution**: Delete the old virtual environment and recreate with correct Python version:
```bash
rm -rf venv_quvine
python3.12 -m venv venv_quvine  # Use specific version
source venv_quvine/bin/activate
python --version  # Verify
```

### Issue 3: System Python is 3.13, can't change

**Solution**: Use pyenv or conda to manage Python versions:
```bash
# Using pyenv
curl https://pyenv.run | bash
pyenv install 3.12.0
pyenv local 3.12.0

# Using conda
conda create -n quvine python=3.12
conda activate quvine
```

### Issue 4: HPC cluster only has Python 3.13

**Solution**: Contact HPC support to request Python 3.12 module, or use conda:
```bash
# Load conda module (if available)
module load anaconda3

# Create environment with Python 3.12
conda create -n quvine python=3.12
conda activate quvine
```

---

## HPC-Specific Instructions

### Check Available Python Versions:
```bash
# Method 1: Module system
module avail python
module avail anaconda

# Method 2: Direct check
ls /usr/bin/python*
ls /opt/python/*/bin/python*
```

### Load Appropriate Python:
```bash
# Example for LSF/IBM Spectrum
module load python/3.12

# Example for SLURM
module load python/3.12.0

# Example for conda
module load anaconda3
conda create -n quvine python=3.12
```

### Update Job Submission Script:
```bash
# In submit_hpc_jobs_complete.sh, add:
module load python/3.12  # Or appropriate version

# Or if using conda:
module load anaconda3
source activate quvine
```

---

## Quick Reference

| Python Version | QuVINE | QBioCode | Status |
|----------------|--------|----------|--------|
| 3.9.x | ❌ | ✅ | Too old for QuVINE |
| 3.10.x | ✅ | ✅ | ✅ Compatible |
| 3.11.x | ✅ | ✅ | ✅ Compatible |
| 3.12.x | ✅ | ✅ | ✅ **Recommended** |
| 3.13.x | ✅ | ❌ | ❌ Too new for QBioCode |

---

## Summary

**Problem**: Python 3.13 is too new for QBioCode dependency

**Solution**: Use Python 3.10, 3.11, or 3.12 (3.12 recommended)

**Quick Fix**:
```bash
# Delete old environment
rm -rf venv_quvine

# Create new with Python 3.12
python3.12 -m venv venv_quvine
source venv_quvine/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements_frozen.txt

# Verify
python --version  # Should be 3.12.x
python -c "import quvine; print('Success!')"
```

---

## Need Help?

1. Check Python version: `python --version`
2. Check available versions: `module avail python` (HPC) or `brew search python` (macOS)
3. Consult HPC documentation for Python module names
4. Contact HPC support if Python 3.10-3.12 is not available