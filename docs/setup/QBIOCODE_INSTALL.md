# Installing QBioCode (Optional)

QBioCode is **optional** for QuVINE. The system works perfectly without it, using the built-in spectral complexity metrics.

## Why Install QBioCode?

QBioCode provides additional complexity metrics from IBM's biological complexity analysis framework:
- Intrinsic dimension
- Condition number
- Manifold complexity (Isomap reconstruction error)
- Total correlations
- Fractal dimension
- And more...

## Installation Steps

### Option 1: Quick Install (Recommended)

```bash
# Activate QuVINE environment
cd /Users/filippoutro/QuVINE
source venv_quvine/bin/activate

# Clone and install QBioCode
cd /tmp
git clone https://github.com/IBM/QBioCode.git
cd QBioCode
pip install -e .
```

### Option 2: Install in Separate Location

```bash
# Activate QuVINE environment
source venv_quvine/bin/activate

# Clone to your preferred location
cd ~/Projects  # or wherever you want
git clone https://github.com/IBM/QBioCode.git
cd QBioCode
pip install -e .
```

## Verify Installation

```bash
source venv_quvine/bin/activate
python -c "from qbiocode.evaluation import evaluate; print('QBioCode installed successfully!')"
```

## What Happens Without QBioCode?

The notebook and all QuVINE functions work perfectly without QBioCode:

✅ **Still Available:**
- All random graph generators
- Spectral complexity metrics (spectral gap, von Neumann entropy, etc.)
- Quantum complexity score
- QuVINE embedding pipeline
- Gene prioritization
- All visualizations

❌ **Not Available:**
- QBioCode-specific metrics (intrinsic dimension, manifold complexity, etc.)
- The notebook will show: `QBioCode available: False`
- Functions will gracefully skip QBC metrics

## Troubleshooting

### If installation fails:

1. **Check Python version**: QBioCode requires Python 3.8+
   ```bash
   python --version
   ```

2. **Install dependencies manually**:
   ```bash
   pip install scikit-learn scikit-dimension pandas numpy scipy
   ```

3. **Try again**:
   ```bash
   cd QBioCode
   pip install -e .
   ```

### If imports fail:

Make sure you're in the correct virtual environment:
```bash
which python
# Should show: /Users/filippoutro/QuVINE/venv_quvine/bin/python
```

## Using QBioCode in Your Code

Once installed, you can use it:

```python
from quvine.data import (
    generate_barabasi_albert,
    compute_qbc_complexity_from_laplacian,
    check_qbc_available
)

# Check if available
if check_qbc_available():
    G = generate_barabasi_albert(100, 3, seed=42)
    
    # Compute QBC metrics from Laplacian
    qbc_metrics = compute_qbc_complexity_from_laplacian(
        G, 
        normalized=True,
        laplacian_method='eigenvectors'
    )
    
    print(f"Manifold Complexity: {qbc_metrics['Manifold Complexity']}")
else:
    print("QBioCode not available - using standard metrics only")
```

## Summary

- **QBioCode is optional** - QuVINE works great without it
- **Easy to install** - Just clone and `pip install -e .`
- **Adds extra metrics** - But not required for core functionality
- **Graceful fallback** - Code automatically detects and adapts

You can run the demo notebook right now without QBioCode and install it later if you want the additional metrics!