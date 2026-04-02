# QuVINE Setup Instructions

## Virtual Environment Setup

A virtual environment has been created and all packages have been installed successfully!

### Activate the Virtual Environment

```bash
source venv_quvine/bin/activate
```

### Deactivate When Done

```bash
deactivate
```

## Installed Packages

All required packages have been installed including:
- ✅ NumPy, SciPy, NetworkX
- ✅ Pandas, Matplotlib, Seaborn
- ✅ Scikit-learn
- ✅ Gensim (Word2Vec)
- ✅ Node2Vec
- ✅ Hydra-core, OmegaConf
- ✅ Hiperwalk (quantum walks)
- ✅ Jupyter Lab & Notebook
- ✅ QuVINE (editable install)

## Running the Demo Notebook

1. **Activate the environment:**
   ```bash
   source venv_quvine/bin/activate
   ```

2. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

3. **Open the demo notebook:**
   Navigate to `notebooks/complexity_and_embedding_demo.ipynb`

4. **Run all cells** to see the complete workflow

## What the Notebook Demonstrates

1. **Generate Random Graphs** - Scale-free and modular networks with seeds/targets
2. **Compute Complexity** - All spectral metrics + QBioCode integration
3. **Run QuVINE** - Generate embeddings using random walks
4. **Gene Prioritization** - Evaluate target recovery performance
5. **Analysis** - Visualizations and insights

## Optional: Install QBioCode

For additional complexity metrics from IBM's QBioCode:

```bash
source venv_quvine/bin/activate
cd /tmp
git clone https://github.com/IBM/QBioCode.git
cd QBioCode
pip install -e .
```

## Testing the Installation

Run a quick test:

```bash
source venv_quvine/bin/activate
python -c "from quvine.data import generate_barabasi_albert, compute_graph_complexity_metrics; G = generate_barabasi_albert(100, 3, seed=42); m = compute_graph_complexity_metrics(G); print(f'Quantum Complexity: {m[\"quantum_complexity\"]:.4f}')"
```

Expected output: `Quantum Complexity: 0.XXXX`

## Running Examples

### Random Graph Examples
```bash
source venv_quvine/bin/activate
python examples/random_graph_examples.py
```

### Complexity Examples
```bash
source venv_quvine/bin/activate
python examples/graph_complexity_examples.py
```

### Run Tests
```bash
source venv_quvine/bin/activate
python tests/test_random_graphs.py
```

## Troubleshooting

### If Jupyter kernel is not found:
```bash
source venv_quvine/bin/activate
python -m ipykernel install --user --name=venv_quvine --display-name="Python (QuVINE)"
```

### If imports fail:
Make sure you're in the QuVINE directory and the virtual environment is activated:
```bash
cd /Users/filippoutro/QuVINE
source venv_quvine/bin/activate
```

## Directory Structure

```
QuVINE/
├── venv_quvine/              # Virtual environment (created)
├── src/quvine/               # Main package
│   ├── data/
│   │   ├── random_graphs.py       # Random graph generators
│   │   ├── graph_complexity.py    # Complexity metrics
│   │   └── qbc_complexity.py      # QBioCode integration
│   └── complexity_pipeline.py     # Complete pipeline
├── notebooks/
│   └── complexity_and_embedding_demo.ipynb  # Demo notebook
├── examples/
│   ├── random_graph_examples.py
│   └── graph_complexity_examples.py
├── docs/
│   ├── random_graphs_guide.md
│   ├── graph_complexity_guide.md
│   └── random_graphs_quick_reference.md
└── tests/
    └── test_random_graphs.py
```

## Next Steps

1. ✅ Virtual environment created and activated
2. ✅ All packages installed
3. ✅ QuVINE installed in editable mode
4. 🚀 Ready to run the demo notebook!

Start with:
```bash
source venv_quvine/bin/activate
jupyter lab
```

Then open `notebooks/complexity_and_embedding_demo.ipynb` and run all cells!