# QuVINE


<p align="center">
  <a href="./pyproject.toml"><img alt="pypi package" src="https://img.shields.io/badge/pypi_package-0.1.0-52c41a"></a>
  <a href="./pyproject.toml"><img alt="Python >= 3.10" src="https://img.shields.io/badge/Python-%3E%3D%203.10-0b8ecf"></a>
  <a href="./pyproject.toml"><img alt="Python <= 3.12" src="https://img.shields.io/badge/Python-%3C%3D%203.12-8a5bd6"></a>
  <a href="./docs/index.rst"><img alt="docs sphinx" src="https://img.shields.io/badge/docs-sphinx-0b8ecf"></a>
</p>

QuVINE: Quantum-enhanced Multi-View Diffusion for Network Embeddings

<p align="center">
  <img src="./images/quvine_framework.png" alt="QuVINE" width="560"
  height="320">
</p>



**QuVINE** is a framework for **Qu**antum and classical **V**iew-based **N**etwork **E**mbeddings. It supports graph preparation, complexity analysis, quantum and classical walk-based representation learning, embedding fusion, and downstream evaluation for biological and network-science workflows.

The project documentation is organized in a Sphinx-style layout:
- narrative documentation under `docs/`
- API reference pages under `docs/api/`
- documentation entry point at `docs/index.rst`

## Documentation

### Sphinx-style entry points
- **Documentation index**: [`docs/index.rst`](./docs/index.rst)
- **Documentation overview**: [`docs/README.md`](./docs/README.md)
- **API reference root**: [`docs/api/quvine.rst`](./docs/api/quvine.rst)

### Setup
- [`docs/setup/SETUP_INSTRUCTIONS.md`](./docs/setup/SETUP_INSTRUCTIONS.md)
- [`docs/setup/PYTHON_VERSION_GUIDE.md`](./docs/setup/PYTHON_VERSION_GUIDE.md)
- [`docs/setup/HPC_DEPLOYMENT.md`](./docs/setup/HPC_DEPLOYMENT.md)
- [`docs/setup/QBIOCODE_INSTALL.md`](./docs/setup/QBIOCODE_INSTALL.md)

### Guides
- [`docs/guides/QUICK_START.md`](./docs/guides/QUICK_START.md)
- [`docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md`](./docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)
- [`docs/guides/COMPREHENSIVE_DATASET_GUIDE.md`](./docs/guides/COMPREHENSIVE_DATASET_GUIDE.md)
- [`docs/guides/PARALLELIZATION_GUIDE.md`](./docs/guides/PARALLELIZATION_GUIDE.md)
- [`docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md`](./docs/guides/QUANTUM_ADVANTAGE_AND_DOWNSTREAM_TASKS.md)

### API reference
- [`docs/api/quvine.rst`](./docs/api/quvine.rst)
- [`docs/api/pipeline.rst`](./docs/api/pipeline.rst)
- [`docs/api/complexity_pipeline.rst`](./docs/api/complexity_pipeline.rst)
- [`docs/api/data.rst`](./docs/api/data.rst)
- [`docs/api/complexity.rst`](./docs/api/complexity.rst)
- [`docs/api/embedding.rst`](./docs/api/embedding.rst)
- [`docs/api/baselines.rst`](./docs/api/baselines.rst)
- [`docs/api/evaluation.rst`](./docs/api/evaluation.rst)
- [`docs/api/fusion.rst`](./docs/api/fusion.rst)
- [`docs/api/views.rst`](./docs/api/views.rst)
- [`docs/api/walks.rst`](./docs/api/walks.rst)
- [`docs/api/utils.rst`](./docs/api/utils.rst)
- [`docs/api/analysis.rst`](./docs/api/analysis.rst)

### Development notes
- [`docs/development/CONTRIBUTING.md`](./docs/development/CONTRIBUTING.md)
- [`docs/development/IMPLEMENTATION_SUMMARY.md`](./docs/development/IMPLEMENTATION_SUMMARY.md)
- [`docs/development/IMPROVEMENTS_AND_FIXES.md`](./docs/development/IMPROVEMENTS_AND_FIXES.md)
- [`docs/development/BUG_REVIEW_AND_FIXES.md`](./docs/development/BUG_REVIEW_AND_FIXES.md)

## Package structure

```text
QuVINE/
├── configs/                 # experiment configuration files
├── data/                    # raw and processed datasets
├── docs/                    # narrative docs + Sphinx-style API docs
├── examples/                # runnable examples
├── notebooks/               # exploratory notebooks
├── scripts/                 # utility and batch execution scripts
├── src/quvine/              # package source code
│   ├── analysis/            # result analysis and comparison
│   ├── baselines/           # classical and hybrid baselines
│   ├── complexity/          # graph complexity metrics
│   ├── data/                # graph/data preparation utilities
│   ├── embedding/           # embedding and quantum filters
│   ├── evaluation/          # ranking/classification/link prediction
│   ├── fusion/              # embedding fusion
│   ├── utils/               # reusable utilities
│   ├── views/               # graph view generation
│   └── walks/               # classical and quantum walks
└── tests/                   # regression and integration tests
```

## Installation

```bash
git clone <repository-url>
cd QuVINE
pip install -e .
```

For environment-specific instructions, use the setup guides in [`docs/setup/`](./docs/setup/).

## Running QuVINE

CLI entry point:

```bash
quvine
```

Python entry point:

```python
from quvine.main import main
```

For complete workflows, start with:
- [`docs/guides/QUICK_START.md`](./docs/guides/QUICK_START.md)
- [`docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md`](./docs/guides/COMPREHENSIVE_ANALYSIS_GUIDE.md)

## Core capabilities

- Graph preparation and preprocessing
- Complexity-aware graph analysis
- Quantum and classical walk pipelines
- Embedding generation and fusion
- Baseline model comparison
- Ranking, classification, and link prediction evaluation
- Large-scale and HPC-oriented experiment workflows

## Development status

Recent repository changes include:
- targeted bug fixes in graph complexity and GCN-MF utilities
- graceful optional-import handling for baseline and embedding packages
- Sphinx-style API reference pages under `docs/api/`
- streamlined documentation entry point through `docs/index.rst`

## Notes

- The banner uses `./docs/images/quvine.png` so GitHub-style Markdown renderers resolve it from the repository root.
- The docs badge links directly to the Sphinx-style documentation entry page at [`docs/index.rst`](./docs/index.rst).

## License

See [`LICENSE`](./LICENSE).