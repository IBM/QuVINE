# QuVINE Test Suite

Comprehensive test suite for the QuVINE 39-method system.

## Test Files

### 1. `test_quantum_filters.py`
Unit tests for the 4 new quantum filter functions:
- `generate_baseline_heat_embedding`
- `generate_baseline_poly_embedding`
- `generate_rwr_heat_embedding`
- `generate_rwr_poly_embedding`

**Tests:**
- Basic functionality
- Parameter variations (scale, order, restart_prob)
- Normalization
- Reproducibility
- Comparison between methods

### 2. `test_fusion.py`
Unit tests for hierarchical fusion functions:
- `fuse_by_method_type` - Within-type fusion
- `fuse_best_across_types` - Cross-type fusion
- `hierarchical_fusion` - Complete hierarchical strategy
- Helper functions and constants

**Tests:**
- Method filtering by type and quantum/classical
- SVD fusion
- Different fusion methods (svd, concatenate, average)
- Hierarchical fusion workflow
- Method registry validation

### 3. `test_integration_39_methods.py`
Integration tests for the complete 39-method workflow:
- Method dispatch for all 39 methods
- Fast method subset (SGNS + Filters)
- Full workflow with fusion
- Method registry completeness
- Embedding properties and validity
- Reproducibility

**Test Markers:**
- `@pytest.mark.slow` - Tests that require neural network training (GAT, GraphGPS)

## Running Tests

### Install Test Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From QuVINE root directory
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Filter tests only
pytest tests/test_quantum_filters.py -v

# Fusion tests only
pytest tests/test_fusion.py -v

# Integration tests only
pytest tests/test_integration_39_methods.py -v
```

### Run Fast Tests Only (Skip Neural Network Tests)

```bash
pytest tests/ -v -m "not slow"
```

### Run With Coverage

```bash
pytest tests/ --cov=quvine --cov-report=html
```

### Run Specific Test Classes

```bash
# Test only baseline heat embedding
pytest tests/test_quantum_filters.py::TestBaselineHeatEmbedding -v

# Test only fusion by method type
pytest tests/test_fusion.py::TestFuseByMethodType -v

# Test only fast methods workflow
pytest tests/test_integration_39_methods.py::TestFastMethodSubset -v
```

## Test Organization

### Unit Tests
- **test_quantum_filters.py**: Tests individual filter functions
- **test_fusion.py**: Tests fusion functions in isolation

### Integration Tests
- **test_integration_39_methods.py**: Tests complete workflows

## Test Data

Tests use small synthetic graphs for speed:
- **Tiny graph**: 5 nodes, 8 edges (for quick unit tests)
- **Small graph**: Karate club (34 nodes) (for integration tests)

## Expected Test Results

### Fast Tests (SGNS + Filters)
- **9 methods** should pass quickly (< 1 minute total)
- All methods should produce valid embeddings
- Fusion should work correctly

### Slow Tests (GAT + GraphGPS)
- **24 methods** may be skipped if PyTorch not available
- If available, tests will take longer (5-10 minutes)

### Total Coverage
- **39 methods** total
- **16 quantum methods**
- **23 classical methods**

## Continuous Integration

To add to CI pipeline:

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m "not slow"
```

## Troubleshooting

### Import Errors
If you get import errors, make sure QuVINE is installed:
```bash
pip install -e .
```

### PyTorch Not Available
GAT and GraphGPS tests will be skipped. This is expected if PyTorch is not installed.

### Slow Tests Timeout
Use `-m "not slow"` to skip neural network tests:
```bash
pytest tests/ -v -m "not slow"
```

### Memory Issues
Reduce graph sizes in fixtures if running on limited memory systems.

## Adding New Tests

### For New Methods
1. Add method to appropriate test class in `test_integration_39_methods.py`
2. Test basic functionality, shape, validity
3. Add to method registry test

### For New Fusion Strategies
1. Add test class to `test_fusion.py`
2. Test with sample embeddings
3. Verify output shapes and validity

### For New Features
1. Create new test file: `test_<feature>.py`
2. Follow existing test structure
3. Add to this README

## Test Statistics

Current test coverage:
- **Filter functions**: 4/4 (100%)
- **Fusion functions**: 5/5 (100%)
- **Method dispatch**: 39/39 (100%)
- **Integration workflows**: Complete

## Performance Benchmarks

Approximate test times on standard hardware:

| Test Suite | Time | Methods Tested |
|------------|------|----------------|
| test_quantum_filters.py | ~10s | 4 filter functions |
| test_fusion.py | ~5s | 5 fusion functions |
| test_integration_39_methods.py (fast) | ~30s | 9 SGNS + Filter methods |
| test_integration_39_methods.py (all) | ~10min | All 39 methods |

## Contact

For test-related issues, please open an issue on GitHub with:
- Test file and class name
- Error message
- Python version and dependencies
- System information