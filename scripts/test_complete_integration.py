#!/usr/bin/env python3
"""
Complete integration test for all QuVINE embedding methods.

Tests all 17 embedding methods:
- 3 Quantum walks: rwr, ctqw, dtqw
- 4 Q-Caliber methods: heat, poly, hgcnmf, pgcnmf
- 6 Fusion variants: svd, graphreg, attention, hybrid, shared_private, all_fusion
- 4 Baselines: baseline_gcnmf, netmf, node2vec, appnp

Also tests hyperparameter tuning for 8 methods:
- baseline_gcnmf, node2vec, netmf
- hgcnmf, pgcnmf
- rwr, ctqw, dtqw
"""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_all_embeddings():
    """Test all 16 embedding methods."""
    print("=" * 80)
    print("Testing All QuVINE Embedding Methods")
    print("=" * 80)
    
    # Create test graph
    print("\n1. Creating test graph...")
    G = nx.karate_club_graph()
    seeds = [0, 1, 2, 3, 4]
    targets = list(range(5, 15))
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"   Seeds: {len(seeds)}, Targets: {len(targets)}")
    
    # Initialize analyzer
    print("\n2. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir='test_results',
        embedding_dim=64
    )
    
    # Define all methods
    all_methods = [
        # Quantum walks
        'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
        # Q-Caliber filters
        'quvine_heat', 'quvine_poly',
        # Q-Caliber GCN-MF
        'quvine_hgcnmf', 'quvine_pgcnmf',
        # Fusion methods
        'quvine_fused_svd', 'quvine_fused_graphreg', 'quvine_fused_attention',
        'quvine_fused_hybrid', 'quvine_fused_svd_shared_priv_heat_poly',
        'quvine_fused_all',
        # Baselines
        'baseline_gcnmf', 'netmf', 'node2vec', 'appnp'
    ]
    
    # Test each method
    print(f"\n3. Testing {len(all_methods)} embedding methods...")
    results = {}
    
    for i, method in enumerate(all_methods, 1):
        try:
            print(f"\n   [{i}/{len(all_methods)}] Testing {method}...", end=' ')
            embedding = analyzer.run_embedding_method(method, G, seeds, targets)
            
            # Verify embedding shape
            expected_shape = (G.number_of_nodes(), 64)
            if embedding.shape == expected_shape:
                print(f"✓ SUCCESS - Shape: {embedding.shape}")
                results[method] = 'PASS'
            else:
                print(f"✗ FAIL - Wrong shape: {embedding.shape} (expected {expected_shape})")
                results[method] = 'FAIL'
        except Exception as e:
            print(f"✗ ERROR - {str(e)[:50]}")
            results[method] = 'ERROR'
    
    # Summary
    print("\n" + "=" * 80)
    print("Embedding Methods Test Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if v == 'FAIL')
    errors = sum(1 for v in results.values() if v == 'ERROR')
    
    print(f"\nTotal: {len(results)} methods")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Errors: {errors}")
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error methods:")
        for method, status in results.items():
            if status != 'PASS':
                print(f"  - {method}: {status}")
    
    return passed == len(all_methods)


def test_hyperparameter_tuning():
    """Test hyperparameter tuning for all supported methods."""
    print("\n" + "=" * 80)
    print("Testing Hyperparameter Tuning")
    print("=" * 80)
    
    # Create test graph
    print("\n1. Creating test graph...")
    G = nx.karate_club_graph()
    seeds = [0, 1, 2, 3, 4]
    targets = list(range(5, 15))
    
    # Initialize analyzer
    print("\n2. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir='test_results',
        embedding_dim=64
    )
    
    # Define tuning methods
    tuning_methods = {
        'baseline_gcnmf': ('tune_gcnmf_hyperparameters', {}),
        'node2vec': ('tune_node2vec_hyperparameters', {}),
        'netmf': ('tune_netmf_hyperparameters', {}),
        'hgcnmf': ('tune_qcaliber_gcnmf_hyperparameters', {'diffusion_type': 'heat'}),
        'pgcnmf': ('tune_qcaliber_gcnmf_hyperparameters', {'diffusion_type': 'poly'}),
        'rwr': ('tune_quantum_walk_hyperparameters', {'walk_type': 'rwr'}),
        'ctqw': ('tune_quantum_walk_hyperparameters', {'walk_type': 'ctqw'}),
        'dtqw': ('tune_quantum_walk_hyperparameters', {'walk_type': 'dtqw'}),
    }
    
    # Test each tuning method
    print(f"\n3. Testing {len(tuning_methods)} tuning methods (5 trials each)...")
    results = {}
    
    for i, (method_name, (tune_func, kwargs)) in enumerate(tuning_methods.items(), 1):
        try:
            print(f"\n   [{i}/{len(tuning_methods)}] Tuning {method_name}...", end=' ')
            
            # Get tuning function
            tune_method = getattr(analyzer, tune_func)
            
            # Run tuning with minimal trials
            result = tune_method(
                G, seeds, targets,
                n_trials=5,
                timeout=120,
                **kwargs
            )
            
            # Verify result
            if 'best_params' in result and 'best_value' in result:
                print(f"✓ SUCCESS - Best recall@50: {result['best_value']:.3f}")
                results[method_name] = 'PASS'
            else:
                print(f"✗ FAIL - Missing result keys")
                results[method_name] = 'FAIL'
                
        except Exception as e:
            print(f"✗ ERROR - {str(e)[:50]}")
            results[method_name] = 'ERROR'
    
    # Summary
    print("\n" + "=" * 80)
    print("Hyperparameter Tuning Test Summary")
    print("=" * 80)
    
    passed = sum(1 for v in results.values() if v == 'PASS')
    failed = sum(1 for v in results.values() if v == 'FAIL')
    errors = sum(1 for v in results.values() if v == 'ERROR')
    
    print(f"\nTotal: {len(results)} methods")
    print(f"✓ Passed: {passed}")
    print(f"✗ Failed: {failed}")
    print(f"⚠ Errors: {errors}")
    
    if failed > 0 or errors > 0:
        print("\nFailed/Error methods:")
        for method, status in results.items():
            if status != 'PASS':
                print(f"  - {method}: {status}")
    
    return passed == len(tuning_methods)


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("QuVINE Complete Integration Test")
    print("=" * 80)
    
    # Test embeddings
    embeddings_pass = test_all_embeddings()
    
    # Test tuning
    tuning_pass = test_hyperparameter_tuning()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    print(f"\nEmbedding Methods: {'✓ PASS' if embeddings_pass else '✗ FAIL'}")
    print(f"Hyperparameter Tuning: {'✓ PASS' if tuning_pass else '✗ FAIL'}")
    
    if embeddings_pass and tuning_pass:
        print("\n🎉 ALL TESTS PASSED! QuVINE integration is complete.")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED. Please review the output above.")
        sys.exit(1)

