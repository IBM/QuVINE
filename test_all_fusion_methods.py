#!/usr/bin/env python3
"""Test script to verify all fusion methods are integrated."""

import sys
sys.path.insert(0, 'src')

import networkx as nx
import numpy as np
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_fusion_methods():
    """Test that all fusion methods are available."""
    
    # Create a simple test graph
    G = nx.karate_club_graph()
    seeds = [0, 1, 2]
    targets = list(range(len(G)))
    
    # Initialize analyzer
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir='test_output',
        n_jobs=1,
        embedding_dim=32,
        random_seed=42
    )
    
    # List of all expected fusion methods
    fusion_methods = [
        'quvine_fused_svd',
        'quvine_fused_graphreg',
        'quvine_fused_attention',
        'quvine_fused_hybrid',
        'quvine_fused_svd_shared_priv_heat_poly',
        'quvine_fused_svd_shared_priv_moe_heat_poly',
    ]
    
    print("Testing fusion methods integration...")
    print(f"Test graph: {len(G)} nodes, {G.number_of_edges()} edges")
    print(f"\nExpected fusion methods ({len(fusion_methods)}):")
    for i, method in enumerate(fusion_methods, 1):
        print(f"  {i}. {method}")
    
    # Test each fusion method
    print("\n" + "="*60)
    print("Testing each fusion method...")
    print("="*60)
    
    successful = []
    failed = []
    
    for method in fusion_methods:
        try:
            print(f"\nTesting {method}...")
            embedding = analyzer.run_embedding_method(method, G, seeds, targets)
            
            # Verify embedding shape
            expected_shape = (len(G), analyzer.embedding_dim)
            if embedding.shape == expected_shape:
                print(f"  ✓ Success! Shape: {embedding.shape}")
                successful.append(method)
            else:
                print(f"  ✗ Wrong shape: {embedding.shape}, expected {expected_shape}")
                failed.append(method)
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(method)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Successful: {len(successful)}/{len(fusion_methods)}")
    print(f"Failed: {len(failed)}/{len(fusion_methods)}")
    
    if successful:
        print("\n✓ Successful methods:")
        for method in successful:
            print(f"  - {method}")
    
    if failed:
        print("\n✗ Failed methods:")
        for method in failed:
            print(f"  - {method}")
    
    # Also test that all methods are in the default list
    print("\n" + "="*60)
    print("Checking default methods list...")
    print("="*60)
    
    # Get the methods from _process_single_network
    import inspect
    source = inspect.getsource(analyzer._process_single_network)
    
    all_found = True
    for method in fusion_methods:
        if method in source:
            print(f"  ✓ {method} found in methods list")
        else:
            print(f"  ✗ {method} NOT found in methods list")
            all_found = False
    
    if all_found:
        print("\n✓ All fusion methods are in the default methods list!")
    else:
        print("\n✗ Some fusion methods are missing from the default methods list")
    
    return len(failed) == 0 and all_found

if __name__ == '__main__':
    success = test_fusion_methods()
    sys.exit(0 if success else 1)

# Made with Bob
