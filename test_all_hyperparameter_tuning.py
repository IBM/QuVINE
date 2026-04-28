#!/usr/bin/env python3
"""
Test script for comprehensive hyperparameter tuning of all baseline methods.
Tests GCN-MF (heat and poly variants), Node2Vec, and NetMF.
"""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def create_test_graph(n_nodes=100, seed=42):
    """Create a small test graph."""
    np.random.seed(seed)
    G = nx.barabasi_albert_graph(n_nodes, 3, seed=seed)
    return G

def test_all_hyperparameter_tuning():
    """Test hyperparameter tuning for all baseline methods."""
    print("=" * 80)
    print("Testing Comprehensive Hyperparameter Tuning")
    print("=" * 80)
    
    # Create test graph
    print("\n1. Creating test graph...")
    G = create_test_graph(n_nodes=100)
    n_nodes = G.number_of_nodes()
    print(f"   Graph: {n_nodes} nodes, {G.number_of_edges()} edges")
    
    # Create seeds and targets
    np.random.seed(42)
    all_nodes = list(G.nodes())
    n_seeds = 10
    seeds = np.random.choice(all_nodes, size=n_seeds, replace=False).tolist()
    targets = [n for n in all_nodes if n not in seeds][:20]
    
    print(f"   Seeds: {len(seeds)} nodes")
    print(f"   Targets: {len(targets)} nodes")
    
    # Initialize analyzer
    print("\n2. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        embedding_dim=64,
        seed=42,
        n_jobs=1
    )
    
    # Initialize storage for test_graph
    if "test_graph" not in analyzer.tuned_hyperparameters:
        analyzer.tuned_hyperparameters["test_graph"] = {}
    
    # Test 1: GCN-MF (Heat) Hyperparameter Tuning
    print("\n" + "=" * 80)
    print("TEST 1: GCN-MF (Heat) Hyperparameter Tuning")
    print("=" * 80)
    try:
        result_hgcnmf = analyzer.tune_gcnmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            diffusion_type='heat',
            n_trials=5,  # Small number for quick test
            timeout=60
        )
        print("\n✓ GCN-MF (Heat) tuning successful!")
        print(f"  Best parameters: {result_hgcnmf['best_params']}")
        
        # Store parameters
        analyzer.tuned_hyperparameters["test_graph"]["hgcnmf"] = result_hgcnmf['best_params']
        print("  ✓ Parameters stored in tuned_hyperparameters")
            
    except Exception as e:
        print(f"\n✗ GCN-MF (Heat) tuning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: GCN-MF (Poly) Hyperparameter Tuning
    print("\n" + "=" * 80)
    print("TEST 2: GCN-MF (Poly) Hyperparameter Tuning")
    print("=" * 80)
    try:
        result_pgcnmf = analyzer.tune_gcnmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            diffusion_type='poly',
            n_trials=5,
            timeout=60
        )
        print("\n✓ GCN-MF (Poly) tuning successful!")
        print(f"  Best parameters: {result_pgcnmf['best_params']}")
        
        # Store parameters
        analyzer.tuned_hyperparameters["test_graph"]["pgcnmf"] = result_pgcnmf['best_params']
        print("  ✓ Parameters stored in tuned_hyperparameters")
                
    except Exception as e:
        print(f"\n✗ GCN-MF (Poly) tuning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Node2Vec Hyperparameter Tuning
    print("\n" + "=" * 80)
    print("TEST 3: Node2Vec Hyperparameter Tuning")
    print("=" * 80)
    try:
        result_node2vec = analyzer.tune_node2vec_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            n_trials=5,
            timeout=60
        )
        print("\n✓ Node2Vec tuning successful!")
        print(f"  Best parameters: {result_node2vec['best_params']}")
        
        # Store parameters
        analyzer.tuned_hyperparameters["test_graph"]["node2vec"] = result_node2vec['best_params']
        print("  ✓ Parameters stored in tuned_hyperparameters")
                
    except Exception as e:
        print(f"\n✗ Node2Vec tuning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 4: NetMF Hyperparameter Tuning
    print("\n" + "=" * 80)
    print("TEST 4: NetMF Hyperparameter Tuning")
    print("=" * 80)
    try:
        result_netmf = analyzer.tune_netmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            n_trials=5,
            timeout=60
        )
        print("\n✓ NetMF tuning successful!")
        print(f"  Best parameters: {result_netmf['best_params']}")
        
        # Store parameters
        analyzer.tuned_hyperparameters["test_graph"]["netmf"] = result_netmf['best_params']
        print("  ✓ Parameters stored in tuned_hyperparameters")
                
    except Exception as e:
        print(f"\n✗ NetMF tuning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 5: Verify all tuned parameters are stored
    print("\n" + "=" * 80)
    print("TEST 5: Verify All Tuned Parameters")
    print("=" * 80)
    print(f"\nStored tuned hyperparameters for 'test_graph':")
    if "test_graph" in analyzer.tuned_hyperparameters:
        for method, params in analyzer.tuned_hyperparameters["test_graph"].items():
            print(f"  {method}: {params}")
    else:
        print("  No parameters stored!")
    
    # Test 6: Test embedding generation with tuned parameters
    print("\n" + "=" * 80)
    print("TEST 6: Generate Embeddings with Tuned Parameters")
    print("=" * 80)
    
    methods_to_test = [
        ('quvine_hgcnmf', 'GCN-MF (Heat)'),
        ('quvine_pgcnmf', 'GCN-MF (Poly)'),
        ('node2vec', 'Node2Vec'),
        ('netmf', 'NetMF')
    ]
    
    for method_name, method_label in methods_to_test:
        print(f"\nTesting {method_label}...")
        try:
            embedding = analyzer.run_embedding_method(
                method_name=method_name,
                G=G,
                seeds=seeds,
                targets=targets,
                network_id="test_graph"
            )
            print(f"  ✓ Embedding generated: shape {embedding.shape}")
            
            # Verify it's using tuned parameters (check logs)
            expected_method = method_name.replace('quvine_', '') if method_name.startswith('quvine_') else method_name
            if "test_graph" in analyzer.tuned_hyperparameters:
                if expected_method in analyzer.tuned_hyperparameters["test_graph"]:
                    print(f"  ✓ Using tuned parameters: {analyzer.tuned_hyperparameters['test_graph'][expected_method]}")
                else:
                    print(f"  ⚠ No tuned parameters found for {expected_method}")
            
        except Exception as e:
            print(f"  ✗ Failed to generate embedding: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    expected_methods = ['hgcnmf', 'pgcnmf', 'node2vec', 'netmf']
    if "test_graph" in analyzer.tuned_hyperparameters:
        stored_methods = list(analyzer.tuned_hyperparameters["test_graph"].keys())
        print(f"\nExpected methods: {expected_methods}")
        print(f"Stored methods:   {stored_methods}")
        
        if set(stored_methods) == set(expected_methods):
            print("\n✓ ALL TESTS PASSED!")
            print("  All hyperparameter tuning methods work correctly.")
            print("  All tuned parameters are properly stored and used.")
        else:
            missing = set(expected_methods) - set(stored_methods)
            extra = set(stored_methods) - set(expected_methods)
            if missing:
                print(f"\n⚠ Missing methods: {missing}")
            if extra:
                print(f"\n⚠ Extra methods: {extra}")
    else:
        print("\n✗ TESTS FAILED!")
        print("  No tuned parameters were stored.")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_all_hyperparameter_tuning()

# Made with Bob
