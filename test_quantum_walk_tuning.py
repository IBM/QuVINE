#!/usr/bin/env python3
"""
Test Quantum Walk Hyperparameter Tuning

This script tests the Bayesian hyperparameter optimization for quantum walk methods.
"""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_quantum_walk_tuning():
    """Test quantum walk hyperparameter tuning."""
    
    print("="*80)
    print("Testing Quantum Walk Hyperparameter Tuning")
    print("="*80)
    
    # Create test graph (Karate Club)
    print("\n1. Creating test graph...")
    G = nx.karate_club_graph()
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Select seeds and targets
    nodes = list(G.nodes())
    np.random.seed(42)
    np.random.shuffle(nodes)
    seeds = nodes[:3]
    targets = nodes[3:13]
    print(f"   Seeds: {seeds}")
    print(f"   Targets: {len(targets)} nodes")
    
    # Initialize analyzer
    print("\n2. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir='test_output',
        embedding_dim=64,
        seed=42
    )
    
    # Test RWR tuning
    print("\n" + "="*80)
    print("Testing RWR (Random Walk with Restart) Tuning")
    print("="*80)
    try:
        result = analyzer.tune_quantum_walk_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            walk_type='rwr',
            n_trials=5,  # Small number for quick test
            timeout=120
        )
        
        if result:
            print("\n✓ RWR tuning SUCCESS!")
            print(f"  Best recall@50: {result['best_value']:.4f}")
            print(f"  Best params: {result['best_params']}")
        else:
            print("\n✗ RWR tuning returned None (Optuna not available?)")
            
    except Exception as e:
        print(f"\n✗ RWR tuning FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test CTQW tuning
    print("\n" + "="*80)
    print("Testing CTQW (Continuous-Time Quantum Walk) Tuning")
    print("="*80)
    try:
        result = analyzer.tune_quantum_walk_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            walk_type='ctqw',
            n_trials=5,  # Small number for quick test
            timeout=120
        )
        
        if result:
            print("\n✓ CTQW tuning SUCCESS!")
            print(f"  Best recall@50: {result['best_value']:.4f}")
            print(f"  Best params: {result['best_params']}")
        else:
            print("\n✗ CTQW tuning returned None (Optuna not available?)")
            
    except Exception as e:
        print(f"\n✗ CTQW tuning FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test DTQW tuning
    print("\n" + "="*80)
    print("Testing DTQW (Discrete-Time Quantum Walk) Tuning")
    print("="*80)
    try:
        result = analyzer.tune_quantum_walk_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            walk_type='dtqw',
            n_trials=5,  # Small number for quick test
            timeout=120
        )
        
        if result:
            print("\n✓ DTQW tuning SUCCESS!")
            print(f"  Best recall@50: {result['best_value']:.4f}")
            print(f"  Best params: {result['best_params']}")
        else:
            print("\n✗ DTQW tuning returned None (Optuna not available?)")
            
    except Exception as e:
        print(f"\n✗ DTQW tuning FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)

if __name__ == '__main__':
    test_quantum_walk_tuning()

# Made with Bob
