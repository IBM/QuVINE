#!/usr/bin/env python3
"""
Test Q-Caliber GCN-MF Hyperparameter Tuning

This script tests the Bayesian hyperparameter optimization for Q-Caliber methods.
"""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_qcaliber_tuning():
    """Test Q-Caliber hyperparameter tuning."""
    
    print("="*80)
    print("Testing Q-Caliber GCN-MF Hyperparameter Tuning")
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
    
    # Test Heat GCN-MF tuning
    print("\n" + "="*80)
    print("Testing HGCNMF (Heat GCN-MF) Tuning")
    print("="*80)
    try:
        result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            diffusion_type='heat',
            n_trials=5,  # Small number for quick test
            timeout=60
        )
        
        if result:
            print("\n✓ HGCNMF tuning SUCCESS!")
            print(f"  Best recall@50: {result['best_value']:.4f}")
            print(f"  Best params: {result['best_params']}")
        else:
            print("\n✗ HGCNMF tuning returned None (Optuna not available?)")
            
    except Exception as e:
        print(f"\n✗ HGCNMF tuning FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test Poly GCN-MF tuning
    print("\n" + "="*80)
    print("Testing PGCNMF (Poly GCN-MF) Tuning")
    print("="*80)
    try:
        result = analyzer.tune_qcaliber_gcnmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            diffusion_type='poly',
            n_trials=5,  # Small number for quick test
            timeout=60
        )
        
        if result:
            print("\n✓ PGCNMF tuning SUCCESS!")
            print(f"  Best recall@50: {result['best_value']:.4f}")
            print(f"  Best params: {result['best_params']}")
        else:
            print("\n✗ PGCNMF tuning returned None (Optuna not available?)")
            
    except Exception as e:
        print(f"\n✗ PGCNMF tuning FAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("Test Complete")
    print("="*80)

if __name__ == '__main__':
    test_qcaliber_tuning()

# Made with Bob
