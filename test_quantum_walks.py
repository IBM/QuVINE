#!/usr/bin/env python3
"""
Test script to verify quantum walk methods are working.
"""

import sys
import networkx as nx
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_quantum_walks():
    """Test that quantum walk methods generate embeddings."""
    print("=" * 80)
    print("Testing Quantum Walk Methods")
    print("=" * 80)
    
    # Create test graph
    print("\n1. Creating test graph...")
    G = nx.karate_club_graph()
    n_nodes = G.number_of_nodes()
    print(f"   Graph: {n_nodes} nodes, {G.number_of_edges()} edges")
    
    # Create seeds and targets
    np.random.seed(42)
    all_nodes = list(G.nodes())
    seeds = [0, 1, 2]  # First 3 nodes as seeds
    targets = [n for n in all_nodes if n not in seeds][:10]
    
    print(f"   Seeds: {seeds}")
    print(f"   Targets: {len(targets)} nodes")
    
    # Initialize analyzer
    print("\n2. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        embedding_dim=64,
        seed=42,
        n_jobs=1
    )
    
    # Test each quantum walk method
    methods = ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw']
    
    for method in methods:
        print(f"\n{'=' * 80}")
        print(f"Testing {method.upper()}")
        print("=" * 80)
        
        try:
            print(f"Generating embedding for {method}...")
            embedding = analyzer.run_embedding_method(
                method_name=method,
                G=G,
                seeds=seeds,
                targets=targets
            )
            
            print(f"✓ SUCCESS!")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding dtype: {embedding.dtype}")
            print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
            print(f"  Non-zero elements: {np.count_nonzero(embedding)}/{embedding.size}")
            
        except Exception as e:
            print(f"✗ FAILED!")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Test Complete")
    print("=" * 80)

if __name__ == "__main__":
    test_quantum_walks()

# Made with Bob
