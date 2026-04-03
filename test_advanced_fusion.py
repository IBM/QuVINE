#!/usr/bin/env python3
"""
Test script for advanced fusion methods with QuVINE embeddings.
"""

import numpy as np
import networkx as nx
from src.quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_advanced_fusion():
    """Test the new SVD shared/private fusion methods."""
    
    print("=" * 80)
    print("Testing Advanced Fusion Methods")
    print("=" * 80)
    
    # Create a small test graph
    G = nx.karate_club_graph()
    print(f"\nTest graph: Karate Club (n={G.number_of_nodes()}, m={G.number_of_edges()})")
    
    # Select some seed and target nodes
    seeds = [0, 1, 2]
    targets = [10, 11, 12, 13, 14]
    
    print(f"Seeds: {seeds}")
    print(f"Targets: {targets}")
    
    # Initialize analyzer
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir="test_output",
        embedding_dim=32,
        n_jobs=1,
        seed=42
    )
    
    # Test methods to verify
    test_methods = [
        'quvine_heat',
        'quvine_poly',
        'quvine_hgcnmf',
        'quvine_pgcnmf',
        'quvine_fused_svd_heat_poly',  # Standard SVD fusion
        'quvine_fused_svd_shared_priv_heat_poly',  # SVD shared/private with attention gate
        'quvine_fused_svd_shared_priv_moe_heat_poly',  # SVD shared/private with MoE gate
    ]
    
    print("\n" + "=" * 80)
    print("Testing Embedding Methods")
    print("=" * 80)
    
    results = {}
    for method in test_methods:
        try:
            print(f"\n[{method}]")
            print("-" * 40)
            
            embedding = analyzer.run_embedding_method(
                method_name=method,
                G=G,
                seeds=seeds,
                targets=targets,
                cfg=None
            )
            
            results[method] = embedding
            print(f"✓ Success! Shape: {embedding.shape}")
            print(f"  Mean: {embedding.mean():.4f}, Std: {embedding.std():.4f}")
            print(f"  Min: {embedding.min():.4f}, Max: {embedding.max():.4f}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Successful methods: {len(results)}/{len(test_methods)}")
    
    if len(results) == len(test_methods):
        print("\n✓ All methods passed!")
        
        # Compare fusion methods
        print("\n" + "=" * 80)
        print("Fusion Method Comparison")
        print("=" * 80)
        
        svd_emb = results['quvine_fused_svd_heat_poly']
        sp_att_emb = results['quvine_fused_svd_shared_priv_heat_poly']
        sp_moe_emb = results['quvine_fused_svd_shared_priv_moe_heat_poly']
        
        # Compute pairwise similarities
        def cosine_sim(a, b):
            return np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        print(f"\nCosine similarities:")
        print(f"  SVD vs SVD-SP-Attention: {cosine_sim(svd_emb, sp_att_emb):.4f}")
        print(f"  SVD vs SVD-SP-MoE:       {cosine_sim(svd_emb, sp_moe_emb):.4f}")
        print(f"  SVD-SP-Att vs SVD-SP-MoE: {cosine_sim(sp_att_emb, sp_moe_emb):.4f}")
        
        return True
    else:
        print("\n✗ Some methods failed")
        return False

if __name__ == "__main__":
    success = test_advanced_fusion()
    exit(0 if success else 1)

# Made with Bob
