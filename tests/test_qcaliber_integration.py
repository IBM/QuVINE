"""
Test script to verify Q-Caliber integration with comprehensive_embedding_analysis.py
"""

import networkx as nx
import numpy as np
from src.quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_qcaliber_methods():
    """Test that all Q-Caliber methods can be called successfully."""
    
    print("=" * 80)
    print("Testing Q-Caliber Integration")
    print("=" * 80)
    
    # Create a small test graph
    print("\n1. Creating test graph...")
    G = nx.karate_club_graph()
    print(f"   Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Select seeds and targets
    print("\n2. Selecting seeds and targets...")
    nodes = list(G.nodes())
    np.random.seed(42)
    seeds = list(np.random.choice(nodes, size=5, replace=False))
    targets = list(np.random.choice([n for n in nodes if n not in seeds], size=10, replace=False))
    print(f"   Seeds: {seeds}")
    print(f"   Targets: {targets[:5]}... (showing first 5)")
    
    # Initialize analyzer
    print("\n3. Initializing ComprehensiveEmbeddingAnalysis...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir="test_output",
        embedding_dim=32,  # Small for testing
        seed=42
    )
    
    # Test each Q-Caliber method
    methods_to_test = [
        'quvine_heat',
        'quvine_poly',
        'quvine_hgcnmf',
        'quvine_pgcnmf',
        'quvine_fused',  # Fuse all methods
        'quvine_fused_heat_poly',  # Fuse only heat and poly
        'quvine_fused_ctqw_rwr_hgcnmf'  # Fuse ctqw, rwr, and hgcnmf
    ]
    
    results = {}
    
    for method in methods_to_test:
        print(f"\n4. Testing method: {method}")
        print("-" * 80)
        try:
            embeddings = analyzer.run_embedding_method(
                method_name=method,
                G=G,
                seeds=seeds,
                targets=targets
            )
            
            print(f"   ✓ Success!")
            print(f"   Embedding shape: {embeddings.shape}")
            print(f"   Embedding dtype: {embeddings.dtype}")
            print(f"   Embedding range: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
            
            # Verify shape
            expected_shape = (G.number_of_nodes(), analyzer.embedding_dim)
            if embeddings.shape == expected_shape:
                print(f"   ✓ Shape matches expected: {expected_shape}")
            else:
                print(f"   ✗ Shape mismatch! Expected {expected_shape}, got {embeddings.shape}")
            
            results[method] = {
                'status': 'success',
                'shape': embeddings.shape,
                'embeddings': embeddings
            }
            
        except Exception as e:
            print(f"   ✗ Failed with error:")
            print(f"   {type(e).__name__}: {str(e)}")
            results[method] = {
                'status': 'failed',
                'error': str(e)
            }
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    successful = [m for m, r in results.items() if r['status'] == 'success']
    failed = [m for m, r in results.items() if r['status'] == 'failed']
    
    print(f"\nSuccessful methods ({len(successful)}/{len(methods_to_test)}):")
    for method in successful:
        print(f"  ✓ {method}")
    
    if failed:
        print(f"\nFailed methods ({len(failed)}/{len(methods_to_test)}):")
        for method in failed:
            print(f"  ✗ {method}: {results[method]['error']}")
    
    # Test that embeddings are different
    if len(successful) > 1:
        print("\n5. Verifying embeddings are different...")
        emb_list = [results[m]['embeddings'] for m in successful]
        for i in range(len(emb_list)):
            for j in range(i+1, len(emb_list)):
                diff = np.linalg.norm(emb_list[i] - emb_list[j])
                print(f"   Distance between {successful[i]} and {successful[j]}: {diff:.4f}")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    try:
        results = test_qcaliber_methods()
        
        # Check if all methods succeeded
        all_success = all(r['status'] == 'success' for r in results.values())
        
        if all_success:
            print("\n✓ ALL TESTS PASSED!")
            exit(0)
        else:
            print("\n✗ SOME TESTS FAILED!")
            exit(1)
            
    except Exception as e:
        print(f"\n✗ TEST SCRIPT FAILED: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)

# Made with Bob
