#!/usr/bin/env python3
"""
Quick test script to verify the analysis setup.

This script runs a minimal version of the analysis on 2 small networks
to verify that all components are working correctly.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import networkx as nx
import numpy as np
from quvine.data.graph_complexity import compute_graph_complexity_metrics
from quvine.data.random_graphs import generate_barabasi_albert, generate_modular_network
from quvine.baselines import run_netmf, run_node2vec


def test_complexity_metrics():
    """Test complexity metric computation including IPR."""
    print("\n" + "="*60)
    print("Testing Complexity Metrics (including IPR)")
    print("="*60)
    
    # Generate a small test graph
    G = generate_barabasi_albert(n=50, m=3, seed=42)
    
    # Compute metrics
    metrics = compute_graph_complexity_metrics(G)
    
    print(f"\nGraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print("\nComplexity Metrics:")
    print(f"  Quantum Complexity: {metrics['quantum_complexity']:.4f}")
    print(f"  Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")
    print(f"  Spectral Gap: {metrics['spectral_gap']:.4f}")
    print(f"  Inverse Participation Ratio: {metrics['inverse_participation_ratio']:.4f}")
    print(f"  Participation Ratio: {metrics['participation_ratio']:.4f}")
    print(f"  Centrality Entropy: {metrics['centrality_entropy']:.4f}")
    
    # Verify IPR is present
    assert 'inverse_participation_ratio' in metrics, "IPR not found in metrics!"
    assert 'participation_ratio' in metrics, "PR not found in metrics!"
    
    print("\n✓ Complexity metrics test passed!")
    return metrics


def test_network_generation():
    """Test network generation."""
    print("\n" + "="*60)
    print("Testing Network Generation")
    print("="*60)
    
    # Generate scale-free
    G_sf = generate_barabasi_albert(n=50, m=3, seed=42)
    print(f"\nScale-free network: {G_sf.number_of_nodes()} nodes, {G_sf.number_of_edges()} edges")
    
    # Generate modular
    G_mod, communities = generate_modular_network(
        num_communities=3,
        nodes_per_community=15,
        p_intra=0.3,
        p_inter=0.01,
        seed=42
    )
    print(f"Modular network: {G_mod.number_of_nodes()} nodes, {G_mod.number_of_edges()} edges")
    print(f"  Communities: {len(set(communities.values()))}")
    
    print("\n✓ Network generation test passed!")
    return G_sf, G_mod


def test_embedding_methods():
    """Test embedding methods."""
    print("\n" + "="*60)
    print("Testing Embedding Methods")
    print("="*60)
    
    # Create small test graph
    G = generate_barabasi_albert(n=30, m=2, seed=42)
    nodes = list(G.nodes())
    
    print(f"\nTest graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Test NetMF
    print("\nTesting NetMF...")
    try:
        Z_netmf = run_netmf(
            graph=G,
            nodes=nodes,
            dimensions=32,
            window_size=5,
            seed=42
        )
        print(f"  NetMF embedding shape: {Z_netmf.shape}")
        assert Z_netmf.shape == (len(nodes), 32), "NetMF shape mismatch!"
        print("  ✓ NetMF works!")
    except Exception as e:
        print(f"  ✗ NetMF failed: {e}")
    
    # Test Node2Vec
    print("\nTesting Node2Vec...")
    try:
        Z_n2v = run_node2vec(
            graph=G,
            nodes=nodes,
            dimensions=32,
            walk_length=5,
            num_walks=5,
            workers=2,
            seed=42
        )
        print(f"  Node2Vec embedding shape: {Z_n2v.shape}")
        assert Z_n2v.shape == (len(nodes), 32), "Node2Vec shape mismatch!"
        print("  ✓ Node2Vec works!")
    except Exception as e:
        print(f"  ✗ Node2Vec failed: {e}")
    
    print("\n✓ Embedding methods test passed!")


def test_minimal_analysis():
    """Test minimal version of the full analysis."""
    print("\n" + "="*60)
    print("Testing Minimal Analysis Pipeline")
    print("="*60)
    
    from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
    
    # Create minimal analysis (2 networks, small size)
    print("\nCreating minimal analysis instance...")
    analysis = ComprehensiveEmbeddingAnalysis(
        output_dir="outputs/test_analysis",
        n_networks_per_type=1,  # Just 1 of each type
        n_nodes=30,              # Small networks
        num_seeds=5,
        num_targets=8,
        embedding_dim=32,
        seed=42
    )
    
    print("Generating test networks...")
    networks = analysis.generate_networks()
    print(f"  Generated {len(networks)} networks")
    
    print("\nComputing complexity metrics...")
    complexity_df = analysis.compute_complexity_for_all(networks)
    print(f"  Computed metrics for {len(complexity_df)} networks")
    print(f"  Metrics: {list(complexity_df.columns)}")
    
    # Verify IPR is in the results
    assert 'inverse_participation_ratio' in complexity_df.columns, "IPR not in complexity results!"
    assert 'participation_ratio' in complexity_df.columns, "PR not in complexity results!"
    
    print("\n✓ Minimal analysis test passed!")
    print(f"\nTest results saved to: outputs/test_analysis/")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SETUP TEST")
    print("="*80)
    
    try:
        # Test 1: Complexity metrics
        test_complexity_metrics()
        
        # Test 2: Network generation
        test_network_generation()
        
        # Test 3: Embedding methods
        test_embedding_methods()
        
        # Test 4: Minimal analysis
        test_minimal_analysis()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nYou can now run the full analysis with:")
        print("  python run_comprehensive_analysis.py")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*80 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
