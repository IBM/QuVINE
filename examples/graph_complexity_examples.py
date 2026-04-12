"""
Examples of computing graph complexity metrics in QuVINE.

This script demonstrates how to compute various complexity metrics
for different graph types, including Laplacian-based and quantum-inspired measures.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from quvine.data.random_graphs import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_powerlaw_cluster,
    generate_modular_network,
    generate_hierarchical_network,
    generate_core_periphery,
)
from quvine.complexity.graph import (
    compute_laplacian_spectrum,
    compute_spectral_gap,
    compute_algebraic_connectivity,
    compute_spectral_entropy,
    compute_von_neumann_entropy,
    compute_estrada_index,
    compute_quantum_complexity,
    compute_graph_complexity_metrics,
    compare_graph_complexities,
    rank_graphs_by_complexity,
)


def example_single_graph_complexity():
    """Compute complexity metrics for a single graph."""
    print("\n" + "="*60)
    print("Example 1: Complexity Metrics for a Single Graph")
    print("="*60)
    
    # Generate a scale-free network
    G = generate_barabasi_albert(n=100, m=3, seed=42)
    
    # Compute all complexity metrics
    metrics = compute_graph_complexity_metrics(G)
    
    print(f"\nGraph: Barabási-Albert (n=100, m=3)")
    print(f"Nodes: {metrics['num_nodes']}, Edges: {metrics['num_edges']}")
    print("\nComplexity Metrics:")
    print(f"  Spectral Gap: {metrics['spectral_gap']:.4f}")
    print(f"  Algebraic Connectivity: {metrics['algebraic_connectivity']:.4f}")
    print(f"  Spectral Entropy: {metrics['spectral_entropy']:.4f}")
    print(f"  Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")
    print(f"  Estrada Index: {metrics['estrada_index']:.2f}")
    print(f"  Quantum Complexity: {metrics['quantum_complexity']:.4f}")


def example_compare_graph_types():
    """Compare complexity across different graph types."""
    print("\n" + "="*60)
    print("Example 2: Comparing Complexity Across Graph Types")
    print("="*60)
    
    n = 100
    seed = 42
    
    # Generate different graph types
    graphs = {
        'Erdős-Rényi': generate_erdos_renyi(n, p=0.05, seed=seed),
        'Scale-Free': generate_barabasi_albert(n, m=3, seed=seed),
        'Small-World': generate_watts_strogatz(n, k=6, p=0.3, seed=seed),
        'Powerlaw Cluster': generate_powerlaw_cluster(n, m=3, p=0.3, seed=seed),
    }
    
    # Compute complexity for all graphs
    complexities = compare_graph_complexities(graphs)
    
    # Create comparison table
    print("\nComplexity Comparison:")
    print(f"{'Graph Type':<20} {'Spectral Gap':<15} {'VN Entropy':<15} {'Quantum Complexity':<20}")
    print("-" * 70)
    
    for name, metrics in complexities.items():
        print(f"{name:<20} {metrics['spectral_gap']:<15.4f} "
              f"{metrics['von_neumann_entropy']:<15.4f} "
              f"{metrics['quantum_complexity']:<20.4f}")


def example_modular_vs_random():
    """Compare modular network with random network."""
    print("\n" + "="*60)
    print("Example 3: Modular vs Random Network Complexity")
    print("="*60)
    
    n = 100
    seed = 42
    
    # Generate modular network
    G_modular, _ = generate_modular_network(
        num_communities=5,
        nodes_per_community=20,
        p_intra=0.3,
        p_inter=0.01,
        seed=seed
    )
    
    # Generate random network with similar density
    G_random = generate_erdos_renyi(n, p=0.05, seed=seed)
    
    # Compute metrics
    metrics_modular = compute_graph_complexity_metrics(G_modular)
    metrics_random = compute_graph_complexity_metrics(G_random)
    
    print("\nModular Network:")
    print(f"  Spectral Gap: {metrics_modular['spectral_gap']:.4f}")
    print(f"  Von Neumann Entropy: {metrics_modular['von_neumann_entropy']:.4f}")
    print(f"  Quantum Complexity: {metrics_modular['quantum_complexity']:.4f}")
    
    print("\nRandom Network:")
    print(f"  Spectral Gap: {metrics_random['spectral_gap']:.4f}")
    print(f"  Von Neumann Entropy: {metrics_random['von_neumann_entropy']:.4f}")
    print(f"  Quantum Complexity: {metrics_random['quantum_complexity']:.4f}")
    
    print("\nInterpretation:")
    if metrics_modular['spectral_gap'] > metrics_random['spectral_gap']:
        print("  - Modular network has larger spectral gap (better connectivity)")
    else:
        print("  - Random network has larger spectral gap")
    
    if metrics_modular['quantum_complexity'] > metrics_random['quantum_complexity']:
        print("  - Modular network has higher quantum complexity")
    else:
        print("  - Random network has higher quantum complexity")


def example_laplacian_spectrum():
    """Visualize Laplacian spectrum for different graphs."""
    print("\n" + "="*60)
    print("Example 4: Laplacian Spectrum Analysis")
    print("="*60)
    
    n = 50
    seed = 42
    
    graphs = {
        'Regular': generate_watts_strogatz(n, k=4, p=0.0, seed=seed),  # Regular ring
        'Small-World': generate_watts_strogatz(n, k=4, p=0.3, seed=seed),
        'Scale-Free': generate_barabasi_albert(n, m=2, seed=seed),
    }
    
    print("\nLaplacian Eigenvalue Statistics:")
    print(f"{'Graph Type':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
    print("-" * 55)
    
    for name, G in graphs.items():
        eigenvalues = compute_laplacian_spectrum(G, normalized=True)
        print(f"{name:<15} {eigenvalues.min():<10.4f} {eigenvalues.max():<10.4f} "
              f"{eigenvalues.mean():<10.4f} {eigenvalues.std():<10.4f}")


def example_hierarchical_complexity():
    """Analyze complexity of hierarchical networks."""
    print("\n" + "="*60)
    print("Example 5: Hierarchical Network Complexity")
    print("="*60)
    
    seed = 42
    
    # Generate hierarchical networks with different levels
    hierarchies = {}
    for levels in [2, 3, 4]:
        G, _ = generate_hierarchical_network(
            levels=levels,
            branching_factor=3,
            p_level=0.1,
            seed=seed
        )
        hierarchies[f'{levels} levels'] = G
    
    # Compare complexity
    complexities = compare_graph_complexities(hierarchies)
    
    print("\nComplexity vs Hierarchy Depth:")
    print(f"{'Levels':<15} {'Nodes':<10} {'Spectral Gap':<15} {'Quantum Complexity':<20}")
    print("-" * 60)
    
    for name, metrics in complexities.items():
        print(f"{name:<15} {metrics['num_nodes']:<10} "
              f"{metrics['spectral_gap']:<15.4f} "
              f"{metrics['quantum_complexity']:<20.4f}")


def example_core_periphery_complexity():
    """Analyze complexity of core-periphery networks."""
    print("\n" + "="*60)
    print("Example 6: Core-Periphery Network Complexity")
    print("="*60)
    
    seed = 42
    
    # Generate core-periphery networks with different core sizes
    networks = {}
    for core_ratio in [0.1, 0.2, 0.3]:
        n_core = int(100 * core_ratio)
        n_periphery = 100 - n_core
        G, _, _ = generate_core_periphery(
            n_core=n_core,
            n_periphery=n_periphery,
            p_core=0.5,
            p_core_periphery=0.1,
            p_periphery=0.01,
            seed=seed
        )
        networks[f'Core {int(core_ratio*100)}%'] = G
    
    # Compare complexity
    complexities = compare_graph_complexities(networks)
    
    print("\nComplexity vs Core Size:")
    print(f"{'Core Size':<15} {'Spectral Gap':<15} {'VN Entropy':<15} {'Quantum Complexity':<20}")
    print("-" * 65)
    
    for name, metrics in complexities.items():
        print(f"{name:<15} {metrics['spectral_gap']:<15.4f} "
              f"{metrics['von_neumann_entropy']:<15.4f} "
              f"{metrics['quantum_complexity']:<20.4f}")


def example_ranking_by_complexity():
    """Rank graphs by different complexity metrics."""
    print("\n" + "="*60)
    print("Example 7: Ranking Graphs by Complexity")
    print("="*60)
    
    n = 100
    seed = 42
    
    graphs = {
        'Erdős-Rényi': generate_erdos_renyi(n, p=0.05, seed=seed),
        'Scale-Free': generate_barabasi_albert(n, m=3, seed=seed),
        'Small-World': generate_watts_strogatz(n, k=6, p=0.3, seed=seed),
        'Powerlaw Cluster': generate_powerlaw_cluster(n, m=3, p=0.3, seed=seed),
        'Modular': generate_modular_network(5, 20, 0.3, 0.01, seed=seed)[0],
    }
    
    # Rank by quantum complexity
    print("\nRanking by Quantum Complexity:")
    rankings = rank_graphs_by_complexity(graphs, metric='quantum_complexity')
    for i, (name, score) in enumerate(rankings, 1):
        print(f"  {i}. {name:<20} Score: {score:.4f}")
    
    # Rank by von Neumann entropy
    print("\nRanking by Von Neumann Entropy:")
    rankings = rank_graphs_by_complexity(graphs, metric='von_neumann_entropy')
    for i, (name, score) in enumerate(rankings, 1):
        print(f"  {i}. {name:<20} Score: {score:.4f}")
    
    # Rank by spectral gap
    print("\nRanking by Spectral Gap:")
    rankings = rank_graphs_by_complexity(graphs, metric='spectral_gap')
    for i, (name, score) in enumerate(rankings, 1):
        print(f"  {i}. {name:<20} Score: {score:.4f}")


def example_complexity_dataframe():
    """Create a pandas DataFrame with all complexity metrics."""
    print("\n" + "="*60)
    print("Example 8: Complexity Metrics DataFrame")
    print("="*60)
    
    n = 100
    seed = 42
    
    graphs = {
        'Erdős-Rényi': generate_erdos_renyi(n, p=0.05, seed=seed),
        'Scale-Free': generate_barabasi_albert(n, m=3, seed=seed),
        'Small-World': generate_watts_strogatz(n, k=6, p=0.3, seed=seed),
        'Modular': generate_modular_network(5, 20, 0.3, 0.01, seed=seed)[0],
    }
    
    # Compute all metrics
    complexities = compare_graph_complexities(graphs)
    
    # Convert to DataFrame
    df = pd.DataFrame(complexities).T
    
    # Select key metrics
    key_metrics = [
        'num_nodes', 'num_edges', 'spectral_gap', 
        'von_neumann_entropy', 'quantum_complexity'
    ]
    
    print("\nComplexity Metrics Summary:")
    print(df[key_metrics].round(4))
    
    return df


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("QuVINE Graph Complexity Metrics Examples")
    print("="*60)
    
    example_single_graph_complexity()
    example_compare_graph_types()
    example_modular_vs_random()
    example_laplacian_spectrum()
    example_hierarchical_complexity()
    example_core_periphery_complexity()
    example_ranking_by_complexity()
    example_complexity_dataframe()
    example_centrality_complexity()
    example_fiedler_sparse()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()




def example_centrality_complexity():
    """Demonstrate Laplacian centrality complexity metrics."""
    print("\n" + "="*60)
    print("Example 9: Laplacian Centrality Complexity")
    print("="*60)
    
    from quvine.complexity.graph import compute_laplacian_centrality_complexity
    
    n = 100
    seed = 42
    
    # Compare scale-free vs random networks
    graphs = {
        'Scale-Free': generate_barabasi_albert(n, m=3, seed=seed),
        'Random': generate_erdos_renyi(n, p=0.05, seed=seed),
        'Modular': generate_modular_network(5, 20, 0.3, 0.01, seed=seed)[0],
    }
    
    print("\nCentrality Complexity Comparison:")
    print(f"{'Graph Type':<15} {'Entropy':<12} {'Gini':<12} {'Range':<12} {'Variance':<12}")
    print("-" * 63)
    
    for name, G in graphs.items():
        metrics = compute_laplacian_centrality_complexity(G, normalized=True)
        print(f"{name:<15} {metrics['centrality_entropy']:<12.4f} "
              f"{metrics['centrality_gini']:<12.4f} "
              f"{metrics['centrality_range']:<12.4f} "
              f"{metrics['centrality_variance']:<12.4f}")
    
    print("\nInterpretation:")
    print("  - Higher Gini coefficient = more centralized (hub-based) structure")
    print("  - Higher entropy = more uniform centrality distribution")
    print("  - Higher range = greater disparity between most/least central nodes")


def example_fiedler_sparse():
    """Demonstrate efficient Fiedler eigenvalue computation."""
    print("\n" + "="*60)
    print("Example 10: Sparse Fiedler Eigenvalue Computation")
    print("="*60)
    
    from quvine.complexity.graph import fiedler_eigenvalue_sparse
    import time
    
    print("\nComputing Fiedler eigenvalue for graphs of different sizes:")
    print(f"{'Nodes':<10} {'Edges':<10} {'λ₂':<15} {'Time (s)':<12}")
    print("-" * 47)
    
    for n in [100, 500, 1000, 2000]:
        G = generate_barabasi_albert(n, m=3, seed=42)
        
        start = time.time()
        lambda2, fiedler_vec = fiedler_eigenvalue_sparse(G, normalized=True)
        elapsed = time.time() - start
        
        print(f"{n:<10} {G.number_of_edges():<10} {lambda2:<15.6f} {elapsed:<12.4f}")
    
    print("\nNote: Sparse method is efficient even for large graphs (2000+ nodes)")
    print("      Fiedler vector can be used for graph partitioning/clustering")

