"""
Examples of using the random graph generators in QuVINE.

This script demonstrates how to generate various types of random graphs
with known structures or specific properties suitable for embedding tasks.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx
import matplotlib.pyplot as plt
from quvine.data.random_graphs import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_powerlaw_cluster,
    generate_stochastic_block_model,
    generate_random_geometric,
    generate_modular_network,
    generate_hierarchical_network,
    generate_core_periphery,
    generate_bipartite_random,
    add_hub_nodes,
    generate_graph_with_seeds_and_targets,
    get_graph_statistics
)


def example_erdos_renyi():
    """Generate and visualize an Erdős-Rényi random graph."""
    print("\n" + "="*60)
    print("Example 1: Erdős-Rényi Random Graph")
    print("="*60)
    
    # G(n,p) model
    G = generate_erdos_renyi(n=100, p=0.05, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    
    return G


def example_scale_free():
    """Generate and visualize a scale-free network."""
    print("\n" + "="*60)
    print("Example 2: Barabási-Albert Scale-Free Network")
    print("="*60)
    
    G = generate_barabasi_albert(n=100, m=3, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Max degree (hub): {stats['max_degree']}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    
    return G


def example_small_world():
    """Generate and visualize a small-world network."""
    print("\n" + "="*60)
    print("Example 3: Watts-Strogatz Small-World Network")
    print("="*60)
    
    G = generate_watts_strogatz(n=100, k=6, p=0.3, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    if stats['is_connected']:
        print(f"Average shortest path: {stats['avg_shortest_path']:.2f}")
    
    return G


def example_modular_network():
    """Generate a modular network with community structure."""
    print("\n" + "="*60)
    print("Example 4: Modular Network with Communities")
    print("="*60)
    
    G, communities = generate_modular_network(
        num_communities=5,
        nodes_per_community=20,
        p_intra=0.3,
        p_inter=0.01,
        seed=42
    )
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Number of communities: {len(set(communities.values()))}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    
    return G, communities


def example_hierarchical_network():
    """Generate a hierarchical network."""
    print("\n" + "="*60)
    print("Example 5: Hierarchical Network")
    print("="*60)
    
    G, node_levels = generate_hierarchical_network(
        levels=4,
        branching_factor=3,
        p_level=0.1,
        seed=42
    )
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Number of levels: {len(set(node_levels.values()))}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    
    return G, node_levels


def example_core_periphery():
    """Generate a core-periphery network."""
    print("\n" + "="*60)
    print("Example 6: Core-Periphery Network")
    print("="*60)
    
    G, core_nodes, periphery_nodes = generate_core_periphery(
        n_core=20,
        n_periphery=80,
        p_core=0.5,
        p_core_periphery=0.1,
        p_periphery=0.01,
        seed=42
    )
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Core nodes: {len(core_nodes)}")
    print(f"Periphery nodes: {len(periphery_nodes)}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    
    return G, core_nodes, periphery_nodes


def example_bipartite():
    """Generate a bipartite network."""
    print("\n" + "="*60)
    print("Example 7: Bipartite Network")
    print("="*60)
    
    G, set1, set2 = generate_bipartite_random(
        n1=30,
        n2=50,
        p=0.1,
        seed=42
    )
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Set 1 size: {len(set1)}")
    print(f"Set 2 size: {len(set2)}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Is bipartite: {nx.is_bipartite(G)}")
    
    return G, set1, set2


def example_with_hubs():
    """Generate a network and add hub nodes."""
    print("\n" + "="*60)
    print("Example 8: Network with Added Hub Nodes")
    print("="*60)
    
    # Start with a random graph
    G = generate_erdos_renyi(n=80, p=0.05, seed=42)
    print(f"Original graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Add hub nodes
    G, hub_nodes = add_hub_nodes(G, num_hubs=5, hub_degree=20, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"After adding hubs - Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
    print(f"Hub nodes: {hub_nodes}")
    print(f"Max degree: {stats['max_degree']}")
    
    return G, hub_nodes


def example_seeds_and_targets():
    """Generate a graph with designated seed and target nodes for embedding evaluation."""
    print("\n" + "="*60)
    print("Example 9: Graph with Seeds and Targets for Embedding")
    print("="*60)
    
    G, seeds, targets = generate_graph_with_seeds_and_targets(
        n=100,
        num_seeds=10,
        num_targets=15,
        graph_type='barabasi_albert',
        m=3,
        seed=42
    )
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Seed nodes: {len(seeds)}")
    print(f"Target nodes: {len(targets)}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"\nFirst 5 seeds: {seeds[:5]}")
    print(f"First 5 targets: {targets[:5]}")
    
    return G, seeds, targets


def example_stochastic_block_model():
    """Generate a stochastic block model with custom community structure."""
    print("\n" + "="*60)
    print("Example 10: Stochastic Block Model")
    print("="*60)
    
    # Define 3 communities with different connection patterns
    sizes = [30, 40, 30]
    p_matrix = [
        [0.4, 0.05, 0.01],  # Community 0: high internal, low to others
        [0.05, 0.3, 0.1],   # Community 1: medium internal, some to community 2
        [0.01, 0.1, 0.5]    # Community 2: very high internal
    ]
    
    G = generate_stochastic_block_model(sizes, p_matrix, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Communities: {len(sizes)}")
    print(f"Community sizes: {sizes}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    
    return G


def example_geometric_graph():
    """Generate a random geometric graph."""
    print("\n" + "="*60)
    print("Example 11: Random Geometric Graph")
    print("="*60)
    
    G = generate_random_geometric(n=100, radius=0.15, dim=2, seed=42)
    stats = get_graph_statistics(G)
    
    print(f"Nodes: {stats['num_nodes']}")
    print(f"Edges: {stats['num_edges']}")
    print(f"Average degree: {stats['avg_degree']:.2f}")
    print(f"Clustering coefficient: {stats['avg_clustering']:.3f}")
    print("Note: Nodes have 'pos' attribute with 2D coordinates")
    
    return G


def compare_graph_types():
    """Compare different graph types for embedding suitability."""
    print("\n" + "="*60)
    print("Comparison: Different Graph Types for Embedding")
    print("="*60)
    
    n = 100
    seed = 42
    
    graphs = {
        'Erdős-Rényi': generate_erdos_renyi(n, p=0.05, seed=seed),
        'Scale-Free': generate_barabasi_albert(n, m=3, seed=seed),
        'Small-World': generate_watts_strogatz(n, k=6, p=0.3, seed=seed),
        'Powerlaw Cluster': generate_powerlaw_cluster(n, m=3, p=0.3, seed=seed),
    }
    
    print(f"\n{'Graph Type':<20} {'Nodes':<8} {'Edges':<8} {'Avg Deg':<10} {'Clustering':<12} {'Connected':<10}")
    print("-" * 80)
    
    for name, G in graphs.items():
        stats = get_graph_statistics(G)
        print(f"{name:<20} {stats['num_nodes']:<8} {stats['num_edges']:<8} "
              f"{stats['avg_degree']:<10.2f} {stats['avg_clustering']:<12.3f} "
              f"{str(stats['is_connected']):<10}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("QuVINE Random Graph Generator Examples")
    print("="*60)
    
    # Run individual examples
    example_erdos_renyi()
    example_scale_free()
    example_small_world()
    example_modular_network()
    example_hierarchical_network()
    example_core_periphery()
    example_bipartite()
    example_with_hubs()
    example_seeds_and_targets()
    example_stochastic_block_model()
    example_geometric_graph()
    
    # Comparison
    compare_graph_types()
    
    print("\n" + "="*60)
    print("All examples completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()

# Made with Bob
