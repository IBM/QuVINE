"""
Unit tests for random graph generators.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import networkx as nx
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


def test_erdos_renyi():
    """Test Erdős-Rényi graph generation."""
    G = generate_erdos_renyi(n=50, p=0.1, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    print("✓ Erdős-Rényi test passed")


def test_barabasi_albert():
    """Test Barabási-Albert graph generation."""
    G = generate_barabasi_albert(n=50, m=3, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    print("✓ Barabási-Albert test passed")


def test_watts_strogatz():
    """Test Watts-Strogatz graph generation."""
    G = generate_watts_strogatz(n=50, k=4, p=0.3, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    print("✓ Watts-Strogatz test passed")


def test_powerlaw_cluster():
    """Test powerlaw cluster graph generation."""
    G = generate_powerlaw_cluster(n=50, m=3, p=0.3, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    print("✓ Powerlaw cluster test passed")


def test_stochastic_block_model():
    """Test stochastic block model generation."""
    sizes = [20, 30]
    p_matrix = [[0.3, 0.05], [0.05, 0.3]]
    G = generate_stochastic_block_model(sizes, p_matrix, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    print("✓ Stochastic block model test passed")


def test_random_geometric():
    """Test random geometric graph generation."""
    G = generate_random_geometric(n=50, radius=0.2, seed=42)
    assert G.number_of_nodes() == 50
    assert isinstance(G, nx.Graph)
    assert 'pos' in G.nodes[0]
    print("✓ Random geometric test passed")


def test_modular_network():
    """Test modular network generation."""
    G, communities = generate_modular_network(
        num_communities=3,
        nodes_per_community=15,
        p_intra=0.3,
        p_inter=0.01,
        seed=42
    )
    assert G.number_of_nodes() == 45
    assert len(set(communities.values())) == 3
    assert 'community' in G.nodes[0]
    print("✓ Modular network test passed")


def test_hierarchical_network():
    """Test hierarchical network generation."""
    G, node_levels = generate_hierarchical_network(
        levels=3,
        branching_factor=3,
        p_level=0.1,
        seed=42
    )
    assert G.number_of_nodes() > 0
    assert len(set(node_levels.values())) == 3
    assert 'level' in G.nodes[0]
    print("✓ Hierarchical network test passed")


def test_core_periphery():
    """Test core-periphery network generation."""
    G, core_nodes, periphery_nodes = generate_core_periphery(
        n_core=10,
        n_periphery=40,
        p_core=0.5,
        p_core_periphery=0.1,
        seed=42
    )
    assert G.number_of_nodes() == 50
    assert len(core_nodes) == 10
    assert len(periphery_nodes) == 40
    assert 'type' in G.nodes[0]
    print("✓ Core-periphery test passed")


def test_bipartite_random():
    """Test bipartite random graph generation."""
    G, set1, set2 = generate_bipartite_random(n1=20, n2=30, p=0.1, seed=42)
    assert G.number_of_nodes() == 50
    assert len(set1) == 20
    assert len(set2) == 30
    assert nx.is_bipartite(G)
    print("✓ Bipartite random test passed")


def test_add_hub_nodes():
    """Test adding hub nodes."""
    G = generate_erdos_renyi(n=40, p=0.05, seed=42)
    G, hub_nodes = add_hub_nodes(G, num_hubs=3, hub_degree=10, seed=42)
    assert G.number_of_nodes() == 43
    assert len(hub_nodes) == 3
    print("✓ Add hub nodes test passed")


def test_graph_with_seeds_and_targets():
    """Test graph generation with seeds and targets."""
    G, seeds, targets = generate_graph_with_seeds_and_targets(
        n=50,
        num_seeds=5,
        num_targets=10,
        graph_type='barabasi_albert',
        m=3,
        seed=42
    )
    assert G.number_of_nodes() == 50
    assert len(seeds) == 5
    assert len(targets) == 10
    assert 'role' in G.nodes[0]
    print("✓ Graph with seeds and targets test passed")


def test_get_graph_statistics():
    """Test graph statistics computation."""
    G = generate_barabasi_albert(n=50, m=3, seed=42)
    stats = get_graph_statistics(G)
    
    assert 'num_nodes' in stats
    assert 'num_edges' in stats
    assert 'density' in stats
    assert 'avg_degree' in stats
    assert stats['num_nodes'] == 50
    print("✓ Graph statistics test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Random Graph Generator Tests")
    print("="*60 + "\n")
    
    test_erdos_renyi()
    test_barabasi_albert()
    test_watts_strogatz()
    test_powerlaw_cluster()
    test_stochastic_block_model()
    test_random_geometric()
    test_modular_network()
    test_hierarchical_network()
    test_core_periphery()
    test_bipartite_random()
    test_add_hub_nodes()
    test_graph_with_seeds_and_targets()
    test_get_graph_statistics()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()

# Made with Bob
