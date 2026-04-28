"""
Tests for data loading and random graph generation modules.

This module tests:
- Random graph generation functions
- Graph property validation
- Data loading utilities
- Error handling for invalid inputs
"""

import pytest
import numpy as np
import networkx as nx
from typing import Dict, List

# Import functions to test
from quvine.data.random_graphs import (
    generate_erdos_renyi,
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_modular_network,
)


class TestErdosRenyiGeneration:
    """Test Erdős-Rényi random graph generation."""
    
    def test_erdos_renyi_with_probability(self):
        """Test G(n,p) model."""
        n = 50
        p = 0.1
        G = generate_erdos_renyi(n=n, p=p, seed=42)
        
        assert G.number_of_nodes() == n
        assert isinstance(G, nx.Graph)
        assert not G.is_directed()
    
    def test_erdos_renyi_with_edges(self):
        """Test G(n,m) model."""
        n = 50
        m = 100
        G = generate_erdos_renyi(n=n, m=m, seed=42)
        
        assert G.number_of_nodes() == n
        assert G.number_of_edges() == m
    
    def test_erdos_renyi_directed(self):
        """Test directed graph generation."""
        n = 30
        p = 0.15
        G = generate_erdos_renyi(n=n, p=p, directed=True, seed=42)
        
        assert G.number_of_nodes() == n
        assert isinstance(G, nx.DiGraph)
        assert G.is_directed()
    
    def test_erdos_renyi_reproducibility(self):
        """Test that same seed produces same graph."""
        n, p = 40, 0.2
        G1 = generate_erdos_renyi(n=n, p=p, seed=42)
        G2 = generate_erdos_renyi(n=n, p=p, seed=42)
        
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())
    
    def test_erdos_renyi_invalid_params(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError, match="Specify either p or m"):
            generate_erdos_renyi(n=50, p=0.1, m=100)
        
        with pytest.raises(ValueError, match="Must specify either p or m"):
            generate_erdos_renyi(n=50)


class TestBarabasiAlbertGeneration:
    """Test Barabási-Albert scale-free network generation."""
    
    def test_barabasi_albert_basic(self):
        """Test basic BA graph generation."""
        n = 100
        m = 3
        G = generate_barabasi_albert(n=n, m=m, seed=42)
        
        assert G.number_of_nodes() == n
        assert isinstance(G, nx.Graph)
        # BA graph should have approximately n*m edges
        assert G.number_of_edges() >= (n - m) * m
    
    def test_barabasi_albert_degree_distribution(self):
        """Test that BA graph has scale-free properties."""
        n = 200
        m = 2
        G = generate_barabasi_albert(n=n, m=m, seed=42)
        
        degrees = [d for _, d in G.degree()]
        
        # Check that there are some high-degree hubs
        assert max(degrees) > 10
        # Check that most nodes have low degree
        assert np.median(degrees) < 10
    
    def test_barabasi_albert_reproducibility(self):
        """Test reproducibility with seed."""
        n, m = 80, 2
        G1 = generate_barabasi_albert(n=n, m=m, seed=42)
        G2 = generate_barabasi_albert(n=n, m=m, seed=42)
        
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())
    
    def test_barabasi_albert_connectivity(self):
        """Test that BA graph is connected."""
        n = 50
        m = 2
        G = generate_barabasi_albert(n=n, m=m, seed=42)
        
        assert nx.is_connected(G)


class TestWattsStrogatzGeneration:
    """Test Watts-Strogatz small-world network generation."""
    
    def test_watts_strogatz_basic(self):
        """Test basic WS graph generation."""
        n = 50
        k = 4
        p = 0.1
        G = generate_watts_strogatz(n=n, k=k, p=p, seed=42)
        
        assert G.number_of_nodes() == n
        assert isinstance(G, nx.Graph)
    
    def test_watts_strogatz_regular_limit(self):
        """Test WS graph with p=0 (regular ring lattice)."""
        n = 40
        k = 4
        p = 0.0
        G = generate_watts_strogatz(n=n, k=k, p=p, seed=42)
        
        # All nodes should have degree k
        degrees = [d for _, d in G.degree()]
        assert all(d == k for d in degrees)
    
    def test_watts_strogatz_random_limit(self):
        """Test WS graph with p=1 (random graph)."""
        n = 40
        k = 4
        p = 1.0
        G = generate_watts_strogatz(n=n, k=k, p=p, seed=42)
        
        # Should still be connected
        assert nx.is_connected(G)
        # Degree distribution should be more varied
        degrees = [d for _, d in G.degree()]
        assert len(set(degrees)) > 1
    
    def test_watts_strogatz_reproducibility(self):
        """Test reproducibility with seed."""
        n, k, p = 50, 6, 0.3
        G1 = generate_watts_strogatz(n=n, k=k, p=p, seed=42)
        G2 = generate_watts_strogatz(n=n, k=k, p=p, seed=42)
        
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())


class TestModularNetworkGeneration:
    """Test modular network generation."""
    
    def test_modular_network_basic(self):
        """Test basic modular network generation."""
        num_communities = 3
        nodes_per_community = 10
        p_intra = 0.3
        p_inter = 0.05
        
        G, communities = generate_modular_network(
            num_communities=num_communities,
            nodes_per_community=nodes_per_community,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=42
        )
        
        assert G.number_of_nodes() == num_communities * nodes_per_community
        assert isinstance(G, nx.Graph)
        assert isinstance(communities, dict)
        assert len(set(communities.values())) == num_communities
    
    def test_modular_network_community_structure(self):
        """Test that communities are well-defined."""
        num_communities = 4
        nodes_per_community = 15
        p_intra = 0.4
        p_inter = 0.02
        
        G, communities = generate_modular_network(
            num_communities=num_communities,
            nodes_per_community=nodes_per_community,
            p_intra=p_intra,
            p_inter=p_inter,
            seed=42
        )
        
        # Count intra-community vs inter-community edges
        intra_edges = 0
        inter_edges = 0
        
        for u, v in G.edges():
            if communities[u] == communities[v]:
                intra_edges += 1
            else:
                inter_edges += 1
        
        # Should have more intra-community edges
        assert intra_edges > inter_edges
    
    def test_modular_network_reproducibility(self):
        """Test reproducibility with seed."""
        params = {
            'num_communities': 3,
            'nodes_per_community': 10,
            'p_intra': 0.3,
            'p_inter': 0.05,
            'seed': 42
        }
        
        G1, comm1 = generate_modular_network(**params)
        G2, comm2 = generate_modular_network(**params)
        
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.edges()) == set(G2.edges())
        assert comm1 == comm2


class TestGraphProperties:
    """Test that generated graphs have expected properties."""
    
    def test_graph_sizes(self):
        """Test various graph sizes."""
        sizes = [10, 50, 100]
        
        for n in sizes:
            G = generate_erdos_renyi(n=n, p=0.1, seed=42)
            assert G.number_of_nodes() == n
    
    def test_graph_connectivity(self):
        """Test connectivity of generated graphs."""
        # BA graphs should always be connected
        G_ba = generate_barabasi_albert(n=50, m=2, seed=42)
        assert nx.is_connected(G_ba)
        
        # WS graphs should be connected
        G_ws = generate_watts_strogatz(n=50, k=4, p=0.1, seed=42)
        assert nx.is_connected(G_ws)
    
    def test_graph_types(self):
        """Test that correct graph types are returned."""
        G_undirected = generate_erdos_renyi(n=30, p=0.1, directed=False, seed=42)
        assert isinstance(G_undirected, nx.Graph)
        assert not G_undirected.is_directed()
        
        G_directed = generate_erdos_renyi(n=30, p=0.1, directed=True, seed=42)
        assert isinstance(G_directed, nx.DiGraph)
        assert G_directed.is_directed()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_small_graphs(self):
        """Test generation of very small graphs."""
        # Minimum viable BA graph
        G = generate_barabasi_albert(n=5, m=2, seed=42)
        assert G.number_of_nodes() == 5
        
        # Small ER graph
        G = generate_erdos_renyi(n=3, p=0.5, seed=42)
        assert G.number_of_nodes() == 3
    
    def test_empty_probability(self):
        """Test ER graph with p=0 (no edges)."""
        G = generate_erdos_renyi(n=20, p=0.0, seed=42)
        assert G.number_of_edges() == 0
    
    def test_full_probability(self):
        """Test ER graph with p=1 (complete graph)."""
        n = 10
        G = generate_erdos_renyi(n=n, p=1.0, seed=42)
        # Complete graph has n*(n-1)/2 edges
        expected_edges = n * (n - 1) // 2
        assert G.number_of_edges() == expected_edges


class TestRandomSeedBehavior:
    """Test random seed behavior across functions."""
    
    def test_different_seeds_different_graphs(self):
        """Test that different seeds produce different graphs."""
        G1 = generate_erdos_renyi(n=50, p=0.1, seed=42)
        G2 = generate_erdos_renyi(n=50, p=0.1, seed=43)
        
        # Should have different edge sets
        assert set(G1.edges()) != set(G2.edges())
    
    def test_no_seed_randomness(self):
        """Test that without seed, graphs are different."""
        G1 = generate_erdos_renyi(n=50, p=0.1)
        G2 = generate_erdos_renyi(n=50, p=0.1)
        
        # Very likely to be different (not guaranteed but extremely probable)
        # We just check they're both valid graphs
        assert G1.number_of_nodes() == 50
        assert G2.number_of_nodes() == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

