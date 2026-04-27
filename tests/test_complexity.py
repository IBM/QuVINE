# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for complexity calculation modules."""

import pytest
import numpy as np
import networkx as nx
from quvine.complexity.graph import (
    compute_graph_complexity_metrics,
    compute_spectral_gap,
    compute_algebraic_connectivity,
    compute_spectral_entropy,
    compute_von_neumann_entropy,
    compute_quantum_complexity,
    compute_estrada_index,
)


@pytest.fixture
def small_graph():
    """Create a small test graph."""
    G = nx.karate_club_graph()
    return G


@pytest.fixture
def tiny_graph():
    """Create a tiny test graph for quick tests."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    return G


@pytest.fixture
def disconnected_graph():
    """Create a disconnected graph."""
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2)])  # Component 1
    G.add_edges_from([(3, 4), (4, 5)])  # Component 2
    return G


class TestSpectralMetrics:
    """Tests for spectral metric functions."""
    
    def test_spectral_gap(self, tiny_graph):
        """Test spectral gap computation."""
        gap = compute_spectral_gap(tiny_graph)
        
        assert isinstance(gap, (int, float))
        assert gap >= 0
        assert not np.isnan(gap)
    
    def test_algebraic_connectivity(self, tiny_graph):
        """Test algebraic connectivity computation."""
        conn = compute_algebraic_connectivity(tiny_graph)
        
        assert isinstance(conn, (int, float))
        assert conn >= 0
        assert not np.isnan(conn)
    
    def test_spectral_entropy(self, tiny_graph):
        """Test spectral entropy computation."""
        entropy = compute_spectral_entropy(tiny_graph)
        
        assert isinstance(entropy, (int, float))
        assert entropy >= 0
        assert not np.isnan(entropy)
    
    def test_complete_graph(self):
        """Test spectral metrics on complete graph."""
        G = nx.complete_graph(5)
        
        gap = compute_spectral_gap(G)
        conn = compute_algebraic_connectivity(G)
        
        # Complete graph should have high algebraic connectivity
        assert conn > 0
        assert not np.isnan(gap)
    
    def test_path_graph(self):
        """Test spectral metrics on path graph."""
        G = nx.path_graph(10)
        
        conn = compute_algebraic_connectivity(G)
        
        # Path graph should have low algebraic connectivity
        assert conn > 0
        assert conn < 1.0
    
    def test_disconnected_graph(self, disconnected_graph):
        """Test spectral metrics on disconnected graph."""
        conn = compute_algebraic_connectivity(disconnected_graph)
        
        # Disconnected graph should have zero algebraic connectivity
        assert conn == pytest.approx(0.0, abs=1e-6)


class TestQuantumMetrics:
    """Tests for quantum metric functions."""
    
    def test_von_neumann_entropy(self, tiny_graph):
        """Test von Neumann entropy computation."""
        entropy = compute_von_neumann_entropy(tiny_graph)
        
        assert isinstance(entropy, (int, float))
        assert entropy >= 0
        assert not np.isnan(entropy)
    
    def test_quantum_complexity(self, tiny_graph):
        """Test quantum complexity computation."""
        complexity = compute_quantum_complexity(tiny_graph)
        
        assert isinstance(complexity, (int, float))
        assert complexity >= 0
        assert not np.isnan(complexity)
    
    def test_estrada_index(self, tiny_graph):
        """Test Estrada index computation."""
        estrada = compute_estrada_index(tiny_graph)
        
        assert isinstance(estrada, (int, float))
        assert estrada > 0
        assert not np.isnan(estrada)
    
    def test_complete_graph(self):
        """Test quantum metrics on complete graph."""
        G = nx.complete_graph(5)
        
        entropy = compute_von_neumann_entropy(G)
        complexity = compute_quantum_complexity(G)
        estrada = compute_estrada_index(G)
        
        assert not np.isnan(entropy)
        assert not np.isnan(complexity)
        assert estrada > 0
    
    def test_star_graph(self):
        """Test quantum metrics on star graph."""
        G = nx.star_graph(5)
        
        entropy = compute_von_neumann_entropy(G)
        complexity = compute_quantum_complexity(G)
        
        assert entropy >= 0
        assert complexity >= 0


class TestComputeGraphComplexityMetrics:
    """Tests for compute_graph_complexity_metrics."""
    
    def test_basic_functionality(self, small_graph):
        """Test complete complexity metrics computation."""
        metrics = compute_graph_complexity_metrics(small_graph)
        
        # Check essential metrics are present
        essential_metrics = [
            'num_nodes', 'num_edges', 'density',
            'spectral_gap', 'algebraic_connectivity', 'spectral_entropy',
            'von_neumann_entropy', 'quantum_complexity', 'estrada_index'
        ]
        
        for metric in essential_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_metric_types(self, tiny_graph):
        """Test that metrics have correct types."""
        metrics = compute_graph_complexity_metrics(tiny_graph)
        
        # Integer metrics
        assert isinstance(metrics['num_nodes'], int)
        assert isinstance(metrics['num_edges'], int)
        
        # Float metrics
        float_metrics = ['density', 'spectral_gap', 'von_neumann_entropy']
        for metric in float_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float))
    
    def test_metric_ranges(self, tiny_graph):
        """Test that metrics are in valid ranges."""
        metrics = compute_graph_complexity_metrics(tiny_graph)
        
        # Density should be between 0 and 1
        assert 0 <= metrics['density'] <= 1
        
        # Non-negative metrics
        non_negative = ['spectral_gap', 'algebraic_connectivity', 
                       'von_neumann_entropy', 'quantum_complexity']
        for metric in non_negative:
            if metric in metrics:
                assert metrics[metric] >= 0, f"{metric} should be non-negative"
    
    def test_empty_graph(self):
        """Test handling of empty graph."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])  # Nodes but no edges
        
        metrics = compute_graph_complexity_metrics(G)
        
        assert metrics['num_nodes'] == 3
        assert metrics['num_edges'] == 0
        assert metrics['density'] == 0.0
    
    def test_single_node(self):
        """Test handling of single node graph."""
        G = nx.Graph()
        G.add_node(0)
        
        metrics = compute_graph_complexity_metrics(G)
        
        assert metrics['num_nodes'] == 1
        assert metrics['num_edges'] == 0
    
    def test_reproducibility(self, tiny_graph):
        """Test that metrics are reproducible."""
        metrics1 = compute_graph_complexity_metrics(tiny_graph)
        metrics2 = compute_graph_complexity_metrics(tiny_graph)
        
        # Compare key metrics
        for key in ['spectral_gap', 'von_neumann_entropy', 'quantum_complexity']:
            if key in metrics1 and key in metrics2:
                assert np.isclose(metrics1[key], metrics2[key], rtol=1e-5)


class TestComplexityEdgeCases:
    """Test edge cases and error handling."""
    
    def test_very_small_graph(self):
        """Test with very small graph (2 nodes)."""
        G = nx.Graph()
        G.add_edge(0, 1)
        
        metrics = compute_graph_complexity_metrics(G)
        
        assert metrics['num_nodes'] == 2
        assert metrics['num_edges'] == 1
        assert not np.isnan(metrics['density'])
    
    def test_self_loops(self):
        """Test graph with self-loops."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0)])
        G.add_edge(1, 1)  # Self-loop
        
        # Should handle self-loops gracefully
        metrics = compute_graph_complexity_metrics(G)
        assert metrics['num_nodes'] == 3


class TestComplexityComparison:
    """Test complexity comparisons between different graph types."""
    
    def test_complete_vs_path(self):
        """Complete graph should have higher complexity than path."""
        G_complete = nx.complete_graph(10)
        G_path = nx.path_graph(10)
        
        metrics_complete = compute_graph_complexity_metrics(G_complete)
        metrics_path = compute_graph_complexity_metrics(G_path)
        
        # Complete graph should have higher density
        assert metrics_complete['density'] > metrics_path['density']
        
        # Complete graph should have higher algebraic connectivity
        assert metrics_complete['algebraic_connectivity'] > metrics_path['algebraic_connectivity']
    
    def test_random_vs_regular(self):
        """Test complexity differences between random and regular graphs."""
        G_random = nx.erdos_renyi_graph(20, 0.3, seed=42)
        G_regular = nx.random_regular_graph(6, 20, seed=42)
        
        metrics_random = compute_graph_complexity_metrics(G_random)
        metrics_regular = compute_graph_complexity_metrics(G_regular)
        
        # Both should have valid metrics
        assert metrics_random['num_nodes'] == 20
        assert metrics_regular['num_nodes'] == 20
        assert not np.isnan(metrics_random['spectral_gap'])
        assert not np.isnan(metrics_regular['spectral_gap'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob