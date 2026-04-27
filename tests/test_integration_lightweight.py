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

"""
Lightweight integration tests that don't require heavy dependencies.

These tests verify that core workflows function correctly without requiring
PyTorch, gensim, or other heavy dependencies.
"""

import pytest
import numpy as np
import networkx as nx
from quvine.complexity.graph import compute_graph_complexity_metrics
from quvine.embedding.quantum_filters import (
    generate_baseline_heat_embedding,
    generate_baseline_poly_embedding,
    generate_rwr_heat_embedding,
    generate_rwr_poly_embedding,
)
from quvine.fusion.fuse import (
    fuse_by_method_type,
    _fuse_via_svd,
    ALL_39_METHODS,
    QUANTUM_METHODS,
)
from quvine.data.random_graphs import (
    generate_barabasi_albert,
    generate_modular_network,
)


@pytest.fixture
def test_graph():
    """Create a test graph for integration tests."""
    return nx.karate_club_graph()


@pytest.fixture
def small_test_graph():
    """Create a small test graph."""
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3)]
    G.add_edges_from(edges)
    return G


class TestComplexityToEmbeddingWorkflow:
    """Test workflow from complexity calculation to embedding generation."""
    
    def test_complexity_then_embedding(self, small_test_graph):
        """Test computing complexity then generating embeddings."""
        # Step 1: Compute complexity
        complexity = compute_graph_complexity_metrics(small_test_graph)
        
        assert 'num_nodes' in complexity
        assert 'spectral_gap' in complexity
        assert complexity['num_nodes'] == 5
        
        # Step 2: Generate embedding based on complexity
        embedding_dim = 8
        embedding = generate_baseline_heat_embedding(
            small_test_graph,
            embedding_dim=embedding_dim,
            scale=complexity['spectral_gap'],
            random_state=42
        )
        
        assert embedding.shape == (5, embedding_dim)
        assert not np.isnan(embedding).any()
    
    def test_multiple_embeddings_workflow(self, small_test_graph):
        """Test generating multiple embeddings and checking consistency."""
        embedding_dim = 8
        
        # Generate multiple embeddings
        emb_heat = generate_baseline_heat_embedding(
            small_test_graph, embedding_dim, random_state=42
        )
        emb_poly = generate_baseline_poly_embedding(
            small_test_graph, embedding_dim, random_state=42
        )
        emb_rwr_heat = generate_rwr_heat_embedding(
            small_test_graph, embedding_dim, random_state=42
        )
        
        # All should have same shape
        assert emb_heat.shape == emb_poly.shape == emb_rwr_heat.shape
        
        # All should be valid
        for emb in [emb_heat, emb_poly, emb_rwr_heat]:
            assert not np.isnan(emb).any()
            assert not np.isinf(emb).any()


class TestGraphGenerationToAnalysis:
    """Test workflow from graph generation to analysis."""
    
    def test_generate_and_analyze_barabasi(self):
        """Test generating Barabasi-Albert graph and analyzing it."""
        # Generate graph
        G = generate_barabasi_albert(n=50, m=3, seed=42)
        
        assert G.number_of_nodes() == 50
        assert G.number_of_edges() > 0
        
        # Analyze complexity
        metrics = compute_graph_complexity_metrics(G)
        
        assert metrics['num_nodes'] == 50
        assert metrics['spectral_gap'] > 0
        assert not np.isnan(metrics['von_neumann_entropy'])
    
    def test_generate_and_analyze_modular(self):
        """Test generating modular network and analyzing it."""
        # Generate graph - API expects nodes_per_community, not n
        G, _ = generate_modular_network(
            num_communities=4,
            nodes_per_community=10,  # 4 * 10 = 40 nodes
            p_intra=0.3,
            p_inter=0.05,
            seed=42
        )
        
        assert G.number_of_nodes() == 40
        
        # Analyze complexity
        metrics = compute_graph_complexity_metrics(G)
        
        assert metrics['num_nodes'] == 40
        assert not np.isnan(metrics['quantum_complexity'])
    
    def test_compare_graph_types(self):
        """Test comparing complexity of different graph types."""
        # Generate different graph types
        G_ba = generate_barabasi_albert(n=30, m=2, seed=42)
        G_mod, _ = generate_modular_network(
            num_communities=3,
            nodes_per_community=10,  # 3 * 10 = 30 nodes
            p_intra=0.3,
            p_inter=0.05,
            seed=42
        )
        
        # Compute metrics
        metrics_ba = compute_graph_complexity_metrics(G_ba)
        metrics_mod = compute_graph_complexity_metrics(G_mod)
        
        # Both should have valid metrics
        assert not np.isnan(metrics_ba['spectral_gap'])
        assert not np.isnan(metrics_mod['spectral_gap'])
        
        # Metrics should be different
        assert metrics_ba['spectral_gap'] != metrics_mod['spectral_gap']


class TestEmbeddingFusionWorkflow:
    """Test workflow for embedding fusion."""
    
    def test_generate_and_fuse_embeddings(self, small_test_graph):
        """Test generating multiple embeddings and fusing them."""
        embedding_dim = 8
        n_nodes = small_test_graph.number_of_nodes()  # 5 nodes
        
        # Generate embeddings
        embeddings = {
            'baseline_heat': generate_baseline_heat_embedding(
                small_test_graph, embedding_dim, random_state=42
            ),
            'baseline_poly': generate_baseline_poly_embedding(
                small_test_graph, embedding_dim, random_state=42
            ),
            'rwr_heat': generate_rwr_heat_embedding(
                small_test_graph, embedding_dim, random_state=42
            ),
        }
        
        # Fuse embeddings
        embeddings_list = list(embeddings.values())
        fused = _fuse_via_svd(embeddings_list, target_dim=embedding_dim)
        
        # SVD can only produce min(n_nodes, n_features) components
        # With 5 nodes, we get at most 5 dimensions
        expected_dim = min(embedding_dim, n_nodes)
        assert fused.shape == (n_nodes, expected_dim)
        assert not np.isnan(fused).any()
    
    def test_fusion_preserves_information(self, small_test_graph):
        """Test that fusion preserves key information."""
        embedding_dim = 16
        n_nodes = small_test_graph.number_of_nodes()  # 5 nodes
        
        # Generate embeddings
        emb1 = generate_baseline_heat_embedding(
            small_test_graph, embedding_dim, random_state=42
        )
        emb2 = generate_baseline_poly_embedding(
            small_test_graph, embedding_dim, random_state=42
        )
        
        # Fuse - will be limited to n_nodes dimensions
        fused = _fuse_via_svd([emb1, emb2], target_dim=embedding_dim)
        
        # SVD limits output to min(n_nodes, target_dim)
        expected_dim = min(embedding_dim, n_nodes)
        assert fused.shape == (n_nodes, expected_dim)
        
        # Fused embedding should capture information from both
        # Check that it's not identical to either input (compare first expected_dim columns)
        assert not np.allclose(fused, emb1[:, :expected_dim])
        assert not np.allclose(fused, emb2[:, :expected_dim])
        
        # But should have similar scale
        assert np.abs(np.std(fused) - np.std(emb1)) < 2.0


class TestMethodRegistry:
    """Test method registry and categorization."""
    
    def test_all_methods_defined(self):
        """Test that all 39 methods are defined."""
        assert len(ALL_39_METHODS) == 39
    
    def test_quantum_methods_subset(self):
        """Test that quantum methods are a subset of all methods."""
        assert QUANTUM_METHODS.issubset(set(ALL_39_METHODS))
    
    def test_quantum_classical_split(self):
        """Test quantum/classical split is correct."""
        quantum_count = len(QUANTUM_METHODS)
        classical_count = len(ALL_39_METHODS) - quantum_count
        
        assert quantum_count == 16
        assert classical_count == 23
    
    def test_method_naming_convention(self):
        """Test that method names follow conventions."""
        for method in ALL_39_METHODS:
            # Should be lowercase with underscores
            assert method == method.lower()
            assert ' ' not in method
            
            # Should start with known prefix
            valid_prefixes = ['quvine_', 'gat_', 'graphgps_', 'node2vec', 
                            'netmf', 'graphsage', 'appnp', 'baseline_']
            assert any(method.startswith(prefix) for prefix in valid_prefixes)


class TestEndToEndLightweight:
    """End-to-end tests with lightweight operations."""
    
    def test_full_pipeline_no_heavy_deps(self):
        """Test full pipeline without heavy dependencies."""
        # 1. Generate graph
        G = generate_barabasi_albert(n=20, m=2, seed=42)
        
        # 2. Compute complexity
        complexity = compute_graph_complexity_metrics(G)
        assert 'quantum_complexity' in complexity
        
        # 3. Generate embeddings (filter-based, no PyTorch/gensim)
        embedding_dim = 8
        embeddings = []
        
        for method_func in [generate_baseline_heat_embedding,
                           generate_baseline_poly_embedding,
                           generate_rwr_heat_embedding]:
            emb = method_func(G, embedding_dim, random_state=42)
            embeddings.append(emb)
            assert emb.shape == (20, embedding_dim)
        
        # 4. Fuse embeddings
        fused = _fuse_via_svd(embeddings, target_dim=embedding_dim)
        assert fused.shape == (20, embedding_dim)
        
        # 5. Verify output quality
        assert not np.isnan(fused).any()
        assert not np.isinf(fused).any()
        assert np.std(fused) > 0  # Has variance
    
    def test_reproducibility_across_pipeline(self):
        """Test that pipeline is reproducible with same seed."""
        seed = 42
        
        # Run 1
        G1 = generate_barabasi_albert(n=15, m=2, seed=seed)
        emb1 = generate_baseline_heat_embedding(G1, 8, random_state=seed)
        
        # Run 2
        G2 = generate_barabasi_albert(n=15, m=2, seed=seed)
        emb2 = generate_baseline_heat_embedding(G2, 8, random_state=seed)
        
        # Should be identical
        assert nx.is_isomorphic(G1, G2)
        assert np.allclose(emb1, emb2)


class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_empty_graph_handling(self):
        """Test handling of empty graph."""
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])  # Nodes but no edges
        
        # Should not crash
        metrics = compute_graph_complexity_metrics(G)
        assert metrics['num_nodes'] == 3
        assert metrics['num_edges'] == 0
    
    def test_single_node_handling(self):
        """Test handling of single node."""
        G = nx.Graph()
        G.add_node(0)
        
        metrics = compute_graph_complexity_metrics(G)
        assert metrics['num_nodes'] == 1
    
    def test_disconnected_graph_handling(self):
        """Test handling of disconnected graph."""
        G = nx.Graph()
        G.add_edges_from([(0, 1), (2, 3)])  # Two components
        
        metrics = compute_graph_complexity_metrics(G)
        assert metrics['num_nodes'] == 4
        # Algebraic connectivity should be 0 for disconnected graph
        assert metrics['algebraic_connectivity'] == pytest.approx(0.0, abs=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
