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

"""Unit tests for quantum filter functions."""

import pytest
import numpy as np
import networkx as nx
from quvine.embedding.quantum_filters import (
    generate_baseline_heat_embedding,
    generate_baseline_poly_embedding,
    generate_rwr_heat_embedding,
    generate_rwr_poly_embedding,
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


class TestBaselineHeatEmbedding:
    """Tests for generate_baseline_heat_embedding."""
    
    def test_basic_functionality(self, tiny_graph):
        """Test basic embedding generation."""
        embedding = generate_baseline_heat_embedding(
            G=tiny_graph,
            embedding_dim=8,
            scale=1.0,
            normalize=True,
            random_state=42
        )
        
        assert embedding.shape == (4, 8), "Embedding shape mismatch"
        assert not np.isnan(embedding).any(), "Embedding contains NaN"
        assert not np.isinf(embedding).any(), "Embedding contains Inf"
    
    def test_embedding_dimension(self, small_graph):
        """Test different embedding dimensions."""
        for dim in [16, 32, 64]:
            embedding = generate_baseline_heat_embedding(
                G=small_graph,
                embedding_dim=dim,
                scale=1.0,
                random_state=42
            )
            assert embedding.shape == (small_graph.number_of_nodes(), dim)
    
    def test_scale_parameter(self, tiny_graph):
        """Test different scale parameters."""
        emb1 = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, scale=0.5, random_state=42
        )
        emb2 = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, scale=2.0, random_state=42
        )
        
        # Different scales should produce different embeddings
        assert not np.allclose(emb1, emb2), "Scale parameter has no effect"
    
    def test_normalization(self, tiny_graph):
        """Test normalization parameter."""
        emb_norm = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, normalize=True, random_state=42
        )
        emb_no_norm = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, normalize=False, random_state=42
        )
        
        # Normalized and non-normalized should differ
        assert not np.allclose(emb_norm, emb_no_norm), "Normalization has no effect"
        
        # Both should be valid embeddings
        assert not np.isnan(emb_norm).any()
        assert not np.isnan(emb_no_norm).any()
    
    def test_reproducibility(self, tiny_graph):
        """Test that same seed produces same results."""
        emb1 = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, random_state=42
        )
        emb2 = generate_baseline_heat_embedding(
            G=tiny_graph, embedding_dim=8, random_state=42
        )
        
        assert np.allclose(emb1, emb2), "Results not reproducible"


class TestBaselinePolyEmbedding:
    """Tests for generate_baseline_poly_embedding."""
    
    def test_basic_functionality(self, tiny_graph):
        """Test basic embedding generation."""
        embedding = generate_baseline_poly_embedding(
            G=tiny_graph,
            embedding_dim=8,
            order=4,
            normalize=True,
            random_state=42
        )
        
        assert embedding.shape == (4, 8), "Embedding shape mismatch"
        assert not np.isnan(embedding).any(), "Embedding contains NaN"
        assert not np.isinf(embedding).any(), "Embedding contains Inf"
    
    def test_polynomial_order(self, tiny_graph):
        """Test different polynomial orders."""
        emb_order2 = generate_baseline_poly_embedding(
            G=tiny_graph, embedding_dim=8, order=2, random_state=42
        )
        emb_order8 = generate_baseline_poly_embedding(
            G=tiny_graph, embedding_dim=8, order=8, random_state=42
        )
        
        # Different orders should produce different embeddings
        assert not np.allclose(emb_order2, emb_order8), "Order parameter has no effect"
    
    def test_reproducibility(self, tiny_graph):
        """Test reproducibility."""
        emb1 = generate_baseline_poly_embedding(
            G=tiny_graph, embedding_dim=8, order=4, random_state=42
        )
        emb2 = generate_baseline_poly_embedding(
            G=tiny_graph, embedding_dim=8, order=4, random_state=42
        )
        
        assert np.allclose(emb1, emb2), "Results not reproducible"


class TestRWRHeatEmbedding:
    """Tests for generate_rwr_heat_embedding."""
    
    def test_basic_functionality(self, tiny_graph):
        """Test basic embedding generation."""
        embedding = generate_rwr_heat_embedding(
            G=tiny_graph,
            embedding_dim=8,
            restart_prob=0.15,
            scale=1.0,
            normalize=True,
            random_state=42
        )
        
        assert embedding.shape == (4, 8), "Embedding shape mismatch"
        assert not np.isnan(embedding).any(), "Embedding contains NaN"
        assert not np.isinf(embedding).any(), "Embedding contains Inf"
    
    def test_restart_probability(self, tiny_graph):
        """Test different restart probabilities."""
        emb_low = generate_rwr_heat_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.1, random_state=42
        )
        emb_high = generate_rwr_heat_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.3, random_state=42
        )
        
        # Different restart probs should produce different embeddings
        assert not np.allclose(emb_low, emb_high), "Restart prob has no effect"
    
    def test_scale_and_restart(self, tiny_graph):
        """Test combination of scale and restart parameters."""
        embedding = generate_rwr_heat_embedding(
            G=tiny_graph,
            embedding_dim=8,
            restart_prob=0.2,
            scale=2.0,
            random_state=42
        )
        
        assert embedding.shape == (4, 8)
        assert not np.isnan(embedding).any()
    
    def test_reproducibility(self, tiny_graph):
        """Test reproducibility."""
        emb1 = generate_rwr_heat_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.15, random_state=42
        )
        emb2 = generate_rwr_heat_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.15, random_state=42
        )
        
        assert np.allclose(emb1, emb2), "Results not reproducible"


class TestRWRPolyEmbedding:
    """Tests for generate_rwr_poly_embedding."""
    
    def test_basic_functionality(self, tiny_graph):
        """Test basic embedding generation."""
        embedding = generate_rwr_poly_embedding(
            G=tiny_graph,
            embedding_dim=8,
            restart_prob=0.15,
            order=4,
            normalize=True,
            random_state=42
        )
        
        assert embedding.shape == (4, 8), "Embedding shape mismatch"
        assert not np.isnan(embedding).any(), "Embedding contains NaN"
        assert not np.isinf(embedding).any(), "Embedding contains Inf"
    
    def test_restart_and_order(self, tiny_graph):
        """Test combination of restart probability and polynomial order."""
        embedding = generate_rwr_poly_embedding(
            G=tiny_graph,
            embedding_dim=8,
            restart_prob=0.2,
            order=6,
            random_state=42
        )
        
        assert embedding.shape == (4, 8)
        assert not np.isnan(embedding).any()
    
    def test_parameter_variations(self, tiny_graph):
        """Test various parameter combinations."""
        params = [
            {'restart_prob': 0.1, 'order': 2},
            {'restart_prob': 0.2, 'order': 4},
            {'restart_prob': 0.3, 'order': 8},
        ]
        
        embeddings = []
        for p in params:
            emb = generate_rwr_poly_embedding(
                G=tiny_graph,
                embedding_dim=8,
                restart_prob=p['restart_prob'],
                order=p['order'],
                random_state=42
            )
            embeddings.append(emb)
        
        # All embeddings should be different
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                assert not np.allclose(embeddings[i], embeddings[j])
    
    def test_reproducibility(self, tiny_graph):
        """Test reproducibility."""
        emb1 = generate_rwr_poly_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.15, order=4, random_state=42
        )
        emb2 = generate_rwr_poly_embedding(
            G=tiny_graph, embedding_dim=8, restart_prob=0.15, order=4, random_state=42
        )
        
        assert np.allclose(emb1, emb2), "Results not reproducible"


class TestFilterComparison:
    """Compare different filter methods."""
    
    def test_all_filters_produce_valid_embeddings(self, small_graph):
        """Test that all filter methods produce valid embeddings."""
        embedding_dim = 16
        
        emb_baseline_heat = generate_baseline_heat_embedding(
            small_graph, embedding_dim, random_state=42
        )
        emb_baseline_poly = generate_baseline_poly_embedding(
            small_graph, embedding_dim, random_state=42
        )
        emb_rwr_heat = generate_rwr_heat_embedding(
            small_graph, embedding_dim, random_state=42
        )
        emb_rwr_poly = generate_rwr_poly_embedding(
            small_graph, embedding_dim, random_state=42
        )
        
        n_nodes = small_graph.number_of_nodes()
        
        for emb in [emb_baseline_heat, emb_baseline_poly, emb_rwr_heat, emb_rwr_poly]:
            assert emb.shape == (n_nodes, embedding_dim)
            assert not np.isnan(emb).any()
            assert not np.isinf(emb).any()
    
    def test_filters_produce_different_embeddings(self, small_graph):
        """Test that different filters produce different embeddings."""
        embedding_dim = 16
        
        emb_baseline_heat = generate_baseline_heat_embedding(
            small_graph, embedding_dim, random_state=42
        )
        emb_rwr_heat = generate_rwr_heat_embedding(
            small_graph, embedding_dim, random_state=42
        )
        
        # Baseline and RWR should produce different embeddings
        assert not np.allclose(emb_baseline_heat, emb_rwr_heat)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob
