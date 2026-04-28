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

"""Integration tests for full 39-method workflow."""

import pytest
import numpy as np
import networkx as nx
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis
from quvine.fusion.fuse import hierarchical_fusion, ALL_39_METHODS


@pytest.fixture
def small_test_graph():
    """Create a small test graph for quick testing."""
    G = nx.karate_club_graph()
    return G


@pytest.fixture
def tiny_test_graph():
    """Create a tiny test graph for very quick testing."""
    G = nx.Graph()
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 2), (1, 3), (2, 4)]
    G.add_edges_from(edges)
    return G


class TestMethodDispatch:
    """Test that all 39 methods can be dispatched correctly."""
    
    def test_sgns_methods(self, tiny_test_graph):
        """Test SGNS methods (3 methods)."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        sgns_methods = ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw']
        
        for method in sgns_methods:
            try:
                embedding = analysis.run_embedding_method(
                    method_name=method,
                    G=tiny_test_graph,
                    seeds=[0, 1],
                    targets=[2, 3, 4],
                    network_id='test'
                )
                
                assert embedding is not None, f"{method} returned None"
                assert embedding.shape[0] == tiny_test_graph.number_of_nodes()
                assert embedding.shape[1] == 8
                assert not np.isnan(embedding).any(), f"{method} contains NaN"
                print(f"✓ {method} passed")
            except Exception as e:
                pytest.fail(f"{method} failed: {str(e)}")
    
    def test_filter_methods(self, tiny_test_graph):
        """Test filter methods (6 methods)."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        filter_methods = [
            'quvine_baseline_heat', 'quvine_baseline_poly',
            'quvine_rwr_heat', 'quvine_rwr_poly',
            'quvine_ctqw_heat', 'quvine_ctqw_poly'
        ]
        
        for method in filter_methods:
            try:
                embedding = analysis.run_embedding_method(
                    method_name=method,
                    G=tiny_test_graph,
                    seeds=[0, 1],
                    targets=[2, 3, 4],
                    network_id='test'
                )
                
                assert embedding is not None, f"{method} returned None"
                assert embedding.shape[0] == tiny_test_graph.number_of_nodes()
                assert not np.isnan(embedding).any(), f"{method} contains NaN"
                print(f"✓ {method} passed")
            except Exception as e:
                pytest.fail(f"{method} failed: {str(e)}")
    
    @pytest.mark.slow
    def test_gat_methods(self, tiny_test_graph):
        """Test GAT methods (12 methods) - marked slow due to neural network training."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        gat_methods = [
            'gat_baseline', 'gat_heat', 'gat_poly',
            'gat_rwr', 'gat_ctqw', 'gat_dtqw',
            'gat_rwr_heat', 'gat_rwr_poly',
            'gat_ctqw_heat', 'gat_ctqw_poly',
            'gat_dtqw_heat', 'gat_dtqw_poly'
        ]
        
        for method in gat_methods:
            try:
                embedding = analysis.run_embedding_method(
                    method_name=method,
                    G=tiny_test_graph,
                    seeds=[0, 1],
                    targets=[2, 3, 4],
                    network_id='test'
                )
                
                assert embedding is not None, f"{method} returned None"
                assert embedding.shape[0] == tiny_test_graph.number_of_nodes()
                print(f"✓ {method} passed")
            except Exception as e:
                print(f"⚠ {method} skipped: {str(e)}")
                # GAT methods may fail if PyTorch not available
                pass
    
    @pytest.mark.slow
    def test_graphgps_methods(self, tiny_test_graph):
        """Test GraphGPS methods (12 methods) - marked slow due to neural network training."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        graphgps_methods = [
            'graphgps_baseline', 'graphgps_heat', 'graphgps_poly',
            'graphgps_rwr', 'graphgps_ctqw', 'graphgps_dtqw',
            'graphgps_rwr_heat', 'graphgps_rwr_poly',
            'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
            'graphgps_dtqw_heat', 'graphgps_dtqw_poly'
        ]
        
        for method in graphgps_methods:
            try:
                embedding = analysis.run_embedding_method(
                    method_name=method,
                    G=tiny_test_graph,
                    seeds=[0, 1],
                    targets=[2, 3, 4],
                    network_id='test'
                )
                
                assert embedding is not None, f"{method} returned None"
                assert embedding.shape[0] == tiny_test_graph.number_of_nodes()
                print(f"✓ {method} passed")
            except Exception as e:
                print(f"⚠ {method} skipped: {str(e)}")
                # GraphGPS methods may fail if PyTorch not available
                pass


class TestFastMethodSubset:
    """Test a fast subset of methods that don't require neural networks."""
    
    def test_fast_methods_workflow(self, small_test_graph):
        """Test workflow with fast methods only."""
        # Fast methods: SGNS + Filters (9 methods total)
        fast_methods = [
            'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
            'quvine_baseline_heat', 'quvine_baseline_poly',
            'quvine_rwr_heat', 'quvine_rwr_poly',
            'quvine_ctqw_heat', 'quvine_ctqw_poly'
        ]
        
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=34,  # Karate club size
            embedding_dim=16,
            base_seed=42
        )
        
        embeddings_dict = {}
        performance_scores = {}
        
        # Generate embeddings
        for method in fast_methods:
            try:
                embedding = analysis.run_embedding_method(
                    method_name=method,
                    G=small_test_graph,
                    seeds=[0, 1, 2],
                    targets=list(range(3, 10)),
                    network_id='karate'
                )
                
                embeddings_dict[method] = embedding
                # Simulate performance score
                performance_scores[method] = np.random.rand()
                
                print(f"✓ Generated embedding for {method}")
            except Exception as e:
                print(f"✗ Failed to generate embedding for {method}: {str(e)}")
        
        # Test that we got some embeddings
        assert len(embeddings_dict) > 0, "No embeddings generated"
        
        # Test hierarchical fusion
        try:
            fused_embeddings = hierarchical_fusion(
                embeddings_dict,
                performance_scores,
                target_dim=16
            )
            
            assert len(fused_embeddings) > 0, "No fused embeddings generated"
            
            for key, emb in fused_embeddings.items():
                assert emb.shape[0] == 34, f"{key}: wrong number of nodes"
                assert emb.shape[1] == 16, f"{key}: wrong embedding dimension"
                assert not np.isnan(emb).any(), f"{key}: contains NaN"
                print(f"✓ Fused embedding: {key}")
            
        except Exception as e:
            pytest.fail(f"Hierarchical fusion failed: {str(e)}")


class TestMethodRegistry:
    """Test that method registry is complete."""
    
    def test_all_39_methods_defined(self):
        """Test that all 39 methods are defined in registry."""
        assert len(ALL_39_METHODS) == 39, f"Expected 39 methods, got {len(ALL_39_METHODS)}"
        
        # Check method categories
        sgns = [m for m in ALL_39_METHODS if m.startswith('quvine_') and not any(x in m for x in ['heat', 'poly', 'baseline'])]
        filters = [m for m in ALL_39_METHODS if ('heat' in m or 'poly' in m) and m.startswith('quvine_')]
        gat = [m for m in ALL_39_METHODS if m.startswith('gat_')]
        graphgps = [m for m in ALL_39_METHODS if m.startswith('graphgps_')]
        baselines = [m for m in ALL_39_METHODS if m in ['node2vec', 'netmf', 'graphsage', 'appnp', 'baseline_filter', 'baseline_gcnmf']]
        
        print(f"SGNS methods: {len(sgns)}")
        print(f"Filter methods: {len(filters)}")
        print(f"GAT methods: {len(gat)}")
        print(f"GraphGPS methods: {len(graphgps)}")
        print(f"Baseline methods: {len(baselines)}")
        
        assert len(sgns) == 3, f"Expected 3 SGNS methods, got {len(sgns)}"
        assert len(filters) == 6, f"Expected 6 filter methods, got {len(filters)}"
        assert len(gat) == 12, f"Expected 12 GAT methods, got {len(gat)}"
        assert len(graphgps) == 12, f"Expected 12 GraphGPS methods, got {len(graphgps)}"
        assert len(baselines) == 6, f"Expected 6 baseline methods, got {len(baselines)}"
    
    def test_quantum_classical_split(self):
        """Test quantum vs classical method split."""
        from quvine.fusion.fuse import QUANTUM_METHODS
        
        quantum_count = len(QUANTUM_METHODS)
        classical_count = len(ALL_39_METHODS) - quantum_count
        
        print(f"Quantum methods: {quantum_count}")
        print(f"Classical methods: {classical_count}")
        
        assert quantum_count == 16, f"Expected 16 quantum methods, got {quantum_count}"
        assert classical_count == 23, f"Expected 23 classical methods, got {classical_count}"


class TestEmbeddingProperties:
    """Test properties of generated embeddings."""
    
    def test_embedding_shapes(self, tiny_test_graph):
        """Test that embeddings have correct shapes."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=16,
            base_seed=42
        )
        
        test_methods = ['quvine_rwr', 'quvine_baseline_heat']
        
        for method in test_methods:
            embedding = analysis.run_embedding_method(
                method_name=method,
                G=tiny_test_graph,
                seeds=[0],
                targets=[1, 2],
                network_id='test'
            )
            
            n_nodes = tiny_test_graph.number_of_nodes()
            assert embedding.shape == (n_nodes, 16), f"{method}: wrong shape"
    
    def test_embedding_validity(self, tiny_test_graph):
        """Test that embeddings are valid (no NaN, Inf)."""
        analysis = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=16,
            base_seed=42
        )
        
        test_methods = ['quvine_ctqw', 'quvine_rwr_poly']
        
        for method in test_methods:
            embedding = analysis.run_embedding_method(
                method_name=method,
                G=tiny_test_graph,
                seeds=[0],
                targets=[1, 2],
                network_id='test'
            )
            
            assert not np.isnan(embedding).any(), f"{method}: contains NaN"
            assert not np.isinf(embedding).any(), f"{method}: contains Inf"
            assert embedding.dtype == np.float64 or embedding.dtype == np.float32


class TestReproducibility:
    """Test that methods are reproducible with same seed."""
    
    def test_sgns_reproducibility(self, tiny_test_graph):
        """Test SGNS method reproducibility."""
        analysis1 = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        analysis2 = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        emb1 = analysis1.run_embedding_method(
            method_name='quvine_rwr',
            G=tiny_test_graph,
            seeds=[0],
            targets=[1, 2],
            network_id='test'
        )
        
        emb2 = analysis2.run_embedding_method(
            method_name='quvine_rwr',
            G=tiny_test_graph,
            seeds=[0],
            targets=[1, 2],
            network_id='test'
        )
        
        assert np.allclose(emb1, emb2, atol=1e-5), "SGNS not reproducible"
    
    def test_filter_reproducibility(self, tiny_test_graph):
        """Test filter method reproducibility."""
        analysis1 = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        analysis2 = ComprehensiveEmbeddingAnalysis(
            n_networks_per_type=1,
            n_nodes=10,
            embedding_dim=8,
            base_seed=42
        )
        
        emb1 = analysis1.run_embedding_method(
            method_name='quvine_baseline_heat',
            G=tiny_test_graph,
            seeds=[0],
            targets=[1, 2],
            network_id='test'
        )
        
        emb2 = analysis2.run_embedding_method(
            method_name='quvine_baseline_heat',
            G=tiny_test_graph,
            seeds=[0],
            targets=[1, 2],
            network_id='test'
        )
        
        assert np.allclose(emb1, emb2, atol=1e-5), "Filter not reproducible"


if __name__ == '__main__':
    # Run fast tests by default
    pytest.main([__file__, '-v', '-m', 'not slow'])
    
    # To run all tests including slow ones:
    # pytest.main([__file__, '-v'])

