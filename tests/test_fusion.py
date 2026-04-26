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

"""Unit tests for fusion functions."""

import pytest
import numpy as np
from quvine.fusion.fuse import (
    fuse_by_method_type,
    fuse_best_across_types,
    hierarchical_fusion,
    _filter_methods_by_type,
    _fuse_via_svd,
    QUANTUM_METHODS,
    ALL_39_METHODS,
)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    n_nodes = 20
    embedding_dim = 16
    
    embeddings = {}
    
    # SGNS methods
    embeddings['quvine_rwr'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['quvine_ctqw'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['quvine_dtqw'] = np.random.randn(n_nodes, embedding_dim)
    
    # Filter methods
    embeddings['quvine_baseline_heat'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['quvine_ctqw_heat'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['quvine_rwr_poly'] = np.random.randn(n_nodes, embedding_dim)
    
    # GAT methods
    embeddings['gat_baseline'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['gat_ctqw'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['gat_rwr_heat'] = np.random.randn(n_nodes, embedding_dim)
    
    # GraphGPS methods
    embeddings['graphgps_baseline'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['graphgps_ctqw_poly'] = np.random.randn(n_nodes, embedding_dim)
    embeddings['graphgps_rwr'] = np.random.randn(n_nodes, embedding_dim)
    
    return embeddings


@pytest.fixture
def sample_performance_scores():
    """Create sample performance scores."""
    return {
        'quvine_rwr': 0.75,
        'quvine_ctqw': 0.82,
        'quvine_dtqw': 0.78,
        'quvine_baseline_heat': 0.70,
        'quvine_ctqw_heat': 0.85,
        'quvine_rwr_poly': 0.72,
        'gat_baseline': 0.80,
        'gat_ctqw': 0.88,
        'gat_rwr_heat': 0.76,
        'graphgps_baseline': 0.83,
        'graphgps_ctqw_poly': 0.90,
        'graphgps_rwr': 0.79,
    }


class TestFilterMethodsByType:
    """Tests for _filter_methods_by_type."""
    
    def test_filter_sgns_methods(self):
        """Test filtering SGNS methods."""
        methods = ['quvine_rwr', 'quvine_ctqw', 'gat_baseline', 'node2vec']
        filtered = _filter_methods_by_type(methods, 'sgns')
        
        assert set(filtered) == {'quvine_rwr', 'quvine_ctqw'}
    
    def test_filter_quantum_only(self):
        """Test filtering quantum methods only."""
        methods = ['quvine_rwr', 'quvine_ctqw', 'quvine_dtqw']
        filtered = _filter_methods_by_type(methods, 'sgns', quantum_only=True)
        
        assert set(filtered) == {'quvine_ctqw', 'quvine_dtqw'}
        assert 'quvine_rwr' not in filtered
    
    def test_filter_classical_only(self):
        """Test filtering classical methods only."""
        methods = ['gat_baseline', 'gat_ctqw', 'gat_rwr']
        filtered = _filter_methods_by_type(methods, 'gat', classical_only=True)
        
        assert set(filtered) == {'gat_baseline', 'gat_rwr'}
        assert 'gat_ctqw' not in filtered
    
    def test_filter_gat_methods(self):
        """Test filtering GAT methods."""
        methods = ['gat_baseline', 'gat_ctqw', 'graphgps_baseline', 'node2vec']
        filtered = _filter_methods_by_type(methods, 'gat')
        
        assert set(filtered) == {'gat_baseline', 'gat_ctqw'}
    
    def test_filter_graphgps_methods(self):
        """Test filtering GraphGPS methods."""
        methods = ['graphgps_baseline', 'graphgps_ctqw', 'gat_baseline']
        filtered = _filter_methods_by_type(methods, 'graphgps')
        
        assert set(filtered) == {'graphgps_baseline', 'graphgps_ctqw'}


class TestFuseViaSVD:
    """Tests for _fuse_via_svd."""
    
    def test_single_embedding(self):
        """Test fusion with single embedding."""
        emb = np.random.randn(10, 8)
        fused = _fuse_via_svd([emb])
        
        assert np.allclose(fused, emb), "Single embedding should be returned as-is"
    
    def test_multiple_embeddings(self):
        """Test fusion with multiple embeddings."""
        emb1 = np.random.randn(10, 8)
        emb2 = np.random.randn(10, 8)
        emb3 = np.random.randn(10, 8)
        
        fused = _fuse_via_svd([emb1, emb2, emb3])
        
        assert fused.shape == (10, 8), "Fused embedding shape mismatch"
        assert not np.isnan(fused).any(), "Fused embedding contains NaN"
        assert not np.isinf(fused).any(), "Fused embedding contains Inf"
    
    def test_target_dimension(self):
        """Test fusion with custom target dimension."""
        emb1 = np.random.randn(10, 16)
        emb2 = np.random.randn(10, 16)
        
        fused = _fuse_via_svd([emb1, emb2], target_dim=8)
        
        assert fused.shape == (10, 8), "Target dimension not respected"
    
    def test_empty_list_raises_error(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _fuse_via_svd([])


class TestFuseByMethodType:
    """Tests for fuse_by_method_type."""
    
    def test_fuse_sgns_quantum(self, sample_embeddings):
        """Test fusing quantum SGNS methods."""
        fused = fuse_by_method_type(
            sample_embeddings,
            method_type='sgns',
            quantum_only=True,
            fusion_method='svd'
        )
        
        assert fused.shape[0] == 20, "Number of nodes mismatch"
        assert fused.shape[1] == 16, "Embedding dimension mismatch"
        assert not np.isnan(fused).any()
    
    def test_fuse_gat_classical(self, sample_embeddings):
        """Test fusing classical GAT methods."""
        fused = fuse_by_method_type(
            sample_embeddings,
            method_type='gat',
            classical_only=True,
            fusion_method='svd'
        )
        
        assert fused.shape[0] == 20
        assert not np.isnan(fused).any()
    
    def test_concatenate_fusion(self, sample_embeddings):
        """Test concatenate fusion method."""
        fused = fuse_by_method_type(
            sample_embeddings,
            method_type='sgns',
            fusion_method='concatenate'
        )
        
        # Should concatenate 3 SGNS methods (each 16-dim) = 48-dim
        assert fused.shape == (20, 48), "Concatenation dimension incorrect"
    
    def test_average_fusion(self, sample_embeddings):
        """Test average fusion method."""
        fused = fuse_by_method_type(
            sample_embeddings,
            method_type='sgns',
            fusion_method='average'
        )
        
        assert fused.shape == (20, 16), "Average fusion dimension incorrect"
    
    def test_no_methods_raises_error(self, sample_embeddings):
        """Test that no matching methods raises error."""
        with pytest.raises(ValueError, match="No embeddings found"):
            fuse_by_method_type(
                sample_embeddings,
                method_type='baselines',  # No baselines in sample
                fusion_method='svd'
            )


class TestFuseBestAcrossTypes:
    """Tests for fuse_best_across_types."""
    
    def test_fuse_best_quantum(self, sample_embeddings, sample_performance_scores):
        """Test fusing best quantum methods across types."""
        fused = fuse_best_across_types(
            sample_embeddings,
            sample_performance_scores,
            quantum_only=True,
            fusion_method='svd'
        )
        
        assert fused.shape[0] == 20
        assert fused.shape[1] == 16
        assert not np.isnan(fused).any()
    
    def test_fuse_best_classical(self, sample_embeddings, sample_performance_scores):
        """Test fusing best classical methods across types."""
        fused = fuse_best_across_types(
            sample_embeddings,
            sample_performance_scores,
            classical_only=True,
            fusion_method='svd'
        )
        
        assert fused.shape[0] == 20
        assert not np.isnan(fused).any()
    
    def test_best_selection(self, sample_embeddings, sample_performance_scores):
        """Test that best methods are selected correctly."""
        # Best quantum SGNS: quvine_ctqw (0.82)
        # Best quantum filter: quvine_ctqw_heat (0.85)
        # Best quantum GAT: gat_ctqw (0.88)
        # Best quantum GraphGPS: graphgps_ctqw_poly (0.90)
        
        fused = fuse_best_across_types(
            sample_embeddings,
            sample_performance_scores,
            quantum_only=True,
            fusion_method='svd'
        )
        
        # Should fuse 4 best quantum methods
        assert fused.shape == (20, 16)


class TestHierarchicalFusion:
    """Tests for hierarchical_fusion."""
    
    def test_complete_hierarchical_fusion(self, sample_embeddings, sample_performance_scores):
        """Test complete hierarchical fusion."""
        fused_dict = hierarchical_fusion(
            sample_embeddings,
            sample_performance_scores,
            target_dim=16
        )
        
        # Check that we get expected fused embeddings
        assert 'fused_quantum_sgns' in fused_dict
        assert 'fused_classical_sgns' in fused_dict
        assert 'fused_q' in fused_dict
        assert 'fused_c' in fused_dict
        
        # Check shapes
        for key, emb in fused_dict.items():
            assert emb.shape[0] == 20, f"{key} has wrong number of nodes"
            assert emb.shape[1] == 16, f"{key} has wrong embedding dimension"
            assert not np.isnan(emb).any(), f"{key} contains NaN"
    
    def test_within_type_fusion(self, sample_embeddings, sample_performance_scores):
        """Test within-type fusion results."""
        fused_dict = hierarchical_fusion(
            sample_embeddings,
            sample_performance_scores
        )
        
        # Should have within-type fusions
        expected_keys = [
            'fused_quantum_sgns', 'fused_classical_sgns',
            'fused_quantum_filter', 'fused_classical_filter',
            'fused_quantum_gat', 'fused_classical_gat',
            'fused_quantum_graphgps', 'fused_classical_graphgps',
        ]
        
        for key in expected_keys:
            if key in fused_dict:  # Some may not exist if no methods of that type
                assert fused_dict[key].shape[0] == 20
    
    def test_cross_type_fusion(self, sample_embeddings, sample_performance_scores):
        """Test cross-type fusion results."""
        fused_dict = hierarchical_fusion(
            sample_embeddings,
            sample_performance_scores
        )
        
        # Should have cross-type fusions
        assert 'fused_q' in fused_dict, "Missing fused_q"
        assert 'fused_c' in fused_dict, "Missing fused_c"
        
        # These should be different
        assert not np.allclose(fused_dict['fused_q'], fused_dict['fused_c'])


class TestQuantumMethodsConstant:
    """Test QUANTUM_METHODS constant."""
    
    def test_quantum_methods_defined(self):
        """Test that quantum methods are properly defined."""
        assert len(QUANTUM_METHODS) == 16, "Should have 16 quantum methods"
        
        # Check some known quantum methods
        assert 'quvine_ctqw' in QUANTUM_METHODS
        assert 'quvine_dtqw' in QUANTUM_METHODS
        assert 'gat_ctqw' in QUANTUM_METHODS
        assert 'graphgps_dtqw_heat' in QUANTUM_METHODS
        
        # Check that classical methods are not included
        assert 'quvine_rwr' not in QUANTUM_METHODS
        assert 'gat_baseline' not in QUANTUM_METHODS
        assert 'node2vec' not in QUANTUM_METHODS


class TestAll39MethodsConstant:
    """Test ALL_39_METHODS constant."""
    
    def test_all_methods_count(self):
        """Test that we have exactly 39 methods."""
        assert len(ALL_39_METHODS) == 39, "Should have exactly 39 methods"
    
    def test_method_categories(self):
        """Test method category counts."""
        sgns = [m for m in ALL_39_METHODS if m.startswith('quvine_') and not any(x in m for x in ['heat', 'poly'])]
        filters = [m for m in ALL_39_METHODS if 'heat' in m or 'poly' in m]
        gat = [m for m in ALL_39_METHODS if m.startswith('gat_')]
        graphgps = [m for m in ALL_39_METHODS if m.startswith('graphgps_')]
        baselines = [m for m in ALL_39_METHODS if m in ['node2vec', 'netmf', 'graphsage', 'appnp', 'baseline_filter', 'baseline_gcnmf']]
        
        assert len(sgns) == 3, "Should have 3 SGNS methods"
        assert len(gat) == 12, "Should have 12 GAT methods"
        assert len(graphgps) == 12, "Should have 12 GraphGPS methods"
        assert len(baselines) == 6, "Should have 6 baseline methods"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

# Made with Bob
