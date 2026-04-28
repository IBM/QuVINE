#!/usr/bin/env python3
"""
Test script for GCN-MF hyperparameter tuning.

This script demonstrates how to use the Bayesian optimization
for tuning GCN-MF hyperparameters.
"""

import sys
sys.path.insert(0, 'src')

import networkx as nx
from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

def test_hyperparameter_tuning():
    """Test hyperparameter tuning for GCN-MF methods."""
    
    print("="*70)
    print("GCN-MF HYPERPARAMETER TUNING TEST")
    print("="*70)
    
    # Create a test graph
    print("\n1. Creating test graph (Karate Club)...")
    G = nx.karate_club_graph()
    print(f"   Graph: {len(G)} nodes, {G.number_of_edges()} edges")
    
    # Select seeds and targets
    print("\n2. Selecting seeds and targets...")
    nodes = list(G.nodes())
    seeds = nodes[:5]  # First 5 nodes as seeds
    targets = nodes[5:15]  # Next 10 as targets
    print(f"   Seeds: {len(seeds)}, Targets: {len(targets)}")
    
    # Initialize analyzer
    print("\n3. Initializing analyzer...")
    analyzer = ComprehensiveEmbeddingAnalysis(
        output_dir='test_tuning_output',
        embedding_dim=32,
        n_jobs=1
    )
    
    # Test tuning for Heat GCN-MF
    print("\n4. Tuning Heat GCN-MF hyperparameters...")
    print("   This will run 10 trials (should take ~2-3 minutes)")
    print("   Using Bayesian optimization (Optuna TPE sampler)")
    
    try:
        result_heat = analyzer.tune_gcnmf_hyperparameters(
            G=G,
            seeds=seeds,
            targets=targets,
            diffusion_type='heat',
            n_trials=10,  # Small number for quick test
            timeout=300,  # 5 minutes max
            n_jobs_optuna=1
        )
        
        print("\n   ✓ Tuning complete!")
        print(f"   Best validation recall@50: {result_heat['best_value']:.4f}")
        print(f"   Best hyperparameters:")
        for param, value in result_heat['best_params'].items():
            print(f"     - {param}: {value}")
        
        # Store tuned parameters
        analyzer.tuned_hyperparameters['hgcnmf'] = result_heat['best_params']
        
        # Test using tuned parameters
        print("\n5. Testing embedding generation with tuned parameters...")
        embedding = analyzer.run_embedding_method(
            'quvine_hgcnmf',
            G,
            seeds,
            targets
        )
        print(f"   ✓ Embedding generated: shape {embedding.shape}")
        
        # Save tuning results
        print("\n6. Saving tuning results...")
        trials_df = result_heat['trials_df']
        output_path = 'test_tuning_output/heat_gcnmf_tuning_results.csv'
        trials_df.to_csv(output_path, index=False)
        print(f"   ✓ Results saved to {output_path}")
        
        print("\n" + "="*70)
        print("TEST PASSED!")
        print("="*70)
        print("\nKey features demonstrated:")
        print("  ✓ Bayesian optimization with Optuna")
        print("  ✓ Train/validation split for hyperparameter selection")
        print("  ✓ Automatic parameter tuning (n_layers, hidden_dim, lr, etc.)")
        print("  ✓ Integration with existing embedding pipeline")
        print("  ✓ Results saved for analysis")
        
        return True
        
    except ImportError as e:
        print(f"\n✗ Error: {e}")
        print("\nTo use hyperparameter tuning, install optuna:")
        print("  pip install optuna")
        return False
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_hyperparameter_tuning()
    sys.exit(0 if success else 1)

# Made with Bob
