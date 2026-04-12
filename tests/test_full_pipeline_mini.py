#!/usr/bin/env python3
"""
Mini full pipeline test to check for bugs in logs.

Runs a small version of the comprehensive analysis with:
- 2 small networks (1 scale-free, 1 modular)
- All embedding methods
- Captures all warnings and errors

Author: QuVINE Team
"""

import sys
import logging
import warnings
import networkx as nx
from src.quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis

# Capture all warnings
warnings.filterwarnings('error')  # Convert warnings to errors so we catch them

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler('pipeline_test.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Run mini pipeline test."""
    logger.info("="*80)
    logger.info("MINI FULL PIPELINE TEST")
    logger.info("="*80)
    
    try:
        # Create analyzer with minimal settings
        analyzer = ComprehensiveEmbeddingAnalysis(
            output_dir="outputs/pipeline_test",
            n_networks_per_type=1,  # Just 1 network per type
            n_nodes=50,  # Small networks
            num_seeds=5,  # Few seeds
            embedding_dim=32,  # Small embeddings
            n_jobs=1  # Single thread for easier debugging
        )
        
        # Generate 2 small test networks
        logger.info("\n" + "="*80)
        logger.info("STEP 1: Generating test networks")
        logger.info("="*80)
        networks = analyzer.generate_networks()
        logger.info(f"✓ Generated {len(networks)} networks")
        
        # Compute complexity
        logger.info("\n" + "="*80)
        logger.info("STEP 2: Computing complexity metrics")
        logger.info("="*80)
        complexity_df = analyzer.compute_complexity_for_all(networks)
        logger.info(f"✓ Computed complexity for {len(complexity_df)} networks")
        
        # Test each embedding method on first network
        logger.info("\n" + "="*80)
        logger.info("STEP 3: Testing all embedding methods")
        logger.info("="*80)
        
        network_id, G, seeds, targets = networks[0]
        logger.info(f"Testing on network: {network_id}")
        logger.info(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        logger.info(f"  Seeds: {len(seeds)}, Targets: {len(targets)}")
        
        # List of methods to test
        methods = [
            'netmf',
            'node2vec',
            'baseline_gcnmf',
            'baseline_filter',
            'quvine_rwr',
            'quvine_ctqw',
            # Skip DTQW (power-of-2 requirement)
            'quvine_heat',
            'quvine_poly',
            'quvine_hgcnmf',
            'quvine_pgcnmf',
            'quvine_fused_svd_ctqw_rwr'  # Test fusion
        ]
        
        results = {}
        for method in methods:
            logger.info(f"\n--- Testing {method} ---")
            try:
                emb = analyzer.run_embedding_method(
                    method_name=method,
                    G=G,
                    seeds=seeds,
                    targets=targets,
                    network_id=network_id
                )
                logger.info(f"✓ {method}: shape {emb.shape}, "
                           f"mean={emb.mean():.4f}, std={emb.std():.4f}, "
                           f"min={emb.min():.4f}, max={emb.max():.4f}")
                
                # Check for NaN or Inf
                if not np.isfinite(emb).all():
                    logger.error(f"✗ {method}: Contains NaN or Inf values!")
                    results[method] = 'FAILED: NaN/Inf'
                else:
                    results[method] = 'PASSED'
                    
            except Warning as w:
                logger.error(f"✗ {method}: Warning raised: {w}")
                results[method] = f'FAILED: {type(w).__name__}'
            except Exception as e:
                logger.error(f"✗ {method}: Exception: {e}")
                results[method] = f'FAILED: {type(e).__name__}'
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        
        passed = sum(1 for r in results.values() if r == 'PASSED')
        total = len(results)
        
        for method, result in results.items():
            status = "✓" if result == 'PASSED' else "✗"
            logger.info(f"{status} {method}: {result}")
        
        logger.info(f"\nTotal: {passed}/{total} methods passed")
        
        if passed == total:
            logger.info("\n🎉 All methods passed!")
            return 0
        else:
            logger.warning(f"\n⚠️  {total - passed} method(s) failed")
            return 1
            
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import numpy as np
    exit_code = main()
    sys.exit(exit_code)

