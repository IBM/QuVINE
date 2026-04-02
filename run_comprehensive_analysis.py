#!/usr/bin/env python3
"""
Runner script for comprehensive embedding analysis (PARALLELIZED).

This script runs the complete analysis pipeline comparing different
embedding methods across scale-free and modular networks with varying
complexity characteristics.

PARALLELIZATION:
- Networks are processed in parallel
- Complexity computation is parallelized
- Each network runs all embedding methods independently
- Dramatically reduces total execution time

Usage:
    python run_comprehensive_analysis.py [--n-jobs N]
    
    --n-jobs N : Number of parallel workers (default: all CPU cores)
"""

import sys
import os
import argparse
from pathlib import Path
import time
import multiprocessing

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from quvine.comprehensive_embedding_analysis import ComprehensiveEmbeddingAnalysis


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive embedding analysis with parallelization"
    )
    parser.add_argument(
        '--n-jobs',
        type=int,
        default=-1,
        help='Number of parallel workers (-1 = all CPU cores)'
    )
    parser.add_argument(
        '--n-networks',
        type=int,
        default=20,
        help='Number of networks per type (default: 20)'
    )
    parser.add_argument(
        '--n-nodes',
        type=int,
        default=200,
        help='Number of nodes per network (default: 200)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (5 networks, 100 nodes)'
    )
    return parser.parse_args()


def main():
    """Run the comprehensive analysis."""
    args = parse_args()
    
    # Quick mode overrides
    if args.quick:
        n_networks = 5
        n_nodes = 100
        mode_str = "QUICK TEST MODE"
    else:
        n_networks = args.n_networks
        n_nodes = args.n_nodes
        mode_str = "FULL ANALYSIS MODE"
    
    n_cpus = multiprocessing.cpu_count()
    n_jobs = n_cpus if args.n_jobs == -1 else min(args.n_jobs, n_cpus)
    
    print("\n" + "="*80)
    print(f"COMPREHENSIVE EMBEDDING ANALYSIS - {mode_str}")
    print("Comparing QuVINE variants, NetMF, and Node2Vec")
    print("="*80)
    print(f"\nParallelization: {n_jobs} workers (out of {n_cpus} available CPUs)")
    print(f"Networks: {n_networks * 2} total ({n_networks} scale-free + {n_networks} modular)")
    print(f"Network size: {n_nodes} nodes")
    print(f"Methods: 6 (RWR, CTQW, DTQW, fused, NetMF, Node2Vec)")
    print(f"Total tasks: {n_networks * 2 * 6} = {n_networks * 2} networks × 6 methods")
    print("="*80 + "\n")
    
    # Estimate time
    if args.quick:
        est_time = "5-15 minutes"
    else:
        est_time = "30-90 minutes (with parallelization)"
    print(f"Estimated time: {est_time}")
    print("Starting analysis...\n")
    
    start_time = time.time()
    
    # Configure analysis
    analysis = ComprehensiveEmbeddingAnalysis(
        output_dir="outputs/comprehensive_analysis",
        n_networks_per_type=n_networks,
        n_nodes=n_nodes,
        num_seeds=15,
        num_targets=25,
        embedding_dim=128,
        seed=42,
        n_jobs=n_jobs
    )
    
    # Run complete analysis
    results = analysis.run_complete_analysis()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nTotal time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)")
    print(f"Average time per network: {elapsed/(n_networks*2):.1f} seconds")
    print(f"\nNetworks analyzed: {len(results['complexity'])}")
    print(f"Methods compared: {results['performance']['method'].nunique()}")
    print(f"Complexity metrics: {len([c for c in results['complexity'].columns if c not in ['network_id', 'network_type']])}")
    print(f"\nResults saved to: outputs/comprehensive_analysis/")
    print("\nKey outputs:")
    print("  - complexity_metrics.csv")
    print("  - embedding_performance.csv")
    print("  - complexity_performance_correlations.csv")
    print("  - method_recommendations.csv")
    print("  - recommendations_report.txt")
    print("  - visualizations/")
    
    # Performance summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    perf_summary = results['performance'].groupby('method')['recall@50_centroid'].agg(['mean', 'std'])
    perf_summary = perf_summary.sort_values('mean', ascending=False)
    print("\nRecall@50 (centroid) by method:")
    for method, row in perf_summary.iterrows():
        print(f"  {method:12s}: {row['mean']:.3f} ± {row['std']:.3f}")
    
    print("\n" + "="*80 + "\n")
    
    return results


if __name__ == "__main__":
    main()

# Made with Bob
