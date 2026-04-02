#!/usr/bin/env python
"""
Collect and Aggregate HPC Results

This script collects results from all network analyses run on HPC
and aggregates them into a single comprehensive CSV file.

Usage:
    python scripts/collect_hpc_results.py --results-dir outputs/hpc_results/results

The script will:
1. Scan for all network subdirectories
2. Load complexity metrics and task results for each network
3. Merge all data into a single comprehensive DataFrame
4. Save to comprehensive_results.csv

Each row in the output CSV represents one network × one method combination,
with columns for:
- Network metadata (id, type, n_nodes, n_edges)
- Complexity metrics (35+ metrics)
- Ranking performance (precision@K, recall@K, MRR)
- Classification performance (F1, accuracy for multiple strategies)
- Link prediction performance (AUC-ROC, AUC-PR, MRR for multiple methods)

Author: QuVINE Team
Date: 2026-04-02
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quvine.comprehensive_embedding_analysis import collect_and_aggregate_results


def main():
    parser = argparse.ArgumentParser(
        description='Collect and aggregate HPC results into comprehensive CSV'
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing network subdirectories with results'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='comprehensive_results.csv',
        help='Name of output CSV file (default: comprehensive_results.csv)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Collect and aggregate
    comprehensive_df = collect_and_aggregate_results(
        results_dir=args.results_dir,
        output_file=args.output_file,
        verbose=not args.quiet
    )
    
    if not args.quiet:
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(f"Total rows: {len(comprehensive_df)}")
        print(f"Total columns: {len(comprehensive_df.columns)}")
        print(f"\nNetworks analyzed: {comprehensive_df['network_id'].nunique()}")
        print(f"Methods compared: {comprehensive_df['method'].nunique()}")
        
        if 'network_type' in comprehensive_df.columns:
            print(f"\nNetwork types:")
            for net_type, count in comprehensive_df['network_type'].value_counts().items():
                print(f"  {net_type}: {count} rows")
        
        print(f"\nMethods:")
        for method, count in comprehensive_df['method'].value_counts().items():
            print(f"  {method}: {count} rows")
        
        # Show sample of columns
        print(f"\nSample columns:")
        complexity_cols = [c for c in comprehensive_df.columns if not c.startswith(('ranking_', 'classification_', 'link_prediction_'))]
        ranking_cols = [c for c in comprehensive_df.columns if c.startswith('ranking_')]
        classification_cols = [c for c in comprehensive_df.columns if c.startswith('classification_')]
        link_pred_cols = [c for c in comprehensive_df.columns if c.startswith('link_prediction_')]
        
        print(f"  Complexity metrics: {len(complexity_cols)} columns")
        print(f"  Ranking metrics: {len(ranking_cols)} columns")
        print(f"  Classification metrics: {len(classification_cols)} columns")
        print(f"  Link prediction metrics: {len(link_pred_cols)} columns")
        
        print("\n" + "="*80)
        print("✓ Results aggregation complete!")
        print(f"Output saved to: {Path(args.results_dir) / args.output_file}")
        print("="*80)
    
    return comprehensive_df


if __name__ == "__main__":
    main()

# Made with Bob
