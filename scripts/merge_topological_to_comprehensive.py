#!/usr/bin/env python3
"""
Merge topological complexity metrics into comprehensive results CSV.

This script reads the individual complexity CSV files (which now contain
topological metrics) and merges them into the comprehensive results file.
"""

import sys
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Paths
RESULTS_DIR = Path("/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/results")
COMPREHENSIVE_CSV = RESULTS_DIR / "comprehensive_results.csv"
OUTPUT_CSV = RESULTS_DIR / "comprehensive_results_with_topology.csv"

def main():
    print("="*80)
    print("Merging Topological Complexity Metrics into Comprehensive Results")
    print("="*80)
    
    # Read comprehensive results
    print(f"\nReading comprehensive results from: {COMPREHENSIVE_CSV}")
    comp_df = pd.read_csv(COMPREHENSIVE_CSV)
    print(f"  Shape: {comp_df.shape}")
    print(f"  Unique networks: {comp_df['network_id'].nunique()}")
    
    # Collect topological metrics from individual complexity files
    print("\nCollecting topological metrics from individual complexity files...")
    topo_data = []
    
    for network_id in comp_df['network_id'].unique():
        complexity_csv = RESULTS_DIR / network_id / f"{network_id}_complexity.csv"
        
        if not complexity_csv.exists():
            print(f"  Warning: {complexity_csv} not found, skipping...")
            continue
        
        try:
            complexity_df = pd.read_csv(complexity_csv)
            
            # Extract topological metrics
            topo_cols = [
                'betti_0', 'betti_1', 'betti_2', 'betti_sum', 
                'euler_characteristic', 'persistence_entropy_H0',
                'persistence_entropy_H1', 'persistence_entropy_H2'
            ]
            
            # Check if topological columns exist
            if all(col in complexity_df.columns for col in topo_cols):
                row = {'network_id': network_id}
                for col in topo_cols:
                    row[col] = complexity_df[col].iloc[0]
                topo_data.append(row)
            else:
                print(f"  Warning: {network_id} missing topological columns")
                
        except Exception as e:
            print(f"  Error processing {network_id}: {e}")
    
    print(f"\nCollected topological metrics for {len(topo_data)} networks")
    
    if len(topo_data) == 0:
        print("ERROR: No topological metrics found!")
        sys.exit(1)
    
    # Create DataFrame with topological metrics
    topo_df = pd.DataFrame(topo_data)
    print(f"Topological metrics DataFrame shape: {topo_df.shape}")
    print(f"Columns: {list(topo_df.columns)}")
    
    # Merge with comprehensive results
    print("\nMerging with comprehensive results...")
    merged_df = comp_df.merge(topo_df, on='network_id', how='left')
    print(f"Merged shape: {merged_df.shape}")
    
    # Check for missing values
    topo_cols = [col for col in topo_df.columns if col != 'network_id']
    print("\nMissing values in topological columns:")
    for col in topo_cols:
        n_missing = merged_df[col].isna().sum()
        if n_missing > 0:
            print(f"  {col}: {n_missing}/{len(merged_df)} missing")
    
    # Save merged results
    print(f"\nSaving merged results to: {OUTPUT_CSV}")
    merged_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved! Final shape: {merged_df.shape}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics for Topological Metrics")
    print("="*80)
    for col in topo_cols:
        if col in merged_df.columns:
            print(f"\n{col}:")
            print(f"  Mean: {merged_df[col].mean():.4f}")
            print(f"  Std:  {merged_df[col].std():.4f}")
            print(f"  Min:  {merged_df[col].min():.4f}")
            print(f"  Max:  {merged_df[col].max():.4f}")
            print(f"  Non-zero: {(merged_df[col] != 0).sum()}/{len(merged_df)}")
    
    print("\n" + "="*80)
    print("Merge completed successfully!")
    print("="*80)

if __name__ == "__main__":
    main()
