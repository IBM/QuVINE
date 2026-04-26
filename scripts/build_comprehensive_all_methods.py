#!/usr/bin/env python3
"""
Build comprehensive results with proper aggregation for all methods.
This script aggregates classification, link prediction, and ranking results
with all the statistics that were missing in the PPI dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import glob
import os


def build_comprehensive_all_methods(
    results_dir: str,
    output_file: str = "comprehensive_results_all_ppi.csv"
):
    """
    Build comprehensive results with proper aggregation including:
    - Classification accuracy per strategy
    - Classification min/max stats
    - Link prediction min stats
    - Link prediction negative_strategy column
    """
    
    print("="*80)
    print("BUILDING COMPREHENSIVE RESULTS WITH ALL METHODS")
    print(f"Results directory: {results_dir}")
    print("="*80)
    
    results_path = Path(results_dir)
    all_results = []
    
    # Find all network directories
    network_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    print(f"\nFound {len(network_dirs)} network directories")
    
    for network_dir in sorted(network_dirs):
        network_id = network_dir.name
        print(f"\nProcessing: {network_id}")
        
        try:
            # Load complexity metrics
            complexity_files = list(network_dir.glob("*_complexity.csv"))
            if not complexity_files:
                print(f"  ⚠ No complexity file found")
                continue
            
            complexity_df = pd.read_csv(complexity_files[0])
            complexity_dict = complexity_df.iloc[0].to_dict()
            
            # Load classification results
            classification_files = list(network_dir.glob("*_classification_results.csv"))
            classification_df = pd.read_csv(classification_files[0]) if classification_files else pd.DataFrame()
            
            # Load link prediction results  
            link_pred_files = list(network_dir.glob("*_link_prediction_results.csv"))
            link_pred_df = pd.read_csv(link_pred_files[0]) if link_pred_files else pd.DataFrame()
            
            # Load ranking results
            ranking_files = list(network_dir.glob("*_ranking_results.csv"))
            ranking_df = pd.read_csv(ranking_files[0]) if ranking_files else pd.DataFrame()
            
            # Get all unique methods
            methods = set()
            if not classification_df.empty:
                methods.update(classification_df['method'].unique())
            if not link_pred_df.empty:
                methods.update(link_pred_df['method'].unique())
            if not ranking_df.empty:
                methods.update(ranking_df['method'].unique())
            
            # Process classification results - data is already aggregated!
            clf_aggregated = {}
            if not classification_df.empty:
                # The data is already aggregated per method, just rename columns
                clf_aggregated = classification_df.copy()
                # Rename columns to add classification_ prefix
                clf_aggregated.columns = [
                    'method' if col == 'method' else
                    'network_id' if col == 'network_id' else
                    f'classification_{col}'
                    for col in clf_aggregated.columns
                ]
                # Drop network_id as we'll add it later
                if 'classification_network_id' in clf_aggregated.columns:
                    clf_aggregated = clf_aggregated.drop(columns=['classification_network_id'])
            
            # Process link prediction results - data is already aggregated!
            lp_aggregated = {}
            if not link_pred_df.empty:
                # The data is already aggregated per method, just rename columns
                lp_aggregated = link_pred_df.copy()
                # Rename columns to add link_prediction_ prefix
                lp_aggregated.columns = [
                    'method' if col == 'method' else
                    'network_id' if col == 'network_id' else
                    f'link_prediction_{col}'
                    for col in lp_aggregated.columns
                ]
                # Drop network_id as we'll add it later
                if 'link_prediction_network_id' in lp_aggregated.columns:
                    lp_aggregated = lp_aggregated.drop(columns=['link_prediction_network_id'])
            
            # Process ranking results
            ranking_aggregated = {}
            if not ranking_df.empty:
                ranking_aggregated = ranking_df.copy()
                # Rename columns to have ranking_ prefix
                ranking_aggregated.columns = ['method' if col == 'method' else f'ranking_{col}' 
                                             for col in ranking_aggregated.columns]
            
            # Create one row per method
            for method in methods:
                row_data = complexity_dict.copy()
                row_data['method'] = method
                row_data['network_id'] = network_id
                
                # Add classification data
                if isinstance(clf_aggregated, pd.DataFrame) and not clf_aggregated.empty:
                    method_clf = clf_aggregated[clf_aggregated['method'] == method]
                    if not method_clf.empty:
                        for col in method_clf.columns:
                            if col != 'method':
                                row_data[col] = method_clf.iloc[0][col]
                
                # Add link prediction data
                if isinstance(lp_aggregated, pd.DataFrame) and not lp_aggregated.empty:
                    method_lp = lp_aggregated[lp_aggregated['method'] == method]
                    if not method_lp.empty:
                        for col in method_lp.columns:
                            if col != 'method':
                                row_data[col] = method_lp.iloc[0][col]
                
                # Add ranking data
                if isinstance(ranking_aggregated, pd.DataFrame) and not ranking_aggregated.empty:
                    method_rank = ranking_aggregated[ranking_aggregated['method'] == method]
                    if not method_rank.empty:
                        for col in method_rank.columns:
                            if col != 'method':
                                row_data[col] = method_rank.iloc[0][col]
                
                all_results.append(row_data)
            
            print(f"  ✓ Processed {len(methods)} methods")
            
        except Exception as e:
            print(f"  ✗ Error processing {network_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create DataFrame
    if not all_results:
        print("\n⚠ No results collected!")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(all_results)
    
    # Save
    output_path = results_path / output_file
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"Total rows: {len(results_df)}")
    print(f"Total columns: {len(results_df.columns)}")
    print(f"Unique networks: {results_df['network_id'].nunique()}")
    print(f"Unique methods: {results_df['method'].nunique()}")
    print(f"\nSaved to: {output_path}")
    print("="*80)
    
    return results_df


if __name__ == '__main__':
    import sys
    
    # Default path
    results_dir = '/Users/aritrabose/OneDrive - IBM/Research/Quantum/quvine/ppi_disease_v3/results'
    
    # Allow command line override
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    
    df = build_comprehensive_all_methods(results_dir)
    
    if not df.empty:
        print("\nColumn names:")
        for col in sorted(df.columns):
            print(f"  - {col}")

