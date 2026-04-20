#!/usr/bin/env python3
"""
Compile comprehensive results from individual PPI experiment directories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import glob

def compile_ppi_results(base_dir='/Users/aritrabose/OneDrive - IBM/Research/Quantum/quvine/ppi_disease_v3/results',
                        output_file='results/comprehensive_results_ppi_complete.csv'):
    """Compile all PPI results from individual experiment directories"""
    
    print("Compiling PPI comprehensive results from individual directories...")
    
    # Find all experiment directories
    exp_dirs = []
    for pattern in ['BioPlex3_*_rep*', 'HumanNet_*_rep*', 'PCNet_*_rep*', 
                    'ProteomeHD_*_rep*', 'STRING_*_rep*']:
        exp_dirs.extend(glob.glob(os.path.join(base_dir, pattern)))
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    all_results = []
    
    for exp_dir in sorted(exp_dirs):
        exp_name = os.path.basename(exp_dir)
        print(f"Processing {exp_name}...")
        
        # Parse experiment name
        parts = exp_name.rsplit('_rep', 1)
        if len(parts) == 2:
            network_disease = parts[0]
            rep_num = parts[1]
            
            # Further split network and disease
            network_parts = network_disease.split('_')
            if len(network_parts) >= 2:
                network_type = network_parts[0]
                disease = '_'.join(network_parts[1:])
            else:
                network_type = network_disease
                disease = 'unknown'
        else:
            continue
        
        # Load complexity
        complexity_file = os.path.join(exp_dir, f'{exp_name}_complexity.csv')
        complexity_data = {}
        if os.path.exists(complexity_file):
            complexity_df = pd.read_csv(complexity_file)
            if len(complexity_df) > 0:
                complexity_data = complexity_df.iloc[0].to_dict()
        
        # Load classification results
        class_file = os.path.join(exp_dir, f'{exp_name}_classification_results.csv')
        if os.path.exists(class_file):
            class_df = pd.read_csv(class_file)
            for _, row in class_df.iterrows():
                result = {
                    'network_type': 'real_ppi',
                    'ppi_network': network_type,
                    'disease': disease,
                    'network_id': exp_name,
                    'rep_num': rep_num,
                    'method': row['method'],
                    'task': 'node_classification'
                }
                
                # Add complexity metrics
                result.update(complexity_data)
                
                # Add classification metrics
                for col in class_df.columns:
                    if col != 'method' and col != 'network_id':
                        result[f'classification_{col}'] = row[col]
                
                all_results.append(result.copy())
        
        # Load ranking results
        rank_file = os.path.join(exp_dir, f'{exp_name}_ranking_results.csv')
        if os.path.exists(rank_file):
            rank_df = pd.read_csv(rank_file)
            for _, row in rank_df.iterrows():
                result = {
                    'network_type': 'real_ppi',
                    'ppi_network': network_type,
                    'disease': disease,
                    'network_id': exp_name,
                    'rep_num': rep_num,
                    'method': row['method'],
                    'task': 'node_ranking'
                }
                
                # Add complexity metrics
                result.update(complexity_data)
                
                # Add ranking metrics
                for col in rank_df.columns:
                    if col != 'method' and col != 'network_id':
                        result[f'ranking_{col}'] = row[col]
                
                all_results.append(result.copy())
        
        # Load link prediction results
        lp_file = os.path.join(exp_dir, f'{exp_name}_link_prediction_results.csv')
        if os.path.exists(lp_file):
            lp_df = pd.read_csv(lp_file)
            for _, row in lp_df.iterrows():
                result = {
                    'network_type': 'real_ppi',
                    'ppi_network': network_type,
                    'disease': disease,
                    'network_id': exp_name,
                    'rep_num': rep_num,
                    'method': row['method'],
                    'task': 'link_prediction'
                }
                
                # Add complexity metrics
                result.update(complexity_data)
                
                # Add link prediction metrics
                for col in lp_df.columns:
                    if col != 'method' and col != 'network_id':
                        result[f'link_prediction_{col}'] = row[col]
                
                all_results.append(result.copy())
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    print(f"\nCompiled {len(results_df)} total results")
    print(f"Unique methods: {results_df['method'].nunique()}")
    print(f"Methods: {sorted(results_df['method'].unique())}")
    print(f"Unique networks: {results_df['ppi_network'].nunique()}")
    print(f"Networks: {sorted(results_df['ppi_network'].unique())}")
    print(f"Unique diseases: {results_df['disease'].nunique()}")
    print(f"Diseases: {sorted(results_df['disease'].unique())}")
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved comprehensive PPI results to: {output_file}")
    
    return results_df

if __name__ == '__main__':
    df = compile_ppi_results()
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nBy task:")
    print(df['task'].value_counts())
    
    print(f"\nBy method:")
    print(df['method'].value_counts().sort_index())
    
    print(f"\nBy PPI network:")
    print(df['ppi_network'].value_counts().sort_index())
    
    print(f"\nBy disease:")
    print(df['disease'].value_counts().sort_index())

# Made with Bob
