#!/usr/bin/env python3
"""
Fix the complexity column selection to use only specific base measures
"""

import json

def fix_complexity_columns(notebook_path):
    """Fix the complexity column selection"""
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find and fix the data cleaning cell
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix the complexity column identification
            if 'Identify complexity feature columns' in source and 'complexity_cols = [col for col' in source:
                print("Fixing complexity column selection...")
                
                new_source = '''# Calculate NaN percentage for each column
nan_percentages = (delta_df.isna().sum() / len(delta_df)) * 100

# Identify columns with >50% NaNs
high_nan_cols = nan_percentages[nan_percentages > 50].index.tolist()
print(f"\\nColumns with >50% NaNs: {len(high_nan_cols)}")
if len(high_nan_cols) > 0:
    print("Dropping:", high_nan_cols[:10], "..." if len(high_nan_cols) > 10 else "")

# Drop high-NaN columns
delta_df_clean = delta_df.drop(columns=high_nan_cols)

# Define specific complexity features to use (base measures only)
base_complexity_features = [
    'num_nodes', 'num_edges', 'spectral_gap', 'algebraic_connectivity', 
    'spectral_entropy', 'von_neumann_entropy', 'quantum_complexity', 
    'modularity', 'clustering_mean', 'degree_heterogeneity', 
    'quantum_advantage_score', 'cyclomatic_number', 'kirchhoff_index',
    'qbc_intrinsic_dimension', 'qbc_total_correlations', 'qbc_variation',
    'qbc_num_non_zero_entries', 'qbc_num_low_variance_features',
    'qbc_coefficient_of_variation', 'qbc_skewness', 'qbc_kurtosis',
    'qbc_mean_log_kernel_density', 'qbc_isomap_reconstruction_error',
    'qbc_fractal_dimension', 'qbc_mutual_information'
]

# Filter to only include columns that exist in the dataframe
complexity_cols = [col for col in base_complexity_features if col in delta_df_clean.columns]

print(f"\\nAfter cleaning:")
print(f"  Total columns: {delta_df_clean.shape[1]}")
print(f"  Complexity features (base measures): {len(complexity_cols)}")
print(f"  Rows: {delta_df_clean.shape[0]}")
print(f"\\nUsing complexity features:")
for col in complexity_cols:
    print(f"  - {col}")

# Show remaining NaN percentages for complexity features
remaining_nans = (delta_df_clean[complexity_cols].isna().sum() / len(delta_df_clean)) * 100
print(f"\\nRemaining NaN percentages in complexity features:")
print(f"  Max: {remaining_nans.max():.1f}%")
print(f"  Mean: {remaining_nans.mean():.1f}%")
print(f"  Features with >20% NaNs: {(remaining_nans > 20).sum()}")
'''
                cell['source'] = new_source.split('\n')
                # Add newline to each line except last
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                 for i, line in enumerate(cell['source'])]
                break
    
    # Write fixed notebook
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\nFixed notebook saved to: {notebook_path}")
    print("\nChanges made:")
    print("1. ✓ Changed from _mean/_median/_std suffixes to base complexity measures")
    print("2. ✓ Using only the 25 specific complexity features you specified")
    print("3. ✓ Added list of features being used for transparency")
    
    return notebook_path

if __name__ == '__main__':
    notebook_path = 'delta_analysis.ipynb'
    fix_complexity_columns(notebook_path)

