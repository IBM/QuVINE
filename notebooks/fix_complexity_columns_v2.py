#!/usr/bin/env python3
"""
Fix the complexity column selection to use _mean versions of base measures
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
            if 'Define specific complexity features' in source and 'base_complexity_features' in source:
                print("Fixing complexity column selection to use _mean versions...")
                
                new_source = '''# Calculate NaN percentage for each column
nan_percentages = (delta_df.isna().sum() / len(delta_df)) * 100

# Identify columns with >50% NaNs
high_nan_cols = nan_percentages[nan_percentages > 50].index.tolist()
print(f"\\nColumns with >50% NaNs: {len(high_nan_cols)}")
if len(high_nan_cols) > 0:
    print("Dropping:", high_nan_cols[:10], "..." if len(high_nan_cols) > 10 else "")

# Drop high-NaN columns
delta_df_clean = delta_df.drop(columns=high_nan_cols)

# Define specific complexity features to use (using _mean versions)
base_complexity_features = [
    'num_nodes_mean', 'num_edges_mean', 'spectral_gap_mean', 
    'algebraic_connectivity_mean', 'spectral_entropy_mean', 
    'von_neumann_entropy_mean', 'quantum_complexity_mean', 
    'modularity_mean', 'clustering_mean_mean', 'degree_heterogeneity_mean', 
    'quantum_advantage_score_mean', 'cyclomatic_number_mean', 
    'kirchhoff_index_mean', 'qbc_intrinsic_dimension_mean', 
    'qbc_total_correlations_mean', 'qbc_variation_mean',
    'qbc_num_non_zero_entries_mean', 'qbc_num_low_variance_features_mean',
    'qbc_coefficient_of_variation_mean', 'qbc_skewness_mean', 
    'qbc_kurtosis_mean', 'qbc_mean_log_kernel_density_mean', 
    'qbc_isomap_reconstruction_error_mean', 'qbc_fractal_dimension_mean', 
    'qbc_mutual_information_mean'
]

# Filter to only include columns that exist in the dataframe
complexity_cols = [col for col in base_complexity_features if col in delta_df_clean.columns]

print(f"\\nAfter cleaning:")
print(f"  Total columns: {delta_df_clean.shape[1]}")
print(f"  Complexity features (using _mean): {len(complexity_cols)}")
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
    print("1. ✓ Using _mean versions of the 25 complexity features")
    print("2. ✓ This matches the actual column names in the CSV file")
    print("3. ✓ All features should now appear in the forest plots")
    
    return notebook_path

if __name__ == '__main__':
    notebook_path = 'delta_analysis.ipynb'
    fix_complexity_columns(notebook_path)

# Made with Bob
