#!/usr/bin/env python3
"""
Script to fix delta_analysis.ipynb:
1. Fix RGBA color error in forest plot
2. Add grid search for ridge regression
3. Fix bootstrap to use optimal alpha
"""

import json
import sys

def fix_notebook(notebook_path):
    """Fix the delta_analysis notebook"""
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find and fix the bootstrap_ridge_regression function
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix bootstrap function
            if 'def bootstrap_ridge_regression' in source:
                print("Fixing bootstrap_ridge_regression function...")
                new_source = '''def bootstrap_ridge_regression(X_df, y, feature_names, n_bootstrap=1000, alpha_range=np.logspace(-3, 3, 50)):
    """
    Perform ridge regression with bootstrapping
    Uses grid search to find optimal alpha, then fixes it for bootstrap
    """
    n_samples, n_features = X_df.shape
    X = X_df.values  # Convert to numpy array
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Grid search for optimal alpha using cross-validation
    print(f"  Running grid search over {len(alpha_range)} alpha values...")
    ridge_cv = RidgeCV(alphas=alpha_range, cv=5, scoring='r2')
    ridge_cv.fit(X_scaled, y)
    optimal_alpha = ridge_cv.alpha_
    
    # Get cross-validation score with optimal alpha
    cv_scores = cross_val_score(Ridge(alpha=optimal_alpha), X_scaled, y, cv=5, scoring='r2')
    
    print(f"  Optimal alpha: {optimal_alpha:.4f}")
    print(f"  CV R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Bootstrap with FIXED optimal alpha
    print(f"  Running {n_bootstrap} bootstrap iterations with fixed alpha={optimal_alpha:.4f}...")
    bootstrap_coefs = np.zeros((n_bootstrap, n_features))
    
    for i in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[indices]
        y_boot = y[indices]
        
        ridge = Ridge(alpha=optimal_alpha)
        ridge.fit(X_boot, y_boot)
        bootstrap_coefs[i] = ridge.coef_
    
    # Calculate statistics
    coefs_mean = bootstrap_coefs.mean(axis=0)
    coefs_std = bootstrap_coefs.std(axis=0)
    
    # Calculate p-values (two-tailed test)
    p_values = np.zeros(n_features)
    for j in range(n_features):
        if coefs_mean[j] > 0:
            p_values[j] = 2 * (bootstrap_coefs[:, j] < 0).mean()
        else:
            p_values[j] = 2 * (bootstrap_coefs[:, j] > 0).mean()
        p_values[j] = max(p_values[j], 1/n_bootstrap)
    
    return coefs_mean, coefs_std, p_values, feature_names, optimal_alpha
'''
                cell['source'] = new_source.split('\n')
                # Add newline to each line except last
                cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line 
                                 for i, line in enumerate(cell['source'])]
            
            # Fix forest plot color issue
            if 'def create_forest_plot' in source and 'ecolor=colors' in source:
                print("Fixing forest plot color issue...")
                # Replace the problematic errorbar call
                source_lines = cell['source']
                new_lines = []
                for i, line in enumerate(source_lines):
                    if 'ax.errorbar(df[\'coef\'], y_pos, xerr=1.96*df[\'std\']' in line:
                        # Find the complete errorbar call
                        j = i
                        errorbar_lines = []
                        while j < len(source_lines):
                            errorbar_lines.append(source_lines[j])
                            if 'elinewidth=2)' in source_lines[j]:
                                break
                            j += 1
                        
                        # Replace with fixed version
                        new_lines.append('    # Plot error bars individually to avoid RGBA issue\n')
                        new_lines.append('    for idx, (coef_val, std_val, color) in enumerate(zip(df[\'coef\'], df[\'std\'], colors)):\n')
                        new_lines.append('        ax.errorbar(coef_val, y_pos[idx], xerr=1.96*std_val,\n')
                        new_lines.append('                    fmt=\'o\', markersize=8, capsize=5, capthick=2,\n')
                        new_lines.append('                    color=\'none\', ecolor=color, elinewidth=2)\n')
                        
                        # Skip the old errorbar lines
                        i = j
                        continue
                    new_lines.append(line)
                
                cell['source'] = new_lines
    
    # Write fixed notebook
    output_path = notebook_path.replace('.ipynb', '_fixed.ipynb')
    with open(output_path, 'w') as f:
        json.dump(nb, f, indent=1)
    
    print(f"\nFixed notebook saved to: {output_path}")
    print("\nChanges made:")
    print("1. ✓ Added grid search with cross-validation for optimal alpha")
    print("2. ✓ Fixed bootstrap to use optimal alpha consistently")
    print("3. ✓ Fixed RGBA color error in forest plot")
    print("4. ✓ Added detailed CV score reporting")
    
    return output_path

if __name__ == '__main__':
    notebook_path = 'delta_analysis.ipynb'
    if len(sys.argv) > 1:
        notebook_path = sys.argv[1]
    
    fix_notebook(notebook_path)

