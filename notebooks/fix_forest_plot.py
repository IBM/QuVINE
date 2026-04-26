#!/usr/bin/env python3
"""
Script to fix the duplicate line in create_forest_plot function
"""

import json

def fix_forest_plot(notebook_path):
    """Fix the forest plot function in the notebook"""
    
    # Read notebook
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # Find and fix the create_forest_plot function
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Fix forest plot function
            if 'def create_forest_plot' in source and 'ecolor=colors' in source:
                print("Fixing create_forest_plot function...")
                
                new_source = '''def create_forest_plot(task_name, coefficients, std_errors, p_values_fdr, features, 
                        significant, top_n=20, output_dir=None):
    """Create a forest plot showing top features with confidence intervals"""
    
    df = pd.DataFrame({
        'feature': features,
        'coef': coefficients,
        'std': std_errors,
        'p_fdr': p_values_fdr,
        'sig': significant
    })
    
    df['abs_coef'] = np.abs(df['coef'])
    df = df.sort_values('abs_coef', ascending=False).head(top_n)
    df = df.sort_values('coef')
    
    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
    
    colors = ['#E74C3C' if sig else '#95A5A6' for sig in df['sig']]
    
    y_pos = np.arange(len(df))
    
    # Plot error bars individually to avoid RGBA issue
    for idx, (coef_val, std_val, color) in enumerate(zip(df['coef'], df['std'], colors)):
        ax.errorbar(coef_val, y_pos[idx], xerr=1.96*std_val,
                    fmt='o', markersize=8, capsize=5, capthick=2,
                    color='none', ecolor=color, elinewidth=2)
    
    ax.scatter(df['coef'], y_pos, c=colors, s=100, zorder=3, edgecolors='black', linewidths=1)
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    display_names = []
    for feat in df['feature']:
        name = feat.replace('_mean', '').replace('_median', '').replace('_std', '')
        name = name.replace('qbc_', '').replace('num_', '')
        if len(name) > 30:
            name = name[:27] + '...'
        display_names.append(name)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel('Standardized Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'{task_name}\\nTop {top_n} Features Predicting Quantum Advantage', 
                fontsize=13, fontweight='bold', pad=20)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', label='Significant (FDR < 0.05)'),
        Patch(facecolor='#95A5A6', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_dir:
        safe_name = task_name.lower().replace(' ', '_')
        plt.savefig(output_dir / f'forest_plot_{safe_name}.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: forest_plot_{safe_name}.png")
    
    plt.show()
    return fig
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
    print("1. ✓ Removed duplicate errorbar line")
    print("2. ✓ Fixed RGBA color error by plotting error bars individually")
    
    return notebook_path

if __name__ == '__main__':
    notebook_path = 'delta_analysis.ipynb'
    fix_forest_plot(notebook_path)

