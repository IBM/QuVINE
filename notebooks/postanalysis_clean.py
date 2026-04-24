# This will be converted to notebook
# Cell 1: Imports and setup
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
import matplotlib.patches as mpatches

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

sns.set_style('whitegrid', {'grid.linestyle': '--', 'grid.alpha': 0.3})

QUANTUM_COLOR = '#E74C3C'
CLASSICAL_COLOR = '#3498DB'

# Cell 2: Load data
PATH = '../../results/'
dg = ['hn4', 'ppi', 'hg']
res_dict = {}
for f in glob(PATH + '*.csv'):
    for d in dg: 
        if d in f: 
            res_dict[d] = pd.read_csv(f)
            print(f"Loaded {d.upper()}: {res_dict[d].shape}")

# Cell 3: Compile
compiled_dfs = []
for dataset_name, df in res_dict.items():
    df_copy = df.copy()
    df_copy['dataset'] = dataset_name
    compiled_dfs.append(df_copy)

compiled_df = pd.concat(compiled_dfs, axis=0, ignore_index=True, sort=False)
print(f"Compiled: {compiled_df.shape}")

# Cell 4: Filter methods
quantum_methods = ['quvine_heat', 'quvine_fused-filt', 'quvine_rwr', 'quvine_ctqw', 
                   'quvine_poly', 'quvine_fused', 'quvine_hgcnmf', 'quvine_pgcnmf',
                   'quvine_fused-walk', 'quvine_fused-gcnmf', 'quvine_dtqw']
classical_methods = ['node2vec', 'netmf', 'gcn', 'graphsage', 'appnp', 'baseline_gcnmf']

if 'method' in compiled_df.columns:
    all_methods = compiled_df['method'].unique()
    present_quantum = [m for m in quantum_methods if m in all_methods]
    present_classical = [m for m in classical_methods if m in all_methods]
    
    df_filtered = compiled_df[compiled_df['method'].isin(present_quantum + present_classical)].copy()
    df_filtered['method_type'] = df_filtered['method'].apply(
        lambda x: 'Quantum' if x in quantum_methods else 'Classical'
    )
    
    # Create base_dataset column
    if 'network_id' in df_filtered.columns:
        df_filtered['base_dataset'] = df_filtered['network_id'].str.replace(r'_rep\d+$', '', regex=True)
        print(f"Base datasets: {df_filtered['base_dataset'].nunique()}")

# Cell 5: Create plots per dataset
meta_output_dir = Path(PATH) / 'meta_analysis'
meta_output_dir.mkdir(exist_ok=True)

task_metrics = {
    'Ranking': 'ranking_precision@10_centroid',
    'Classification': 'classification_mean_f1_macro',
    'Link Prediction': 'link_prediction_mean_auc_roc'
}

if 'base_dataset' in df_filtered.columns:
    base_datasets = sorted(df_filtered['base_dataset'].unique())
    print(f"Creating {len(base_datasets)} plots...")
    
    for base_ds in base_datasets:
        df_ds = df_filtered[df_filtered['base_dataset'] == base_ds]
        if len(df_ds) == 0:
            continue
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (task_name, metric) in enumerate(task_metrics.items()):
            ax = axes[idx]
            
            if metric in df_ds.columns and not df_ds[metric].isna().all():
                quantum_data = df_ds[df_ds['method_type'] == 'Quantum'][metric].dropna()
                classical_data = df_ds[df_ds['method_type'] == 'Classical'][metric].dropna()
                
                if len(quantum_data) > 0 and len(classical_data) > 0:
                    bp = ax.boxplot(
                        [quantum_data, classical_data],
                        labels=['Quantum', 'Classical'],
                        patch_artist=True,
                        widths=0.6
                    )
                    
                    bp['boxes'][0].set_facecolor(QUANTUM_COLOR)
                    bp['boxes'][0].set_alpha(0.7)
                    bp['boxes'][1].set_facecolor(CLASSICAL_COLOR)
                    bp['boxes'][1].set_alpha(0.7)
                    
                    ax.set_title(task_name, fontsize=13, fontweight='bold')
                    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
                    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        
        fig.suptitle(f'{base_ds}', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        safe_name = base_ds.replace('/', '_').replace(' ', '_')
        plt.savefig(meta_output_dir / f'boxplot_{safe_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ {safe_name}")

print(f"\n✓ Done: {len(base_datasets)} plots created")
