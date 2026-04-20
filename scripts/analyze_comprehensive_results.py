#!/usr/bin/env python3
"""
Comprehensive analysis of quantum vs classical methods across different network types and tasks.
This script:
1. Merges comprehensive results CSV files
2. Creates boxplots for performance comparison
3. Calculates delta performance between best quantum and classical methods
4. Generates statistics and meta-analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_and_merge_results(results_dir='results'):
    """Load and merge all comprehensive_results*.csv files"""
    results_path = Path(results_dir)
    csv_files = list(results_path.glob('comprehensive_results*.csv'))
    
    print(f"Found {len(csv_files)} CSV files to merge:")
    for f in csv_files:
        print(f"  - {f.name}")
    
    # Load and concatenate all CSV files
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\nMerged dataset shape: {merged_df.shape}")
    print(f"Total rows: {len(merged_df)}")
    
    return merged_df

def categorize_methods(df):
    """Categorize methods into quantum, classical, and baseline"""
    quantum_methods = ['quvine_heat', 'quvine_ctqw', 'quvine_dtqw', 'quvine_rwr', 
                       'quvine_poly', 'quvine_fused-filt', 'quvine_fused-walk',
                       'quvine_pgcnmf', 'quvine_hgcnmf', 'quvine_fused-gcnmf']
    
    classical_methods = ['baseline_gcnmf', 'graphsage', 'node2vec', 'netmf', 'appnp']
    
    baseline_filter = ['baseline_filter']
    
    df['method_category'] = df['method'].apply(
        lambda x: 'quantum' if x in quantum_methods 
        else ('classical' if x in classical_methods 
              else ('baseline_filter' if x in baseline_filter else 'unknown'))
    )
    
    return df

def extract_task_metrics(df):
    """Extract metrics for each task"""
    # Node Ranking metrics (using precision@10_max as primary metric)
    ranking_cols = [col for col in df.columns if 'ranking_precision@10_max' in col]
    
    # Node Classification metrics (using mean_f1_macro as primary metric)
    classification_cols = [col for col in df.columns if 'classification_mean_f1_macro' in col]
    
    # Link Prediction metrics (using mean_auc_roc as primary metric)
    link_pred_cols = [col for col in df.columns if 'link_prediction_mean_auc_roc' in col]
    
    return ranking_cols, classification_cols, link_pred_cols

def create_boxplots(df, output_dir='visualization'):
    """Create boxplots for performance across tasks and network types"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Exclude baseline_filter
    df_filtered = df[df['method_category'] != 'baseline_filter'].copy()
    
    # Define tasks and their metrics
    tasks = {
        'Node Ranking': 'ranking_precision@10_max',
        'Node Classification': 'classification_mean_f1_macro',
        'Link Prediction': 'link_prediction_mean_auc_roc'
    }
    
    # Get unique network types
    network_types = df_filtered['network_type'].unique()
    
    print(f"\nCreating boxplots for {len(network_types)} network types...")
    
    for task_name, metric in tasks.items():
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, net_type in enumerate(sorted(network_types)):
            if idx >= len(axes):
                break
                
            data = df_filtered[df_filtered['network_type'] == net_type]
            
            # Create boxplot
            ax = axes[idx]
            sns.boxplot(data=data, x='method_category', y=metric, ax=ax,
                       palette={'quantum': '#FF6B6B', 'classical': '#4ECDC4'})
            
            ax.set_title(f'{net_type}\n(n={len(data)})', fontsize=12, fontweight='bold')
            ax.set_xlabel('Method Category', fontsize=10)
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            
            # Add mean markers
            for category in ['quantum', 'classical']:
                cat_data = data[data['method_category'] == category][metric].dropna()
                if len(cat_data) > 0:
                    mean_val = cat_data.mean()
                    x_pos = 0 if category == 'quantum' else 1
                    ax.plot(x_pos, mean_val, 'D', color='black', markersize=8, 
                           markeredgewidth=2, markerfacecolor='yellow', zorder=10)
        
        # Remove empty subplots
        for idx in range(len(network_types), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle(f'{task_name} Performance by Network Type', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_file = output_path / f'boxplot_{task_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_file}")
        plt.close()

def calculate_delta_performance(df, output_dir='visualization'):
    """Calculate delta performance between best quantum and classical methods"""
    output_path = Path(output_dir)
    
    # Exclude baseline_filter
    df_filtered = df[df['method_category'] != 'baseline_filter'].copy()
    
    tasks = {
        'Node Ranking': 'ranking_precision@10_max',
        'Node Classification': 'classification_mean_f1_macro',
        'Link Prediction': 'link_prediction_mean_auc_roc'
    }
    
    results = []
    
    for task_name, metric in tasks.items():
        for net_type in df_filtered['network_type'].unique():
            data = df_filtered[df_filtered['network_type'] == net_type]
            
            # Get mean performance for quantum and classical
            quantum_mean = data[data['method_category'] == 'quantum'][metric].mean()
            classical_mean = data[data['method_category'] == 'classical'][metric].mean()
            
            # Get best performance for quantum and classical
            quantum_best = data[data['method_category'] == 'quantum'][metric].max()
            classical_best = data[data['method_category'] == 'classical'][metric].max()
            
            delta_mean = quantum_mean - classical_mean
            delta_best = quantum_best - classical_best
            
            results.append({
                'task': task_name,
                'network_type': net_type,
                'quantum_mean': quantum_mean,
                'classical_mean': classical_mean,
                'quantum_best': quantum_best,
                'classical_best': classical_best,
                'delta_mean': delta_mean,
                'delta_best': delta_best
            })
    
    delta_df = pd.DataFrame(results)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, task_name in enumerate(tasks.keys()):
        task_data = delta_df[delta_df['task'] == task_name]
        
        ax = axes[idx]
        x = np.arange(len(task_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, task_data['delta_mean'], width, 
                      label='Mean Delta', alpha=0.8, color='#FF6B6B')
        bars2 = ax.bar(x + width/2, task_data['delta_best'], width,
                      label='Best Delta', alpha=0.8, color='#4ECDC4')
        
        ax.set_xlabel('Network Type', fontsize=12)
        ax.set_ylabel('Performance Delta\n(Quantum - Classical)', fontsize=12)
        ax.set_title(task_name, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(task_data['network_type'], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)
    
    plt.suptitle('Delta Performance: Quantum vs Classical Methods', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / 'delta_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved delta performance plot: {output_file}")
    plt.close()
    
    return delta_df

def generate_statistics(df, delta_df, output_dir='visualization'):
    """Generate comprehensive statistics document"""
    output_path = Path(output_dir)
    
    # Exclude baseline_filter
    df_filtered = df[df['method_category'] != 'baseline_filter'].copy()
    
    tasks = {
        'Node Ranking': 'ranking_precision@10_max',
        'Node Classification': 'classification_mean_f1_macro',
        'Link Prediction': 'link_prediction_mean_auc_roc'
    }
    
    stats_lines = []
    stats_lines.append("=" * 80)
    stats_lines.append("COMPREHENSIVE RESULTS ANALYSIS: QUANTUM VS CLASSICAL METHODS")
    stats_lines.append("=" * 80)
    stats_lines.append("")
    
    # Dataset overview
    stats_lines.append("## DATASET OVERVIEW")
    stats_lines.append("-" * 80)
    stats_lines.append(f"Total experiments: {len(df_filtered)}")
    stats_lines.append(f"Network types: {df_filtered['network_type'].nunique()}")
    stats_lines.append(f"  - {', '.join(sorted(df_filtered['network_type'].unique()))}")
    stats_lines.append(f"Quantum methods tested: {len(df_filtered[df_filtered['method_category']=='quantum']['method'].unique())}")
    stats_lines.append(f"Classical methods tested: {len(df_filtered[df_filtered['method_category']=='classical']['method'].unique())}")
    stats_lines.append("")
    
    # Win rates by task
    stats_lines.append("## WIN RATES: QUANTUM VS CLASSICAL")
    stats_lines.append("-" * 80)
    
    for task_name, metric in tasks.items():
        stats_lines.append(f"\n### {task_name}")
        
        wins = {'quantum': 0, 'classical': 0, 'tie': 0}
        
        for net_type in df_filtered['network_type'].unique():
            data = df_filtered[df_filtered['network_type'] == net_type]
            
            quantum_best = data[data['method_category'] == 'quantum'][metric].max()
            classical_best = data[data['method_category'] == 'classical'][metric].max()
            
            if pd.notna(quantum_best) and pd.notna(classical_best):
                if quantum_best > classical_best:
                    wins['quantum'] += 1
                elif classical_best > quantum_best:
                    wins['classical'] += 1
                else:
                    wins['tie'] += 1
        
        total = sum(wins.values())
        if total > 0:
            stats_lines.append(f"  Quantum wins: {wins['quantum']} ({wins['quantum']/total*100:.1f}%)")
            stats_lines.append(f"  Classical wins: {wins['classical']} ({wins['classical']/total*100:.1f}%)")
            stats_lines.append(f"  Ties: {wins['tie']} ({wins['tie']/total*100:.1f}%)")
    
    stats_lines.append("")
    
    # Best methods by task
    stats_lines.append("## BEST PERFORMING METHODS BY TASK")
    stats_lines.append("-" * 80)
    
    for task_name, metric in tasks.items():
        stats_lines.append(f"\n### {task_name}")
        
        # Overall best
        best_idx = df_filtered[metric].idxmax()
        if pd.notna(best_idx):
            best_row = df_filtered.loc[best_idx]
            stats_lines.append(f"  Overall best: {best_row['method']} ({best_row['method_category']})")
            stats_lines.append(f"    Score: {best_row[metric]:.4f}")
            stats_lines.append(f"    Network: {best_row['network_type']}")
        
        # Best quantum
        quantum_data = df_filtered[df_filtered['method_category'] == 'quantum']
        best_q_idx = quantum_data[metric].idxmax()
        if pd.notna(best_q_idx):
            best_q = quantum_data.loc[best_q_idx]
            stats_lines.append(f"  Best quantum: {best_q['method']}")
            stats_lines.append(f"    Score: {best_q[metric]:.4f}")
            stats_lines.append(f"    Network: {best_q['network_type']}")
        
        # Best classical
        classical_data = df_filtered[df_filtered['method_category'] == 'classical']
        best_c_idx = classical_data[metric].idxmax()
        if pd.notna(best_c_idx):
            best_c = classical_data.loc[best_c_idx]
            stats_lines.append(f"  Best classical: {best_c['method']}")
            stats_lines.append(f"    Score: {best_c[metric]:.4f}")
            stats_lines.append(f"    Network: {best_c['network_type']}")
    
    stats_lines.append("")
    
    # Average performance by network type
    stats_lines.append("## AVERAGE PERFORMANCE BY NETWORK TYPE")
    stats_lines.append("-" * 80)
    
    for net_type in sorted(df_filtered['network_type'].unique()):
        stats_lines.append(f"\n### {net_type}")
        data = df_filtered[df_filtered['network_type'] == net_type]
        
        for task_name, metric in tasks.items():
            quantum_mean = data[data['method_category'] == 'quantum'][metric].mean()
            classical_mean = data[data['method_category'] == 'classical'][metric].mean()
            
            stats_lines.append(f"  {task_name}:")
            stats_lines.append(f"    Quantum mean: {quantum_mean:.4f}")
            stats_lines.append(f"    Classical mean: {classical_mean:.4f}")
            stats_lines.append(f"    Delta: {quantum_mean - classical_mean:+.4f}")
    
    stats_lines.append("")
    stats_lines.append("=" * 80)
    
    # Save to file
    output_file = output_path / 'comprehensive_statistics.txt'
    with open(output_file, 'w') as f:
        f.write('\n'.join(stats_lines))
    
    print(f"\nSaved statistics document: {output_file}")
    
    return '\n'.join(stats_lines)

def create_win_rate_plots(df, output_dir='visualization'):
    """Create stacked bar plots showing win rates"""
    output_path = Path(output_dir)
    
    df_filtered = df[df['method_category'] != 'baseline_filter'].copy()
    
    tasks = {
        'Node Ranking': 'ranking_precision@10_max',
        'Node Classification': 'classification_mean_f1_macro',
        'Link Prediction': 'link_prediction_mean_auc_roc'
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (task_name, metric) in enumerate(tasks.items()):
        wins_by_network = []
        
        for net_type in sorted(df_filtered['network_type'].unique()):
            data = df_filtered[df_filtered['network_type'] == net_type]
            
            quantum_best = data[data['method_category'] == 'quantum'][metric].max()
            classical_best = data[data['method_category'] == 'classical'][metric].max()
            
            if pd.notna(quantum_best) and pd.notna(classical_best):
                if quantum_best > classical_best:
                    winner = 'quantum'
                elif classical_best > quantum_best:
                    winner = 'classical'
                else:
                    winner = 'tie'
            else:
                winner = 'unknown'
            
            wins_by_network.append({'network_type': net_type, 'winner': winner})
        
        wins_df = pd.DataFrame(wins_by_network)
        win_counts = wins_df['winner'].value_counts()
        
        ax = axes[idx]
        colors = {'quantum': '#FF6B6B', 'classical': '#4ECDC4', 'tie': '#95E1D3'}
        
        win_counts.plot(kind='bar', ax=ax, color=[colors.get(x, 'gray') for x in win_counts.index])
        ax.set_title(task_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Winner', fontsize=12)
        ax.set_ylabel('Number of Network Types', fontsize=12)
        ax.tick_params(axis='x', rotation=0)
        
        # Add percentage labels
        total = len(wins_by_network)
        for i, (label, value) in enumerate(win_counts.items()):
            ax.text(i, value, f'{value}\n({value/total*100:.1f}%)', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.suptitle('Win Rates: Quantum vs Classical by Task', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / 'win_rates_by_task.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved win rate plot: {output_file}")
    plt.close()

def analyze_complexity_features(df, output_dir='visualization'):
    """Analyze which complexity features correlate with quantum advantage"""
    output_path = Path(output_dir)
    
    df_filtered = df[df['method_category'] != 'baseline_filter'].copy()
    
    # Complexity features to analyze
    complexity_features = [
        'spectral_gap', 'algebraic_connectivity', 'spectral_entropy',
        'quantum_complexity', 'modularity', 'clustering_mean',
        'degree_heterogeneity', 'quantum_advantage_score',
        'num_nodes', 'num_edges'
    ]
    
    tasks = {
        'Node Ranking': 'ranking_precision@10_max',
        'Node Classification': 'classification_mean_f1_macro',
        'Link Prediction': 'link_prediction_mean_auc_roc'
    }
    
    # Calculate quantum advantage for each experiment
    results = []
    
    for idx, row in df_filtered.iterrows():
        net_type = row['network_type']
        method_cat = row['method_category']
        
        for task_name, metric in tasks.items():
            if pd.notna(row[metric]):
                # Get classical baseline for this network
                classical_data = df_filtered[
                    (df_filtered['network_type'] == net_type) & 
                    (df_filtered['method_category'] == 'classical')
                ][metric]
                
                if len(classical_data) > 0:
                    classical_mean = classical_data.mean()
                    advantage = row[metric] - classical_mean if method_cat == 'quantum' else 0
                    
                    result = {'task': task_name, 'advantage': advantage}
                    for feat in complexity_features:
                        if feat in row:
                            result[feat] = row[feat]
                    results.append(result)
    
    advantage_df = pd.DataFrame(results)
    
    # Calculate correlations
    print("\n## COMPLEXITY FEATURE CORRELATIONS WITH QUANTUM ADVANTAGE")
    print("-" * 80)
    
    for task_name in tasks.keys():
        task_data = advantage_df[advantage_df['task'] == task_name]
        
        print(f"\n### {task_name}")
        correlations = []
        
        for feat in complexity_features:
            if feat in task_data.columns:
                corr = task_data[['advantage', feat]].corr().iloc[0, 1]
                if pd.notna(corr):
                    correlations.append((feat, corr))
        
        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for feat, corr in correlations[:10]:  # Top 10
            print(f"  {feat:30s}: {corr:+.4f}")
    
    # Create correlation heatmap
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, task_name in enumerate(tasks.keys()):
        task_data = advantage_df[advantage_df['task'] == task_name]
        
        # Select features that exist
        available_features = [f for f in complexity_features if f in task_data.columns]
        corr_data = task_data[['advantage'] + available_features].corr()['advantage'].drop('advantage')
        
        ax = axes[idx]
        corr_data.sort_values(ascending=False).plot(kind='barh', ax=ax, 
                                                     color=['#FF6B6B' if x > 0 else '#4ECDC4' for x in corr_data.sort_values(ascending=False)])
        ax.set_title(task_name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Correlation with Quantum Advantage', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Complexity Features Correlated with Quantum Advantage', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = output_path / 'complexity_correlations.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved complexity correlation plot: {output_file}")
    plt.close()

def main():
    """Main analysis pipeline"""
    print("=" * 80)
    print("COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)
    
    # Step 1: Load and merge results
    print("\n[1/7] Loading and merging CSV files...")
    df = load_and_merge_results('results')
    
    # Save merged results
    output_file = 'results/comprehensive_results_merged.csv'
    df.to_csv(output_file, index=False)
    print(f"Saved merged results to: {output_file}")
    
    # Step 2: Categorize methods
    print("\n[2/7] Categorizing methods...")
    df = categorize_methods(df)
    
    # Step 3: Create visualization directory
    print("\n[3/7] Creating visualization directory...")
    Path('visualization').mkdir(exist_ok=True)
    
    # Step 4: Create boxplots
    print("\n[4/7] Creating boxplots...")
    create_boxplots(df, 'visualization')
    
    # Step 5: Calculate delta performance
    print("\n[5/7] Calculating delta performance...")
    delta_df = calculate_delta_performance(df, 'visualization')
    delta_df.to_csv('visualization/delta_performance.csv', index=False)
    
    # Step 6: Create win rate plots
    print("\n[6/7] Creating win rate plots...")
    create_win_rate_plots(df, 'visualization')
    
    # Step 7: Generate statistics
    print("\n[7/7] Generating statistics and meta-analysis...")
    stats_text = generate_statistics(df, delta_df, 'visualization')
    
    # Complexity analysis
    analyze_complexity_features(df, 'visualization')
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nOutputs saved to:")
    print(f"  - results/comprehensive_results_merged.csv")
    print(f"  - visualization/ (all plots and statistics)")
    print("\nKey files:")
    print(f"  - visualization/comprehensive_statistics.txt")
    print(f"  - visualization/delta_performance.csv")
    print(f"  - visualization/boxplot_*.png")
    print(f"  - visualization/delta_performance.png")
    print(f"  - visualization/win_rates_by_task.png")
    print(f"  - visualization/complexity_correlations.png")

if __name__ == '__main__':
    main()

# Made with Bob
