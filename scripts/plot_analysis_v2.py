#!/usr/bin/env python3
"""
Extended analysis and visualization for hard_negatives_v2 results.

Q3: NC performance per network type across all labeling strategies
Q4: LP performance (AUC-ROC vs AUC-PR) per network type
Q5: Degree/distance matched plots aggregated per network type
"""

import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as scipy_stats

RESULTS_DIR = Path("/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v2/results")
VIZ_DIR     = Path("/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v2/visualizations")
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling constants (mirror collect_hpc_results.py) ─────────────────────────

CANONICAL_ORDER = [
    'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'quvine_fused-walk',
    'quvine_heat', 'quvine_poly', 'quvine_fused-filt',
    'quvine_hgcnmf', 'quvine_pgcnmf', 'quvine_fused-gcnmf',
    'node2vec', 'netmf', 'graphsage', 'baseline_filter', 'baseline_gcnmf',
]
_CLASSICAL = frozenset({'node2vec', 'netmf', 'graphsage', 'baseline_filter', 'baseline_gcnmf'})

METHOD_COLORS = {
    'quvine_rwr':          '#1565c0',
    'quvine_ctqw':         '#1e88e5',
    'quvine_dtqw':         '#90caf9',
    'quvine_fused-walk':   '#0d47a1',
    'quvine_heat':         '#2e7d32',
    'quvine_poly':         '#66bb6a',
    'quvine_fused-filt':   '#1b5e20',
    'quvine_hgcnmf':       '#6a1b9a',
    'quvine_pgcnmf':       '#ba68c8',
    'quvine_fused-gcnmf':  '#4a148c',
    'node2vec':            '#e65100',
    'netmf':               '#b71c1c',
    'graphsage':           '#00897b',
    'baseline_filter':     '#4e342e',
    'baseline_gcnmf':      '#f57f17',
}
METHOD_LABELS = {
    'quvine_rwr':          'RWR',
    'quvine_ctqw':         'CTQW',
    'quvine_dtqw':         'DTQW',
    'quvine_fused-walk':   'Fused-Walk',
    'quvine_heat':         'Heat',
    'quvine_poly':         'Poly',
    'quvine_fused-filt':   'Fused-Filt',
    'quvine_hgcnmf':       'HGCN-MF',
    'quvine_pgcnmf':       'PGCN-MF',
    'quvine_fused-gcnmf':  'Fused-GCN',
    'node2vec':            'Node2Vec',
    'netmf':               'NetMF',
    'graphsage':           'GraphSAGE',
    'baseline_filter':     'BL-Filter',
    'baseline_gcnmf':      'BL-GCN',
}
_DEFAULT_COLOR = '#888888'

# Labeling strategy columns and their display names
LABELING_COLS = {
    'classification_f1_community_louvain':            'Louvain',
    'classification_f1_community_label_propagation':  'Label Prop.',
    'classification_f1_degree_based':                 'Degree',
    'classification_f1_centrality_betweenness':       'Betweenness',
    'classification_f1_centrality_pagerank':          'PageRank',
    'classification_f1_core_periphery':               'Core-Periph.',
    'classification_f1_homophily_based':              'Homophily',
}
LABELING_ORDER = list(LABELING_COLS.keys())
LABELING_LABELS = list(LABELING_COLS.values())

NETWORK_LABELS = {
    'erdos_renyi':             'Erdős-Rényi',
    'random_geometric':        'Random Geometric',
    'watts_strogatz_high_p':   'WS High-p',
    'watts_strogatz_low_p':    'WS Low-p',
    'modular_strong':          'Modular Strong',
    'modular_medium':          'Modular Medium',
    'modular_many_communities':'Modular Many Comm.',
    'scale_free':              'Scale-Free',
    'core_periphery':          'Core-Periphery',
    'stochastic_block_model':  'SBM',
    'powerlaw_cluster':        'Powerlaw Cluster',
    'real_karate':             'Karate Club',
    'real_lesmis':             'Les Misérables',
    'real_polbooks':           'Polbooks',
}

PUB_RC = {
    'font.family':       'sans-serif',
    'font.size':         9,
    'axes.titlesize':    9,
    'axes.labelsize':    9,
    'xtick.labelsize':   7.5,
    'ytick.labelsize':   8,
    'legend.fontsize':   7.5,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
}


def _method_order(methods):
    present = set(methods)
    known_q = [m for m in CANONICAL_ORDER if m not in _CLASSICAL and m in present]
    known_c = [m for m in CANONICAL_ORDER if m in _CLASSICAL and m in present]
    extra_q = sorted(m for m in present if m not in set(CANONICAL_ORDER) and m.startswith('quvine_'))
    extra_c = sorted(m for m in present if m not in set(CANONICAL_ORDER) and not m.startswith('quvine_'))
    return known_q + extra_q + known_c + extra_c


def _ci95(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    mean = np.mean(vals)
    if len(vals) < 3:
        return mean, mean, mean
    se = scipy_stats.sem(vals)
    t  = scipy_stats.t.ppf(0.975, df=len(vals) - 1)
    return mean, mean - t * se, mean + t * se


# ─────────────────────────────────────────────────────────────────────────────
# Q3: NC performance per network, across all labeling strategies
# ─────────────────────────────────────────────────────────────────────────────

def plot_nc_by_labeling_strategy(df: pd.DataFrame) -> None:
    """
    For each network type: grouped bar chart where x=labeling strategy,
    bars = best quantum vs best classical mean F1, with individual method
    lines overlaid for detail.
    Also produces a summary heatmap: network_type × method, coloured by
    mean F1 using the best labeling strategy per row.
    """
    net_types = sorted(df['network_type'].dropna().unique())
    quantum_methods  = [m for m in _method_order(df['method'].unique()) if m not in _CLASSICAL]
    classical_methods = [m for m in _method_order(df['method'].unique()) if m in _CLASSICAL]

    # ── Figure 1: per-network subplot grid, x=labeling strategy ──────────────
    ncols = 4
    nrows = int(np.ceil(len(net_types) / ncols))
    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(nrows, ncols,
                                 figsize=(ncols * 4.5, nrows * 3.8),
                                 sharey=False)
        axes_flat = axes.flatten() if nrows > 1 else (axes if ncols > 1 else [axes])

        for ax_idx, nt in enumerate(net_types):
            ax = axes_flat[ax_idx]
            nt_df = df[df['network_type'] == nt]
            nt_label = NETWORK_LABELS.get(nt, nt)

            x = np.arange(len(LABELING_ORDER))
            width = 0.35

            # Quantum mean (best quantum method per labeling strategy)
            q_means, q_errs = [], []
            c_means, c_errs = [], []

            for col in LABELING_ORDER:
                if col not in nt_df.columns:
                    q_means.append(np.nan); q_errs.append(0)
                    c_means.append(np.nan); c_errs.append(0)
                    continue
                q_vals = nt_df[nt_df['method'].isin(quantum_methods)][col].dropna().values
                c_vals = nt_df[nt_df['method'].isin(classical_methods)][col].dropna().values
                qm, ql, qh = _ci95(q_vals)
                cm, cl, ch = _ci95(c_vals)
                q_means.append(qm); q_errs.append((qh - ql) / 2 if not np.isnan(qh) else 0)
                c_means.append(cm); c_errs.append((ch - cl) / 2 if not np.isnan(ch) else 0)

            q_means = np.array(q_means, dtype=float)
            c_means = np.array(c_means, dtype=float)
            q_errs  = np.array(q_errs,  dtype=float)
            c_errs  = np.array(c_errs,  dtype=float)

            bar_q = ax.bar(x - width/2, q_means, width, label='Quantum (mean)',
                           color='#1e88e5', alpha=0.8, yerr=q_errs, capsize=3,
                           error_kw=dict(elinewidth=0.7, capthick=0.7))
            bar_c = ax.bar(x + width/2, c_means, width, label='Classical (mean)',
                           color='#e65100', alpha=0.8, yerr=c_errs, capsize=3,
                           error_kw=dict(elinewidth=0.7, capthick=0.7))

            # Overlay dots for individual methods (best quantum + best classical)
            best_q = max(quantum_methods,
                         key=lambda m: nt_df[nt_df['method'] == m][
                             [c for c in LABELING_ORDER if c in nt_df.columns]].mean().mean()
                         if not nt_df[nt_df['method'] == m].empty else -1)
            best_c = max(classical_methods,
                         key=lambda m: nt_df[nt_df['method'] == m][
                             [c for c in LABELING_ORDER if c in nt_df.columns]].mean().mean()
                         if not nt_df[nt_df['method'] == m].empty else -1)

            for xi, col in enumerate(LABELING_ORDER):
                if col not in nt_df.columns:
                    continue
                bq_vals = nt_df[nt_df['method'] == best_q][col].dropna().values
                bc_vals = nt_df[nt_df['method'] == best_c][col].dropna().values
                if len(bq_vals):
                    ax.scatter(xi - width/2, np.mean(bq_vals), marker='*',
                               s=40, color='#0d47a1', zorder=5)
                if len(bc_vals):
                    ax.scatter(xi + width/2, np.mean(bc_vals), marker='*',
                               s=40, color='#b71c1c', zorder=5)

            ax.set_xticks(x)
            ax.set_xticklabels(LABELING_LABELS, rotation=35, ha='right', fontsize=7)
            ax.set_ylabel('F1-macro', fontsize=8)
            ax.set_title(f'{nt_label}\n(★={METHOD_LABELS.get(best_q,"Q")} vs '
                         f'{METHOD_LABELS.get(best_c,"C")})', fontsize=8, fontweight='bold')
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            if ax_idx == 0:
                ax.legend(fontsize=7, loc='upper right')

        for ax in axes_flat[len(net_types):]:
            ax.set_visible(False)

        fig.suptitle('Node Classification F1 per Network Type across Labeling Strategies\n'
                     '(bars = quantum/classical mean ± 95% CI; ★ = best individual method)',
                     fontsize=11, fontweight='bold', y=1.01)
        plt.tight_layout()
        out = VIZ_DIR / 'nc_by_labeling_strategy_per_network.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    # ── Figure 2: heatmap — method × network, value = best-labeling-strategy F1 ─
    with plt.rc_context(PUB_RC):
        all_methods = _method_order(df['method'].unique())
        heat_data = np.full((len(all_methods), len(net_types)), np.nan)

        for j, nt in enumerate(net_types):
            nt_df = df[df['network_type'] == nt]
            for i, m in enumerate(all_methods):
                m_df = nt_df[nt_df['method'] == m]
                if m_df.empty:
                    continue
                valid_cols = [c for c in LABELING_ORDER if c in m_df.columns]
                if not valid_cols:
                    continue
                # use mean_f1_macro (mean across all strategies)
                if 'classification_mean_f1_macro' in m_df.columns:
                    heat_data[i, j] = m_df['classification_mean_f1_macro'].mean()
                else:
                    heat_data[i, j] = m_df[valid_cols].mean().mean()

        heat_df = pd.DataFrame(
            heat_data,
            index=[METHOD_LABELS.get(m, m) for m in all_methods],
            columns=[NETWORK_LABELS.get(nt, nt) for nt in net_types],
        )

        fig, ax = plt.subplots(figsize=(len(net_types) * 0.9 + 2, len(all_methods) * 0.55 + 1.5))
        sns.heatmap(heat_df, ax=ax, cmap='RdYlGn', vmin=0, vmax=1,
                    annot=True, fmt='.2f', linewidths=0.3, linecolor='#dddddd',
                    cbar_kws={'label': 'Mean F1-macro', 'shrink': 0.7})
        # Draw separator between quantum and classical rows
        n_quantum = sum(1 for m in all_methods if m not in _CLASSICAL)
        ax.axhline(n_quantum, color='black', linewidth=2)
        ax.text(-0.5, n_quantum / 2, 'Quantum', fontsize=9, color='#1565c0',
                fontweight='bold', va='center', ha='right', rotation=90)
        ax.text(-0.5, n_quantum + (len(all_methods) - n_quantum) / 2, 'Classical',
                fontsize=9, color='#e65100', fontweight='bold', va='center',
                ha='right', rotation=90)

        ax.set_xlabel('Network Type', fontsize=9)
        ax.set_ylabel('Method', fontsize=9)
        ax.set_title('Node Classification — Mean F1-macro\n(mean across all labeling strategies, averaged over replicates)',
                     fontweight='bold')
        plt.xticks(rotation=35, ha='right')
        plt.tight_layout()
        out = VIZ_DIR / 'nc_heatmap_method_vs_network.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

    # ── Figure 3: winner per (network_type, labeling_strategy) ───────────────
    # Which method wins (highest mean F1) in each cell?
    winner_q = np.full((len(net_types), len(LABELING_ORDER)), np.nan)
    winner_c = np.full((len(net_types), len(LABELING_ORDER)), np.nan)
    for j, col in enumerate(LABELING_ORDER):
        if col not in df.columns:
            continue
        for i, nt in enumerate(net_types):
            nt_df = df[df['network_type'] == nt]
            q_vals = nt_df[nt_df['method'].isin(quantum_methods)][col].dropna().values
            c_vals = nt_df[nt_df['method'].isin(classical_methods)][col].dropna().values
            if len(q_vals):
                winner_q[i, j] = np.mean(q_vals)
            if len(c_vals):
                winner_c[i, j] = np.mean(c_vals)

    advantage = winner_q - winner_c  # positive = quantum wins

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(len(LABELING_ORDER) * 1.1 + 1, len(net_types) * 0.55 + 1.5))
        adv_df = pd.DataFrame(
            advantage,
            index=[NETWORK_LABELS.get(nt, nt) for nt in net_types],
            columns=LABELING_LABELS,
        )
        sns.heatmap(adv_df, ax=ax, cmap='RdBu_r', center=0,
                    annot=True, fmt='.2f', linewidths=0.3, linecolor='#dddddd',
                    cbar_kws={'label': 'ΔF1 (Quantum − Classical)', 'shrink': 0.7})
        ax.set_title('Quantum vs Classical Advantage per Labeling Strategy\n'
                     '(blue = quantum wins, red = classical wins)',
                     fontweight='bold')
        ax.set_xlabel('Labeling Strategy', fontsize=9)
        ax.set_ylabel('Network Type', fontsize=9)
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        out = VIZ_DIR / 'nc_quantum_vs_classical_advantage.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Q4: LP performance (AUC-ROC vs AUC-PR) per network type
# ─────────────────────────────────────────────────────────────────────────────

def plot_lp_per_network(df: pd.DataFrame) -> None:
    """
    Two-panel figure per LP metric (AUC-ROC, AUC-PR):
    - Left panel: bar chart across network types, best quantum vs best classical
    - Heatmap: method × network, coloured by mean AUC
    Also a consistency summary (rank correlation) showing which method is
    most consistent across network types.
    """
    net_types     = sorted(df['network_type'].dropna().unique())
    all_methods   = _method_order(df['method'].unique())
    quantum_methods   = [m for m in all_methods if m not in _CLASSICAL]
    classical_methods = [m for m in all_methods if m in _CLASSICAL]

    for metric_col, metric_label in [
        ('link_prediction_mean_auc_roc', 'AUC-ROC'),
        ('link_prediction_mean_auc_pr',  'AUC-PR'),
    ]:
        if metric_col not in df.columns:
            print(f'  [SKIP] {metric_col} not in dataframe')
            continue

        # ── Bar chart: per network type, quantum avg vs classical avg ─────────
        q_means, q_errs, c_means, c_errs = [], [], [], []
        best_q_by_nt, best_c_by_nt = [], []
        for nt in net_types:
            nt_df = df[df['network_type'] == nt]
            q_vals = nt_df[nt_df['method'].isin(quantum_methods)][metric_col].dropna().values
            c_vals = nt_df[nt_df['method'].isin(classical_methods)][metric_col].dropna().values
            qm, ql, qh = _ci95(q_vals)
            cm, cl, ch = _ci95(c_vals)
            q_means.append(qm); q_errs.append((qh - ql) / 2 if not np.isnan(qh) else 0)
            c_means.append(cm); c_errs.append((ch - cl) / 2 if not np.isnan(ch) else 0)
            # best individual method per nt
            bq = max(quantum_methods,
                     key=lambda m: nt_df[nt_df['method'] == m][metric_col].mean()
                     if not nt_df[nt_df['method'] == m].empty else -1)
            bc = max(classical_methods,
                     key=lambda m: nt_df[nt_df['method'] == m][metric_col].mean()
                     if not nt_df[nt_df['method'] == m].empty else -1)
            best_q_by_nt.append(bq)
            best_c_by_nt.append(bc)

        q_means = np.array(q_means, dtype=float)
        c_means = np.array(c_means, dtype=float)

        with plt.rc_context(PUB_RC):
            fig, axes = plt.subplots(1, 2, figsize=(16, 5.5),
                                     gridspec_kw={'width_ratios': [2, 1]})

            # Left: grouped bar per network type
            ax = axes[0]
            x = np.arange(len(net_types))
            w = 0.35
            ax.bar(x - w/2, q_means, w, label='Quantum (mean)',
                   color='#1e88e5', alpha=0.85,
                   yerr=q_errs, capsize=3,
                   error_kw=dict(elinewidth=0.7, capthick=0.7))
            ax.bar(x + w/2, c_means, w, label='Classical (mean)',
                   color='#e65100', alpha=0.85,
                   yerr=c_errs, capsize=3,
                   error_kw=dict(elinewidth=0.7, capthick=0.7))

            # Scatter best individual method
            for xi, (nt, bq, bc) in enumerate(zip(net_types, best_q_by_nt, best_c_by_nt)):
                nt_df = df[df['network_type'] == nt]
                bq_v = nt_df[nt_df['method'] == bq][metric_col].mean()
                bc_v = nt_df[nt_df['method'] == bc][metric_col].mean()
                ax.scatter(xi - w/2, bq_v, marker='*', s=45, color='#0d47a1', zorder=5)
                ax.scatter(xi + w/2, bc_v, marker='*', s=45, color='#b71c1c', zorder=5)

            ax.axhline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([NETWORK_LABELS.get(nt, nt) for nt in net_types],
                               rotation=38, ha='right', fontsize=7.5)
            ax.set_ylabel(f'Mean {metric_label}', fontsize=9)
            ax.set_title(f'Link Prediction {metric_label} per Network Type\n'
                         f'(★ = best individual method)', fontweight='bold')
            ax.set_ylim(0.4, 1.02)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.legend(fontsize=8)

            # Right: heatmap method × network
            ax2 = axes[1]
            heat_data = np.full((len(all_methods), len(net_types)), np.nan)
            for j, nt in enumerate(net_types):
                nt_df = df[df['network_type'] == nt]
                for i, m in enumerate(all_methods):
                    v = nt_df[nt_df['method'] == m][metric_col].mean()
                    heat_data[i, j] = v

            heat_df = pd.DataFrame(
                heat_data,
                index=[METHOD_LABELS.get(m, m) for m in all_methods],
                columns=[NETWORK_LABELS.get(nt, nt)[:8] for nt in net_types],
            )
            sns.heatmap(heat_df, ax=ax2, cmap='RdYlGn', vmin=0.4, vmax=1.0,
                        annot=True, fmt='.2f', linewidths=0.3, linecolor='#dddddd',
                        annot_kws={'size': 5.5},
                        cbar_kws={'label': metric_label, 'shrink': 0.7})
            n_quantum = sum(1 for m in all_methods if m not in _CLASSICAL)
            ax2.axhline(n_quantum, color='black', linewidth=1.5)
            ax2.set_xlabel('Network Type', fontsize=8)
            ax2.set_ylabel('')
            ax2.set_title(f'Method × Network\n{metric_label}', fontweight='bold', fontsize=9)
            ax2.tick_params(axis='x', labelsize=6.5, rotation=40)
            ax2.tick_params(axis='y', labelsize=7)

            plt.tight_layout()
            fname = f'lp_{metric_label.lower().replace("-","_")}_per_network.png'
            out = VIZ_DIR / fname
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out}')

    # ── Combined: both metrics side-by-side with consistency ranking ──────────
    auc_roc_col = 'link_prediction_mean_auc_roc'
    auc_pr_col  = 'link_prediction_mean_auc_pr'
    if auc_roc_col not in df.columns or auc_pr_col not in df.columns:
        return

    # Compute per-method mean and std across networks (consistency = low std)
    method_summary = []
    for m in all_methods:
        m_df = df[df['method'] == m]
        roc_means = m_df.groupby('network_type')[auc_roc_col].mean()
        pr_means  = m_df.groupby('network_type')[auc_pr_col].mean()
        method_summary.append({
            'method':         m,
            'label':          METHOD_LABELS.get(m, m),
            'group':          'Quantum' if m not in _CLASSICAL else 'Classical',
            'color':          METHOD_COLORS.get(m, _DEFAULT_COLOR),
            'mean_roc':       roc_means.mean(),
            'std_roc':        roc_means.std(),
            'mean_pr':        pr_means.mean(),
            'std_pr':         pr_means.std(),
            'consistency_roc': -roc_means.std(),  # higher = more consistent
        })
    summary_df = pd.DataFrame(method_summary).sort_values('mean_roc', ascending=False)

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Panel 1: Scatter AUC-ROC vs AUC-PR per method
        ax = axes[0]
        for _, row in summary_df.iterrows():
            marker = 'o' if row['group'] == 'Quantum' else 's'
            ax.scatter(row['mean_roc'], row['mean_pr'],
                       color=row['color'], marker=marker, s=80, alpha=0.9,
                       edgecolors='white', linewidths=0.5, zorder=3)
            ax.annotate(row['label'], (row['mean_roc'], row['mean_pr']),
                        fontsize=6.5, ha='left', va='bottom',
                        xytext=(2, 2), textcoords='offset points')
        ax.plot([0.4, 1], [0.4, 1], 'k--', linewidth=0.7, alpha=0.4)
        ax.set_xlim(0.4, 1.0); ax.set_ylim(0.4, 1.0)
        ax.set_xlabel('Mean AUC-ROC (across all networks)', fontsize=9)
        ax.set_ylabel('Mean AUC-PR (across all networks)', fontsize=9)
        ax.set_title('AUC-ROC vs AUC-PR per Method\n(○=Quantum, □=Classical)',
                     fontweight='bold')

        # Panel 2: Bar chart of mean AUC-ROC ranked by performance
        ax = axes[1]
        s = summary_df.copy()
        ax.barh(s['label'], s['mean_roc'],
                xerr=s['std_roc'], capsize=3,
                color=s['color'], alpha=0.85,
                error_kw=dict(elinewidth=0.7, capthick=0.7))
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax.set_xlabel('Mean AUC-ROC', fontsize=9)
        ax.set_title('Method Ranking by\nMean AUC-ROC (± std)', fontweight='bold')
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # Panel 3: Consistency plot (mean vs std — want high mean, low std)
        ax = axes[2]
        for _, row in summary_df.iterrows():
            marker = 'o' if row['group'] == 'Quantum' else 's'
            ax.scatter(row['std_roc'], row['mean_roc'],
                       color=row['color'], marker=marker, s=80, alpha=0.9,
                       edgecolors='white', linewidths=0.5, zorder=3)
            ax.annotate(row['label'], (row['std_roc'], row['mean_roc']),
                        fontsize=6.5, ha='left', va='bottom',
                        xytext=(2, 2), textcoords='offset points')
        ax.set_xlabel('Std of AUC-ROC across network types\n(lower = more consistent)', fontsize=9)
        ax.set_ylabel('Mean AUC-ROC', fontsize=9)
        ax.set_title('Performance Consistency\n(○=Quantum, □=Classical)', fontweight='bold')
        ax.invert_xaxis()  # lower std on right = more consistent

        plt.tight_layout()
        out = VIZ_DIR / 'lp_both_metrics_summary.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f'  Saved: {out}')

        # Print ranking
        print('\n── Link Prediction Method Ranking (mean across all networks) ──')
        print(summary_df[['label', 'group', 'mean_roc', 'std_roc', 'mean_pr', 'std_pr']].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Q5: Degree/distance matched plots per network type
# ─────────────────────────────────────────────────────────────────────────────

def plot_degree_distance_per_network(dd_df: pd.DataFrame) -> None:
    """
    For each network type, produce a 2×2 figure:
      Row 1: degree-matched AUC-PR (lines per method + quantum-classical gap bar)
      Row 2: distance-matched AUC-PR (lines per method + gap bar)
    Uses cosine AUC-PR as the score (auc_pr_cosine).
    """
    # Method set present
    methods   = [m for m in CANONICAL_ORDER if m in dd_df['method'].unique()]
    q_methods = [m for m in methods if m not in _CLASSICAL]
    c_methods = [m for m in methods if m in _CLASSICAL]

    deg_bins  = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    dist_bins = ['2', '3', '4', '5+']
    _LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    net_types = sorted(dd_df['network_type'].dropna().unique())

    for nt in net_types:
        nt_df    = dd_df[dd_df['network_type'] == nt]
        nt_label = NETWORK_LABELS.get(nt, nt)

        with plt.rc_context(PUB_RC):
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle(f'Degree & Distance Matched Analysis — {nt_label}',
                         fontsize=11, fontweight='bold', y=1.01)

            for row_idx, (bin_type, bins, xlabel, annotate_hub) in enumerate([
                ('degree',   deg_bins,  'Max-degree bin (Q1=low, Q5=hub)', True),
                ('distance', dist_bins, 'Shortest-path distance',          False),
            ]):
                sub = nt_df[nt_df['bin_type'] == bin_type]
                ax_lines = axes[row_idx][0]
                ax_gap   = axes[row_idx][1]

                # ── Line plot: AUC-PR per method ─────────────────────────────
                for mi, method in enumerate(methods):
                    mdf = sub[sub['method'] == method]
                    means, lows, highs = [], [], []
                    for bl in bins:
                        vals = mdf[mdf['bin_label'] == bl]['auc_pr_cosine'].dropna().values
                        m_, lo, hi = _ci95(vals)
                        means.append(m_); lows.append(lo); highs.append(hi)
                    means = np.array(means, dtype=float)
                    lows  = np.array(lows,  dtype=float)
                    highs = np.array(highs, dtype=float)
                    x_  = np.arange(len(bins))
                    valid = ~np.isnan(means)
                    if not valid.any():
                        continue
                    color = METHOD_COLORS.get(method, _DEFAULT_COLOR)
                    ls    = _LINESTYLES[mi % len(_LINESTYLES)]
                    lw    = 2.0 if method.startswith('quvine_fused') else 1.4
                    ax_lines.plot(x_[valid], means[valid],
                                  color=color, linestyle=ls, linewidth=lw,
                                  marker='o', markersize=4,
                                  label=METHOD_LABELS.get(method, method))
                    ax_lines.fill_between(x_[valid], lows[valid], highs[valid],
                                          alpha=0.08, color=color)

                ax_lines.set_xticks(range(len(bins)))
                ax_lines.set_xticklabels(bins, fontsize=8)
                ax_lines.set_xlabel(xlabel, fontsize=8.5)
                ax_lines.set_ylabel('AUC-PR (cosine)', fontsize=8.5)
                ax_lines.set_title(f'{bin_type.title()}-Matched AUC-PR — {nt_label}',
                                   fontweight='bold', pad=4)
                ax_lines.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                ax_lines.set_ylim(bottom=0.3)
                ax_lines.legend(fontsize=6.5, loc='best', ncol=2)

                if annotate_hub:
                    ylo, yhi = ax_lines.get_ylim()
                    ax_lines.annotate('← low-degree',
                                      xy=(0, ylo + 0.03 * (yhi - ylo)),
                                      fontsize=6, color='#555', style='italic')
                    ax_lines.annotate('hub-dominated →',
                                      xy=(len(bins) - 1, ylo + 0.03 * (yhi - ylo)),
                                      fontsize=6, color='#555', style='italic', ha='right')

                # ── Gap bar: Quantum − Classical ─────────────────────────────
                gaps = []
                for bl in bins:
                    bdf = sub[sub['bin_label'] == bl]
                    q_mean = bdf[bdf['method'].isin(q_methods)]['auc_pr_cosine'].mean()
                    c_mean = bdf[bdf['method'].isin(c_methods)]['auc_pr_cosine'].mean()
                    gaps.append(q_mean - c_mean)
                gaps = np.array(gaps, dtype=float)
                x_   = np.arange(len(bins))
                bar_colors = ['#1e88e5' if g >= 0 else '#e65100' for g in gaps]
                ax_gap.bar(x_, gaps, color=bar_colors, alpha=0.75,
                           edgecolor='white', linewidth=0.5)
                ax_gap.axhline(0, color='black', linewidth=0.8)
                ax_gap.set_xticks(x_)
                ax_gap.set_xticklabels(bins, fontsize=8)
                ax_gap.set_xlabel(xlabel, fontsize=8.5)
                ax_gap.set_ylabel('ΔAUC-PR  (Quantum − Classical)', fontsize=8.5)
                ax_gap.set_title(f'Quantum vs Classical Gap — {bin_type.title()} Bins',
                                 fontweight='bold', pad=4)
                from matplotlib.patches import Patch
                ax_gap.legend(handles=[
                    Patch(facecolor='#1e88e5', alpha=0.75, label='Quantum wins'),
                    Patch(facecolor='#e65100', alpha=0.75, label='Classical wins'),
                ], fontsize=7.5, loc='best')

            plt.tight_layout()
            fname = f'degree_dist_matched_{nt}.png'
            out   = VIZ_DIR / fname
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out}')

    # ── Summary heatmap: ΔAUC-PR across all network_types × bins ─────────────
    for bin_type, bins, xlabel in [
        ('degree',   deg_bins,  'Degree bin'),
        ('distance', dist_bins, 'Distance bin'),
    ]:
        sub = dd_df[dd_df['bin_type'] == bin_type]
        if sub.empty:
            continue
        pivot_data = {}
        for nt in net_types:
            nt_sub = sub[sub['network_type'] == nt]
            gaps = []
            for bl in bins:
                bdf = nt_sub[nt_sub['bin_label'] == bl]
                q_m = bdf[bdf['method'].isin(q_methods)]['auc_pr_cosine'].mean()
                c_m = bdf[bdf['method'].isin(c_methods)]['auc_pr_cosine'].mean()
                gaps.append(q_m - c_m)
            pivot_data[NETWORK_LABELS.get(nt, nt)] = gaps

        heat_df = pd.DataFrame(pivot_data, index=bins).T
        with plt.rc_context(PUB_RC):
            fig, ax = plt.subplots(figsize=(max(6, len(bins) * 1.4),
                                            max(5, len(net_types) * 0.55 + 1)))
            sns.heatmap(heat_df, ax=ax, cmap='RdBu_r', center=0,
                        annot=True, fmt='.3f', linewidths=0.3, linecolor='#cccccc',
                        cbar_kws={'label': 'ΔAUC-PR (Quantum − Classical)'})
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Network Type', fontsize=9)
            ax.set_title(f'Quantum vs Classical — {bin_type.title()}-Matched ΔAUC-PR\n'
                         f'(blue = quantum wins, red = classical wins)',
                         fontweight='bold')
            plt.tight_layout()
            out = VIZ_DIR / f'heatmap_delta_auc_{bin_type}_per_network.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f'  Saved: {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print('=' * 70)
    print('Extended Analysis — hard_negatives_v2')
    print('=' * 70)

    # Load comprehensive results
    comp_csv = RESULTS_DIR / 'comprehensive_results.csv'
    print(f'Loading {comp_csv} …')
    df = pd.read_csv(comp_csv)
    print(f'  {len(df)} rows × {len(df.columns)} cols')
    print(f'  Network types: {sorted(df["network_type"].unique())}')
    print(f'  Methods: {sorted(df["method"].unique())}')

    # Load degree_distance_matched CSVs (per-rep pre-computed)
    print('\nLoading degree_distance_matched CSVs …')
    dd_files = sorted(RESULTS_DIR.glob('*/*_degree_distance_matched.csv'))
    print(f'  Found {len(dd_files)} files')
    dd_df = pd.concat((pd.read_csv(f) for f in dd_files), ignore_index=True)
    print(f'  Total rows: {len(dd_df)}')
    print(f'  Network types: {sorted(dd_df["network_type"].unique())}')
    print(f'  Methods: {sorted(dd_df["method"].unique())}')

    print('\n── Q3: NC per network × labeling strategy ──')
    plot_nc_by_labeling_strategy(df)

    print('\n── Q4: LP AUC-ROC and AUC-PR per network ──')
    plot_lp_per_network(df)

    print('\n── Q5: Degree/distance matched per network type ──')
    plot_degree_distance_per_network(dd_df)

    print('\n' + '=' * 70)
    print(f'All figures saved to: {VIZ_DIR}')
    print('=' * 70)


if __name__ == '__main__':
    main()
