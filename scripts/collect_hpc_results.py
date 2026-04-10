#!/usr/bin/env python
"""
Collect, Aggregate, and Visualize HPC Results

Usage:
    python scripts/collect_hpc_results.py \\
        --results-dir outputs/hpc_results/results \\
        [--output-file comprehensive_results.csv] \\
        [--viz-dir outputs/hpc_results/visualizations] \\
        [--n-networks 20] \\
        [--quiet]
"""

import argparse
import re
import sys
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.lines as mlines
import seaborn as sns
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from quvine.comprehensive_embedding_analysis import collect_and_aggregate_results


# ---------------------------------------------------------------------------
# Method ordering and styling
# ---------------------------------------------------------------------------

# Canonical display order — quantum first (with fused "super-methods" grouped
# after the components they aggregate), then classical.
CANONICAL_ORDER = [
    'quvine_rwr',
    'quvine_ctqw',
    'quvine_dtqw',
    'quvine_fused-walk',    # walk-based fusion
    'quvine_heat',
    'quvine_poly',
    'quvine_fused-filt',    # filter-based fusion
    'quvine_hgcnmf',
    'quvine_pgcnmf',
    'quvine_fused-gcnmf',   # GCN-MF fusion
    # -- classical boundary --
    'node2vec',
    'netmf',
    'graphsage',
    'baseline_filter',
    'baseline_gcnmf',
]
_CLASSICAL = frozenset({'node2vec', 'netmf', 'graphsage', 'baseline_filter', 'baseline_gcnmf'})
_N_QUANTUM = sum(1 for m in CANONICAL_ORDER if m not in _CLASSICAL)  # 10

# Per-method colors: blues/teals for walk-based, greens for filter-based,
# purples for GCN-MF, warm tones for classical.
METHOD_COLORS = {
    'quvine_rwr':          '#1565c0',   # deep blue
    'quvine_ctqw':         '#1e88e5',   # blue
    'quvine_dtqw':         '#90caf9',   # light blue
    'quvine_fused-walk':   '#0d47a1',   # navy (bold — fused)
    'quvine_heat':         '#2e7d32',   # dark green
    'quvine_poly':         '#66bb6a',   # green
    'quvine_fused-filt':   '#1b5e20',   # forest (bold — fused)
    'quvine_hgcnmf':       '#6a1b9a',   # deep purple
    'quvine_pgcnmf':       '#ba68c8',   # light purple
    'quvine_fused-gcnmf':  '#4a148c',   # dark purple (bold — fused)
    'node2vec':            '#e65100',   # deep orange
    'netmf':               '#b71c1c',   # deep red
    'graphsage':           '#00897b',   # teal
    'baseline_filter':     '#4e342e',   # brown
    'baseline_gcnmf':      '#f57f17',   # amber
}
_DEFAULT_COLOR = '#888888'

# Display labels (shortened for tick readability)
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

# Metric candidate columns (priority order — first found is used)
RANKING_METRIC_CANDIDATES = [
    'ranking_precision@10_centroid',
    'ranking_precision@20_centroid',
    'ranking_precision@40_centroid',
    'ranking_precision@80_centroid',
    'ranking_recall@10_centroid',
]
CLASSIFICATION_METRIC_CANDIDATES = [
    'classification_mean_f1_macro',
    'classification_max_f1_macro',
    'classification_mean_accuracy',
]
LINK_PREDICTION_METRIC_CANDIDATES = [
    'link_prediction_mean_auc_roc',
    'link_prediction_max_auc_roc',
    'link_prediction_mean_auc_pr',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_available(df: pd.DataFrame, candidates: list):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _method_order(methods_present) -> list:
    """
    Return a sorted method list using CANONICAL_ORDER, with any unrecognised
    quantum-prefixed methods appended before the classical block.
    """
    present = set(methods_present)
    known_quantum   = [m for m in CANONICAL_ORDER if m not in _CLASSICAL and m in present]
    known_classical = [m for m in CANONICAL_ORDER if m in _CLASSICAL and m in present]
    extra_quantum   = sorted(m for m in present
                             if m not in set(CANONICAL_ORDER) and m.startswith('quvine_'))
    extra_classical = sorted(m for m in present
                             if m not in set(CANONICAL_ORDER) and not m.startswith('quvine_'))
    return known_quantum + extra_quantum + known_classical + extra_classical


def _method_colors(order: list) -> list:
    return [METHOD_COLORS.get(m, _DEFAULT_COLOR) for m in order]


def _method_labels(order: list) -> list:
    return [METHOD_LABELS.get(m, m) for m in order]


def _clean_metric_label(col: str) -> str:
    return (
        col.replace('ranking_', '')
           .replace('classification_', '')
           .replace('link_prediction_', '')
           .replace('mean_', 'Mean ')
           .replace('max_', 'Max ')
           .replace('_macro', ' (Macro)')
           .replace('_centroid', '')
           .replace('_', ' ')
           .replace('auc roc', 'AUC-ROC')
           .replace('auc pr', 'AUC-PR')
           .replace('f1', 'F1')
           .title()
    )


def _divider_x(rendered_order: list) -> float | None:
    """Return x position (between ticks) where quantum ends and classical begins."""
    for i, m in enumerate(rendered_order):
        if m in _CLASSICAL:
            return i - 0.5
    return None


# ---------------------------------------------------------------------------
# Publication-quality rcParams
# ---------------------------------------------------------------------------

PUB_RC = {
    'font.family':       'sans-serif',
    'font.size':         9,
    'axes.titlesize':    10,
    'axes.labelsize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'legend.fontsize':   8,
    'figure.dpi':        150,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linewidth':    0.5,
}


# ---------------------------------------------------------------------------
# Single box-plot panel
# ---------------------------------------------------------------------------

def _boxplot_panel(ax, df: pd.DataFrame, metric_col: str, title: str,
                   network_type: str = None) -> None:
    """Draw one box-plot panel for a single (task, network_type) combination."""
    subset = df.copy()
    if network_type is not None and 'network_type' in subset.columns:
        subset = subset[subset['network_type'] == network_type]

    if metric_col not in subset.columns or subset[metric_col].isna().all():
        ax.text(0.5, 0.5, f'No data\n({metric_col})',
                ha='center', va='center', transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title(title, fontweight='bold')
        return

    order = _method_order(subset['method'].unique())
    plot_df = subset[['method', metric_col]].dropna()
    rendered = [m for m in order if m in plot_df['method'].values]
    colors   = _method_colors(rendered)

    sns.boxplot(
        data=plot_df,
        x='method', y=metric_col,
        order=rendered,
        palette=colors,
        linewidth=0.8,
        width=0.6,
        flierprops=dict(marker='o', markersize=2.5, alpha=0.4, linewidth=0),
        medianprops=dict(color='black', linewidth=1.5),
        ax=ax,
    )

    # Tick labels
    ax.set_xticks(range(len(rendered)))
    ax.set_xticklabels(_method_labels(rendered), rotation=35, ha='right', fontsize=7.5)
    ax.set_xlabel('')
    ax.set_ylabel(_clean_metric_label(metric_col), fontsize=8.5)
    ax.set_title(title, fontweight='bold', pad=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Quantum / classical divider
    div_x = _divider_x(rendered)
    if div_x is not None:
        ax.axvline(div_x, color='#555555', linewidth=1.2, linestyle='--', alpha=0.7, zorder=3)
        yhi = ax.get_ylim()[1]
        ax.text(div_x - 0.15, yhi, 'Quantum', fontsize=6.5, color='#555555',
                ha='right', va='top', style='italic')
        ax.text(div_x + 0.15, yhi, 'Classical', fontsize=6.5, color='#555555',
                ha='left', va='top', style='italic')


# ---------------------------------------------------------------------------
# Per-network-type box-plot figures  (3 task panels per figure)
# ---------------------------------------------------------------------------

def generate_task_boxplots(df: pd.DataFrame, viz_dir: Path, n_networks: int = 20) -> None:
    """
    For each network type present in the data, produce one figure with three
    side-by-side panels: Ranking | Classification | Link Prediction.

    Files: boxplot_{network_type}.png
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    net_types = (
        sorted(df['network_type'].dropna().unique())
        if 'network_type' in df.columns
        else [None]
    )

    task_cfgs = [
        dict(candidates=RANKING_METRIC_CANDIDATES,         task_label='Node Prioritization'),
        dict(candidates=CLASSIFICATION_METRIC_CANDIDATES,  task_label='Node Classification'),
        dict(candidates=LINK_PREDICTION_METRIC_CANDIDATES, task_label='Link Prediction'),
    ]

    with plt.rc_context(PUB_RC):
        for nt in net_types:
            nt_label  = nt.replace('_', '-').title() if nt else 'All Networks'
            nt_suffix = nt if nt else 'all'

            fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
            fig.suptitle(
                f'{nt_label} Networks  —  N = {n_networks}',
                fontsize=11, fontweight='bold', y=1.01,
            )

            for ax, cfg in zip(axes, task_cfgs):
                col = _first_available(df, cfg['candidates'])
                if col is None:
                    ax.set_visible(False)
                    continue
                _boxplot_panel(ax, df, col, cfg['task_label'], network_type=nt)

            plt.tight_layout()
            out = viz_dir / f'boxplot_{nt_suffix}.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Combined pooled summary (all network types together)
# ---------------------------------------------------------------------------

def generate_combined_figure(df: pd.DataFrame, viz_dir: Path) -> None:
    """
    Single 1×3 figure pooled across all network types — quick overall summary.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        _first_available(df, RANKING_METRIC_CANDIDATES),
        _first_available(df, CLASSIFICATION_METRIC_CANDIDATES),
        _first_available(df, LINK_PREDICTION_METRIC_CANDIDATES),
    ]
    labels = ['Node Prioritization', 'Node Classification', 'Link Prediction']

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
        fig.suptitle('Method Comparison — All Networks (Pooled)',
                     fontsize=11, fontweight='bold', y=1.01)
        for ax, col, lbl in zip(axes, cols, labels):
            if col is None:
                ax.set_visible(False)
                continue
            _boxplot_panel(ax, df, col, lbl, network_type=None)
        plt.tight_layout()
        out = viz_dir / 'method_comparison_combined.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Ranking @ K line plots
# ---------------------------------------------------------------------------

def _extract_k_cols(df: pd.DataFrame, metric: str):
    """
    Return sorted list of (K, col) for columns matching
    'ranking_{metric}@{K}_centroid'.
    """
    pat = re.compile(rf'^ranking_{metric}@(\d+)_centroid$')
    hits = []
    for col in df.columns:
        m = pat.match(col)
        if m:
            hits.append((int(m.group(1)), col))
    return sorted(hits)


def generate_ranking_k_curves(df: pd.DataFrame, viz_dir: Path) -> None:
    """
    For each network type, produce a publication-quality line plot of
    Precision@K and Recall@K (centroid scoring) across all available K values.

    Each method is one line; quantum methods use cool tones, classical warm
    tones.  Error bands show ± 1 std across networks.

    File: ranking_k_curve_{network_type}.png
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    prec_k = _extract_k_cols(df, 'precision')
    rec_k  = _extract_k_cols(df, 'recall')

    if not prec_k and not rec_k:
        logger.warning('No ranking@K columns found — skipping K-curve plots.')
        return

    net_types = (
        sorted(df['network_type'].dropna().unique())
        if 'network_type' in df.columns
        else [None]
    )

    # Linestyles and markers for per-method visual differentiation
    _LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    _MARKERS    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']

    with plt.rc_context({**PUB_RC, 'figure.dpi': 150}):
        for nt in net_types:
            nt_label  = nt.replace('_', '-').title() if nt else 'All Networks'
            nt_suffix = nt if nt else 'all'

            subset = df.copy()
            if nt is not None and 'network_type' in subset.columns:
                subset = subset[subset['network_type'] == nt]

            n_panels = (1 if prec_k else 0) + (1 if rec_k else 0)
            if n_panels == 0:
                continue

            fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5.0),
                                     squeeze=False)
            fig.suptitle(
                f'Ranking Performance vs. K  —  {nt_label} Networks',
                fontsize=11, fontweight='bold', y=1.01,
            )

            panel_cfgs = []
            if prec_k:
                panel_cfgs.append(('Precision@K', prec_k))
            if rec_k:
                panel_cfgs.append(('Recall@K', rec_k))

            all_methods = _method_order(subset['method'].unique())
            legend_handles = []

            for ax, (panel_title, k_cols) in zip(axes[0], panel_cfgs):
                k_vals = [k for k, _ in k_cols]

                for mi, method in enumerate(all_methods):
                    mdf = subset[subset['method'] == method]
                    if mdf.empty:
                        continue

                    means, stds = [], []
                    for k, col in k_cols:
                        if col not in mdf.columns:
                            means.append(np.nan); stds.append(np.nan)
                        else:
                            means.append(mdf[col].mean())
                            stds.append(mdf[col].std())

                    means = np.array(means)
                    stds  = np.array(stds)
                    valid = ~np.isnan(means)
                    if not valid.any():
                        continue

                    color = METHOD_COLORS.get(method, _DEFAULT_COLOR)
                    ls    = _LINESTYLES[mi % len(_LINESTYLES)]
                    mk    = _MARKERS[mi % len(_MARKERS)]
                    lw    = 2.0 if method in ('quvine_fused-walk',
                                              'quvine_fused-filt',
                                              'quvine_fused-gcnmf') else 1.4

                    xv = np.array(k_vals)[valid]
                    yv = means[valid]
                    sv = stds[valid]

                    line, = ax.plot(xv, yv,
                                   color=color, linewidth=lw,
                                   linestyle=ls, marker=mk,
                                   markersize=5, zorder=3,
                                   label=METHOD_LABELS.get(method, method))
                    ax.fill_between(xv, yv - sv, yv + sv,
                                    color=color, alpha=0.12, zorder=2)

                    if panel_cfgs.index((panel_title, k_cols)) == 0:
                        legend_handles.append(line)

                ax.set_xlabel('K', fontsize=9)
                ax.set_ylabel(panel_title, fontsize=9)
                ax.set_title(panel_title, fontweight='bold')
                ax.set_xticks(k_vals)
                ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
                ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

                # Quantum / classical divider annotation in line plot:
                # draw a subtle background band behind classical methods' lines.
                # Instead, we add a text annotation in the legend.

            # Shared legend — one column per group (quantum / classical)
            quantum_handles  = [h for h in legend_handles
                                if h.get_label() in [METHOD_LABELS.get(m, m)
                                                     for m in all_methods
                                                     if m not in _CLASSICAL]]
            classic_handles  = [h for h in legend_handles
                                if h.get_label() not in [METHOD_LABELS.get(m, m)
                                                         for m in all_methods
                                                         if m not in _CLASSICAL]]

            divider_line = mlines.Line2D([], [], color='none', label='— Classical —')
            ordered_handles = quantum_handles + [divider_line] + classic_handles

            axes[0][-1].legend(
                handles=ordered_handles,
                loc='upper right',
                framealpha=0.85,
                fontsize=7.5,
                ncol=1,
                handlelength=2,
            )

            plt.tight_layout()
            out = viz_dir / f'ranking_k_curve_{nt_suffix}.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Timing box-plots
# ---------------------------------------------------------------------------

def generate_timing_boxplot(results_dir: Path, viz_dir: Path) -> None:
    """
    Scan every per-network subdirectory for *_timing_results.csv files,
    pool them, and produce two figures:

    1. timing_boxplot_all.png      — wall-clock embedding time per method,
                                     pooled across all network types.
    2. timing_boxplot_{type}.png   — same, faceted by network type.

    The y-axis uses a log scale so that fast (classical) and slow (quantum)
    methods are both readable.  A quantum/classical divider line is drawn as
    in the task box-plots.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Collect all timing CSVs
    timing_frames = []
    for net_dir in sorted(results_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        for csv_path in net_dir.glob("*_timing_results.csv"):
            try:
                timing_frames.append(pd.read_csv(csv_path))
            except Exception as e:
                logger.warning(f"  Could not read timing file {csv_path}: {e}")

    if not timing_frames:
        logger.warning("  No timing CSVs found — skipping timing box-plots.")
        return

    tdf = pd.concat(timing_frames, ignore_index=True)

    if 'embedding_time_s' not in tdf.columns or 'method' not in tdf.columns:
        logger.warning("  Timing data missing expected columns — skipping.")
        return

    # Add log-time column (for log-scale plots)
    tdf['log_time_s'] = np.log10(tdf['embedding_time_s'].clip(lower=1e-6))

    def _draw_timing_panel(ax, data: pd.DataFrame, title: str) -> None:
        if data.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(title, fontweight='bold')
            return

        order   = _method_order(data['method'].unique())
        rendered = [m for m in order if m in data['method'].values]
        colors   = _method_colors(rendered)

        sns.boxplot(
            data=data,
            x='method', y='embedding_time_s',
            order=rendered,
            palette=colors,
            linewidth=0.8,
            width=0.6,
            flierprops=dict(marker='o', markersize=2.5, alpha=0.4, linewidth=0),
            medianprops=dict(color='black', linewidth=1.5),
            ax=ax,
        )

        ax.set_yscale('log')
        ax.set_xticks(range(len(rendered)))
        ax.set_xticklabels(_method_labels(rendered), rotation=35, ha='right', fontsize=7.5)
        ax.set_xlabel('')
        ax.set_ylabel('Embedding time (s, log scale)', fontsize=8.5)
        ax.set_title(title, fontweight='bold', pad=4)

        # Quantum / classical divider
        div_x = _divider_x(rendered)
        if div_x is not None:
            ax.axvline(div_x, color='#555555', linewidth=1.2, linestyle='--', alpha=0.7, zorder=3)
            yhi = ax.get_ylim()[1]
            ax.text(div_x - 0.15, yhi, 'Quantum', fontsize=6.5, color='#555555',
                    ha='right', va='top', style='italic')
            ax.text(div_x + 0.15, yhi, 'Classical', fontsize=6.5, color='#555555',
                    ha='left', va='top', style='italic')

    net_types = (
        sorted(tdf['network_type'].dropna().unique())
        if 'network_type' in tdf.columns
        else []
    )

    with plt.rc_context(PUB_RC):
        # --- Figure 1: pooled across all network types ---
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle('Embedding Computation Time — All Networks (Pooled)',
                     fontsize=11, fontweight='bold', y=1.01)
        _draw_timing_panel(ax, tdf, 'Wall-clock embedding time')
        plt.tight_layout()
        out = viz_dir / 'timing_boxplot_all.png'
        fig.savefig(out, dpi=200, bbox_inches='tight')
        plt.close(fig)
        logger.info(f'  Saved: {out}')

        # --- Figure 2: one panel per network type ---
        if net_types:
            ncols = min(len(net_types), 3)
            nrows = (len(net_types) + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(12, 4.5 * nrows),
                                     squeeze=False)
            fig.suptitle('Embedding Computation Time — Per Network Type',
                         fontsize=11, fontweight='bold', y=1.01)
            for idx, nt in enumerate(net_types):
                r, c = divmod(idx, ncols)
                ax = axes[r][c]
                subset = tdf[tdf['network_type'] == nt] if 'network_type' in tdf.columns else tdf
                _draw_timing_panel(ax, subset, nt.replace('_', '-').title())
            # Hide unused panels
            total = nrows * ncols
            for idx in range(len(net_types), total):
                r, c = divmod(idx, ncols)
                axes[r][c].set_visible(False)
            plt.tight_layout()
            out = viz_dir / 'timing_boxplot_by_type.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  Saved: {out}')

        # --- Figure 3: timing vs. graph size scatter (one trace per method) ---
        if 'n_nodes' in tdf.columns:
            order = _method_order(tdf['method'].unique())
            colors = dict(zip(order, _method_colors(order)))
            _LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
            fig, ax = plt.subplots(figsize=(9, 5))
            for mi, method in enumerate(order):
                mdf = tdf[tdf['method'] == method].sort_values('n_nodes')
                if mdf.empty:
                    continue
                # Bin by n_nodes and show mean ± std
                mdf = mdf.copy()
                mdf['n_nodes_bin'] = pd.cut(mdf['n_nodes'], bins=8)
                grp = mdf.groupby('n_nodes_bin', observed=True)['embedding_time_s']
                means = grp.mean()
                stds  = grp.std().fillna(0)
                xs    = [iv.mid for iv in means.index]
                ls    = _LINESTYLES[mi % len(_LINESTYLES)]
                ax.plot(xs, means.values,
                        color=colors.get(method, '#888'),
                        linestyle=ls,
                        label=METHOD_LABELS.get(method, method),
                        linewidth=1.2)
                ax.fill_between(xs,
                                (means - stds).clip(lower=0).values,
                                (means + stds).values,
                                alpha=0.08,
                                color=colors.get(method, '#888'))
            ax.set_yscale('log')
            ax.set_xlabel('Number of nodes', fontsize=9)
            ax.set_ylabel('Embedding time (s, log scale)', fontsize=9)
            ax.set_title('Scaling of Embedding Time with Graph Size', fontweight='bold')
            ax.legend(fontsize=7, ncol=2, loc='upper left')
            plt.tight_layout()
            out = viz_dir / 'timing_scaling_curve.png'
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            logger.info(f'  Saved: {out}')


# ---------------------------------------------------------------------------
# Degree- and distance-matched binned AUC-PR plots
# ---------------------------------------------------------------------------

def generate_degree_distance_plots(results_dir: Path, viz_dir: Path) -> None:
    """
    Collect all *_degree_distance_matched.csv files written by
    run_single_network_analysis / the hard-negatives jobs, pool them, and
    produce:

      - degree_dist_matched_{case}_{bin_type}.png   per (case, degree|distance)
      - heatmap_delta_auc_{degree|distance}.png      summary heatmap across cases
      - degree_dist_combined_{strategy}_{bin_type}.png  pooled across cases per
                                                         network type

    Negative-strategy dimension is handled automatically: each CSV may contain
    rows for both 'hard_2hop' and 'same_community'.
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect all CSVs ──────────────────────────────────────────────────────
    frames = []
    for net_dir in sorted(results_dir.iterdir()):
        if not net_dir.is_dir():
            continue
        for csv_path in net_dir.glob("*_degree_distance_matched.csv"):
            try:
                frames.append(pd.read_csv(csv_path))
            except Exception as e:
                logger.warning(f"  Could not read {csv_path}: {e}")

    if not frames:
        logger.warning("  No degree_distance_matched CSVs found — skipping.")
        return

    df = pd.concat(frames, ignore_index=True)
    required = {'method', 'bin_type', 'bin_label', 'auc_pr'}
    if not required.issubset(df.columns):
        logger.warning(f"  Degree/distance CSV missing columns {required - set(df.columns)} — skipping.")
        return

    logger.info(f"  Loaded {len(df)} degree/distance records from {len(frames)} files.")

    _CLASSICAL_SET = frozenset({'node2vec', 'netmf', 'baseline_filter', 'baseline_gcnmf'})

    # ── Colour/style helpers ──────────────────────────────────────────────────
    def _colour(m):
        return METHOD_COLORS.get(m, _DEFAULT_COLOR)

    def _label(m):
        return METHOD_LABELS.get(m, m)

    _LS = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    # ── CI helper ────────────────────────────────────────────────────────────
    def _ci(vals):
        vals = vals.dropna().values
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        m = vals.mean()
        if len(vals) < 3:
            return m, m, m
        from scipy import stats as _st
        se = _st.sem(vals)
        t  = _st.t.ppf(0.975, df=max(len(vals) - 1, 1))
        return m, m - t * se, m + t * se

    # ── Line-plot helper ──────────────────────────────────────────────────────
    def _lineplot(ax, sub: pd.DataFrame, bins: list, title: str, xlabel: str,
                  annotate_hub: bool = False):
        if sub.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(title, fontweight='bold')
            return
        order = _method_order(sub['method'].unique())
        for mi, method in enumerate(order):
            mdf = sub[sub['method'] == method]
            means, lows, highs = [], [], []
            for bl in bins:
                m, lo, hi = _ci(mdf[mdf['bin_label'] == bl]['auc_pr'])
                means.append(m); lows.append(lo); highs.append(hi)
            means, lows, highs = map(np.array, (means, lows, highs))
            x = np.arange(len(bins))
            valid = ~np.isnan(means)
            if not valid.any():
                continue
            c  = _colour(method)
            ls = _LS[mi % len(_LS)]
            ax.plot(x[valid], means[valid], color=c, linestyle=ls, linewidth=1.5,
                    marker='o', markersize=4, label=_label(method))
            ax.fill_between(x[valid], lows[valid], highs[valid], alpha=0.08, color=c)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel('AUC-PR (hard negatives)', fontsize=8.5)
        ax.set_title(title, fontweight='bold', pad=4)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, loc='best', ncol=2)
        if annotate_hub:
            yl = ax.get_ylim()
            ax.annotate('← low-degree',
                        xy=(0, yl[0] + 0.02*(yl[1]-yl[0])),
                        fontsize=6, color='#555', style='italic', ha='left')
            ax.annotate('hub-dominated →',
                        xy=(len(bins)-1, yl[0] + 0.02*(yl[1]-yl[0])),
                        fontsize=6, color='#555', style='italic', ha='right')

    # ── Gap-bar helper ────────────────────────────────────────────────────────
    def _gapbar(ax, sub: pd.DataFrame, bins: list, title: str, xlabel: str):
        if sub.empty:
            ax.set_title(title, fontweight='bold')
            return
        q_meths = [m for m in sub['method'].unique() if m.startswith('quvine_')]
        c_meths = [m for m in sub['method'].unique() if m in _CLASSICAL_SET]
        if not q_meths or not c_meths:
            ax.set_title(title + ' (need quantum+classical)', fontweight='bold')
            return
        gaps, colors = [], []
        for bl in bins:
            bdf = sub[sub['bin_label'] == bl]
            q = bdf[bdf['method'].isin(q_meths)]['auc_pr'].mean()
            c = bdf[bdf['method'].isin(c_meths)]['auc_pr'].mean()
            gaps.append(q - c)
            colors.append('#1e88e5' if (q - c) >= 0 else '#e65100')
        ax.bar(range(len(bins)), gaps, color=colors, alpha=0.75,
               edgecolor='white', linewidth=0.5)
        ax.axhline(0, color='black', linewidth=0.8)
        ax.set_xticks(range(len(bins)))
        ax.set_xticklabels(bins, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel('ΔAUC-PR  (Quantum − Classical)', fontsize=8.5)
        ax.set_title(title, fontweight='bold', pad=4)
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor='#1e88e5', alpha=0.75, label='Quantum wins'),
            Patch(facecolor='#e65100', alpha=0.75, label='Classical wins'),
        ], fontsize=7)

    # ── Bin label lists ───────────────────────────────────────────────────────
    _N_DEG_BINS = 5
    _DIST_MAX   = 5
    deg_bins  = [f"Q{i+1}" for i in range(_N_DEG_BINS)]
    dist_bins = [str(d) if d < _DIST_MAX else f"{_DIST_MAX}+"
                 for d in range(2, _DIST_MAX + 1)]

    bin_cfgs = [
        ('degree',   deg_bins,  'Max-degree bin of pair (Q1=low → Q5=hub)', True),
        ('distance', dist_bins, 'Shortest-path distance between pair',       False),
    ]

    strategies = sorted(df['negative_strategy'].dropna().unique()) \
        if 'negative_strategy' in df.columns else ['hard_2hop']

    cases = sorted(df['case'].dropna().unique()) \
        if ('case' in df.columns and df['case'].notna().any()) else [None]

    # ── Figure A: per-case, per-bin-type, per-strategy ──────────────���────────
    with plt.rc_context(PUB_RC):
        for strategy in strategies:
            strat_df = df[df['negative_strategy'] == strategy] \
                if 'negative_strategy' in df.columns else df

            for bin_type, bins, xlabel, annotate_hub in bin_cfgs:
                sub = strat_df[strat_df['bin_type'] == bin_type]
                if sub.empty:
                    continue

                for case in cases:
                    if case is not None and 'case' in sub.columns:
                        case_df = sub[sub['case'] == case]
                        nt  = case_df['network_type'].iloc[0] \
                            if ('network_type' in case_df.columns and not case_df.empty) else ''
                        ew  = case_df['expected_winner'].iloc[0] \
                            if ('expected_winner' in case_df.columns and not case_df.empty) else ''
                        suptitle = (f"Case {case}  |  {nt.replace('_','-')}  |  "
                                    f"neg={strategy}  |  expected: {ew}")
                        fname = f"degree_dist_matched_{case}_{strategy}_{bin_type}.png"
                    else:
                        case_df  = sub
                        suptitle = f"All cases  |  neg={strategy}  |  {bin_type}-binned"
                        fname    = f"degree_dist_matched_all_{strategy}_{bin_type}.png"

                    if case_df.empty:
                        continue

                    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                    fig.suptitle(suptitle, fontsize=10, fontweight='bold', y=1.02)
                    _lineplot(axes[0], case_df, bins,
                              f'AUC-PR by {bin_type} bin', xlabel, annotate_hub)
                    _gapbar(axes[1], case_df, bins,
                            f'Quantum − Classical gap by {bin_type} bin', xlabel)
                    plt.tight_layout()
                    fig.savefig(viz_dir / fname, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f'  Saved: {viz_dir / fname}')

        # ── Figure B: pooled per-network-type, per-strategy ──────────────────
        if 'network_type' in df.columns:
            for strategy in strategies:
                strat_df = df[df['negative_strategy'] == strategy] \
                    if 'negative_strategy' in df.columns else df

                net_types = sorted(strat_df['network_type'].dropna().unique())
                for bin_type, bins, xlabel, annotate_hub in bin_cfgs:
                    sub = strat_df[strat_df['bin_type'] == bin_type]
                    if sub.empty:
                        continue
                    ncols = min(3, len(net_types))
                    nrows = (len(net_types) + ncols - 1) // ncols
                    fig, axes = plt.subplots(nrows, ncols,
                                             figsize=(6.5 * ncols, 5 * nrows),
                                             squeeze=False)
                    fig.suptitle(
                        f'AUC-PR by {bin_type} bin  |  neg={strategy}  '
                        f'(per network type)',
                        fontsize=11, fontweight='bold', y=1.01,
                    )
                    for idx, nt in enumerate(net_types):
                        r, c = divmod(idx, ncols)
                        _lineplot(axes[r][c],
                                  sub[sub['network_type'] == nt],
                                  bins,
                                  nt.replace('_', '-').title(),
                                  xlabel, annotate_hub)
                    for idx in range(len(net_types), nrows * ncols):
                        r, c = divmod(idx, ncols)
                        axes[r][c].set_visible(False)
                    plt.tight_layout()
                    fname = f"degree_dist_by_type_{strategy}_{bin_type}.png"
                    fig.savefig(viz_dir / fname, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    logger.info(f'  Saved: {viz_dir / fname}')

        # ── Figure C: ΔAUC-PR heatmap — all cases × bins ─────────────────────
        for strategy in strategies:
            strat_df = df[df['negative_strategy'] == strategy] \
                if 'negative_strategy' in df.columns else df

            for bin_type, bins, xlabel, _ in bin_cfgs:
                sub = strat_df[strat_df['bin_type'] == bin_type]
                if sub.empty or 'case' not in sub.columns:
                    continue
                valid_cases = sorted(sub['case'].dropna().unique())
                if not valid_cases:
                    continue

                q_meths = [m for m in sub['method'].unique() if m.startswith('quvine_')]
                c_meths = [m for m in sub['method'].unique() if m in _CLASSICAL_SET]
                if not q_meths or not c_meths:
                    continue

                pivot = {}
                for case in valid_cases:
                    cdf = sub[sub['case'] == case]
                    gaps = []
                    for bl in bins:
                        bdf = cdf[cdf['bin_label'] == bl]
                        q   = bdf[bdf['method'].isin(q_meths)]['auc_pr'].mean()
                        c   = bdf[bdf['method'].isin(c_meths)]['auc_pr'].mean()
                        gaps.append(q - c)
                    pivot[case] = gaps

                heat_df = pd.DataFrame(pivot, index=bins).T
                fig, ax = plt.subplots(
                    figsize=(max(7, len(bins) * 1.3), max(5, len(valid_cases) * 0.7))
                )
                import seaborn as _sns
                _sns.heatmap(
                    heat_df, ax=ax,
                    cmap='RdBu_r', center=0, annot=True, fmt='.3f',
                    linewidths=0.3, linecolor='#cccccc',
                    cbar_kws={'label': 'ΔAUC-PR (Quantum − Classical)'},
                )
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel('Experiment case', fontsize=9)
                ax.set_title(
                    f'Quantum vs Classical  |  {bin_type}-binned  |  neg={strategy}\n'
                    '(blue = quantum wins, red = classical wins)',
                    fontweight='bold',
                )
                plt.tight_layout()
                fname = f"heatmap_delta_auc_{strategy}_{bin_type}.png"
                fig.savefig(viz_dir / fname, dpi=200, bbox_inches='tight')
                plt.close(fig)
                logger.info(f'  Saved: {viz_dir / fname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Collect HPC results and generate publication-quality visualizations'
    )
    parser.add_argument('--results-dir', required=True,
                        help='Directory containing per-network result subdirectories')
    parser.add_argument('--output-file', default='comprehensive_results.csv',
                        help='Aggregated CSV filename (saved inside --results-dir)')
    parser.add_argument('--viz-dir', default=None,
                        help='Output directory for plots (default: <results-dir>/../visualizations)')
    parser.add_argument('--n-networks', type=int, default=20,
                        help='Networks per type (used in plot titles)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress messages')
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    viz_dir = Path(args.viz_dir) if args.viz_dir else results_path.parent / 'visualizations'

    # ------------------------------------------------------------------
    # Step 1: Collect & aggregate
    # ------------------------------------------------------------------
    logger.info('=' * 70)
    logger.info('STEP 1 — Collecting and aggregating results')
    logger.info('=' * 70)

    df = collect_and_aggregate_results(
        results_dir=args.results_dir,
        output_file=args.output_file,
        verbose=not args.quiet,
    )

    if df.empty:
        logger.error('Aggregated DataFrame is empty — nothing to visualize.')
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2: Summary statistics
    # ------------------------------------------------------------------
    if not args.quiet:
        print('\n' + '=' * 70)
        print('AGGREGATION SUMMARY')
        print('=' * 70)
        print(f'Rows   : {len(df)}')
        print(f'Columns: {len(df.columns)}')
        if 'network_id'   in df.columns: print(f'Networks : {df["network_id"].nunique()}')
        if 'method'       in df.columns: print(f'Methods  : {df["method"].nunique()}')
        if 'network_type' in df.columns:
            print('\nNetwork types:')
            for nt, cnt in df['network_type'].value_counts().items():
                print(f'  {nt}: {cnt} rows')

        for label, candidates in [
            ('Ranking',         RANKING_METRIC_CANDIDATES),
            ('Classification',  CLASSIFICATION_METRIC_CANDIDATES),
            ('Link Prediction', LINK_PREDICTION_METRIC_CANDIDATES),
        ]:
            col = _first_available(df, candidates)
            if col and 'method' in df.columns:
                print(f'\n{label} — {col} (median per method):')
                medians = df.groupby('method')[col].median().sort_values(ascending=False)
                for m, v in medians.items():
                    print(f'  {m:40s}: {v:.4f}')

        print('\n' + '=' * 70)
        print(f'CSV: {results_path / args.output_file}')
        print('=' * 70)

    # ------------------------------------------------------------------
    # Step 3: Visualizations
    # ------------------------------------------------------------------
    logger.info('=' * 70)
    logger.info('STEP 2 — Generating visualizations')
    logger.info(f'Output directory: {viz_dir}')
    logger.info('=' * 70)

    # One figure per network type (3 task panels each)
    generate_task_boxplots(df, viz_dir=viz_dir, n_networks=args.n_networks)

    # Pooled combined summary
    generate_combined_figure(df, viz_dir=viz_dir)

    # Ranking @ K line curves (per network type)
    generate_ranking_k_curves(df, viz_dir=viz_dir)

    # Computation time box-plots (timing CSVs written by run_single_network_analysis)
    generate_timing_boxplot(results_path, viz_dir=viz_dir)

    # Degree- and distance-matched binned AUC-PR plots
    generate_degree_distance_plots(results_path, viz_dir=viz_dir)

    logger.info('=' * 70)
    logger.info('DONE')
    logger.info('=' * 70)

    return df


if __name__ == '__main__':
    main()
