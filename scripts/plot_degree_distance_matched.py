#!/usr/bin/env python3
"""
Degree-Matched and Distance-Matched Link-Prediction Analysis

For each (case, network_type) in the hard-negatives experiment this script
answers the question:

  "Do classical methods fail specifically because they are confused by
   hub nodes or by topologically distant pairs?"

Two sets of figures are produced:

FIGURE SET 1 — Degree-matched AUC-PR
─────────────────────────────────────
Hard negative pairs are binned by max_degree(u, v).  For each bin we report
per-method AUC-PR.  If classical methods are dominated by hubs, their AUC-PR
should drop sharply in high-degree bins while quantum methods stay flat.

FIGURE SET 2 — Distance-matched AUC-PR
───────────────────────────────────────
Hard negative pairs are binned by their shortest-path distance in the graph
(distance 2 = direct 2-hop, distance ≥ 3 = deeper non-edges).  Classical
methods that rely on biased random walks tend to score distant pairs as
similar when hubs mediate the path.

Algorithm for each network replicate
─────────────────────────────────────
1. Load the .graphml file.
2. Load the per-method embeddings (saved as .npy by run_single_network_analysis).
3. Sample hard negatives using the same strategy recorded in metadata.
4. For each negative pair (u, v):
   a. max_degree_bin  = percentile bucket of max(deg(u), deg(v))
   b. distance_bin    = min(shortest_path(u, v), 5)  [capped at 5+]
5. Compute AUC-PR separately within each bin for each method.
6. Plot mean ± 95 % CI across replicates.

Usage:
    python scripts/plot_degree_distance_matched.py \
        --results-dir outputs/hard_negatives/results \
        --viz-dir     outputs/hard_negatives/visualizations

Options:
    --results-dir DIR    Directory of per-network result subdirectories
    --viz-dir     DIR    Where to write PNG files  (default: <results-dir>/../visualizations)
    --n-neg       INT    Hard negatives to sample per replicate (default: 500)
    --seed        INT    Random seed (default: 42)
    --quiet              Suppress progress messages
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from scipy import stats as scipy_stats
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quvine.evaluation.link_prediction import (
    split_edges,
    sample_negative_edges,
)

# ── Style constants (mirror collect_hpc_results.py) ───────────────────────────

CANONICAL_ORDER = [
    'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'quvine_fused-walk',
    'node2vec', 'netmf',
]
_CLASSICAL = frozenset({'node2vec', 'netmf', 'baseline_filter', 'baseline_gcnmf'})

METHOD_COLORS = {
    'quvine_rwr':        '#1565c0',
    'quvine_ctqw':       '#1e88e5',
    'quvine_dtqw':       '#90caf9',
    'quvine_fused-walk': '#0d47a1',
    'node2vec':          '#e65100',
    'netmf':             '#b71c1c',
}
_DEFAULT_COLOR = '#888888'

METHOD_LABELS = {
    'quvine_rwr':        'RWR',
    'quvine_ctqw':       'CTQW',
    'quvine_dtqw':       'DTQW',
    'quvine_fused-walk': 'Fused-Walk',
    'node2vec':          'Node2Vec',
    'netmf':             'NetMF',
}

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

DEGREE_N_BINS = 5   # quintile bins by max-degree of the pair
DIST_MAX_BIN  = 5   # distances ≥ DIST_MAX_BIN are merged into "5+"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _method_order(methods):
    present = set(methods)
    known   = [m for m in CANONICAL_ORDER if m in present]
    extra_q = sorted(m for m in present if m not in set(CANONICAL_ORDER) and m.startswith('quvine_'))
    extra_c = sorted(m for m in present if m not in set(CANONICAL_ORDER) and not m.startswith('quvine_'))
    return known + extra_q + extra_c


def _ci95(values):
    """Return (mean, lower, upper) of a 1-D array using 95 % bootstrap CI."""
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    mean = np.nanmean(values)
    if len(values) < 3:
        return mean, mean, mean
    se   = scipy_stats.sem(values, nan_policy='omit')
    t    = scipy_stats.t.ppf(0.975, df=max(len(values) - 1, 1))
    return mean, mean - t * se, mean + t * se


def _cosine_sim(u_emb, v_emb):
    nu = np.linalg.norm(u_emb)
    nv = np.linalg.norm(v_emb)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.dot(u_emb, v_emb) / (nu * nv))


def _auc_pr_for_pairs(emb, node_list, pos_pairs, neg_pairs):
    """
    Compute AUC-PR using cosine similarity as score.
    Returns np.nan when either set is empty.
    """
    if len(pos_pairs) == 0 or len(neg_pairs) == 0:
        return np.nan

    node_idx = {n: i for i, n in enumerate(node_list)}
    scores, labels = [], []

    for u, v in pos_pairs:
        ui, vi = node_idx.get(u), node_idx.get(v)
        if ui is None or vi is None:
            continue
        scores.append(_cosine_sim(emb[ui], emb[vi]))
        labels.append(1)

    for u, v in neg_pairs:
        ui, vi = node_idx.get(u), node_idx.get(v)
        if ui is None or vi is None:
            continue
        scores.append(_cosine_sim(emb[ui], emb[vi]))
        labels.append(0)

    if len(set(labels)) < 2:
        return np.nan
    return average_precision_score(labels, scores)


# ─────────────────────────────────────────────────────────────────────────────
# Per-network analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_network(net_dir: Path, n_neg: int, seed: int):
    """
    Analyse one replicate directory.

    Returns a list of dicts, one per (method, bin_type, bin_label), with
    keys: method, bin_type ('degree'|'distance'), bin_label, auc_pr,
          network_id, network_type, case, expected_winner, negative_strategy.
    """
    network_id = net_dir.name

    # Load metadata
    meta_file = net_dir / f"{network_id}_metadata.json"
    if not meta_file.exists():
        return []
    with open(meta_file) as f:
        metadata = json.load(f)

    neg_strategy    = metadata.get('negative_strategy', 'hard_2hop')
    network_type    = metadata.get('type', 'unknown')
    case            = metadata.get('case', '')
    expected_winner = metadata.get('expected_winner', '')

    # Load graph
    graphml_file = net_dir / f"{network_id}.graphml"
    if not graphml_file.exists():
        return []
    try:
        G = nx.read_graphml(str(graphml_file))
        G = nx.convert_node_labels_to_integers(G)
    except Exception as e:
        print(f"  [WARN] Could not load graph {graphml_file}: {e}")
        return []

    if G.number_of_nodes() < 10:
        return []

    node_list = list(G.nodes())
    degrees   = dict(G.degree())

    # Edge split (identical to training split used during embedding)
    try:
        _, _, pos_edges, _ = split_edges(G, test_ratio=0.2, val_ratio=0.0, seed=seed)
    except Exception as e:
        print(f"  [WARN] Edge split failed for {network_id}: {e}")
        return []

    if len(pos_edges) == 0:
        return []

    # Sample hard negatives using stored strategy
    try:
        neg_edges = sample_negative_edges(
            G, n_samples=n_neg, strategy=neg_strategy, seed=seed
        )
    except Exception as e:
        print(f"  [WARN] Negative sampling failed for {network_id}: {e}")
        return []

    if len(neg_edges) == 0:
        return []

    # ── Precompute per-pair features ──────────────────────────────────────────

    # Max degree of each negative pair
    neg_max_deg = np.array([
        max(degrees.get(u, 0), degrees.get(v, 0)) for u, v in neg_edges
    ])

    # Shortest path distance for each negative pair (expensive — capped at DIST_MAX_BIN)
    neg_dist = []
    for u, v in neg_edges:
        try:
            d = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            d = DIST_MAX_BIN + 1
        neg_dist.append(min(d, DIST_MAX_BIN))
    neg_dist = np.array(neg_dist)

    # Degree bins (quintiles of max_degree across all negative pairs)
    deg_percentiles = np.percentile(neg_max_deg, np.linspace(0, 100, DEGREE_N_BINS + 1))
    deg_bins_labels = [f"Q{i+1}" for i in range(DEGREE_N_BINS)]

    def _deg_bin(d):
        for i in range(DEGREE_N_BINS):
            lo = deg_percentiles[i]
            hi = deg_percentiles[i + 1]
            if d <= hi:
                return deg_bins_labels[i]
        return deg_bins_labels[-1]

    neg_deg_bin = [_deg_bin(d) for d in neg_max_deg]

    # Distance bins: 2, 3, 4, 5+
    dist_bin_labels = [str(d) if d < DIST_MAX_BIN else f"{DIST_MAX_BIN}+" for d in range(2, DIST_MAX_BIN + 1)]

    def _dist_bin(d):
        return f"{DIST_MAX_BIN}+" if d >= DIST_MAX_BIN else str(d)

    neg_dist_bin = [_dist_bin(d) for d in neg_dist]

    # ── Load embeddings and compute binned AUC-PR ─────────────────────────────
    records = []

    emb_files = sorted(net_dir.glob(f"{network_id}_*_embedding.npy"))
    for emb_file in emb_files:
        # Extract method name: {network_id}_{method}_embedding.npy
        stem   = emb_file.stem                         # e.g. "QW1_rep00_quvine_ctqw_embedding"
        suffix = "_embedding"
        if not stem.endswith(suffix):
            continue
        mid    = stem[len(network_id) + 1 : -len(suffix)]   # "quvine_ctqw"
        method = mid

        try:
            emb = np.load(str(emb_file))
        except Exception:
            continue

        if emb.shape[0] != len(node_list):
            continue

        # ── Degree-matched analysis ───────────────────────────────────────────
        for bl in deg_bins_labels:
            idx = [i for i, b in enumerate(neg_deg_bin) if b == bl]
            if len(idx) == 0:
                continue
            bin_negs = [neg_edges[i] for i in idx]
            auc = _auc_pr_for_pairs(emb, node_list, pos_edges, bin_negs)
            records.append(dict(
                network_id=network_id, network_type=network_type,
                case=case, expected_winner=expected_winner,
                negative_strategy=neg_strategy,
                method=method, bin_type='degree', bin_label=bl,
                bin_n_negs=len(bin_negs),
                auc_pr=auc,
            ))

        # ── Distance-matched analysis ─────────────────────────────────────────
        for bl in dist_bin_labels:
            idx = [i for i, b in enumerate(neg_dist_bin) if b == bl]
            if len(idx) == 0:
                continue
            bin_negs = [neg_edges[i] for i in idx]
            auc = _auc_pr_for_pairs(emb, node_list, pos_edges, bin_negs)
            records.append(dict(
                network_id=network_id, network_type=network_type,
                case=case, expected_winner=expected_winner,
                negative_strategy=neg_strategy,
                method=method, bin_type='distance', bin_label=bl,
                bin_n_negs=len(bin_negs),
                auc_pr=auc,
            ))

    return records


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_binned_auc(ax, sub_df: pd.DataFrame, bin_labels: list,
                     title: str, xlabel: str, annotate_hub: bool = False):
    """
    Line plot of mean AUC-PR per bin, one line per method, with 95 % CI bands.
    """
    if sub_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title(title, fontweight='bold')
        return

    order = _method_order(sub_df['method'].unique())
    _LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]

    for mi, method in enumerate(order):
        mdf = sub_df[sub_df['method'] == method]
        means, lows, highs = [], [], []
        for bl in bin_labels:
            vals = mdf[mdf['bin_label'] == bl]['auc_pr'].dropna().values
            m, lo, hi = _ci95(vals)
            means.append(m); lows.append(lo); highs.append(hi)

        means  = np.array(means)
        lows   = np.array(lows)
        highs  = np.array(highs)
        x      = np.arange(len(bin_labels))
        valid  = ~np.isnan(means)
        if not valid.any():
            continue

        color = METHOD_COLORS.get(method, _DEFAULT_COLOR)
        ls    = _LINESTYLES[mi % len(_LINESTYLES)]
        label = METHOD_LABELS.get(method, method)

        ax.plot(x[valid], means[valid], color=color, linestyle=ls,
                linewidth=1.5, marker='o', markersize=4, label=label)
        ax.fill_between(x[valid], lows[valid], highs[valid],
                        alpha=0.10, color=color)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('AUC-PR (hard negatives)', fontsize=8.5)
    ax.set_title(title, fontweight='bold', pad=4)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7, loc='best', ncol=2)

    if annotate_hub:
        ylo, yhi = ax.get_ylim()
        ax.annotate('← easier\n(low-degree pairs)',
                    xy=(0, ylo + 0.02 * (yhi - ylo)),
                    fontsize=6.5, color='#555', style='italic', ha='left')
        ax.annotate('hub-dominated →\n(high-degree pairs)',
                    xy=(len(bin_labels) - 1, ylo + 0.02 * (yhi - ylo)),
                    fontsize=6.5, color='#555', style='italic', ha='right')


def _plot_quantum_gap(ax, sub_df: pd.DataFrame, bin_labels: list,
                      title: str, xlabel: str):
    """
    Bar plot of ΔAUC-PR = mean_quantum - mean_classical per bin.
    Positive = quantum wins; negative = classical wins.
    """
    if sub_df.empty:
        ax.text(0.5, 0.5, "No data", ha='center', va='center',
                transform=ax.transAxes, fontsize=8, color='gray')
        ax.set_title(title, fontweight='bold')
        return

    quantum_methods  = [m for m in sub_df['method'].unique() if m.startswith('quvine_')]
    classical_methods = [m for m in sub_df['method'].unique() if m in _CLASSICAL]

    if not quantum_methods or not classical_methods:
        ax.set_title(title + ' (insufficient methods)', fontweight='bold')
        return

    gaps = []
    for bl in bin_labels:
        bdf = sub_df[sub_df['bin_label'] == bl]
        q_mean = bdf[bdf['method'].isin(quantum_methods)]['auc_pr'].mean()
        c_mean = bdf[bdf['method'].isin(classical_methods)]['auc_pr'].mean()
        gaps.append(q_mean - c_mean)

    gaps = np.array(gaps)
    x    = np.arange(len(bin_labels))
    colors = ['#1e88e5' if g >= 0 else '#e65100' for g in gaps]

    ax.bar(x, gaps, color=colors, alpha=0.75, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8.5)
    ax.set_ylabel('ΔAUC-PR  (Quantum − Classical)', fontsize=8.5)
    ax.set_title(title, fontweight='bold', pad=4)

    # Legend proxy patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#1e88e5', alpha=0.75, label='Quantum wins'),
        Patch(facecolor='#e65100', alpha=0.75, label='Classical wins'),
    ], fontsize=7, loc='best')


# ─────────────────────────────────────────────────────────────────────────────
# Main figure generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_figures(df: pd.DataFrame, viz_dir: Path):
    """
    Produce one figure per (case, bin_type) combination:
      - Top row  : AUC-PR per method per bin
      - Bottom row: ΔAUC-PR (quantum − classical) per bin
    """
    viz_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        print("  [WARN] Empty DataFrame — no figures generated.")
        return

    cases = sorted(df['case'].dropna().unique()) if 'case' in df.columns else ['']

    deg_bins  = [f"Q{i+1}" for i in range(DEGREE_N_BINS)]
    dist_bins = [str(d) if d < DIST_MAX_BIN else f"{DIST_MAX_BIN}+"
                 for d in range(2, DIST_MAX_BIN + 1)]

    for case in cases:
        case_df = df[df['case'] == case] if case else df
        if case_df.empty:
            continue

        net_type    = case_df['network_type'].iloc[0] if 'network_type' in case_df else 'unknown'
        exp_winner  = case_df['expected_winner'].iloc[0] if 'expected_winner' in case_df else ''
        neg_strat   = case_df['negative_strategy'].iloc[0] if 'negative_strategy' in case_df else ''

        for bin_type, bins, xlabel, annotate_hub in [
            ('degree',   deg_bins,  'Max-degree bin of pair (Q1=low, Q5=hub)', True),
            ('distance', dist_bins, 'Shortest-path distance between pair',     False),
        ]:
            sub = case_df[case_df['bin_type'] == bin_type]
            if sub.empty:
                continue

            with plt.rc_context(PUB_RC):
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                title_base = (
                    f"Case {case}  |  {net_type.replace('_', '-')}  |  "
                    f"neg={neg_strat}  |  expected winner: {exp_winner}"
                )
                fig.suptitle(title_base, fontsize=10, fontweight='bold', y=1.02)

                _plot_binned_auc(
                    axes[0], sub, bins,
                    title=f'AUC-PR by {bin_type} bin',
                    xlabel=xlabel,
                    annotate_hub=annotate_hub,
                )
                _plot_quantum_gap(
                    axes[1], sub, bins,
                    title=f'Quantum − Classical gap by {bin_type} bin',
                    xlabel=xlabel,
                )

                plt.tight_layout()
                fname = f"degree_dist_matched_{case}_{bin_type}.png"
                out   = viz_dir / fname
                fig.savefig(out, dpi=200, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {out}")

    # ── Summary heatmap: ΔAUC-PR across all cases × bins ─────────────────────
    for bin_type, bins, xlabel in [
        ('degree',   deg_bins,  'Degree bin'),
        ('distance', dist_bins, 'Distance bin'),
    ]:
        sub = df[df['bin_type'] == bin_type]
        if sub.empty:
            continue

        quantum_methods  = [m for m in sub['method'].unique() if m.startswith('quvine_')]
        classical_methods = [m for m in sub['method'].unique() if m in _CLASSICAL]
        if not quantum_methods or not classical_methods:
            continue

        pivot_data = {}
        for case in sorted(sub['case'].dropna().unique()):
            case_sub = sub[sub['case'] == case]
            gaps = []
            for bl in bins:
                bdf = case_sub[case_sub['bin_label'] == bl]
                q_m = bdf[bdf['method'].isin(quantum_methods)]['auc_pr'].mean()
                c_m = bdf[bdf['method'].isin(classical_methods)]['auc_pr'].mean()
                gaps.append(q_m - c_m)
            pivot_data[case] = gaps

        heat_df = pd.DataFrame(pivot_data, index=bins).T

        with plt.rc_context(PUB_RC):
            fig, ax = plt.subplots(figsize=(max(7, len(bins) * 1.2), max(5, len(cases) * 0.6)))
            sns.heatmap(
                heat_df, ax=ax,
                cmap='RdBu_r', center=0, annot=True, fmt='.3f',
                linewidths=0.3, linecolor='#cccccc',
                cbar_kws={'label': 'ΔAUC-PR (Quantum − Classical)'},
            )
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel('Experiment case', fontsize=9)
            ax.set_title(
                f'Quantum vs Classical Advantage by {bin_type.title()} Bin\n'
                f'(blue = quantum wins, red = classical wins)',
                fontweight='bold',
            )
            plt.tight_layout()
            out = viz_dir / f"heatmap_delta_auc_{bin_type}.png"
            fig.savefig(out, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Degree- and distance-matched hard-negative analysis'
    )
    parser.add_argument('--results-dir', required=True,
                        help='Per-network result directories from the hard-negatives experiment')
    parser.add_argument('--viz-dir', default=None,
                        help='Output directory for plots')
    parser.add_argument('--n-neg', type=int, default=500,
                        help='Hard negatives to sample per replicate (default: 500)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    results_path = Path(args.results_dir)
    viz_dir = Path(args.viz_dir) if args.viz_dir else results_path.parent / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 70)
        print("Degree- and Distance-Matched Hard-Negative Analysis")
        print(f"Results dir: {results_path}")
        print(f"Viz dir    : {viz_dir}")
        print("=" * 70)

    # ── Collect per-network records ───────────────────────────────────────────
    all_records = []
    net_dirs = sorted(d for d in results_path.iterdir() if d.is_dir())

    for i, net_dir in enumerate(net_dirs):
        if not args.quiet:
            print(f"  [{i+1}/{len(net_dirs)}] Analysing {net_dir.name} …")
        try:
            records = analyse_network(net_dir, n_neg=args.n_neg, seed=args.seed)
            all_records.extend(records)
            if not args.quiet and records:
                methods = set(r['method'] for r in records)
                print(f"    → {len(records)} records from {len(methods)} methods")
        except Exception as exc:
            print(f"  [WARN] Failed for {net_dir.name}: {exc}")
            import traceback; traceback.print_exc()

    if not all_records:
        print("[ERROR] No records collected — check that .graphml and *_embedding.npy"
              " files exist in the result directories.")
        sys.exit(1)

    df = pd.DataFrame(all_records)

    # Save raw table
    csv_path = viz_dir / 'degree_distance_matched_results.csv'
    df.to_csv(csv_path, index=False)
    if not args.quiet:
        print(f"\nRaw table saved: {csv_path}")
        print(f"Total records : {len(df)}")
        print(f"Cases         : {sorted(df['case'].dropna().unique()) if 'case' in df.columns else 'N/A'}")
        print(f"Methods       : {sorted(df['method'].unique())}")

    # ── Generate figures ──────────────────────────────────────────────────────
    if not args.quiet:
        print("\nGenerating figures …")
    generate_figures(df, viz_dir)

    if not args.quiet:
        print("=" * 70)
        print("Done.")
        print("=" * 70)


if __name__ == '__main__':
    main()
