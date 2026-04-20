#!/usr/bin/env python3
"""
Run ranking and link-prediction evaluation for all baseline methods on a
single network, using pre-computed .npy embeddings.

This script is called by submit_ranking_lp_jobs.sh.  It intentionally avoids
re-generating embeddings (which already exist) and skips classification (which
was already captured in node_embedding.csv).

Results are written (or merged) into:
  <output-dir>/<network-id>_ranking_results.csv        (all methods, incl. appnp)
  <output-dir>/<network-id>_link_prediction.csv        (tidy, all methods)
  <output-dir>/<network-id>_link_prediction_results.csv (summary, all methods)

Existing appnp rows are preserved if --resume is passed.

Usage:
    python scripts/run_ranking_lp_single_network.py \\
        --graphml-path /path/to/network.graphml \\
        --output-dir   /path/to/results/network_id \\
        --network-id   network_id \\
        --network-type real_ppi \\
        --methods      quvine_fused,quvine_rwr,quvine_heat,quvine_poly,\\
                       quvine_hgcnmf,quvine_pgcnmf,netmf,node2vec,\\
                       baseline_filter,baseline_gcnmf,graphsage \\
        [--negative-strategy random] \\
        [--num-seeds 15] \\
        [--num-targets 25] \\
        [--resume] \\
        [--verbose]
"""

import argparse
import logging
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from quvine.comprehensive_embedding_analysis import (
    ComprehensiveEmbeddingAnalysis,
    _select_seeds_targets_structured,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Ordered list of all baseline methods (not appnp, handled separately)
ALL_BASELINES = [
    "quvine_fused",
    "quvine_rwr",
    "quvine_heat",
    "quvine_poly",
    "quvine_hgcnmf",
    "quvine_pgcnmf",
    "netmf",
    "node2vec",
    "baseline_filter",
    "baseline_gcnmf",
    "graphsage",
    "quvine_ctqw",
    "quvine_dtqw",
]


def _merge_and_save(new_rows: list, existing_path: Path, key_cols: list) -> None:
    """Append new_rows to existing CSV, deduplicating by key_cols (last wins)."""
    new_df = pd.DataFrame(new_rows)
    if existing_path.exists():
        try:
            old_df = pd.read_csv(existing_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        except Exception:
            combined = new_df
    else:
        combined = new_df
    combined.to_csv(existing_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ranking + link-prediction eval for all baselines on one network"
    )
    parser.add_argument("--graphml-path",      required=True)
    parser.add_argument("--output-dir",        required=True)
    parser.add_argument("--network-id",        required=True)
    parser.add_argument("--network-type",      default="real_ppi")
    parser.add_argument("--methods",           default=",".join(ALL_BASELINES),
                        help="Comma-separated list of methods to evaluate")
    parser.add_argument("--negative-strategy", default="random")
    parser.add_argument("--num-seeds",         type=int, default=15)
    parser.add_argument("--num-targets",       type=int, default=25)
    parser.add_argument("--resume",            action="store_true",
                        help="Skip methods already present in ranking_results.csv")
    parser.add_argument("--verbose",           action="store_true", default=True)
    args = parser.parse_args()

    graphml_path = Path(args.graphml_path)
    output_dir   = Path(args.output_dir)
    network_id   = args.network_id
    methods      = [m.strip() for m in args.methods.split(",") if m.strip()]

    if not graphml_path.exists():
        logger.error(f"GraphML not found: {graphml_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info(f"Ranking + Link-Prediction Evaluation — {network_id}")
    logger.info(f"Methods   : {methods}")
    logger.info(f"Output    : {output_dir}")
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Load graph
    # ------------------------------------------------------------------
    logger.info("Loading graph …")
    G = nx.read_graphml(str(graphml_path))
    G = nx.convert_node_labels_to_integers(G)
    logger.info(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ------------------------------------------------------------------
    # Determine which methods to skip (resume mode)
    # ------------------------------------------------------------------
    done_methods: set = set()
    if args.resume:
        rank_csv = output_dir / f"{network_id}_ranking_results.csv"
        if rank_csv.exists():
            try:
                done_methods = set(pd.read_csv(rank_csv)["method"].unique())
                if done_methods:
                    logger.info(f"  Resume: skipping {sorted(done_methods)}")
            except Exception:
                pass

    methods_to_run = [m for m in methods if m not in done_methods]
    if not methods_to_run:
        logger.info("All methods already done — nothing to do.")
        return 0

    # ------------------------------------------------------------------
    # Select seeds and targets (same logic as run_single_network_analysis)
    # ------------------------------------------------------------------
    metadata = {
        "type":              args.network_type,
        "network_id":        network_id,
        "negative_strategy": args.negative_strategy,
    }
    seeds, targets = _select_seeds_targets_structured(
        G=G,
        network_metadata=metadata,
        num_seeds=args.num_seeds,
        num_targets=args.num_targets,
        base_seed=42,
    )
    logger.info(f"  Seeds: {len(seeds)}, Targets: {len(targets)}")

    # ------------------------------------------------------------------
    # Initialise evaluator (no embedding generation needed)
    # ------------------------------------------------------------------
    evaluator = ComprehensiveEmbeddingAnalysis(
        output_dir=str(output_dir),
        n_networks_per_type=1,
        n_nodes=G.number_of_nodes(),
        num_seeds=args.num_seeds,
        num_targets=args.num_targets,
        embedding_dim=128,
        seed=42,
        n_jobs=1,
    )

    # ------------------------------------------------------------------
    # Evaluate each method
    # ------------------------------------------------------------------
    ranking_rows   = []
    lp_tidy_rows   = []
    lp_summary_rows = []

    succeeded = 0
    failed    = 0

    for method in methods_to_run:
        emb_path = output_dir / f"{network_id}_{method}_embedding.npy"
        if not emb_path.exists():
            logger.warning(f"  [{method}] Embedding not found: {emb_path} — skipping")
            failed += 1
            continue

        try:
            embedding = np.load(str(emb_path))
        except Exception as e:
            logger.warning(f"  [{method}] Could not load embedding: {e} — skipping")
            failed += 1
            continue

        logger.info(f"  Processing {method} …")

        # Ranking
        try:
            rank_result = evaluator.evaluate_embedding(
                embedding, G, seeds, targets, method, network_id
            )
            ranking_rows.append(rank_result)
            logger.info(f"    ✓ ranking  p@10={rank_result.get('precision@10_centroid', 0):.3f}")
        except Exception as e:
            logger.warning(f"    ✗ ranking failed: {e}")

        # Link prediction
        try:
            lp_summary, lp_tidy = evaluator.evaluate_embedding_link_prediction(
                embedding, G, method, network_id,
                negative_strategy=args.negative_strategy,
            )
            lp_summary_rows.append(lp_summary)
            lp_tidy_rows.extend(lp_tidy)
            logger.info(f"    ✓ link-pred auc_roc={lp_summary.get('mean_auc_roc', 0):.3f}")
        except Exception as e:
            logger.warning(f"    ✗ link prediction failed: {e}")

        succeeded += 1

    logger.info(f"\n  Done: {succeeded} succeeded, {failed} skipped (no embedding)")

    # ------------------------------------------------------------------
    # Write / merge results
    # ------------------------------------------------------------------
    if ranking_rows:
        rank_path = output_dir / f"{network_id}_ranking_results.csv"
        _merge_and_save(ranking_rows, rank_path, key_cols=["network_id", "method"])
        logger.info(f"  ✓ Ranking results → {rank_path}")

    if lp_tidy_rows:
        lp_tidy_path = output_dir / f"{network_id}_link_prediction.csv"
        _merge_and_save(
            lp_tidy_rows, lp_tidy_path,
            key_cols=["network_id", "method", "edge_feature_method"],
        )
        logger.info(f"  ✓ Link prediction (tidy) → {lp_tidy_path}")

    if lp_summary_rows:
        lp_sum_path = output_dir / f"{network_id}_link_prediction_results.csv"
        _merge_and_save(lp_summary_rows, lp_sum_path, key_cols=["network_id", "method"])
        logger.info(f"  ✓ Link prediction (summary) → {lp_sum_path}")

    logger.info("=" * 80)
    logger.info(f"✓ Evaluation complete: {network_id}")
    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
