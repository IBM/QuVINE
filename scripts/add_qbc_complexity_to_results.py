#!/usr/bin/env python3
"""
Compute QBC complexity metrics for all networks in hard_negatives_v3 and
merge them into comprehensive_results.csv.

Runs in parallel using joblib. Saves a checkpoint every CHECKPOINT_EVERY
networks so that if killed it can resume from where it left off.

Usage:
    python add_qbc_complexity_to_results.py [--n-jobs N] [--reset-checkpoint]
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(
    "/dccstor/boseukb/Q/NetMed/QuVINE/results/hard_negatives_v3/results"
)
COMP_CSV       = RESULTS_DIR / "comprehensive_results.csv"
OUT_CSV        = RESULTS_DIR / "comprehensive_results.csv"
CHECKPOINT_CSV = RESULTS_DIR / "qbc_checkpoint.csv"
CHECKPOINT_EVERY = 100

# ── Add quvine to path ────────────────────────────────────────────────────────
_QUVINE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_QUVINE_SRC) not in sys.path:
    sys.path.insert(0, str(_QUVINE_SRC))

from quvine.complexity.qbc import compute_qbc_metrics   # noqa: E402


# ── Per-network worker ────────────────────────────────────────────────────────

def _process_one(gml_path: Path) -> dict | None:
    """Load graphml, compute QBC metrics, return row dict or None on failure."""
    network_id = gml_path.parent.name
    try:
        G = nx.read_graphml(gml_path)
        G = nx.convert_node_labels_to_integers(G)
        row = {"network_id": network_id}
        row.update(compute_qbc_metrics(G))
        del G
        gc.collect()
        return row
    except Exception as exc:
        warnings.warn(f"Failed for {network_id}: {exc}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-jobs", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--reset-checkpoint", action="store_true",
                        help="Ignore existing checkpoint and start fresh")
    args = parser.parse_args()

    print(f"Results dir  : {RESULTS_DIR}")
    print(f"Input CSV    : {COMP_CSV}")
    print(f"Checkpoint   : {CHECKPOINT_CSV}")
    print(f"Workers      : {args.n_jobs}")

    comp_df = pd.read_csv(COMP_CSV)
    print(f"Comprehensive results: {comp_df.shape}, "
          f"{comp_df['network_id'].nunique()} unique networks")

    graphml_files = sorted(RESULTS_DIR.glob("*/*.graphml"))
    print(f"GraphML files found : {len(graphml_files)}")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    qbc_rows: list[dict] = []
    done_ids: set[str] = set()
    if not args.reset_checkpoint and CHECKPOINT_CSV.exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        qbc_rows = ckpt.to_dict("records")
        done_ids = {r["network_id"] for r in qbc_rows}
        print(f"Resuming: {len(done_ids)} already in checkpoint.\n")
    else:
        print()

    todo = [p for p in graphml_files if p.parent.name not in done_ids]
    print(f"Remaining: {len(todo)} networks to process.\n")

    if not todo:
        print("Nothing to do — all networks already in checkpoint.")
    else:
        # Process in chunks so we can save checkpoints periodically
        chunk_size = max(CHECKPOINT_EVERY, args.n_jobs)
        for chunk_start in range(0, len(todo), chunk_size):
            chunk = todo[chunk_start: chunk_start + chunk_size]
            chunk_end = min(chunk_start + chunk_size, len(todo))
            print(f"Processing [{chunk_start+1}–{chunk_end}] / {len(todo)} ...")
            sys.stdout.flush()

            results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=0)(
                delayed(_process_one)(p) for p in chunk
            )

            n_ok = 0
            for r in results:
                if r is not None:
                    qbc_rows.append(r)
                    n_ok += 1

            print(f"  Done chunk: {n_ok}/{len(chunk)} succeeded. "
                  f"Total rows: {len(qbc_rows)}")

            # Save checkpoint after each chunk
            pd.DataFrame(qbc_rows).to_csv(CHECKPOINT_CSV, index=False)
            print(f"  Checkpoint saved.")
            sys.stdout.flush()

    print(f"\nTotal succeeded: {len(qbc_rows)} / {len(graphml_files)}")

    qbc_df = pd.DataFrame(qbc_rows)
    new_cols = [c for c in qbc_df.columns if c != "network_id"]
    print(f"QBC columns ({len(new_cols)}): {new_cols}")

    # Drop stale qbc_* columns from comp_df to avoid duplication
    existing_qbc = [c for c in comp_df.columns if c.startswith("qbc_")]
    if existing_qbc:
        print(f"Dropping {len(existing_qbc)} stale qbc_* columns.")
        comp_df = comp_df.drop(columns=existing_qbc)

    merged = comp_df.merge(qbc_df, on="network_id", how="left")
    print(f"Merged shape: {merged.shape}")
    assert len(merged) == len(comp_df), \
        f"Row count mismatch: {len(comp_df)} → {len(merged)}"

    merged.to_csv(OUT_CSV, index=False)
    print(f"Saved to: {OUT_CSV}")

    print(f"\nQBC columns NaN summary:")
    for c in new_cols:
        n_na = merged[c].isna().sum()
        if n_na > 0:
            print(f"  {c:<50s}  nan={n_na}/{len(merged)}")
    print("Done.")


if __name__ == "__main__":
    main()
