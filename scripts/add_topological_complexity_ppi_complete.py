#!/usr/bin/env python3
"""
Compute topological complexity metrics (Betti numbers and persistence entropy)
for all graphs in ppi_disease_v3 and add them to existing complexity CSV files.

Metrics computed:
- betti_0, betti_1, betti_2
- betti_sum
- euler_characteristic
- persistence_entropy_H0, persistence_entropy_H1, persistence_entropy_H2

Usage:
    python add_topological_complexity_ppi_complete.py [--n-jobs N] [--reset-checkpoint]
"""

import argparse
import gc
import sys
import warnings
from pathlib import Path

import networkx as nx
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(
    "/Users/aritrabose/Library/CloudStorage/OneDrive-IBM/Research/Quantum/quvine/ppi_disease_v3/results"
)
CHECKPOINT_CSV = RESULTS_DIR / "topological_checkpoint.csv"
CHECKPOINT_EVERY = 50

# ── Add quvine to path ────────────────────────────────────────────────────────
_QUVINE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_QUVINE_SRC) not in sys.path:
    sys.path.insert(0, str(_QUVINE_SRC))

from quvine.complexity.graph import compute_topological_metrics  # noqa: E402


# ── Per-network worker ────────────────────────────────────────────────────────

def _process_one(graphml_path: Path) -> dict | None:
    """Load graphml, compute topological metrics, return row dict or None on failure."""
    network_id = graphml_path.stem  # e.g., BioPlex3_asthma_rep00
    try:
        G = nx.read_graphml(graphml_path)
        G = nx.convert_node_labels_to_integers(G)
        
        # Compute topological metrics
        metrics = compute_topological_metrics(
            G,
            include_betti=True,
            include_persistence_entropy=True,
            maxdim=2,
            filtration_scale=1.0
        )
        
        # Extract only the metrics we need
        row = {
            "network_id": network_id,
            "betti_0": metrics.get("betti_0", 0),
            "betti_1": metrics.get("betti_1", 0),
            "betti_2": metrics.get("betti_2", 0),
            "betti_sum": metrics.get("betti_sum", 0),
            "euler_characteristic": metrics.get("euler_characteristic", 0),
            "persistence_entropy_H0": metrics.get("persistence_entropy_H0", 0.0),
            "persistence_entropy_H1": metrics.get("persistence_entropy_H1", 0.0),
            "persistence_entropy_H2": metrics.get("persistence_entropy_H2", 0.0),
        }
        
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
    print(f"Checkpoint   : {CHECKPOINT_CSV}")
    print(f"Workers      : {args.n_jobs}")

    # Find all graphml files
    graphml_files = sorted(RESULTS_DIR.glob("*/*.graphml"))
    print(f"GraphML files found : {len(graphml_files)}")

    # ── Resume from checkpoint ───────────────────────────────────────────────
    topo_rows: list[dict] = []
    done_ids: set[str] = set()
    if not args.reset_checkpoint and CHECKPOINT_CSV.exists():
        ckpt = pd.read_csv(CHECKPOINT_CSV)
        topo_rows = ckpt.to_dict("records")
        done_ids = {r["network_id"] for r in topo_rows}
        print(f"Resuming: {len(done_ids)} already in checkpoint.\n")
    else:
        print()

    todo = [p for p in graphml_files if p.stem not in done_ids]
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

            results = Parallel(n_jobs=args.n_jobs, backend="threading", verbose=0)(
                delayed(_process_one)(p) for p in chunk
            )

            n_ok = 0
            for r in results:
                if r is not None:
                    topo_rows.append(r)
                    n_ok += 1

            print(f"  Done chunk: {n_ok}/{len(chunk)} succeeded. "
                  f"Total rows: {len(topo_rows)}")

            # Save checkpoint after each chunk
            pd.DataFrame(topo_rows).to_csv(CHECKPOINT_CSV, index=False)
            print(f"  Checkpoint saved.")
            sys.stdout.flush()

    print(f"\nTotal succeeded: {len(topo_rows)} / {len(graphml_files)}")

    # Create DataFrame with topological metrics
    topo_df = pd.DataFrame(topo_rows)
    print(f"\nTopological metrics computed for {len(topo_df)} networks")
    print(f"Columns: {list(topo_df.columns)}")

    # Now update each individual complexity CSV file
    print("\n" + "="*80)
    print("Updating individual complexity CSV files...")
    print("="*80)
    
    updated_count = 0
    failed_count = 0
    
    for _, row in topo_df.iterrows():
        network_id = row["network_id"]
        complexity_csv = RESULTS_DIR / network_id / f"{network_id}_complexity.csv"
        
        if not complexity_csv.exists():
            print(f"Warning: {complexity_csv} does not exist, skipping...")
            failed_count += 1
            continue
        
        try:
            # Read existing complexity file
            comp_df = pd.read_csv(complexity_csv)
            
            # Add topological metrics as new columns
            for col in topo_df.columns:
                if col != "network_id":
                    comp_df[col] = row[col]
            
            # Save updated file
            comp_df.to_csv(complexity_csv, index=False)
            updated_count += 1
            
            if updated_count % 50 == 0:
                print(f"  Updated {updated_count} files...")
                
        except Exception as e:
            print(f"Error updating {complexity_csv}: {e}")
            failed_count += 1
    
    print(f"\nCompleted!")
    print(f"  Successfully updated: {updated_count} files")
    print(f"  Failed: {failed_count} files")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics:")
    print("="*80)
    for col in ["betti_0", "betti_1", "betti_2", "betti_sum", "euler_characteristic",
                "persistence_entropy_H0", "persistence_entropy_H1", "persistence_entropy_H2"]:
        if col in topo_df.columns:
            print(f"{col:30s}: mean={topo_df[col].mean():.4f}, "
                  f"std={topo_df[col].std():.4f}, "
                  f"min={topo_df[col].min():.4f}, "
                  f"max={topo_df[col].max():.4f}")


if __name__ == "__main__":
    main()

