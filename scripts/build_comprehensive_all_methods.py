#!/usr/bin/env python3
"""
Build comprehensive results CSV with all methods (baselines + APPNP).

Output (long format, one row per network × method):
  - Classification metrics: all 14 methods from node_embedding.csv
  - Ranking metrics: appnp only (others were overwritten)
  - Link prediction metrics: appnp only (others were overwritten)
  - Network-level graph + topology features: broadcast from complexity.csv
"""

import glob
from pathlib import Path
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("/dccstor/boseukb/Q/NetMed/QuVINE/results/ppi_disease_v3/results")
OUTPUT_CSV  = RESULTS_DIR / "comprehensive_results_all_methods.csv"

print("=" * 80)
print(" Building Comprehensive Results — All Methods")
print("=" * 80)

# ---------------------------------------------------------------------------
# 1. Network-level features from complexity.csv
# ---------------------------------------------------------------------------
print("\n[1/4] Reading network-level features from complexity.csv ...")
complexity_frames = []
for f in sorted(glob.glob(str(RESULTS_DIR / "*" / "*_complexity.csv"))):
    try:
        complexity_frames.append(pd.read_csv(f))
    except Exception as e:
        print(f"  WARN: {f}: {e}")

complexity_df = pd.concat(complexity_frames, ignore_index=True)
# De-duplicate: keep last (post-topology update)
complexity_df = complexity_df.drop_duplicates(subset=["network_id"], keep="last")
print(f"  Network features: {complexity_df.shape[0]} networks, {complexity_df.shape[1]} columns")

# ---------------------------------------------------------------------------
# 2. Classification results from node_embedding.csv (all methods)
# ---------------------------------------------------------------------------
print("\n[2/4] Reading classification results from node_embedding.csv ...")
nc_frames = []
for f in sorted(glob.glob(str(RESULTS_DIR / "*" / "*_node_embedding.csv"))):
    try:
        nc_frames.append(pd.read_csv(f))
    except Exception as e:
        print(f"  WARN: {f}: {e}")

nc_raw = pd.concat(nc_frames, ignore_index=True)
print(f"  Raw rows: {len(nc_raw):,}  |  Methods: {sorted(nc_raw['method'].unique())}")

# Aggregate over label strategies per (network_id, method)
nc_agg = (
    nc_raw
    .groupby(["network_id", "method"])
    .agg(
        clf_mean_f1_macro   =("f1_macro",  "mean"),
        clf_std_f1_macro    =("f1_macro",  "std"),
        clf_max_f1_macro    =("f1_macro",  "max"),
        clf_mean_accuracy   =("accuracy",  "mean"),
        clf_std_accuracy    =("accuracy",  "std"),
        clf_n_strategies    =("f1_macro",  "count"),
    )
    .reset_index()
)

# Per-strategy f1 pivot (wide columns like clf_f1_community_louvain)
strategy_pivot = nc_raw.pivot_table(
    index=["network_id", "method"],
    columns="label_strategy",
    values="f1_macro",
    aggfunc="first",
).reset_index()
strategy_pivot.columns = (
    ["network_id", "method"]
    + [f"clf_f1_{c}" for c in strategy_pivot.columns[2:]]
)

nc_df = nc_agg.merge(strategy_pivot, on=["network_id", "method"], how="left")
print(f"  Aggregated classification: {nc_df.shape[0]:,} rows, {nc_df.shape[1]} cols")

# ---------------------------------------------------------------------------
# 3. Ranking results (appnp only)
# ---------------------------------------------------------------------------
print("\n[3/4] Reading ranking results ...")
rank_frames = []
for f in sorted(glob.glob(str(RESULTS_DIR / "*" / "*_ranking_results.csv"))):
    try:
        rank_frames.append(pd.read_csv(f))
    except Exception as e:
        print(f"  WARN: {f}: {e}")

rank_df = pd.concat(rank_frames, ignore_index=True)
# Rename columns to avoid clashes
rank_df = rank_df.rename(columns={
    c: f"ranking_{c}" for c in rank_df.columns if c not in ("network_id", "method")
})
print(f"  Ranking rows: {len(rank_df):,}  |  Methods: {rank_df['method'].unique().tolist()}")

# ---------------------------------------------------------------------------
# 4. Link prediction results (appnp only)
# ---------------------------------------------------------------------------
print("\n[4/4] Reading link prediction results ...")
lp_frames = []
for f in sorted(glob.glob(str(RESULTS_DIR / "*" / "*_link_prediction.csv"))):
    try:
        lp_frames.append(pd.read_csv(f))
    except Exception as e:
        print(f"  WARN: {f}: {e}")

lp_raw = pd.concat(lp_frames, ignore_index=True)

# Aggregate over edge_feature_method per (network_id, method, negative_strategy)
lp_agg = (
    lp_raw
    .groupby(["network_id", "method", "negative_strategy"])
    .agg(
        lp_mean_auc_roc =("auc_roc", "mean"),
        lp_std_auc_roc  =("auc_roc", "std"),
        lp_max_auc_roc  =("auc_roc", "max"),
        lp_mean_auc_pr  =("auc_pr",  "mean"),
        lp_std_auc_pr   =("auc_pr",  "std"),
        lp_max_auc_pr   =("auc_pr",  "max"),
        lp_mean_mrr     =("mrr",     "mean"),
        lp_mean_f1      =("f1",      "mean"),
        lp_n_methods    =("auc_roc", "count"),
    )
    .reset_index()
    .drop(columns=["negative_strategy"])  # only 'random' strategy present
)

# Per-edge-feature-method AUC columns (hadamard, average, l1, l2, inner_product, cosine)
ef_pivot = lp_raw.pivot_table(
    index=["network_id", "method"],
    columns="edge_feature_method",
    values=["auc_roc", "auc_pr", "f1"],
    aggfunc="first",
).reset_index()
ef_pivot.columns = (
    ["network_id", "method"]
    + [f"lp_{stat}_{ef}" for stat, ef in ef_pivot.columns[2:]]
)

lp_df = lp_agg.merge(ef_pivot, on=["network_id", "method"], how="left")
print(f"  Link prediction rows: {len(lp_df):,}  |  Methods: {lp_raw['method'].unique().tolist()}")

# ---------------------------------------------------------------------------
# 5. Assemble final table
# ---------------------------------------------------------------------------
print("\nAssembling final table ...")

# Start from classification (all methods)
final = nc_df.copy()

# Left-join ranking (appnp only → NaN for others)
final = final.merge(rank_df, on=["network_id", "method"], how="left")

# Left-join link prediction (appnp only → NaN for others)
final = final.merge(lp_df, on=["network_id", "method"], how="left")

# Left-join network-level features (broadcast per network)
net_cols = [c for c in complexity_df.columns if c != "network_id"]
final = final.merge(complexity_df[["network_id"] + net_cols], on="network_id", how="left")

print(f"\nFinal shape: {final.shape}")
print(f"  Networks: {final['network_id'].nunique()}")
print(f"  Methods:  {sorted(final['method'].unique())}")
print(f"  Rows:     {len(final):,}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
final.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved → {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# 7. Coverage summary
# ---------------------------------------------------------------------------
print("\n--- Coverage per method ---")
method_order = sorted(final["method"].unique())
for m in method_order:
    sub = final[final["method"] == m]
    n = len(sub)
    clf_ok  = sub["clf_mean_f1_macro"].notna().sum()
    rank_ok = sub["ranking_precision@10_centroid"].notna().sum() if "ranking_precision@10_centroid" in sub else 0
    lp_ok   = sub["lp_mean_auc_roc"].notna().sum()
    print(f"  {m:<30s} networks={n:3d}  clf={clf_ok:3d}  ranking={rank_ok:3d}  lp={lp_ok:3d}")

print("\nDone.")
