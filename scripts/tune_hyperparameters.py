#!/dccstor/boseukb/Q/NetMed/QuVINE/Python-3.12.2/venv_quvine/bin/python3
"""
tune_hyperparameters.py

Pilot hyperparameter search for all QuVINE and classical embedding methods
across 14 synthetic network types and 5 real PPI networks (optionally with
disease gene data for the ranking task).

Design
------
- Pilot graphs: 100 nodes, 3 random seeds per network type (for stochasticity)
- Real networks: BFS-subsampled to `--pilot-nodes` nodes, 3 random seeds
- Search: Optuna TPE (with random-search fallback if Optuna not installed)
- Metric: mean(NC F1-macro CV, LP AUC-ROC) averaged across 3 pilot graphs
- Output: best_hyperparams.json  (best params per network_type × method)
          tuning_trials.csv      (all trial results for diagnostics)

Environment
-----------
Run with the project venv:
    /dccstor/boseukb/Q/NetMed/QuVINE/Python-3.12.2/venv_quvine/bin/python3 tune_hyperparameters.py
Or make executable and run directly:
    chmod +x tune_hyperparameters.py && ./tune_hyperparameters.py

Install optuna for TPE search (recommended, otherwise falls back to random search):
    /dccstor/boseukb/Q/NetMed/QuVINE/Python-3.12.2/venv_quvine/bin/pip install optuna

Usage
-----
# Tune everything (runs for a while)
python tune_hyperparameters.py --output-dir ./tuning_results --n-trials 30

# Tune a single network type on HPC (parallelize externally)
python tune_hyperparameters.py --network-type erdos_renyi --n-trials 50

# Tune a single real network + disease
python tune_hyperparameters.py --network-type real_STRING --disease autism --n-trials 40

# Tune only specific methods
python tune_hyperparameters.py --methods node2vec netmf --n-trials 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ── QuVINE source path ──────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent   # .../QuVINE/QuVINE/scripts/
_REPO_ROOT  = _SCRIPT_DIR.parent                # .../QuVINE/QuVINE/
_SRC_DIR    = _REPO_ROOT / "src"               # .../QuVINE/QuVINE/src/
for _p in [str(_SRC_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quvine.data.random_graphs import (
    generate_core_periphery,
    generate_erdos_renyi,
    generate_modular_network,
    generate_powerlaw_cluster,
    generate_random_geometric,
    generate_stochastic_block_model,
    generate_watts_strogatz,
    generate_barabasi_albert,
)
from quvine.views.generator import ViewBuilder
from quvine.walks.base import BaseWalker
from quvine.corpus.builder import CorpusBuilder
from quvine.embedding.word2vec import corpus_to_embedding
from quvine.baselines.node2vec import run_node2vec
from quvine.baselines.netmf import run_netmf
from quvine.baselines.graphsage import run_graphsage
from quvine.baselines.appnp import run_appnp
from quvine.baselines.gcn_mf import (
    generate_baseline_gcnmf_embedding,
    generate_baseline_filter_embedding_wrapper,
)
from quvine.evaluation.classification import evaluate_all_label_strategies
from quvine.evaluation.link_prediction import split_edges, evaluate_link_prediction

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[WARNING] optuna not installed — using random search (install with: pip install optuna)")

# ── Constants ────────────────────────────────────────────────────────────────

_DATA_DIR = _SCRIPT_DIR.parent / "data" / "networks"

REAL_NETWORK_PATHS: Dict[str, Path] = {
    "BioPlex3":  _DATA_DIR / "BioPlex3_shared" / "edges_list_ncbi.csv",
    "HumanNet":  _DATA_DIR / "HumanNetV3"      / "edges_list_ncbi.csv",
    "PCNet":     _DATA_DIR / "PCNet"            / "edges_list_ncbi.csv",
    "ProteomeHD":_DATA_DIR / "ProteomeHD"       / "edges_list_ncbi.csv",
    "STRING":    _DATA_DIR / "STRING"           / "edges_list_ncbi.csv",
}

_PROCESSED = _SCRIPT_DIR.parent / "data" / "processed_data"
DISEASE_SEED_PATHS: Dict[str, Path] = {
    "asthma":       _PROCESSED / "gene_seeds" / "asthma_ncbi_seeds.json",
    "autism":       _PROCESSED / "gene_seeds" / "autism_ncbi_seeds.json",
    "schizophrenia":_PROCESSED / "gene_seeds" / "schizophrenia_ncbi_seeds.json",
}
DISEASE_TARGET_PATHS: Dict[str, Path] = {
    "asthma":       _PROCESSED / "gwas_catalog_targets" / "asthma_targets_gene2ncbi.json",
    "autism":       _PROCESSED / "gwas_catalog_targets" / "autism_targets_gene2ncbi.json",
    "schizophrenia":_PROCESSED / "gwas_catalog_targets" / "schizophrenia_targets_gene2ncbi.json",
}

# All 14 synthetic network types used in the hard_negatives_v2 experiment
SYNTHETIC_NETWORK_TYPES = [
    "erdos_renyi",
    "watts_strogatz_high_p",
    "watts_strogatz_low_p",
    "random_geometric",
    "modular_strong",
    "modular_medium",
    "modular_many_communities",
    "core_periphery",
    "scale_free",
    "powerlaw_cluster",
    "stochastic_block_model",
    # tiny real graphs — use as-is (no subsampling needed)
    "real_karate",
    "real_lesmis",
    "real_polbooks",
]

# Representative methods for tuning (tune these 8, reuse for all 39)
ALL_METHODS = [
    "quvine_walks",        # Representative for all 11 quvine_* methods
    "baseline_filter_heat",
    "baseline_filter_poly",
    "baseline_gcnmf",
    "node2vec",
    "netmf",
    "graphsage",
    "appnp",
]

# Full list of 39 methods (for reference and validation)
ALL_39_METHODS = [
    # Quantum walk-based (11 methods) - use quvine_walks params
    "quvine_rwr", "quvine_ctqw", "quvine_dtqw",
    "quvine_baseline_heat", "quvine_baseline_poly",
    "quvine_rwr_heat", "quvine_rwr_poly",
    "quvine_ctqw_heat", "quvine_ctqw_poly",
    "quvine_dtqw_heat", "quvine_dtqw_poly",
    # GAT variants (12 methods) - use gat_baseline params (not tuned here)
    "gat_baseline", "gat_heat", "gat_poly",
    "gat_rwr", "gat_ctqw", "gat_dtqw",
    "gat_rwr_heat", "gat_rwr_poly",
    "gat_ctqw_heat", "gat_ctqw_poly",
    "gat_dtqw_heat", "gat_dtqw_poly",
    # GraphGPS variants (12 methods) - use graphgps_baseline params (not tuned here)
    "graphgps_baseline", "graphgps_heat", "graphgps_poly",
    "graphgps_rwr", "graphgps_ctqw", "graphgps_dtqw",
    "graphgps_rwr_heat", "graphgps_rwr_poly",
    "graphgps_ctqw_heat", "graphgps_ctqw_poly",
    "graphgps_dtqw_heat", "graphgps_dtqw_poly",
    # Classical baselines (4 methods)
    "node2vec", "netmf", "graphsage", "baseline_gcnmf",
]

# Method mapping: maps each of 39 methods to its tuning representative
METHOD_TUNING_MAP = {
    # All quvine methods use quvine_walks params
    **{m: "quvine_walks" for m in ALL_39_METHODS if m.startswith("quvine_")},
    # All GAT methods use gat_baseline params (default, not tuned)
    **{m: "gat_baseline" for m in ALL_39_METHODS if m.startswith("gat_")},
    # All GraphGPS methods use graphgps_baseline params (default, not tuned)
    **{m: "graphgps_baseline" for m in ALL_39_METHODS if m.startswith("graphgps_")},
    # Classical methods tune individually
    "node2vec": "node2vec",
    "netmf": "netmf",
    "graphsage": "graphsage",
    "baseline_gcnmf": "baseline_gcnmf",
}

# ── Graph generation ─────────────────────────────────────────────────────────

def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component as a new graph."""
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def generate_pilot_graph(network_type: str, n_nodes: int, seed: int) -> nx.Graph:
    """Generate a single pilot graph of the requested type with ~n_nodes nodes."""
    rng = np.random.default_rng(seed)

    if network_type == "erdos_renyi":
        G = generate_erdos_renyi(n_nodes, p=0.08, seed=seed)

    elif network_type == "watts_strogatz_high_p":
        G = generate_watts_strogatz(n_nodes, k=6, p=0.5, seed=seed)

    elif network_type == "watts_strogatz_low_p":
        G = generate_watts_strogatz(n_nodes, k=6, p=0.05, seed=seed)

    elif network_type == "random_geometric":
        G = generate_random_geometric(n_nodes, radius=0.18, seed=seed)

    elif network_type == "modular_strong":
        G, _ = generate_modular_network(
            num_communities=4,
            nodes_per_community=n_nodes // 4,
            p_intra=0.5,
            p_inter=0.01,
            seed=seed,
        )

    elif network_type == "modular_medium":
        G, _ = generate_modular_network(
            num_communities=4,
            nodes_per_community=n_nodes // 4,
            p_intra=0.35,
            p_inter=0.04,
            seed=seed,
        )

    elif network_type == "modular_many_communities":
        G, _ = generate_modular_network(
            num_communities=8,
            nodes_per_community=n_nodes // 8,
            p_intra=0.4,
            p_inter=0.02,
            seed=seed,
        )

    elif network_type == "core_periphery":
        G, _, _ = generate_core_periphery(
            n_core=n_nodes // 5,
            n_periphery=n_nodes - n_nodes // 5,
            p_core=0.7,
            p_core_periphery=0.15,
            p_periphery=0.01,
            seed=seed,
        )

    elif network_type == "scale_free":
        G = generate_barabasi_albert(n_nodes, m=3, seed=seed)

    elif network_type == "powerlaw_cluster":
        G = generate_powerlaw_cluster(n_nodes, m=3, p=0.3, seed=seed)

    elif network_type == "stochastic_block_model":
        k = 5
        block_sizes = [n_nodes // k] * k
        block_sizes[-1] += n_nodes - sum(block_sizes)
        p_matrix = [[0.4 if i == j else 0.02 for j in range(k)] for i in range(k)]
        G = generate_stochastic_block_model(block_sizes, p_matrix, seed=seed)

    elif network_type == "real_karate":
        G = nx.karate_club_graph()

    elif network_type == "real_lesmis":
        G = nx.les_miserables_graph()

    elif network_type == "real_polbooks":
        # polbooks is available as a GML file in networkx datasets
        try:
            G = nx.read_gml(
                str(Path(nx.__file__).parent / "generators" / "data" / "polbooks.gml"),
                label="id",
            )
        except Exception:
            # Fallback: generate a similar modular graph
            G, _ = generate_modular_network(
                num_communities=3,
                nodes_per_community=35,
                p_intra=0.35,
                p_inter=0.04,
                seed=seed,
            )
    else:
        raise ValueError(f"Unknown network type: {network_type!r}")

    G = _largest_cc(G)
    return G


def subsample_graph(G: nx.Graph, max_nodes: int, seed: int) -> nx.Graph:
    """
    Extract a connected subgraph of at most max_nodes nodes via BFS from a
    randomly chosen starting node.
    """
    if G.number_of_nodes() <= max_nodes:
        return G
    rng = np.random.default_rng(seed)
    start = rng.choice(list(G.nodes()))
    bfs_nodes = list(nx.bfs_tree(G, start).nodes())[:max_nodes]
    subG = G.subgraph(bfs_nodes).copy()
    subG = _largest_cc(subG)
    return subG


def load_real_network(name: str) -> nx.Graph:
    """Load a real PPI network CSV and return the largest CC."""
    path = REAL_NETWORK_PATHS[name]
    df = pd.read_csv(path, dtype=str)
    G = nx.from_pandas_edgelist(df, source="node1", target="node2")
    return _largest_cc(G)


def load_disease_data(
    G: nx.Graph, disease: str
) -> Tuple[Optional[List], Optional[List]]:
    """
    Load seed and target gene lists for a disease, filtered to nodes in G.
    Returns (seeds, targets) or (None, None) if files not found.
    """
    seed_path = DISEASE_SEED_PATHS.get(disease)
    target_path = DISEASE_TARGET_PATHS.get(disease)
    if seed_path is None or not seed_path.exists():
        return None, None
    try:
        with open(seed_path) as f:
            seeds_raw = json.load(f)
        with open(target_path) as f:
            targets_raw = json.load(f)
        nodes_set = set(str(n) for n in G.nodes())
        seeds = [s for s in seeds_raw if str(s) in nodes_set]
        targets = [str(v) for v in targets_raw.values() if str(v) in nodes_set]
        if len(seeds) < 2 or len(targets) < 2:
            return None, None
        return seeds, targets
    except Exception:
        return None, None


# ── OmegaConf config builder for QuVINE ─────────────────────────────────────

def make_quvine_cfg(params: Dict[str, Any]) -> Any:
    """Build a minimal OmegaConf DictConfig for ViewBuilder + BaseWalker."""
    return OmegaConf.create({
        "views": {
            "num_views":   params.get("num_views", 6),
            "constrained": params.get("constrained", True),
            "max_degree":  params.get("max_degree", 10),
            "max_nodes":   params.get("max_nodes", 120),
            "max_edges":   params.get("max_edges", 400),
            "degree_norm": params.get("degree_norm", True),
            "degree_alpha":params.get("degree_alpha", 0.5),
        },
        "walks": {
            "kinds":        params.get("walk_kinds", ["rwr", "ctqw", "dtqw"]),
            "num_walks":    params.get("num_walks", 6),
            "walk_length":  params.get("walk_length", 12),
            "restart_prob": params.get("restart_prob", 0.25),
            "max_iter":     params.get("max_iter", 500),
            "steps":        params.get("steps", 24),
            "time":         params.get("time", 1.0),
            "coin":         params.get("coin", "grover"),
        },
    })


# ── Embedding runners ────────────────────────────────────────────────────────

def _nodes_str(G: nx.Graph) -> List[str]:
    """Sorted string node list for corpus_to_embedding alignment."""
    return [str(n) for n in sorted(G.nodes())]


def run_quvine_walks(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """
    Run QuVINE walk-based embeddings (RWR, CTQW, DTQW).
    Returns {walk_kind: embedding_matrix (n_nodes × dim)}.
    """
    cfg = make_quvine_cfg(params)
    nodes_str = _nodes_str(G)
    node2idx = {str(n): i for i, n in enumerate(sorted(G.nodes()))}
    corpus_builders: Dict[str, CorpusBuilder] = {
        kind: CorpusBuilder() for kind in params.get("walk_kinds", ["rwr", "ctqw", "dtqw"])
    }

    for node in G.nodes():
        idx = node2idx[str(node)]
        root_rng = np.random.default_rng(seed * 100_000 + idx)

        view_builder = ViewBuilder(cfg=cfg, rng=root_rng)
        views = view_builder.build(G, node)

        walker = BaseWalker(cfg=cfg, rng=root_rng)
        for view in views:
            view_g = G.subgraph(view)
            view_nodes = list(view_g.nodes())
            walk_output = walker.run(G, node, view_nodes)
            for kind, walks in walk_output.items():
                if kind in corpus_builders and walks:
                    corpus_builders[kind].add(node, walks)

    dim = params.get("embedding_dim", 64)
    embeddings: Dict[str, np.ndarray] = {}
    for kind, builder in corpus_builders.items():
        corpus = builder.build()
        if not corpus:
            continue
        try:
            Z = corpus_to_embedding(
                corpus=corpus,
                nodes=nodes_str,
                vector_size=dim,
                window=params.get("window", 5),
                sg=1,
                negative=params.get("negative", 10),
                min_count=0,
                workers=1,
                epochs=params.get("epochs", 100),
            )
            embeddings[kind] = Z
        except Exception as exc:
            logging.debug("corpus_to_embedding failed for %s: %s", kind, exc)
    return embeddings


def run_filter_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run baseline graph filter embedding (heat or poly)."""
    filter_type = params.get("filter_type", "heat")
    Z = generate_baseline_filter_embedding_wrapper(
        G=G,
        filter_type=filter_type,
        t=params.get("t", 1.0),
        K=params.get("K", 4),
        embedding_dim=params.get("embedding_dim", 64),
        normalize=params.get("normalize", True),
        random_state=seed,
    )
    return {f"filter_{filter_type}": Z}


def run_gcnmf_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run baseline GCN-MF embedding."""
    Z = generate_baseline_gcnmf_embedding(
        G=G,
        embedding_dim=params.get("embedding_dim", 64),
        hidden_dim=params.get("hidden_dim", 64),
        mf_dim=params.get("mf_dim", 64),
        n_layers=params.get("n_layers", 2),
        epochs=params.get("epochs", 200),
        lr=params.get("lr", 0.01),
        weight_decay=params.get("weight_decay", 5e-4),
        random_state=seed,
    )
    return {"gcnmf": Z}


def run_node2vec_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run node2vec embedding."""
    nodes = list(G.nodes())
    Z = run_node2vec(
        graph=G,
        nodes=nodes,
        dimensions=params.get("dimensions", 64),
        walk_length=params.get("walk_length", 10),
        num_walks=params.get("num_walks", 10),
        p=params.get("p", 1.0),
        q=params.get("q", 0.5),
        window=params.get("window", 5),
        min_count=1,
        workers=1,
        seed=seed,
    )
    return {"node2vec": Z}


def run_netmf_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run NetMF embedding."""
    nodes = list(G.nodes())
    Z = run_netmf(
        graph=G,
        nodes=nodes,
        dimensions=params.get("dimensions", 64),
        window_size=params.get("window_size", 10),
        negative=params.get("negative", 1),
        seed=seed,
    )
    return {"netmf": Z}


def run_graphsage_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run GraphSAGE embedding."""
    nodes = list(G.nodes())
    Z = run_graphsage(
        graph=G,
        nodes=nodes,
        dimensions=params.get("dimensions", 64),
        hidden_dim=params.get("hidden_dim", 128),
        n_layers=params.get("n_layers", 2),
        epochs=params.get("epochs", 50),
        lr=params.get("lr", 0.01),
        neg_samples=params.get("neg_samples", 5),
        seed=seed,
    )
    return {"graphsage": Z}


def run_appnp_embedding(
    G: nx.Graph, params: Dict[str, Any], seed: int = 42
) -> Dict[str, np.ndarray]:
    """Run APPNP embedding."""
    nodes = list(G.nodes())
    Z = run_appnp(
        graph=G,
        nodes=nodes,
        dimensions=params.get("dimensions", 64),
        hidden_dim=params.get("hidden_dim", 64),
        n_layers=params.get("n_layers", 2),
        alpha=params.get("alpha", 0.1),
        K=params.get("K", 10),
        dropout=params.get("dropout", 0.5),
        epochs=params.get("epochs", 200),
        lr=params.get("lr", 0.01),
        weight_decay=params.get("weight_decay", 5e-4),
        seed=seed,
    )
    return {"appnp": Z}


# ── Evaluation ───────────────────────────────────────────────────────────────

def _nc_score(G: nx.Graph, Z: np.ndarray, node_list: List, seed: int) -> float:
    """Mean F1-macro (CV) across all label strategies that succeed."""
    try:
        results = evaluate_all_label_strategies(G, Z, node_list, random_state=seed)
    except Exception:
        return 0.0
    f1s = []
    for r in results.values():
        if not isinstance(r, dict):
            continue
        # prefer CV mean if available, else single-split
        val = r.get("f1_macro_cv_mean", r.get("f1_macro", None))
        if val is not None and val > 0:
            f1s.append(val)
    return float(np.mean(f1s)) if f1s else 0.0


def _lp_score(G: nx.Graph, Z: np.ndarray, node_list: List, seed: int) -> float:
    """AUC-ROC averaged over hadamard and cosine edge features."""
    if G.number_of_edges() < 10:
        return 0.0
    try:
        train_G, val_pos, test_pos, test_neg = split_edges(
            G, test_ratio=0.2, val_ratio=0.0, seed=seed
        )
    except Exception:
        return 0.0
    # Build train negatives (same count as test positives)
    from quvine.evaluation.link_prediction import sample_negative_edges
    try:
        train_neg = sample_negative_edges(G, len(list(train_G.edges())), seed=seed)
    except Exception:
        train_neg = []

    aucs = []
    for feat_method in ["hadamard", "cosine"]:
        try:
            metrics = evaluate_link_prediction(
                embeddings=Z,
                node_list=node_list,
                positive_edges=test_pos,
                negative_edges=test_neg,
                edge_feature_method=feat_method,
                train_positive_edges=list(train_G.edges()),
                train_negative_edges=train_neg,
                random_state=seed,
            )
            auc = metrics.get("auc_roc", 0.0)
            if auc > 0:
                aucs.append(auc)
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.0


def evaluate_embeddings(
    G: nx.Graph,
    embeddings_dict: Dict[str, np.ndarray],
    seed: int = 42,
) -> float:
    """
    Score a set of embeddings on NC + LP.
    Returns the mean score across all embeddings and tasks.
    """
    # For NC, node_list must match embedding rows and be same type as G nodes
    nodes = sorted(G.nodes())
    all_scores: List[float] = []

    for name, Z in embeddings_dict.items():
        if Z is None or Z.shape[0] != len(nodes):
            continue

        nc = _nc_score(G, Z, nodes, seed)
        lp = _lp_score(G, Z, nodes, seed)

        if nc > 0:
            all_scores.append(nc)
        if lp > 0:
            all_scores.append(lp)

    return float(np.mean(all_scores)) if all_scores else 0.0


def evaluate_on_graphs(
    G_list: List[nx.Graph],
    method: str,
    params: Dict[str, Any],
) -> Tuple[float, List[float]]:
    """
    Run a method with given params on all pilot graphs, return (mean_score, per_graph_scores).
    """
    RUNNERS = {
        "quvine_walks":         run_quvine_walks,
        "baseline_filter_heat": run_filter_embedding,
        "baseline_filter_poly": run_filter_embedding,
        "baseline_gcnmf":       run_gcnmf_embedding,
        "node2vec":             run_node2vec_embedding,
        "netmf":                run_netmf_embedding,
        "graphsage":            run_graphsage_embedding,
        "appnp":                run_appnp_embedding,
    }
    runner = RUNNERS[method]

    per_graph_scores: List[float] = []
    for i, G in enumerate(G_list):
        seed = 42 + i * 1000
        try:
            emb_dict = runner(G, params, seed=seed)
            score = evaluate_embeddings(G, emb_dict, seed=seed)
        except Exception as exc:
            logging.debug("Trial failed on graph %d: %s", i, exc)
            score = 0.0
        per_graph_scores.append(score)

    mean_score = float(np.mean(per_graph_scores)) if per_graph_scores else 0.0
    return mean_score, per_graph_scores


# ── Optuna parameter spaces ──────────────────────────────────────────────────

def suggest_quvine_walks(trial: Any) -> Dict[str, Any]:
    return {
        "num_views":    trial.suggest_int("num_views", 4, 12),
        "constrained":  trial.suggest_categorical("constrained", [True, False]),
        "max_degree":   trial.suggest_int("max_degree", 5, 25),
        "max_nodes":    trial.suggest_categorical("max_nodes", [60, 80, 120, 160]),
        "max_edges":    trial.suggest_categorical("max_edges", [200, 300, 400, 600]),
        "degree_alpha": trial.suggest_float("degree_alpha", 0.3, 0.9),
        "walk_kinds":   ["rwr", "ctqw", "dtqw"],
        "num_walks":    trial.suggest_int("num_walks", 4, 12),
        "walk_length":  trial.suggest_categorical("walk_length", [6, 8, 12, 16, 24]),
        "restart_prob": trial.suggest_float("restart_prob", 0.1, 0.6),
        "steps":        trial.suggest_categorical("steps", [8, 12, 16, 24, 32]),
        "time":         trial.suggest_float("time", 0.3, 3.0),
        "embedding_dim":trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "window":       trial.suggest_categorical("window", [3, 5, 8, 10]),
        "negative":     trial.suggest_categorical("negative", [5, 10, 15]),
        "epochs":       trial.suggest_categorical("epochs", [50, 100, 150]),
    }


def suggest_filter_heat(trial: Any) -> Dict[str, Any]:
    return {
        "filter_type":  "heat",
        "t":            trial.suggest_float("t", 0.1, 10.0, log=True),
        "embedding_dim":trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "normalize":    trial.suggest_categorical("normalize", [True, False]),
    }


def suggest_filter_poly(trial: Any) -> Dict[str, Any]:
    return {
        "filter_type":  "poly",
        "K":            trial.suggest_int("K", 2, 10),
        "embedding_dim":trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "normalize":    trial.suggest_categorical("normalize", [True, False]),
    }


def suggest_gcnmf(trial: Any) -> Dict[str, Any]:
    return {
        "embedding_dim": trial.suggest_categorical("embedding_dim", [32, 64, 128]),
        "hidden_dim":    trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "mf_dim":        trial.suggest_categorical("mf_dim", [32, 64, 128]),
        "n_layers":      trial.suggest_int("n_layers", 1, 3),
        "epochs":        trial.suggest_categorical("epochs", [50, 100, 200]),
        "lr":            trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "weight_decay":  trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


def suggest_node2vec(trial: Any) -> Dict[str, Any]:
    return {
        "dimensions":trial.suggest_categorical("dimensions", [32, 64, 128]),
        "walk_length":trial.suggest_categorical("walk_length", [8, 16, 30, 40]),
        "num_walks": trial.suggest_categorical("num_walks", [10, 15, 20]),
        "p":         trial.suggest_float("p", 0.25, 4.0, log=True),
        "q":         trial.suggest_float("q", 0.25, 4.0, log=True),
        "window":    trial.suggest_categorical("window", [3, 5, 8, 10]),
    }


def suggest_netmf(trial: Any) -> Dict[str, Any]:
    return {
        "dimensions":  trial.suggest_categorical("dimensions", [32, 64, 128]),
        "window_size": trial.suggest_categorical("window_size", [5, 10, 15, 20]),
        "negative":    trial.suggest_categorical("negative", [1, 3, 5]),
    }


def suggest_graphsage(trial: Any) -> Dict[str, Any]:
    return {
        "dimensions": trial.suggest_categorical("dimensions", [32, 64, 128]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "n_layers":   trial.suggest_int("n_layers", 1, 3),
        "epochs":     trial.suggest_categorical("epochs", [30, 50, 100]),
        "lr":         trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "neg_samples":trial.suggest_categorical("neg_samples", [5, 10, 15]),
    }


def suggest_appnp(trial: Any) -> Dict[str, Any]:
    return {
        "dimensions":  trial.suggest_categorical("dimensions", [32, 64, 128]),
        "hidden_dim":  trial.suggest_categorical("hidden_dim", [32, 64, 128]),
        "n_layers":    trial.suggest_int("n_layers", 1, 3),
        "alpha":       trial.suggest_float("alpha", 0.05, 0.3),
        "K":           trial.suggest_int("K", 5, 20),
        "dropout":     trial.suggest_float("dropout", 0.3, 0.7),
        "epochs":      trial.suggest_categorical("epochs", [100, 200, 300]),
        "lr":          trial.suggest_float("lr", 1e-4, 0.1, log=True),
        "weight_decay":trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


SUGGESTERS = {
    "quvine_walks":         suggest_quvine_walks,
    "baseline_filter_heat": suggest_filter_heat,
    "baseline_filter_poly": suggest_filter_poly,
    "baseline_gcnmf":       suggest_gcnmf,
    "node2vec":             suggest_node2vec,
    "netmf":                suggest_netmf,
    "graphsage":            suggest_graphsage,
    "appnp":                suggest_appnp,
}

# Default params (used as the first trial / fallback best)
DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "quvine_walks": {
        "num_views": 6, "constrained": True, "max_degree": 10,
        "max_nodes": 120, "max_edges": 400, "degree_alpha": 0.5,
        "walk_kinds": ["rwr", "ctqw", "dtqw"], "num_walks": 6,
        "walk_length": 12, "restart_prob": 0.25, "steps": 24, "time": 1.0,
        "embedding_dim": 64, "window": 5, "negative": 10, "epochs": 100,
    },
    "baseline_filter_heat": {
        "filter_type": "heat", "t": 1.0, "embedding_dim": 64, "normalize": True,
    },
    "baseline_filter_poly": {
        "filter_type": "poly", "K": 4, "embedding_dim": 64, "normalize": True,
    },
    "baseline_gcnmf": {
        "embedding_dim": 64, "hidden_dim": 64, "mf_dim": 64, "n_layers": 2,
        "epochs": 100, "lr": 0.01, "weight_decay": 5e-4,
    },
    "node2vec": {
        "dimensions": 64, "walk_length": 10, "num_walks": 10,
        "p": 1.0, "q": 0.5, "window": 5,
    },
    "netmf": {
        "dimensions": 64, "window_size": 10, "negative": 1,
    },
    "graphsage": {
        "dimensions": 64, "hidden_dim": 128, "n_layers": 2,
        "epochs": 50, "lr": 0.01, "neg_samples": 5,
    },
    "appnp": {
        "dimensions": 64, "hidden_dim": 64, "n_layers": 2,
        "alpha": 0.1, "K": 10, "dropout": 0.5,
        "epochs": 200, "lr": 0.01, "weight_decay": 5e-4,
    },
}


# ── Random search fallback ───────────────────────────────────────────────────

class _RandomTrial:
    """Minimal mock trial for random search when Optuna is unavailable."""
    def __init__(self, rng: np.random.Generator, trial_num: int):
        self.rng = rng
        self.number = trial_num

    def suggest_int(self, name: str, low: int, high: int) -> int:
        return int(self.rng.integers(low, high + 1))

    def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
        if log:
            return float(np.exp(self.rng.uniform(np.log(low), np.log(high))))
        return float(self.rng.uniform(low, high))

    def suggest_categorical(self, name: str, choices: List) -> Any:
        idx = int(self.rng.integers(0, len(choices)))
        return choices[idx]


# ── Core tuning routine ──────────────────────────────────────────────────────

def tune_method(
    G_list: List[nx.Graph],
    method: str,
    n_trials: int,
    network_label: str,
    trial_records: List[Dict],
) -> Tuple[Dict[str, Any], float]:
    """
    Run hyperparameter search for a single method on a set of pilot graphs.
    Returns (best_params, best_score).
    """
    suggester = SUGGESTERS[method]

    # Always evaluate default params first
    def_params = DEFAULT_PARAMS[method].copy()
    def_score, def_per = evaluate_on_graphs(G_list, method, def_params)
    best_params = def_params.copy()
    best_score = def_score
    trial_records.append({
        "network": network_label, "method": method, "trial": 0,
        "score": def_score, "per_graph": str(def_per),
        **{f"p_{k}": v for k, v in def_params.items() if k != "walk_kinds"},
    })

    if OPTUNA_AVAILABLE:
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=max(5, n_trials // 4)),
        )

        # Seed with default params
        try:
            enqueue_params = {k: v for k, v in def_params.items() if k != "walk_kinds"}
            study.enqueue_trial(enqueue_params)
        except Exception:
            pass

        def objective(trial):
            params = suggester(trial)
            score, per_graph = evaluate_on_graphs(G_list, method, params)
            trial_records.append({
                "network": network_label, "method": method,
                "trial": trial.number + 1, "score": score, "per_graph": str(per_graph),
                **{f"p_{k}": v for k, v in params.items() if k != "walk_kinds"},
            })
            return score

        study.optimize(objective, n_trials=n_trials, catch=(Exception,))

        if study.best_value > best_score:
            best_score = study.best_value
            best_params = suggester(study.best_trial)

    else:
        # Random search
        rng = np.random.default_rng(42)
        for t in range(1, n_trials + 1):
            trial_mock = _RandomTrial(rng, t)
            try:
                params = suggester(trial_mock)
                score, per_graph = evaluate_on_graphs(G_list, method, params)
            except Exception:
                score, per_graph, params = 0.0, [], def_params
            trial_records.append({
                "network": network_label, "method": method, "trial": t,
                "score": score, "per_graph": str(per_graph),
                **{f"p_{k}": v for k, v in params.items() if k != "walk_kinds"},
            })
            if score > best_score:
                best_score = score
                best_params = params.copy()

    return best_params, best_score


def tune_network(
    network_label: str,
    G_list: List[nx.Graph],
    methods: List[str],
    n_trials: int,
    trial_records: List[Dict],
) -> Dict[str, Any]:
    """Tune all methods for a given network, return best params per method."""
    result: Dict[str, Any] = {}
    for method in methods:
        print(f"  [{network_label}] Tuning {method} ({n_trials} trials)…", flush=True)
        try:
            best_params, best_score = tune_method(
                G_list, method, n_trials, network_label, trial_records
            )
            # Clean up non-serialisable objects
            clean_params = {
                k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in best_params.items()
            }
            result[method] = {"params": clean_params, "score": round(best_score, 6)}
            print(f"    → best score={best_score:.4f}", flush=True)
        except Exception as exc:
            print(f"    [ERROR] {method}: {exc}", flush=True)
            result[method] = {"params": DEFAULT_PARAMS[method].copy(), "score": 0.0}
    return result


# ── Build pilot graph lists ──────────────────────────────────────────────────

def build_synthetic_pilots(
    network_type: str, n_nodes: int, n_seeds: int
) -> List[nx.Graph]:
    G_list = []
    for seed in range(n_seeds):
        try:
            G = generate_pilot_graph(network_type, n_nodes, seed=seed * 7 + 1)
            G_list.append(G)
        except Exception as exc:
            print(f"  [WARN] Could not generate {network_type} seed={seed}: {exc}")
    return G_list


def build_real_pilots(
    network_name: str, n_nodes: int, n_seeds: int
) -> List[nx.Graph]:
    print(f"  Loading {network_name}…", flush=True)
    try:
        full_G = load_real_network(network_name)
    except Exception as exc:
        print(f"  [ERROR] Could not load {network_name}: {exc}")
        return []
    G_list = []
    for seed in range(n_seeds):
        subG = subsample_graph(full_G, n_nodes, seed=seed * 13 + 42)
        if subG.number_of_nodes() >= 20:
            G_list.append(subG)
    return G_list


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for QuVINE methods")
    parser.add_argument("--output-dir", default="./tuning_results",
                        help="Directory for output files")
    parser.add_argument("--n-trials", type=int, default=30,
                        help="Number of Optuna trials per method per network type")
    parser.add_argument("--pilot-nodes", type=int, default=200,
                        help="Number of nodes in pilot graphs (synthetic + subsampled real)")
    parser.add_argument("--pilot-seeds", type=int, default=3,
                        help="Number of random graph instances for stochasticity")
    parser.add_argument("--network-type", nargs="*", default=None,
                        help="Restrict to these network types (omit for all). "
                             "Use 'real_BioPlex3', 'real_STRING' etc. for real networks.")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Restrict to these methods (omit for all)")
    parser.add_argument("--disease", default=None,
                        choices=["asthma", "autism", "schizophrenia"],
                        help="Disease for real network ranking task (optional)")
    parser.add_argument("--skip-real", action="store_true",
                        help="Skip real PPI networks (run synthetic types only)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = args.methods if args.methods else ALL_METHODS
    invalid = [m for m in methods if m not in ALL_METHODS]
    if invalid:
        parser.error(f"Unknown methods: {invalid}. Valid: {ALL_METHODS}")

    # ── Determine which network types to run ───────────────────────────────
    real_networks = [] if args.skip_real else list(REAL_NETWORK_PATHS.keys())

    # real_karate / real_lesmis / real_polbooks are tiny real graphs used as
    # synthetic pilots (handled in generate_pilot_graph), NOT PPI networks.
    _TINY_REAL_AS_SYNTHETIC = {"real_karate", "real_lesmis", "real_polbooks"}

    if args.network_type:
        # User-specified subset
        synthetic_types = [t for t in args.network_type
                           if not t.startswith("real_") or t in _TINY_REAL_AS_SYNTHETIC]
        real_networks = [t.replace("real_", "") for t in args.network_type
                        if t.startswith("real_")
                        and t not in _TINY_REAL_AS_SYNTHETIC
                        and t.replace("real_", "") in REAL_NETWORK_PATHS]
    else:
        synthetic_types = SYNTHETIC_NETWORK_TYPES

    print(f"\n{'='*60}")
    print(f"QuVINE Hyperparameter Tuning")
    print(f"  Synthetic types : {len(synthetic_types)}")
    print(f"  Real networks   : {real_networks}")
    print(f"  Methods         : {methods}")
    print(f"  Trials/method   : {args.n_trials}")
    print(f"  Pilot nodes     : {args.pilot_nodes}")
    print(f"  Pilot seeds     : {args.pilot_seeds}")
    print(f"  Output dir      : {out_dir}")
    print(f"  Optuna available: {OPTUNA_AVAILABLE}")
    print(f"{'='*60}\n")

    all_results: Dict[str, Any] = {"synthetic": {}, "real": {}}
    trial_records: List[Dict] = []

    # ── Synthetic networks ─────────────────────────────────────────────────
    for net_type in synthetic_types:
        print(f"\n[Synthetic] {net_type}")
        G_list = build_synthetic_pilots(net_type, args.pilot_nodes, args.pilot_seeds)
        if not G_list:
            print(f"  [SKIP] No graphs generated for {net_type}")
            continue
        print(f"  Graphs: {[G.number_of_nodes() for G in G_list]} nodes")
        result = tune_network(net_type, G_list, methods, args.n_trials, trial_records)
        all_results["synthetic"][net_type] = result
        # Save incrementally
        _save_results(all_results, trial_records, out_dir)

    # ── Real networks ──────────────────────────────────────────────────────
    for net_name in real_networks:
        label = f"real_{net_name}"
        print(f"\n[Real] {net_name}")
        G_list = build_real_pilots(net_name, args.pilot_nodes, args.pilot_seeds)
        if not G_list:
            print(f"  [SKIP] No graphs loaded for {net_name}")
            continue
        print(f"  Graphs: {[G.number_of_nodes() for G in G_list]} nodes")
        result = tune_network(label, G_list, methods, args.n_trials, trial_records)
        all_results["real"][net_name] = result
        _save_results(all_results, trial_records, out_dir)

    print(f"\n{'='*60}")
    print(f"Done. Results in {out_dir}/")
    print(f"  best_hyperparams.json  — best params per network × method")
    print(f"  tuning_trials.csv      — all trial results for diagnostics")
    print(f"{'='*60}\n")


def _save_results(all_results: Dict, trial_records: List[Dict], out_dir: Path):
    """Save best params JSON and trial CSV (called incrementally)."""
    # Flatten all_results to per-network-type best params for easy consumption
    flat: Dict[str, Dict[str, Any]] = {}
    for category in ["synthetic", "real"]:
        for net_type, net_results in all_results[category].items():
            flat[net_type] = {
                method: entry.get("params", {})
                for method, entry in net_results.items()
            }
            flat[net_type]["_scores"] = {
                method: entry.get("score", 0.0)
                for method, entry in net_results.items()
            }

    json_path = out_dir / "best_hyperparams.json"
    with open(json_path, "w") as f:
        json.dump({"_full": all_results, "best_params": flat}, f, indent=2, default=str)

    if trial_records:
        csv_path = out_dir / "tuning_trials.csv"
        pd.DataFrame(trial_records).to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
