#!/usr/bin/env python3
"""
Local Hyperparameter Tuning Test Script

Tests hyperparameter tuning for erdos_renyi and modular networks
with all methods before deploying to LSF.

Usage:
    python scripts/tune_local_test.py --n-trials 10 --output-dir ./tuning_test
    python scripts/tune_local_test.py --network-type erdos_renyi --methods quvine_walks node2vec
"""

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

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add QuVINE source to path
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
for _p in [str(_SRC_DIR), str(_REPO_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quvine.data.random_graphs import (
    generate_erdos_renyi,
    generate_modular_network,
)
from quvine.views.generator import ViewBuilder
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

# Try to import GAT and GraphGPS (may not be available)
try:
    from quvine.baselines.gat import generate_gat_embedding_by_method_name, GATConfig, TrainConfig as GATTrainConfig
    GAT_AVAILABLE = True
except ImportError:
    GAT_AVAILABLE = False
    logger.warning("GAT not available - will skip GAT methods")

try:
    from quvine.baselines.graphgps import generate_graphgps_embedding_by_method_name, GraphGPSConfig, TrainConfig as GraphGPSTrainConfig
    GRAPHGPS_AVAILABLE = True
except ImportError:
    GRAPHGPS_AVAILABLE = False
    logger.warning("GraphGPS not available - will skip GraphGPS methods")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
    logger.info("Optuna available - using TPE sampler")
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not installed - using random search")

# Network types to test
NETWORK_TYPES = ["erdos_renyi", "modular"]

# Methods to tune (10 representative methods for all 39)
ALL_METHODS = [
    "quvine_walks",        # Representative for all 11 quvine_* methods
    "baseline_filter_heat",
    "baseline_filter_poly",
    "baseline_gcnmf",
    "node2vec",
    "netmf",
    "graphsage",
    "appnp",
    "gat_baseline",        # Representative for all 12 GAT methods
    "graphgps_baseline",   # Representative for all 12 GraphGPS methods
]


def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component."""
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def generate_pilot_graph(network_type: str, n_nodes: int, seed: int) -> nx.Graph:
    """Generate a pilot graph for tuning."""
    logger.info(f"Generating {network_type} graph with {n_nodes} nodes, seed={seed}")
    
    if network_type == "erdos_renyi":
        G = generate_erdos_renyi(n_nodes, p=0.08, seed=seed)
    elif network_type == "modular":
        G, _ = generate_modular_network(
            num_communities=4,
            nodes_per_community=n_nodes // 4,
            p_intra=0.5,
            p_inter=0.01,
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
    
    G = _largest_cc(G)
    G = nx.convert_node_labels_to_integers(G)
    logger.info(f"Generated graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def make_quvine_cfg(params: Dict[str, Any]) -> Dict[str, Any]:
    """Create QuVINE configuration from parameters."""
    return {
        "views": {
            "num_views": params.get("num_views", 3),
            "constrained": True,
            "max_degree": params.get("max_degree", 50),
            "max_nodes": params.get("max_nodes", 500),
            "max_edges": params.get("max_edges", 2000),
            "degree_norm": True,
            "degree_alpha": params.get("degree_alpha", 0.5),
        },
        "walks": {
            "kinds": ["rwr"],  # Use only RWR for simplicity
            "num_walks": params.get("num_walks", 10),
            "walk_length": params.get("walk_length", 80),
            "restart_prob": params.get("restart_prob", 0.15),
            "max_iter": 100,
            "steps": params.get("steps", 5),
            "time": params.get("time", 1.0),
            "coin": params.get("coin", "grover"),
        },
        "embedding": {
            "dim": params.get("embedding_dim", 128),
            "window": params.get("window", 10),
            "sg": 1,
            "negative": params.get("negative", 5),
            "min_count": 1,
            "workers": 4,
            "epochs": params.get("epochs", 10),
        },
    }


def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> np.ndarray:
    """Run QuVINE walks embedding."""
    from types import SimpleNamespace
    from quvine.walks.base import BaseWalker
    
    cfg_dict = make_quvine_cfg(params)
    
    # Convert dict to nested namespace for ViewBuilder and Walker
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    
    cfg = dict_to_namespace(cfg_dict)
    
    # Build views and walks
    rng = np.random.default_rng(42)
    view_builder = ViewBuilder(cfg, rng)
    walker = BaseWalker(cfg, rng)
    corpus_builder = CorpusBuilder()
    
    # Generate views and walks for each seed
    for seed in seeds:
        # view_builder.build returns a list of node sets
        view_node_sets = view_builder.build(G, seed)
        
        # Generate walks for each view
        for view_nodes in view_node_sets:
            # Create subgraph from node set
            view_graph = G.subgraph(view_nodes)
            
            # Generate walks on this view
            view_walks_dict = walker.run(view_graph, seed, view_nodes=list(view_nodes))
            
            # Combine all walk types and convert node IDs to strings
            all_walks = []
            for walk_type, walks in view_walks_dict.items():
                # Convert each walk's nodes to strings
                str_walks = [[str(node) for node in walk] for walk in walks]
                all_walks.extend(str_walks)
            corpus_builder.add(seed, all_walks)
    
    # Build corpus
    corpus = corpus_builder.build()
    
    # Train embedding
    # Convert node IDs to strings for corpus_to_embedding
    nodes_str = [str(n) for n in G.nodes()]
    embedding = corpus_to_embedding(
        corpus,
        nodes=nodes_str,
        vector_size=cfg_dict["embedding"]["dim"],
        window=cfg_dict["embedding"]["window"],
        sg=cfg_dict["embedding"]["sg"],
        negative=cfg_dict["embedding"]["negative"],
        min_count=cfg_dict["embedding"]["min_count"],
        workers=cfg_dict["embedding"]["workers"],
        epochs=cfg_dict["embedding"]["epochs"],
    )
    
    return embedding


def run_filter_embedding(G: nx.Graph, seeds: List[int], params: Dict[str, Any], filter_type: str) -> np.ndarray:
    """Run filter-based embedding."""
    return generate_baseline_filter_embedding_wrapper(
        G,
        filter_type=filter_type,
        t=params.get("filter_t", 3.0),
        K=params.get("filter_order", 5),
        embedding_dim=params.get("embedding_dim", 128),
        normalize=True,
        random_state=42
    )


def run_gcnmf_embedding(G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> np.ndarray:
    """Run GCN-MF embedding."""
    return generate_baseline_gcnmf_embedding(
        G,
        embedding_dim=params.get("embedding_dim", 128),
        hidden_dim=params.get("hidden_dim", 64),
        mf_dim=params.get("mf_dim", 64),
        n_layers=params.get("gcnmf_layers", 2),
        epochs=params.get("epochs", 200),
        lr=params.get("lr", 0.01),
        weight_decay=params.get("weight_decay", 5e-4),
        random_state=42
    )


def evaluate_embeddings(
    embedding: np.ndarray,
    G: nx.Graph,
    test_edges: List[Tuple[int, int]],
    test_non_edges: List[Tuple[int, int]],
) -> Dict[str, float]:
    """Evaluate embedding on node classification and link prediction."""
    metrics = {}
    node_list = list(G.nodes())
    
    # Node classification with community detection
    try:
        nc_results = evaluate_all_label_strategies(
            G,
            embedding,
            node_list=node_list,
            test_size=0.3,
            random_state=42
        )
        # Average F1 scores across all strategies
        f1_scores = []
        for strategy, results in nc_results.items():
            if isinstance(results, dict) and 'f1_macro' in results:
                f1_scores.append(results['f1_macro'])
        metrics['nc_f1_macro'] = np.mean(f1_scores) if f1_scores else 0.0
    except Exception as e:
        logger.warning(f"Node classification failed: {e}")
        metrics['nc_f1_macro'] = 0.0
    
    # Link prediction
    try:
        lp_results = evaluate_link_prediction(
            embeddings=embedding,
            node_list=node_list,
            positive_edges=test_edges,
            negative_edges=test_non_edges,
            edge_feature_method='hadamard',
            classifier='logistic',
            test_size=0.3,
            random_state=42
        )
        metrics['lp_auc'] = lp_results.get('auc_roc', 0.0)
    except Exception as e:
        logger.warning(f"Link prediction failed: {e}")
        metrics['lp_auc'] = 0.0
    
    # Combined score (weighted average, link prediction is more reliable)
    metrics['combined_score'] = 0.3 * metrics['nc_f1_macro'] + 0.7 * metrics['lp_auc']
    
    return metrics


def suggest_params(trial, method: str) -> Dict[str, Any]:
    """Suggest hyperparameters for a method."""
    params = {}
    
    if method == "quvine_walks":
        params['num_views'] = trial.suggest_int('num_views', 2, 5)
        params['max_degree'] = trial.suggest_int('max_degree', 30, 100)
        params['degree_alpha'] = trial.suggest_float('degree_alpha', 0.3, 0.7)
        params['num_walks'] = trial.suggest_int('num_walks', 5, 20)
        params['walk_length'] = trial.suggest_int('walk_length', 40, 120)
        params['restart_prob'] = trial.suggest_float('restart_prob', 0.1, 0.3)
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['window'] = trial.suggest_int('window', 5, 15)
        params['negative'] = trial.suggest_int('negative', 3, 10)
        params['epochs'] = trial.suggest_int('epochs', 5, 15)
    
    elif method in ["baseline_filter_heat", "baseline_filter_poly"]:
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['filter_t'] = trial.suggest_float('filter_t', 1.0, 5.0)
        params['filter_order'] = trial.suggest_int('filter_order', 3, 10)
    
    elif method == "baseline_gcnmf":
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        params['mf_dim'] = trial.suggest_categorical('mf_dim', [32, 64, 128])
        params['gcnmf_layers'] = trial.suggest_int('gcnmf_layers', 1, 3)
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
        params['lr'] = trial.suggest_categorical('lr', [0.001, 0.01, 0.05])
        params['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 5e-4, 5e-3])
    
    elif method == "node2vec":
        params['dimensions'] = trial.suggest_categorical('dimensions', [64, 128, 256])
        params['walk_length'] = trial.suggest_int('walk_length', 40, 120)
        params['num_walks'] = trial.suggest_int('num_walks', 5, 20)
        params['p'] = trial.suggest_float('p', 0.5, 2.0)
        params['q'] = trial.suggest_float('q', 0.5, 2.0)
        params['window'] = trial.suggest_int('window', 5, 15)
    
    elif method == "netmf":
        params['dimensions'] = trial.suggest_categorical('dimensions', [64, 128, 256])
        params['window_size'] = trial.suggest_int('window_size', 5, 15)
        params['negative'] = trial.suggest_int('negative', 1, 10)
    
    elif method == "graphsage":
        params['dimensions'] = trial.suggest_categorical('dimensions', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        params['n_layers'] = trial.suggest_int('n_layers', 1, 3)
        params['epochs'] = trial.suggest_categorical('epochs', [50, 100, 200])
        params['lr'] = trial.suggest_categorical('lr', [0.001, 0.01, 0.05])
    
    elif method == "appnp":
        params['dimensions'] = trial.suggest_categorical('dimensions', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [32, 64, 128])
        params['n_layers'] = trial.suggest_int('n_layers', 1, 3)
        params['alpha'] = trial.suggest_float('alpha', 0.05, 0.2)
        params['K'] = trial.suggest_int('K', 5, 15)
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
        params['lr'] = trial.suggest_categorical('lr', [0.001, 0.01, 0.05])
    
    elif method == "gat_baseline":
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3, 3e-3])
        params['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.2, 0.4, 0.6])
        params['heads'] = trial.suggest_categorical('heads', [2, 4, 8])
        params['attn_dropout'] = trial.suggest_categorical('attn_dropout', [0.0, 0.2, 0.4, 0.6])
        params['concat_heads'] = trial.suggest_categorical('concat_heads', [True, False])
        params['residual'] = trial.suggest_categorical('residual', [True, False])
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
    
    elif method == "graphgps_baseline":
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3, 3e-3])
        params['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.2, 0.4, 0.6])
        params['gps_layers'] = trial.suggest_int('gps_layers', 2, 4)
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
        params['pe_dim'] = trial.suggest_categorical('pe_dim', [8, 16, 32])
        params['attn_dropout'] = trial.suggest_categorical('attn_dropout', [0.0, 0.2, 0.4])
        params['local_gnn'] = trial.suggest_categorical('local_gnn', ['sage', 'gcn', 'gat'])
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
    
    return params


def tune_method(
    method: str,
    graphs: List[nx.Graph],
    seeds_list: List[List[int]],
    test_edges_list: List[List[Tuple[int, int]]],
    test_non_edges_list: List[List[Tuple[int, int]]],
    n_trials: int,
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters for a method."""
    logger.info(f"Tuning {method} with {n_trials} trials on {len(graphs)} graphs")
    
    def objective(trial_or_params):
        """Objective function that works with both Optuna trials and dict params"""
        # If it's a dict, use it directly; otherwise get params from trial
        if isinstance(trial_or_params, dict):
            params = trial_or_params
        else:
            params = suggest_params(trial_or_params, method)
        
        scores = []
        
        for G, seeds, test_edges, test_non_edges in zip(
            graphs, seeds_list, test_edges_list, test_non_edges_list
        ):
            try:
                # Generate embedding
                if method == "quvine_walks":
                    embedding = run_quvine_walks(G, seeds, params)
                elif method == "baseline_filter_heat":
                    embedding = run_filter_embedding(G, seeds, params, "heat")
                elif method == "baseline_filter_poly":
                    embedding = run_filter_embedding(G, seeds, params, "poly")
                elif method == "baseline_gcnmf":
                    embedding = run_gcnmf_embedding(G, seeds, params)
                elif method == "node2vec":
                    embedding = run_node2vec(G, nodes=list(G.nodes()), **params)
                elif method == "netmf":
                    embedding = run_netmf(G, nodes=list(G.nodes()), **params)
                elif method == "graphsage":
                    embedding = run_graphsage(G, nodes=list(G.nodes()), **params)
                elif method == "appnp":
                    embedding = run_appnp(G, nodes=list(G.nodes()), **params)
                elif method == "gat_baseline":
                    if not GAT_AVAILABLE:
                        logger.warning(f"GAT not available, skipping {method}")
                        scores.append(0.0)
                        continue
                    gat_config = GATConfig(
                        hidden_dim=params.get('hidden_dim', 64),
                        output_dim=params.get('embedding_dim', 128),
                        num_layers=params.get('num_layers', 2),
                        heads=params.get('heads', 4),
                        dropout=params.get('dropout', 0.5),
                        attention_dropout=params.get('attn_dropout', 0.2),
                        residual=params.get('residual', True),
                    )
                    train_config = GATTrainConfig(
                        epochs=params.get('epochs', 200),
                        lr=params.get('lr', 0.005),
                        weight_decay=params.get('weight_decay', 5e-4),
                    )
                    embedding = generate_gat_embedding_by_method_name(
                        G,
                        method_name="gat_baseline",
                        embedding_dim=params.get('embedding_dim', 128),
                        nodelist=list(G.nodes()),
                        gat_config=gat_config,
                        train_config=train_config,
                    )
                elif method == "graphgps_baseline":
                    if not GRAPHGPS_AVAILABLE:
                        logger.warning(f"GraphGPS not available, skipping {method}")
                        scores.append(0.0)
                        continue
                    # Note: attn_dropout removed due to PyG version compatibility
                    gps_config = GraphGPSConfig(
                        hidden_dim=params.get('hidden_dim', 64),
                        output_dim=params.get('embedding_dim', 128),
                        num_layers=params.get('num_layers', 2),
                        heads=params.get('num_heads', 4),
                        dropout=params.get('dropout', 0.2),
                        attn_dropout=0.0,  # Set to 0.0 for compatibility
                        local_gnn=params.get('local_gnn', 'gcn'),
                        lap_pe_dim=params.get('pe_dim', 0),
                    )
                    train_config = GraphGPSTrainConfig(
                        epochs=params.get('epochs', 200),
                        lr=params.get('lr', 0.005),
                        weight_decay=params.get('weight_decay', 5e-4),
                    )
                    embedding = generate_graphgps_embedding_by_method_name(
                        G,
                        method_name="graphgps_baseline",
                        embedding_dim=params.get('embedding_dim', 128),
                        nodelist=list(G.nodes()),
                        gps_config=gps_config,
                        train_config=train_config,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Evaluate
                metrics = evaluate_embeddings(embedding, G, test_edges, test_non_edges)
                scores.append(metrics['combined_score'])
                
            except Exception as e:
                logger.warning(f"Trial failed for {method}: {e}")
                scores.append(0.0)
        
        return np.mean(scores)
    
    if OPTUNA_AVAILABLE:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
        best_score = study.best_value
    else:
        # Random search fallback
        best_params = None
        best_score = -np.inf
        
        for trial_idx in range(n_trials):
            # Create a simple trial object for random search
            class RandomTrial:
                def suggest_int(self, name, low, high):
                    return np.random.randint(low, high + 1)
                def suggest_float(self, name, low, high):
                    return np.random.uniform(low, high)
                def suggest_categorical(self, name, choices):
                    return np.random.choice(choices)
            
            trial = RandomTrial()
            params = suggest_params(trial, method)
            score = objective(params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"Trial {trial_idx + 1}/{n_trials}: score={score:.4f}")
    
    logger.info(f"Best score for {method}: {best_score:.4f}")
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='Local hyperparameter tuning test')
    parser.add_argument('--network-type', type=str, choices=NETWORK_TYPES + ['all'],
                        default='all', help='Network type to tune')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        help='Methods to tune')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials per method')
    parser.add_argument('--n-graphs', type=int, default=3,
                        help='Number of pilot graphs per network type')
    parser.add_argument('--n-nodes', type=int, default=100,
                        help='Number of nodes in pilot graphs')
    parser.add_argument('--output-dir', type=str, default='./tuning_test',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine network types to process
    if args.network_type == 'all':
        network_types = NETWORK_TYPES
    else:
        network_types = [args.network_type]
    
    logger.info(f"Starting hyperparameter tuning")
    logger.info(f"Network types: {network_types}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Trials per method: {args.n_trials}")
    logger.info(f"Output directory: {output_dir}")
    
    all_results = {}
    
    for network_type in network_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing network type: {network_type}")
        logger.info(f"{'='*60}")
        
        # Generate pilot graphs
        graphs = []
        seeds_list = []
        test_edges_list = []
        test_non_edges_list = []
        
        for i in range(args.n_graphs):
            seed = args.seed + i
            G = generate_pilot_graph(network_type, args.n_nodes, seed)
            
            # Select seeds (10% of nodes)
            n_seeds = max(5, G.number_of_nodes() // 10)
            seeds = list(np.random.RandomState(seed).choice(
                list(G.nodes()), size=n_seeds, replace=False
            ))
            
            # Split edges for link prediction
            train_graph, val_edges, test_edges, negative_edges = split_edges(
                G, test_ratio=0.2, val_ratio=0.0, seed=seed
            )
            test_non_edges = negative_edges
            
            graphs.append(G)
            seeds_list.append(seeds)
            test_edges_list.append(test_edges)
            test_non_edges_list.append(test_non_edges)
        
        # Tune each method
        network_results = {}
        for method in args.methods:
            try:
                best_params, best_score = tune_method(
                    method,
                    graphs,
                    seeds_list,
                    test_edges_list,
                    test_non_edges_list,
                    args.n_trials,
                )
                
                network_results[method] = {
                    'best_params': best_params,
                    'best_score': best_score,
                }
                
                logger.info(f"✓ {method}: score={best_score:.4f}")
                
            except Exception as e:
                logger.error(f"✗ {method} failed: {e}")
                traceback.print_exc()
                network_results[method] = {
                    'best_params': {},
                    'best_score': 0.0,
                    'error': str(e),
                }
        
        all_results[network_type] = network_results
        
        # Save intermediate results
        results_file = output_dir / f'{network_type}_tuning_results.json'
        with open(results_file, 'w') as f:
            json.dump(network_results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
        logger.info(f"Saved results to {results_file}")
    
    # Save combined results
    combined_file = output_dir / 'all_tuning_results.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x) if isinstance(x, np.floating) else str(x))
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Tuning complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*60}")
    
    # Print summary
    print("\n" + "="*60)
    print("TUNING SUMMARY")
    print("="*60)
    for network_type, network_results in all_results.items():
        print(f"\n{network_type}:")
        for method, result in network_results.items():
            score = result.get('best_score', 0.0)
            print(f"  {method:30s} score={score:.4f}")


if __name__ == '__main__':
    main()

