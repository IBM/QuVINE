#!/usr/bin/env python3
"""
Task-Specific Hyperparameter Tuning Script

Tunes hyperparameters separately for each task:
- node_classification
- link_prediction  
- node_ranking

Output JSON structure:
{
  "method_name": {
    "node_classification": {best_params, best_score},
    "link_prediction": {best_params, best_score},
    "node_ranking": {best_params, best_score}
  }
}

Usage:
    python scripts/tune_by_task.py --n-trials 10 --output-dir ./tuning_results
    python scripts/tune_by_task.py --network-type erdos_renyi --methods quvine_walks node2vec
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
from quvine.evaluation.ranking import SeedTargetEvaluator

# Try to import GAT and GraphGPS
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

# Network types and methods
NETWORK_TYPES = ["erdos_renyi", "modular"]
ALL_METHODS = [
    "quvine_walks",
    "baseline_filter_heat",
    "baseline_filter_poly",
    "baseline_gcnmf",
    "node2vec",
    "netmf",
    "graphsage",
    "appnp",
    "gat_baseline",
    "graphgps_baseline",
]

# Task types
TASKS = ["node_classification", "link_prediction", "node_ranking"]


def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component."""
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def generate_pilot_graph(network_type: str, n_nodes: int = 200, seed: int = 42) -> nx.Graph:
    """Generate a pilot graph for tuning. Default N=200 for better evaluation."""
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
            "kinds": ["rwr"],
            "num_walks": params.get("num_walks", 10),
            "walk_length": params.get("walk_length", 80),
            "restart_prob": params.get("restart_prob", 0.15),
            "max_iter": 100,
            "steps": params.get("steps", 5),
            "time": params.get("time", 1.0),
            "coin": params.get("coin", "grover"),
        },
        "embedding": {
            "dimensions": params.get("embedding_dim", 128),
            "window": params.get("window", 10),
            "negative": params.get("negative", 5),
            "epochs": params.get("epochs", 10),
            "workers": 4,
        },
    }


def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> np.ndarray:
    """Run QuVINE walks method."""
    cfg = make_quvine_cfg(params)
    vb = ViewBuilder(G, **cfg["views"])
    views = vb.build_views()
    cb = CorpusBuilder(views, **cfg["walks"])
    corpus = cb.build_corpus()
    embedding = corpus_to_embedding(corpus, **cfg["embedding"])
    return embedding


def run_filter_embedding(G: nx.Graph, seeds: List[int], params: Dict[str, Any], filter_type: str) -> np.ndarray:
    """Run filter-based embedding."""
    return generate_baseline_filter_embedding_wrapper(
        G,
        filter_type=filter_type,
        t=params.get('filter_t', 2.0),
        K=params.get('filter_order', 5),
        embedding_dim=params.get('embedding_dim', 128),
    )


def run_gcnmf_embedding(G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> np.ndarray:
    """Run GCN-MF embedding."""
    return generate_baseline_gcnmf_embedding(
        G,
        embedding_dim=params.get('embedding_dim', 128),
        hidden_dim=params.get('hidden_dim', 64),
        mf_dim=params.get('mf_dim', 64),
        n_layers=params.get('gcnmf_layers', 2),
        epochs=params.get('epochs', 200),
        lr=params.get('lr', 0.01),
        weight_decay=params.get('weight_decay', 5e-4),
    )


def evaluate_node_classification(embedding: np.ndarray, G: nx.Graph) -> float:
    """
    Evaluate node classification performance using 5 label strategies:
    1. Community-based: Louvain, Label Propagation
    2. Degree-based: Structural role binning
    3. Centrality-based: Betweenness, PageRank
    4. Core-periphery: K-core decomposition
    5. Homophily-based: Graph structure-aware labels
    """
    try:
        node_list = list(G.nodes())
        
        # Use evaluate_all_label_strategies which implements all 5 strategies
        nc_results = evaluate_all_label_strategies(
            G=G,
            embeddings=embedding,
            node_list=node_list,
            test_size=0.3,
            random_state=42
        )
        
        # Collect F1-macro scores from all strategies
        f1_scores = []
        for strategy, results in nc_results.items():
            if isinstance(results, dict) and 'f1_macro' in results:
                f1_scores.append(results['f1_macro'])
        
        # Return average F1-macro across all strategies
        return np.mean(f1_scores) if f1_scores else 0.0
    except Exception as e:
        logger.warning(f"Node classification failed: {e}")
        return 0.0


def evaluate_link_pred(embedding: np.ndarray, G: nx.Graph, test_edges: List, test_non_edges: List) -> float:
    """Evaluate link prediction performance."""
    try:
        node_list = list(G.nodes())
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
        return lp_results.get('auc_roc', 0.0)
    except Exception as e:
        logger.warning(f"Link prediction failed: {e}")
        return 0.0


def evaluate_node_rank(embedding: np.ndarray, G: nx.Graph, seeds: List[int]) -> float:
    """
    Evaluate node ranking performance using K=2 hop targets.
    Uses a simpler, more robust evaluation based on ranking correlation.
    """
    try:
        # Use seeds as query nodes and their K=2 hop neighbors as targets
        targets = set()
        for seed in seeds:
            if seed in G:
                # 1-hop neighbors
                neighbors_1hop = set(G.neighbors(seed))
                # 2-hop neighbors
                neighbors_2hop = set()
                for n1 in neighbors_1hop:
                    if n1 in G:
                        neighbors_2hop.update(G.neighbors(n1))
                
                # Combine all up to 2-hop
                targets.update(neighbors_1hop)
                targets.update(neighbors_2hop)
        
        # Remove seeds from targets
        targets = targets - set(seeds)
        
        if len(targets) < 10:
            # Not enough targets, return 0
            return 0.0
        
        # Compute ranking scores (cosine similarity from seeds)
        node_list = list(G.nodes())
        seed_indices = [node_list.index(s) for s in seeds if s in node_list]
        if not seed_indices:
            return 0.0
        
        # Mean seed embedding
        seed_emb = embedding[seed_indices].mean(axis=0)
        seed_norm = np.linalg.norm(seed_emb)
        if seed_norm < 1e-10:
            return 0.0
        
        # Compute similarity scores for all nodes
        all_scores = []
        for node in node_list:
            idx = node_list.index(node)
            node_emb = embedding[idx]
            node_norm = np.linalg.norm(node_emb)
            if node_norm > 1e-10:
                sim = np.dot(node_emb, seed_emb) / (node_norm * seed_norm)
            else:
                sim = 0.0
            all_scores.append((node, sim))
        
        # Sort by similarity (descending)
        all_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get top-K ranked nodes
        K = min(50, len(all_scores) // 2)  # Top 50 or half the nodes
        top_k_nodes = set([node for node, _ in all_scores[:K]])
        
        # Compute precision: how many of top-K are actual targets
        hits = len(top_k_nodes & targets)
        precision = hits / K if K > 0 else 0.0
        
        # Compute recall: how many targets are in top-K
        recall = hits / len(targets) if len(targets) > 0 else 0.0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        
        return f1
        
    except Exception as e:
        logger.warning(f"Node ranking failed: {e}")
        return 0.0


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
        params['residual'] = trial.suggest_categorical('residual', [True, False])
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
    
    elif method == "graphgps_baseline":
        params['embedding_dim'] = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        params['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        params['num_layers'] = trial.suggest_int('num_layers', 2, 4)
        params['lr'] = trial.suggest_categorical('lr', [1e-4, 3e-4, 1e-3, 3e-3])
        params['weight_decay'] = trial.suggest_categorical('weight_decay', [0, 1e-5, 1e-4, 1e-3])
        params['dropout'] = trial.suggest_categorical('dropout', [0.0, 0.2, 0.4, 0.6])
        params['num_heads'] = trial.suggest_categorical('num_heads', [2, 4, 8])
        params['pe_dim'] = trial.suggest_categorical('pe_dim', [0, 8, 16, 32])
        params['local_gnn'] = trial.suggest_categorical('local_gnn', ['sage', 'gcn', 'gat'])
        params['epochs'] = trial.suggest_categorical('epochs', [100, 200, 300])
    
    return params


def generate_embedding(method: str, G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Generate embedding for a method with given parameters."""
    try:
        if method == "quvine_walks":
            return run_quvine_walks(G, seeds, params)
        elif method == "baseline_filter_heat":
            return run_filter_embedding(G, seeds, params, "heat")
        elif method == "baseline_filter_poly":
            return run_filter_embedding(G, seeds, params, "poly")
        elif method == "baseline_gcnmf":
            return run_gcnmf_embedding(G, seeds, params)
        elif method == "node2vec":
            return run_node2vec(G, nodes=list(G.nodes()), **params)
        elif method == "netmf":
            return run_netmf(G, nodes=list(G.nodes()), **params)
        elif method == "graphsage":
            return run_graphsage(G, nodes=list(G.nodes()), **params)
        elif method == "appnp":
            return run_appnp(G, nodes=list(G.nodes()), **params)
        elif method == "gat_baseline":
            if not GAT_AVAILABLE:
                return None
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
            return generate_gat_embedding_by_method_name(
                G, 
                method_name="gat_baseline",
                embedding_dim=params.get('embedding_dim', 128),
                nodelist=list(G.nodes()),
                gat_config=gat_config,
                train_config=train_config,
            )
        elif method == "graphgps_baseline":
            if not GRAPHGPS_AVAILABLE:
                return None
            gps_config = GraphGPSConfig(
                hidden_dim=params.get('hidden_dim', 64),
                output_dim=params.get('embedding_dim', 128),
                num_layers=params.get('num_layers', 2),
                heads=params.get('num_heads', 4),
                dropout=params.get('dropout', 0.2),
                attn_dropout=0.0,
                local_gnn=params.get('local_gnn', 'gcn'),
                lap_pe_dim=params.get('pe_dim', 0),
            )
            train_config = GraphGPSTrainConfig(
                epochs=params.get('epochs', 200),
                lr=params.get('lr', 0.005),
                weight_decay=params.get('weight_decay', 5e-4),
            )
            return generate_graphgps_embedding_by_method_name(
                G,
                method_name="graphgps_baseline",
                embedding_dim=params.get('embedding_dim', 128),
                nodelist=list(G.nodes()),
                gps_config=gps_config,
                train_config=train_config,
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        logger.warning(f"Embedding generation failed for {method}: {e}")
        return None


def tune_method_for_task(
    method: str,
    task: str,
    graphs: List[nx.Graph],
    seeds_list: List[List[int]],
    test_edges_list: List[List[Tuple[int, int]]],
    test_non_edges_list: List[List[Tuple[int, int]]],
    n_trials: int,
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters for a method on a specific task."""
    logger.info(f"Tuning {method} for {task} with {n_trials} trials on {len(graphs)} graphs")
    
    def objective(trial_or_params):
        """Objective function for a specific task."""
        if isinstance(trial_or_params, dict):
            params = trial_or_params
        else:
            params = suggest_params(trial_or_params, method)
        
        scores = []
        for G, seeds, test_edges, test_non_edges in zip(
            graphs, seeds_list, test_edges_list, test_non_edges_list
        ):
            embedding = generate_embedding(method, G, seeds, params)
            if embedding is None:
                scores.append(0.0)
                continue
            
            # Evaluate based on task
            if task == "node_classification":
                score = evaluate_node_classification(embedding, G)
            elif task == "link_prediction":
                score = evaluate_link_pred(embedding, G, test_edges, test_non_edges)
            elif task == "node_ranking":
                score = evaluate_node_rank(embedding, G, seeds)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            scores.append(score)
        
        return np.mean(scores)
    
    if OPTUNA_AVAILABLE:
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
    else:
        # Random search fallback
        best_params = None
        best_score = -np.inf
        
        for trial_idx in range(n_trials):
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
            
            logger.info(f"  Trial {trial_idx + 1}/{n_trials}: score={score:.4f}")
    
    logger.info(f"  Best {task} score for {method}: {best_score:.4f}")
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='Task-specific hyperparameter tuning')
    parser.add_argument('--network-type', type=str, choices=NETWORK_TYPES + ['all'],
                        default='all', help='Network type to tune')
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        help='Methods to tune')
    parser.add_argument('--n-trials', type=int, default=10,
                        help='Number of trials per method per task')
    parser.add_argument('--n-graphs', type=int, default=3,
                        help='Number of graphs to use for tuning')
    parser.add_argument('--n-nodes', type=int, default=100,
                        help='Number of nodes in each graph')
    parser.add_argument('--output-dir', type=str, default='tuning_by_task',
                        help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    network_types = NETWORK_TYPES if args.network_type == 'all' else [args.network_type]
    
    logger.info("Starting task-specific hyperparameter tuning")
    logger.info(f"Network types: {network_types}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Tasks: {TASKS}")
    logger.info(f"Trials per method per task: {args.n_trials}")
    logger.info(f"Output directory: {args.output_dir}")
    
    for network_type in network_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing network type: {network_type}")
        logger.info(f"{'='*60}")
        
        # Generate graphs
        graphs = []
        seeds_list = []
        test_edges_list = []
        test_non_edges_list = []
        
        for i in range(args.n_graphs):
            G = generate_pilot_graph(network_type, args.n_nodes, args.seed + i)
            graphs.append(G)
            
            # Generate seeds (10% of nodes)
            all_nodes = list(G.nodes())
            n_seeds = max(5, len(all_nodes) // 10)
            seeds = np.random.choice(all_nodes, size=n_seeds, replace=False).tolist()
            seeds_list.append(seeds)
            
            # Split edges for link prediction
            train_graph, val_edges, test_edges, negative_edges = split_edges(
                G, test_ratio=0.2, val_ratio=0.0, seed=args.seed + i
            )
            test_edges_list.append(test_edges)
            test_non_edges_list.append(negative_edges)
        
        # Tune each method for each task
        results = {}
        
        for method in args.methods:
            if method == "gat_baseline" and not GAT_AVAILABLE:
                logger.warning(f"Skipping {method} - not available")
                continue
            if method == "graphgps_baseline" and not GRAPHGPS_AVAILABLE:
                logger.warning(f"Skipping {method} - not available")
                continue
            
            logger.info(f"\nTuning {method} for all tasks")
            results[method] = {}
            
            for task in TASKS:
                best_params, best_score = tune_method_for_task(
                    method, task, graphs, seeds_list,
                    test_edges_list, test_non_edges_list, args.n_trials
                )
                
                results[method][task] = {
                    "best_params": best_params,
                    "best_score": float(best_score)
                }
                
                logger.info(f"✓ {method} - {task}: score={best_score:.4f}")
        
        # Save results - convert numpy types to native Python types
        def convert_to_native(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            elif hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        results_native = convert_to_native(results)
        output_file = os.path.join(args.output_dir, f"{network_type}_tuning_by_task.json")
        with open(output_file, 'w') as f:
            json.dump(results_native, f, indent=2)
        logger.info(f"\nSaved results to {output_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Tuning complete!")
    logger.info(f"Results saved to {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

