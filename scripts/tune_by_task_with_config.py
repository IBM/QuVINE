#!/usr/bin/env python3
"""
Task-Specific Hyperparameter Tuning Script with YAML Configuration

Reads hyperparameter search spaces and fixed parameters from tuning_config.yaml
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
    # Use config file (default: tuning_config.yaml)
    python scripts/tune_by_task_with_config.py --config tuning_config.yaml
    
    # Override specific settings
    python scripts/tune_by_task_with_config.py --config tuning_config.yaml --n-trials 100
    
    # Test with specific methods
    python scripts/tune_by_task_with_config.py --methods quvine_walks node2vec --n-trials 5
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
import yaml

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
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_powerlaw_cluster,
    generate_stochastic_block_model,
    generate_random_geometric,
    generate_core_periphery,
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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        # Try relative to script directory first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_config_path = os.path.join(script_dir, os.path.basename(config_path))
        if os.path.exists(abs_config_path):
            config_path = abs_config_path
        else:
            # Try relative to current working directory
            cwd_config_path = os.path.abspath(config_path)
            if os.path.exists(cwd_config_path):
                config_path = cwd_config_path
            else:
                raise FileNotFoundError(
                    f"Config file not found. Tried:\n"
                    f"  - {abs_config_path}\n"
                    f"  - {cwd_config_path}\n"
                    f"Please provide the correct path to the config file."
                )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    return config


def _largest_cc(G: nx.Graph) -> nx.Graph:
    """Return the largest connected component."""
    cc = max(nx.connected_components(G), key=len)
    return G.subgraph(cc).copy()


def generate_pilot_graph(network_type: str, config: Dict[str, Any], seed: int = 42) -> nx.Graph:
    """Generate a pilot graph for tuning using config parameters."""
    graph_config = config['fixed_params']['graph']
    n_nodes = graph_config['n_nodes']
    
    logger.info(f"Generating {network_type} graph with {n_nodes} nodes, seed={seed}")
    
    if network_type == "erdos_renyi":
        p = graph_config['erdos_renyi']['p']
        G = generate_erdos_renyi(n_nodes, p=p, seed=seed)
    
    elif network_type == "modular" or network_type == "modular_strong":
        mod_config = graph_config['modular']
        G, _ = generate_modular_network(
            num_communities=mod_config['n_communities'],
            nodes_per_community=n_nodes // mod_config['n_communities'],
            p_intra=mod_config['p_in'],
            p_inter=mod_config['p_out'],
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
    
    elif network_type == "watts_strogatz_high_p":
        G = generate_watts_strogatz(n_nodes, k=6, p=0.5, seed=seed)
    
    elif network_type == "watts_strogatz_low_p":
        G = generate_watts_strogatz(n_nodes, k=6, p=0.05, seed=seed)
    
    elif network_type == "random_geometric":
        G = generate_random_geometric(n_nodes, radius=0.18, seed=seed)
    
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
        G = generate_stochastic_block_model(
            sizes=[n_nodes // 4] * 4,
            p_matrix=[[0.5, 0.05, 0.05, 0.05],
                     [0.05, 0.5, 0.05, 0.05],
                     [0.05, 0.05, 0.5, 0.05],
                     [0.05, 0.05, 0.05, 0.5]],
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
            "vector_size": params.get("embedding_dim", 128),
            "window": params.get("window", 10),
            "sg": 1,
            "negative": params.get("negative", 5),
            "min_count": 1,
            "workers": 4,
            "epochs": params.get("epochs", 10),
        },
    }


def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> np.ndarray:
    """Run QuVINE walks method."""
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
        
        # Collect all walks for this seed
        all_walks = []
        
        # Generate walks for each view
        for view_nodes in view_node_sets:
            # Create subgraph from node set
            view_graph = G.subgraph(view_nodes)
            
            # Generate walks on this view
            view_walks_dict = walker.run(view_graph, seed, view_nodes=list(view_nodes))
            
            # Combine all walk types and convert node IDs to strings
            for walk_type, walks in view_walks_dict.items():
                # Convert each walk's nodes to strings
                str_walks = [[str(node) for node in walk] for walk in walks]
                all_walks.extend(str_walks)
        
        # Add walks for this seed to corpus
        corpus_builder.add(seed, all_walks)
    
    # Build corpus and generate embedding
    corpus = corpus_builder.build()
    
    # Convert node IDs to strings for corpus_to_embedding
    nodes_str = [str(n) for n in G.nodes()]
    embedding = corpus_to_embedding(
        corpus,
        nodes=nodes_str,
        **cfg_dict["embedding"]
    )
    
    return embedding


def run_filter_embedding(G: nx.Graph, seeds: List[int], params: Dict[str, Any], filter_type: str) -> np.ndarray:
    """Run filter-based embedding."""
    return generate_baseline_filter_embedding_wrapper(
        G,
        filter_type=filter_type,
        t=params.get('tau', 2.0),
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
        n_layers=params.get('n_layers', 2),
        epochs=params.get('epochs', 200),
        lr=params.get('learning_rate', 0.01),
        weight_decay=params.get('weight_decay', 5e-4),
    )


def evaluate_node_classification(embedding: np.ndarray, G: nx.Graph, config: Dict[str, Any]) -> float:
    """Evaluate node classification performance using config parameters."""
    try:
        eval_config = config['fixed_params']['evaluation']['node_classification']
        node_list = list(G.nodes())
        
        nc_results = evaluate_all_label_strategies(
            G=G,
            embeddings=embedding,
            node_list=node_list,
            test_size=eval_config['test_size'],
            random_state=42
        )
        
        f1_scores = []
        for strategy, results in nc_results.items():
            if isinstance(results, dict) and 'f1_macro' in results:
                f1_scores.append(results['f1_macro'])
        
        return np.mean(f1_scores) if f1_scores else 0.0
    except Exception as e:
        logger.warning(f"Node classification failed: {e}")
        return 0.0


def evaluate_link_pred(embedding: np.ndarray, G: nx.Graph, test_edges: List, test_non_edges: List, config: Dict[str, Any]) -> float:
    """Evaluate link prediction performance using config parameters."""
    try:
        eval_config = config['fixed_params']['evaluation']['link_prediction']
        node_list = list(G.nodes())
        lp_results = evaluate_link_prediction(
            embeddings=embedding,
            node_list=node_list,
            positive_edges=test_edges,
            negative_edges=test_non_edges,
            edge_feature_method=eval_config['edge_feature_method'],
            classifier='logistic',
            test_size=0.3,
            random_state=42
        )
        return lp_results.get('auc_roc', 0.0)
    except Exception as e:
        logger.warning(f"Link prediction failed: {e}")
        return 0.0


def evaluate_node_rank(embedding: np.ndarray, G: nx.Graph, seeds: List[int], config: Dict[str, Any]) -> float:
    """Evaluate node ranking performance using config parameters."""
    try:
        eval_config = config['fixed_params']['evaluation']['node_ranking']
        k_hops = eval_config['k_hops']
        top_k = eval_config['top_k']
        
        # Use seeds as query nodes and their K-hop neighbors as targets
        targets = set()
        for seed in seeds:
            if seed in G:
                # Collect k-hop neighbors
                current_level = {seed}
                for _ in range(k_hops):
                    next_level = set()
                    for node in current_level:
                        if node in G:
                            next_level.update(G.neighbors(node))
                    targets.update(next_level)
                    current_level = next_level
        
        # Remove seeds from targets
        targets = targets - set(seeds)
        
        if len(targets) < 10:
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
        K = min(top_k, len(all_scores) // 2)
        top_k_nodes = set([node for node, _ in all_scores[:K]])
        
        # Compute precision and recall
        hits = len(top_k_nodes & targets)
        precision = hits / K if K > 0 else 0.0
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


def suggest_params_from_config(trial, method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Suggest hyperparameters for a method from config file."""
    params = {}
    
    if method not in config['hyperparameters']:
        raise ValueError(f"Method {method} not found in config hyperparameters")
    
    method_config = config['hyperparameters'][method]
    
    for param_name, param_values in method_config.items():
        if isinstance(param_values, list):
            if all(isinstance(v, int) for v in param_values):
                # Categorical integer
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif all(isinstance(v, float) for v in param_values):
                # Categorical float or range
                if len(param_values) == 2:
                    # Assume range [min, max]
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
                else:
                    # Categorical
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif all(isinstance(v, str) for v in param_values):
                # Categorical string
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                # Mixed types - treat as categorical
                params[param_name] = trial.suggest_categorical(param_name, param_values)
        else:
            # Single value - use as is
            params[param_name] = param_values
    
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
            n2v_params = {
                'dimensions': params.get('embedding_dim', 128),
                'walk_length': params.get('walk_length', 80),
                'num_walks': params.get('num_walks', 10),
                'p': params.get('p', 1.0),
                'q': params.get('q', 1.0),
                'window': params.get('window_size', 10),  # API uses 'window' not 'window_size'
            }
            return run_node2vec(G, nodes=list(G.nodes()), **n2v_params)
        elif method == "netmf":
            netmf_params = {
                'dimensions': params.get('embedding_dim', 128),
                'window': params.get('window_size', 10),  # API uses 'window' not 'window_size'
                'negative': params.get('negative_samples', 5),
            }
            return run_netmf(G, nodes=list(G.nodes()), **netmf_params)
        elif method == "graphsage":
            sage_params = {
                'dimensions': params.get('embedding_dim', 128),
                'hidden_dim': params.get('hidden_dim', 128),
                'n_layers': params.get('n_layers', 2),
                'epochs': params.get('epochs', 100),
                'lr': params.get('learning_rate', 0.01),
            }
            return run_graphsage(G, nodes=list(G.nodes()), **sage_params)
        elif method == "appnp":
            appnp_params = {
                'dimensions': params.get('embedding_dim', 128),
                'hidden_dim': params.get('hidden_dim', 64),
                'n_layers': params.get('n_layers', 2),
                'alpha': params.get('alpha', 0.1),
                'K': params.get('k_hops', 10),
                'epochs': params.get('epochs', 200),
                'lr': params.get('learning_rate', 0.01),
            }
            return run_appnp(G, nodes=list(G.nodes()), **appnp_params)
        elif method == "gat_baseline":
            if not GAT_AVAILABLE:
                return None
            gat_config = GATConfig(
                hidden_dim=params.get('hidden_dim', 64),
                output_dim=params.get('embedding_dim', 128),
                num_layers=params.get('n_layers', 2),
                heads=params.get('n_heads', 4),
                dropout=params.get('dropout', 0.5),
                attention_dropout=params.get('attn_dropout', 0.2),
                residual=True,
            )
            train_config = GATTrainConfig(
                epochs=params.get('epochs', 200),
                lr=params.get('learning_rate', 0.005),
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
                num_layers=params.get('n_layers', 2),
                heads=params.get('n_heads', 4),
                dropout=params.get('dropout', 0.2),
                attn_dropout=0.0,
                local_gnn=params.get('mpnn_type', 'gcn'),
                lap_pe_dim=0,
            )
            train_config = GraphGPSTrainConfig(
                epochs=params.get('epochs', 200),
                lr=params.get('learning_rate', 0.005),
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


def get_n_trials_for_method(method: str, config: Dict[str, Any], override_trials: Optional[int] = None) -> int:
    """
    Get the number of trials for a method based on configuration.
    
    Uses adaptive trials if enabled, otherwise uses base_trials or override.
    """
    if override_trials is not None:
        return override_trials
    
    optuna_config = config.get('optuna', {})
    
    # Check if adaptive trials is enabled
    if optuna_config.get('adaptive_trials', False):
        method_trials = optuna_config.get('method_trials', {})
        if method in method_trials:
            trials = method_trials[method]
            logger.info(f"Using adaptive trials for {method}: {trials} trials")
            return trials
        else:
            # Fallback to base_trials if method not in config
            base = optuna_config.get('base_trials', 50)
            logger.warning(f"Method {method} not in method_trials config, using base_trials: {base}")
            return base
    else:
        # Use base_trials if adaptive is disabled
        return optuna_config.get('base_trials', 50)


def tune_method_for_task(
    method: str,
    task: str,
    graphs: List[nx.Graph],
    seeds_list: List[List[int]],
    test_edges_list: List[List[Tuple[int, int]]],
    test_non_edges_list: List[List[Tuple[int, int]]],
    config: Dict[str, Any],
    n_trials: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters for a method on a specific task."""
    # Get adaptive trials if not overridden
    actual_trials = get_n_trials_for_method(method, config, n_trials)
    logger.info(f"Tuning {method} for {task} with {actual_trials} trials on {len(graphs)} graphs")
    
    def objective(trial_or_params):
        """Objective function for a specific task."""
        if isinstance(trial_or_params, dict):
            params = trial_or_params
        else:
            params = suggest_params_from_config(trial_or_params, method, config)
        
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
                score = evaluate_node_classification(embedding, G, config)
            elif task == "link_prediction":
                score = evaluate_link_pred(embedding, G, test_edges, test_non_edges, config)
            elif task == "node_ranking":
                score = evaluate_node_rank(embedding, G, seeds, config)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            scores.append(score)
        
        return np.mean(scores)
    
    if OPTUNA_AVAILABLE:
        optuna_config = config.get('optuna', {})
        sampler_name = optuna_config.get('sampler', 'TPE')
        
        if sampler_name == 'TPE':
            sampler = optuna.samplers.TPESampler(
                n_startup_trials=optuna_config.get('n_startup_trials', 10)
            )
        else:
            sampler = optuna.samplers.RandomSampler()
        
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
    else:
        # Random search fallback
        best_params = None
        best_score = -np.inf
        
        for trial_idx in range(actual_trials):
            class RandomTrial:
                def suggest_int(self, name, low, high):
                    return np.random.randint(low, high + 1)
                def suggest_float(self, name, low, high):
                    return np.random.uniform(low, high)
                def suggest_categorical(self, name, choices):
                    return np.random.choice(choices)
            
            trial = RandomTrial()
            params = suggest_params_from_config(trial, method, config)
            score = objective(params)
            
            if score > best_score:
                best_score = score
                best_params = params
            
            logger.info(f"  Trial {trial_idx + 1}/{actual_trials}: score={score:.4f}")
    
    logger.info(f"  Best {task} score for {method}: {best_score:.4f}")
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description='Task-specific hyperparameter tuning with config file')
    # Default to tuning_config.yaml in the same directory as this script
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuning_config.yaml')
    parser.add_argument('--config', type=str, default=default_config,
                        help='Path to YAML configuration file')
    parser.add_argument('--network-type', type=str,
                        choices=[
                            'erdos_renyi', 'modular',
                            'watts_strogatz_high_p', 'watts_strogatz_low_p',
                            'random_geometric', 'modular_strong', 'modular_medium',
                            'modular_many_communities', 'core_periphery',
                            'scale_free', 'powerlaw_cluster', 'stochastic_block_model',
                            'all'
                        ],
                        default=None, help='Network type to tune (overrides config)')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Methods to tune (overrides config)')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Number of trials per method per task (overrides config)')
    parser.add_argument('--n-graphs', type=int, default=None,
                        help='Number of graphs to use for tuning (overrides config)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.network_type is not None:
        if args.network_type == 'all':
            # Use all available network types
            network_types = [
                'erdos_renyi', 'watts_strogatz_high_p', 'watts_strogatz_low_p',
                'random_geometric', 'modular_strong', 'modular_medium',
                'modular_many_communities', 'core_periphery', 'scale_free',
                'powerlaw_cluster', 'stochastic_block_model'
            ]
        else:
            network_types = [args.network_type]
    else:
        network_types = config['experiment']['network_types']
    
    if args.methods is not None:
        methods = args.methods
    else:
        methods = config['methods']['all']
    
    # n_trials will be None if not specified, allowing adaptive trials to work
    n_trials = args.n_trials  # Can be None
    n_graphs = args.n_graphs if args.n_graphs is not None else config['experiment']['n_graphs']
    output_dir = args.output_dir if args.output_dir is not None else config['experiment']['output_dir']
    seed = args.seed if args.seed is not None else config['fixed_params']['training']['random_seed']
    
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = ["node_classification", "link_prediction", "node_ranking"]
    
    logger.info("Starting task-specific hyperparameter tuning with config")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Network types: {network_types}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Trials per method per task: {n_trials}")
    logger.info(f"Graphs per trial: {n_graphs}")
    logger.info(f"Output directory: {output_dir}")
    
    for network_type in network_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing network type: {network_type}")
        logger.info(f"{'='*60}")
        
        # Generate graphs
        graphs = []
        seeds_list = []
        test_edges_list = []
        test_non_edges_list = []
        
        for i in range(n_graphs):
            G = generate_pilot_graph(network_type, config, seed + i)
            graphs.append(G)
            
            # Generate seeds (10% of nodes)
            all_nodes = list(G.nodes())
            n_seeds = max(5, len(all_nodes) // 10)
            seeds = np.random.choice(all_nodes, size=n_seeds, replace=False).tolist()
            seeds_list.append(seeds)
            
            # Split edges for link prediction
            test_ratio = config['fixed_params']['evaluation']['link_prediction']['test_ratio']
            train_graph, val_edges, test_edges, negative_edges = split_edges(
                G, test_ratio=test_ratio, val_ratio=0.0, seed=seed + i
            )
            test_edges_list.append(test_edges)
            test_non_edges_list.append(negative_edges)
        
        # Tune each method for each task
        results = {}
        
        for method in methods:
            if method == "gat_baseline" and not GAT_AVAILABLE:
                logger.warning(f"Skipping {method} - not available")
                continue
            if method == "graphgps_baseline" and not GRAPHGPS_AVAILABLE:
                logger.warning(f"Skipping {method} - not available")
                continue
            
            logger.info(f"\nTuning {method} for all tasks")
            results[method] = {}
            
            for task in tasks:
                # Pass n_trials only if explicitly set via command line
                # Otherwise, use adaptive trials from config
                trials_to_use = n_trials if args.n_trials is not None else None
                best_params, best_score = tune_method_for_task(
                    method, task, graphs, seeds_list,
                    test_edges_list, test_non_edges_list, config, trials_to_use
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
        output_file = os.path.join(output_dir, f"{network_type}_tuning_by_task.json")
        with open(output_file, 'w') as f:
            json.dump(results_native, f, indent=2)
        logger.info(f"\nSaved results to {output_file}")
    
    logger.info(f"\n{'='*60}")
    logger.info("Tuning complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()

# Made with Bob
