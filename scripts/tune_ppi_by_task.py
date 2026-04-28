#!/usr/bin/env python3
"""
PPI Network Hyperparameter Tuning Script

Tunes hyperparameters for embedding methods on real PPI networks with GWAS data.
Optimizes separately for each task:
- node_ranking (using GWAS seeds and targets)
- node_classification (using 7 label generation strategies)
- link_prediction (using hadamard edge features)

Networks: STRING, BioPlex3, HumanNet, PCNet, ProteomeHD
Diseases: asthma, autism, schizophrenia

Output JSON structure:
{
  "network_disease": {
    "method_name": {
      "node_classification": {best_params, best_score},
      "link_prediction": {best_params, best_score},
      "node_ranking": {best_params, best_score}
    }
  }
}

Usage:
    # Tune all methods on STRING network with asthma
    python scripts/tune_ppi_by_task.py --network STRING --disease asthma
    
    # Tune specific method
    python scripts/tune_ppi_by_task.py --network STRING --disease asthma --methods quvine_fused
    
    # Override trials
    python scripts/tune_ppi_by_task.py --network STRING --disease asthma --n-trials 20
"""

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
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

from quvine.data.subgraph import subsample_nodes
from quvine.views.generator import ViewBuilder
from quvine.corpus.builder import CorpusBuilder
from quvine.embedding.word2vec import corpus_to_embedding
from quvine.fusion.fuse import fuse_embeddings
from quvine.baselines.node2vec import run_node2vec
from quvine.baselines.netmf import run_netmf
from quvine.baselines.graphsage import run_graphsage
from quvine.baselines.gcn_mf import (
    generate_baseline_gcnmf_embedding,
    generate_baseline_filter_embedding_wrapper,
    generate_quvine_gcnmf_embedding,
)
from quvine.embedding.quantum_filters import (
    generate_quvine_heat_embedding,
    generate_quvine_poly_embedding,
)
from quvine.evaluation.classification import evaluate_all_label_strategies
from quvine.evaluation.link_prediction import split_edges, evaluate_link_prediction
from quvine.evaluation.ranking import SeedTargetEvaluator

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
    if not os.path.isabs(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        abs_config_path = os.path.join(script_dir, os.path.basename(config_path))
        if os.path.exists(abs_config_path):
            config_path = abs_config_path
        else:
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


def load_ppi_network(network_path: str) -> nx.Graph:
    """Load PPI network from edge list CSV."""
    logger.info(f"Loading network from {network_path}")
    df = pd.read_csv(network_path, usecols=[0, 1], dtype=str)
    df.columns = ['node1', 'node2']
    df[['node1', 'node2']] = df[['node1', 'node2']].apply(pd.to_numeric, errors='coerce')
    
    n_before = len(df)
    df = df.dropna(subset=['node1', 'node2'])
    if n_before > len(df):
        logger.info(f'Dropped {n_before - len(df)} rows with non-integer node IDs')
    
    df[['node1', 'node2']] = df[['node1', 'node2']].astype(int)
    
    G = nx.from_pandas_edgelist(df, source='node1', target='node2')
    G = nx.convert_node_labels_to_integers(G, label_attribute='ncbi_id')
    logger.info(f'Loaded network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
    
    return G


def load_gwas_data(seeds_path: str, targets_path: str, G: nx.Graph) -> Tuple[List[int], List[int]]:
    """Load GWAS seeds and targets and map to graph nodes."""
    with open(seeds_path) as f:
        raw_seeds = json.load(f)
    with open(targets_path) as f:
        raw_targets = json.load(f)
    
    # Create mapping from NCBI ID to graph node
    ncbi_to_node = {G.nodes[v]['ncbi_id']: v for v in G.nodes() if 'ncbi_id' in G.nodes[v]}
    
    seeds = [ncbi_to_node[int(g)] for g in raw_seeds if int(g) in ncbi_to_node]
    targets = [ncbi_to_node[int(g)] for g in raw_targets if int(g) in ncbi_to_node]
    
    logger.info(f'Mapped {len(seeds)}/{len(raw_seeds)} seeds, {len(targets)}/{len(raw_targets)} targets')
    
    if len(seeds) == 0 or len(targets) == 0:
        raise ValueError("No seeds or targets mapped to network")
    
    return seeds, targets


def subsample_ppi_network(
    G: nx.Graph,
    seeds: List[int],
    targets: List[int],
    max_nodes: int,
    config: Dict[str, Any],
    seed: int = 42
) -> Tuple[nx.Graph, List[int], List[int]]:
    """Subsample PPI network to max_nodes while preserving seeds and targets."""
    if G.number_of_nodes() <= max_nodes:
        logger.info(f"Network fits within {max_nodes} nodes - no subsampling needed")
        return G, seeds, targets
    
    graph_config = config['fixed_params']['graph']
    radius = graph_config.get('radius', 2)
    degree_matched_fill = graph_config.get('degree_matched_fill', True)
    degree_matched_trim = graph_config.get('degree_matched_trim', True)
    
    logger.info(f'Subsampling to {max_nodes} nodes (seed={seed}) with radius={radius}')
    
    # Guard: if combined seeds+targets exceed budget, trim targets
    combined = list(dict.fromkeys(seeds + targets))
    if len(combined) > max_nodes:
        logger.warning(f'{len(combined)} protected nodes > max_nodes={max_nodes}. Trimming targets.')
        seeds = seeds[:max_nodes // 2]
        targets = targets[:max_nodes - len(seeds)]
        combined = list(dict.fromkeys(seeds + targets))
        logger.info(f'After trim: {len(seeds)} seeds, {len(targets)} targets')
    
    rng = np.random.default_rng(seed)
    
    G_sub = subsample_nodes(
        G=G,
        seeds=seeds,
        targets=targets,
        max_nodes=max_nodes,
        radius=radius,
        rng=rng,
        degree_matched_fill=degree_matched_fill,
        degree_matched_trim=degree_matched_trim,
        degree_reference_nodes=list(G.nodes()),
    )
    
    G_sub = nx.convert_node_labels_to_integers(G_sub)
    logger.info(f'Subsampled: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges')
    
    # Map seeds and targets to new node IDs
    ncbi_to_node_sub = {G_sub.nodes[v].get('ncbi_id'): v
                        for v in G_sub.nodes() if 'ncbi_id' in G_sub.nodes[v]}
    
    seeds_sub = [ncbi_to_node_sub[G.nodes[s]['ncbi_id']]
                 for s in seeds
                 if G.nodes[s]['ncbi_id'] in ncbi_to_node_sub]
    targets_sub = [ncbi_to_node_sub[G.nodes[t]['ncbi_id']]
                   for t in targets
                   if G.nodes[t]['ncbi_id'] in ncbi_to_node_sub]
    
    logger.info(f'After subsampling: {len(seeds_sub)} seeds, {len(targets_sub)} targets retained')
    
    if len(seeds_sub) == 0 or len(targets_sub) == 0:
        raise ValueError("No seeds or targets in subgraph")
    
    return G_sub, seeds_sub, targets_sub


def run_quvine_walks(G: nx.Graph, seeds: List[int], params: Dict[str, Any], walk_type: str = 'rwr') -> np.ndarray:
    """Run QuVINE with specified quantum walk type.
    
    Args:
        G: NetworkX graph
        seeds: List of seed nodes
        params: Hyperparameter dictionary
        walk_type: Type of quantum walk ('rwr', 'ctqw', or 'dtqw')
    """
    from types import SimpleNamespace
    from quvine.walks.base import BaseWalker
    
    embedding_dim = params.get('embedding_dim', 128)
    num_views = params.get('num_views', 4)
    walk_length = params.get('walk_length', 40)
    num_walks = params.get('num_walks', 40)
    window_size = params.get('window_size', 10)
    negative_samples = params.get('negative_samples', 5)
    epochs = params.get('epochs', 20)
    restart_prob = params.get('restart_prob', 0.15)
    max_degree = params.get('max_degree', 100)
    degree_alpha = params.get('degree_alpha', 0.5)
    steps = params.get('steps', 5)
    time = params.get('time', 1.0)
    coin = params.get('coin', 'grover')
    
    # Create configuration for walks
    cfg_dict = {
        "views": {
            "num_views": num_views,
            "constrained": True,
            "max_degree": max_degree,
            "max_nodes": 500,
            "max_edges": 2000,
            "degree_norm": True,
            "degree_alpha": degree_alpha,
        },
        "walks": {
            "kinds": [walk_type],  # Use specified walk type
            "num_walks": num_walks,
            "walk_length": walk_length,
            "restart_prob": restart_prob,
            "max_iter": 100,
            "steps": steps,
            "time": time,
            "coin": coin,
        },
        "embedding": {
            "vector_size": embedding_dim,
            "window": window_size,
            "sg": 1,
            "negative": negative_samples,
            "min_count": 1,
            "workers": 4,
            "epochs": epochs,
        },
    }
    
    # Convert dict to nested namespace
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
        view_node_sets = view_builder.build(G, seed)
        all_walks = []
        
        for view_nodes in view_node_sets:
            view_graph = G.subgraph(view_nodes)
            view_walks_dict = walker.run(view_graph, seed, view_nodes=list(view_nodes))
            
            for wtype, walks in view_walks_dict.items():
                str_walks = [[str(node) for node in walk] for walk in walks]
                all_walks.extend(str_walks)
        
        corpus_builder.add(seed, all_walks)
    
    # Build corpus and generate embedding
    corpus = corpus_builder.build()
    nodes_str = [str(n) for n in G.nodes()]
    embedding = corpus_to_embedding(
        corpus,
        nodes=nodes_str,
        **cfg_dict["embedding"]
    )
    
    return embedding


def run_quvine_filter(G: nx.Graph, seeds: List[int], params: Dict[str, Any], filter_type: str) -> np.ndarray:
    """Run QuVINE with filter-based diffusion."""
    if filter_type == 'heat':
        return generate_quvine_heat_embedding(
            G=G,
            embedding_dim=params.get('embedding_dim', 128),
            tau=params.get('tau', 2.0),
            filter_order=params.get('filter_order', 5),
        )
    elif filter_type == 'poly':
        return generate_quvine_poly_embedding(
            G=G,
            embedding_dim=params.get('embedding_dim', 128),
            K=params.get('filter_order', 5),
            alpha=params.get('alpha', 0.5),
        )
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def run_quvine_gcnmf(G: nx.Graph, seeds: List[int], params: Dict[str, Any], diffusion_type: str) -> np.ndarray:
    """Run QuVINE GCN-MF with quantum calibration."""
    # Generate quantum targets from seeds
    q_targets = seeds  # Simplified - in practice would use quantum walk
    
    embedding, _ = generate_quvine_gcnmf_embedding(
        G=G,
        q_targets=q_targets,
        embedding_dim=params.get('embedding_dim', 128),
        diffusion_type=diffusion_type,
        t_star=params.get('tau', 2.0) if diffusion_type == 'heat' else None,
        K=params.get('filter_order', 5) if diffusion_type == 'poly' else 4,
        poly_coeffs=None,  # Will use default polynomial coefficients
        ridge=1e-6,
        hidden_dim=params.get('hidden_dim', 128),
        mf_dim=params.get('mf_dim', 64),
        n_layers=params.get('n_layers', 2),
        epochs=params.get('epochs', 200),
        lr=params.get('learning_rate', 0.01),
        weight_decay=params.get('weight_decay', 5e-4),
        normalize_laplacian=True,
        random_state=42,
    )
    return embedding


def generate_embedding(method: str, G: nx.Graph, seeds: List[int], params: Dict[str, Any]) -> Optional[np.ndarray]:
    """Generate embedding for a method with given parameters."""
    try:
        # Quantum walk variants (unified 12-method configuration)
        if method == "quvine_rwr":
            return run_quvine_walks(G, seeds, params, walk_type='rwr')
        elif method == "quvine_ctqw":
            return run_quvine_walks(G, seeds, params, walk_type='ctqw')
        elif method == "quvine_dtqw":
            return run_quvine_walks(G, seeds, params, walk_type='dtqw')
        
        # Filter baselines (split from baseline_filter)
        elif method == "baseline_filter_heat":
            return generate_baseline_filter_embedding_wrapper(
                G,
                filter_type="heat",
                t=params.get('tau', 2.0),
                K=params.get('filter_order', 5),
                embedding_dim=params.get('embedding_dim', 128),
            )
        elif method == "baseline_filter_poly":
            return generate_baseline_filter_embedding_wrapper(
                G,
                filter_type="poly",
                K=params.get('filter_order', 5),
                alpha=params.get('alpha', 0.5),
                embedding_dim=params.get('embedding_dim', 128),
            )
        
        # GCN-MF baseline
        elif method == "baseline_gcnmf":
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
        
        # GNN baselines with quantum calibration capability
        elif method == "gat_baseline":
            try:
                from quvine.baselines.gat import GATConfig, GATTrainConfig, generate_gat_embedding_by_method_name
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
            except ImportError:
                logger.warning("GAT not available")
                return None
        
        elif method == "graphgps_baseline":
            try:
                from quvine.baselines.graphgps import GraphGPSConfig, GraphGPSTrainConfig, generate_graphgps_embedding_by_method_name
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
            except ImportError:
                logger.warning("GraphGPS not available")
                return None
        
        # Classical random walk methods
        elif method == "node2vec":
            n2v_params = {
                'dimensions': params.get('embedding_dim', 128),
                'walk_length': params.get('walk_length', 80),
                'num_walks': params.get('num_walks', 10),
                'p': params.get('p', 1.0),
                'q': params.get('q', 1.0),
                'window': params.get('window_size', 10),
            }
            return run_node2vec(G, nodes=list(G.nodes()), **n2v_params)
        elif method == "netmf":
            netmf_params = {
                'dimensions': params.get('embedding_dim', 128),
                'window': params.get('window_size', 10),
                'negative': params.get('negative_samples', 5),
            }
            return run_netmf(G, nodes=list(G.nodes()), **netmf_params)
        
        # Classical GNN methods
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
            try:
                from quvine.baselines.appnp import run_appnp
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
            except ImportError:
                logger.warning("APPNP not available")
                return None
        
        else:
            raise ValueError(f"Unknown method: {method}")
    except Exception as e:
        logger.warning(f"Embedding generation failed for {method}: {e}")
        return None


def evaluate_node_classification(embedding: np.ndarray, G: nx.Graph, config: Dict[str, Any]) -> float:
    """Evaluate node classification performance using 7 label strategies."""
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
    """Evaluate link prediction performance using hadamard."""
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


def evaluate_node_rank(embedding: np.ndarray, G: nx.Graph, seeds: List[int], targets: List[int], config: Dict[str, Any]) -> float:
    """Evaluate node ranking performance using GWAS targets."""
    try:
        eval_config = config['fixed_params']['evaluation']['node_ranking']
        top_k = eval_config['top_k']
        
        # Compute ranking scores using seed centroid
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
        scores = []
        for node in node_list:
            idx = node_list.index(node)
            node_emb = embedding[idx]
            node_norm = np.linalg.norm(node_emb)
            if node_norm > 1e-10:
                sim = np.dot(node_emb, seed_emb) / (node_norm * seed_norm)
            else:
                sim = 0.0
            scores.append(sim)
        
        # Get top-K ranked nodes
        scores_array = np.array(scores)
        top_k_indices = np.argsort(scores_array)[-top_k:]
        top_k_nodes = set([node_list[i] for i in top_k_indices])
        
        # Compute recall on targets
        target_set = set(targets)
        hits = len(top_k_nodes & target_set)
        recall = hits / len(target_set) if len(target_set) > 0 else 0.0
        
        return recall
        
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
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif all(isinstance(v, float) for v in param_values):
                if len(param_values) == 2:
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
                else:
                    params[param_name] = trial.suggest_categorical(param_name, param_values)
            elif all(isinstance(v, str) for v in param_values):
                params[param_name] = trial.suggest_categorical(param_name, param_values)
            else:
                params[param_name] = trial.suggest_categorical(param_name, param_values)
        else:
            params[param_name] = param_values
    
    return params


def get_n_trials_for_method(method: str, config: Dict[str, Any], override_trials: Optional[int] = None) -> int:
    """Get the number of trials for a method based on configuration."""
    if override_trials is not None:
        return override_trials
    
    optuna_config = config.get('optuna', {})
    
    if optuna_config.get('adaptive_trials', False):
        method_trials = optuna_config.get('method_trials', {})
        if method in method_trials:
            return method_trials[method]
        else:
            return optuna_config.get('base_trials', 50)
    else:
        return optuna_config.get('base_trials', 50)


def tune_method_for_task(
    method: str,
    task: str,
    graphs: List[nx.Graph],
    seeds_list: List[List[int]],
    targets_list: List[List[int]],
    test_edges_list: List[List[Tuple[int, int]]],
    test_non_edges_list: List[List[Tuple[int, int]]],
    config: Dict[str, Any],
    n_trials: Optional[int] = None,
) -> Tuple[Dict[str, Any], float]:
    """Tune hyperparameters for a method on a specific task."""
    actual_trials = get_n_trials_for_method(method, config, n_trials)
    logger.info(f"Tuning {method} for {task} with {actual_trials} trials on {len(graphs)} graphs")
    
    def objective(trial_or_params):
        """Objective function for a specific task."""
        if isinstance(trial_or_params, dict):
            params = trial_or_params
        else:
            params = suggest_params_from_config(trial_or_params, method, config)
        
        scores = []
        for G, seeds, targets, test_edges, test_non_edges in zip(
            graphs, seeds_list, targets_list, test_edges_list, test_non_edges_list
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
                score = evaluate_node_rank(embedding, G, seeds, targets, config)
            else:
                raise ValueError(f"Unknown task: {task}")
            
            scores.append(score)
        
        return np.mean(scores)
    
    if OPTUNA_AVAILABLE:
        optuna_config = config.get('optuna', {})
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=optuna_config.get('n_startup_trials', 10)
        )
        
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=actual_trials, show_progress_bar=False)
        best_params = study.best_params
        best_score = study.best_value
    else:
        # Random search fallback
        logger.warning("Using random search (Optuna not available)")
        best_params = {}
        best_score = 0.0
        for _ in range(actual_trials):
            # Random parameter selection
            params = {}
            method_config = config['hyperparameters'][method]
            for param_name, param_values in method_config.items():
                if isinstance(param_values, list):
                    params[param_name] = np.random.choice(param_values)
                else:
                    params[param_name] = param_values
            
            score = objective(params)
            if score > best_score:
                best_score = score
                best_params = params
    
    logger.info(f"Best {task} score for {method}: {best_score:.4f}")
    logger.info(f"Best params: {best_params}")
    
    return best_params, best_score


def main():
    parser = argparse.ArgumentParser(description="PPI Network Hyperparameter Tuning")
    parser.add_argument('--config', type=str, default='scripts/unified_tuning_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--network', type=str, required=True,
                        choices=['STRING', 'BioPlex3', 'HumanNet', 'PCNet', 'ProteomeHD'],
                        help='PPI network to tune on')
    parser.add_argument('--disease', type=str, required=True,
                        choices=['asthma', 'autism', 'schizophrenia'],
                        help='Disease for GWAS data')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                        help='Methods to tune (default: all from config)')
    parser.add_argument('--n-replicates', type=int, default=3,
                        help='Number of subsampling replicates')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='Override number of trials per method')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: from config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get methods to tune
    if args.methods:
        methods = args.methods
    else:
        methods = config['methods']['all']
    
    logger.info(f"Tuning {len(methods)} methods: {methods}")
    
    # Get output directory
    output_dir = args.output_dir or config['experiment']['output_dir']
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PPI network
    network_config = config['ppi_networks'][args.network]
    G_full = load_ppi_network(network_config['path'])
    
    # Load GWAS data
    disease_config = config['diseases'][args.disease]
    seeds_full, targets_full = load_gwas_data(
        disease_config['seeds'],
        disease_config['targets'],
        G_full
    )
    
    # Generate multiple subsampled graphs
    max_nodes = config['fixed_params']['graph']['n_nodes']
    graphs = []
    seeds_list = []
    targets_list = []
    test_edges_list = []
    test_non_edges_list = []
    
    for rep in range(args.n_replicates):
        logger.info(f"\n=== Replicate {rep + 1}/{args.n_replicates} ===")
        G_sub, seeds_sub, targets_sub = subsample_ppi_network(
            G_full, seeds_full, targets_full, max_nodes, config, seed=42 + rep
        )
        
        # Split edges for link prediction
        test_ratio = config['fixed_params']['evaluation']['link_prediction']['test_ratio']
        G_train, val_edges, test_edges, negative_edges = split_edges(
            G_sub, test_ratio=test_ratio, val_ratio=0.0, seed=42 + rep
        )
        
        # split_edges already returns negative_edges, no need to sample again
        graphs.append(G_sub)
        seeds_list.append(seeds_sub)
        targets_list.append(targets_sub)
        test_edges_list.append(test_edges)
        test_non_edges_list.append(negative_edges)
    
    # Tune each method for each task
    results = {}
    tasks = config['experiment']['tasks']
    
    for method in methods:
        logger.info(f"\n{'='*80}")
        logger.info(f"Tuning method: {method}")
        logger.info(f"{'='*80}")
        
        results[method] = {}
        
        for task in tasks:
            logger.info(f"\n--- Task: {task} ---")
            best_params, best_score = tune_method_for_task(
                method=method,
                task=task,
                graphs=graphs,
                seeds_list=seeds_list,
                targets_list=targets_list,
                test_edges_list=test_edges_list,
                test_non_edges_list=test_non_edges_list,
                config=config,
                n_trials=args.n_trials,
            )
            
            results[method][task] = {
                'best_params': best_params,
                'best_score': best_score,
            }
    
    # Save results - each method gets its own file for parallel execution
    network_disease_id = f"{args.network}_{args.disease}"
    
    if len(methods) == 1:
        # Single method: save to method-specific file
        method_name = methods[0]
        output_file = output_dir / f"{network_disease_id}_{method_name}_tuning_by_task.json"
    else:
        # Multiple methods: save to combined file (serial mode)
        output_file = output_dir / f"{network_disease_id}_tuning_by_task.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Tuning complete!")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()

