#!/usr/bin/env python3
"""
Single Graph Pipeline for Large-Scale Experiment

This script processes a single graph through the complete pipeline:
1. Load graph and metadata
2. Compute complexity metrics
3. Generate parameter combinations
4. Train embeddings with all parameter combinations
5. Evaluate performance
6. Save all results

Designed to run as part of an HPC job array where each job processes one graph.

Usage:
    python run_single_graph_pipeline.py --job_id 0 --config configs/large_scale_experiment.yaml
    python run_single_graph_pipeline.py --graph_id erdos_renyi_0001 --config configs/large_scale_experiment.yaml
"""

import sys
import os
import argparse
import yaml
import json
import numpy as np
import networkx as nx
from pathlib import Path
from datetime import datetime
import itertools
import traceback
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quvine.data.graph_complexity import (
    compute_spectral_gap,
    compute_algebraic_connectivity,
    compute_spectral_entropy,
    compute_von_neumann_entropy,
    compute_estrada_index,
    compute_quantum_complexity,
    compute_fiedler_gini,
    compute_fiedler_entropy,
    compute_fiedler_range,
    compute_centrality_entropy,
    compute_centrality_variance,
    compute_centrality_gini,
    compute_centrality_range,
    compute_dominant_eigenvector_centrality,
    compute_eigenvalue_stats
)
from quvine.data.qbc_complexity import compute_qbc_complexity
from quvine.views.generator import ViewBuilder
from quvine.walks.rwr import RandomWalkWithRestart
from quvine.walks.dtqw import DiscreteTimeQuantumWalk
from quvine.walks.ctqw import ContinuousTimeQuantumWalk
from quvine.embedding.word2vec import train_word2vec
from quvine.baselines.node2vec import Node2VecEmbedding
from quvine.baselines.netmf import NetMFEmbedding
from quvine.fusion.fuse import fuse_embeddings


def load_config(config_path: str) -> Dict:
    """Load experiment configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_graph_metadata(experiment_dir: Path, graph_id: str = None, job_id: int = None) -> Tuple[str, Path, Path]:
    """
    Load graph metadata from job mapping or graph_id.
    
    Returns:
        graph_id, graph_path, metadata_path
    """
    if job_id is not None:
        # Load from job mapping
        job_mapping_path = experiment_dir / 'job_mapping.csv'
        with open(job_mapping_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            if job_id >= len(lines):
                raise ValueError(f"Job ID {job_id} out of range (max: {len(lines)-1})")
            
            parts = lines[job_id].strip().split(',')
            graph_id = parts[1]
            graph_path = experiment_dir / parts[2]
            metadata_path = experiment_dir / parts[3]
    else:
        # Use graph_id directly
        graph_path = experiment_dir / 'graphs' / f'{graph_id}.graphml'
        metadata_path = experiment_dir / 'metadata' / f'{graph_id}_metadata.json'
    
    return graph_id, graph_path, metadata_path


def compute_all_complexity_metrics(G: nx.Graph, config: Dict) -> Dict[str, float]:
    """Compute all complexity metrics for a graph."""
    metrics = {}
    
    print("Computing complexity metrics...")
    
    # Spectral metrics
    try:
        metrics['spectral_gap'] = compute_spectral_gap(G)
        metrics['algebraic_connectivity'] = compute_algebraic_connectivity(G)
        metrics['spectral_entropy'] = compute_spectral_entropy(G)
        metrics['von_neumann_entropy'] = compute_von_neumann_entropy(G)
        metrics['estrada_index'] = compute_estrada_index(G)
        metrics['quantum_complexity'] = compute_quantum_complexity(G)
    except Exception as e:
        print(f"Warning: Spectral metrics failed: {e}")
    
    # Centrality metrics
    try:
        metrics['fiedler_gini'] = compute_fiedler_gini(G)
        metrics['fiedler_entropy'] = compute_fiedler_entropy(G)
        metrics['fiedler_range'] = compute_fiedler_range(G)
        metrics['centrality_entropy'] = compute_centrality_entropy(G)
        metrics['centrality_variance'] = compute_centrality_variance(G)
        metrics['centrality_gini'] = compute_centrality_gini(G)
        metrics['centrality_range'] = compute_centrality_range(G)
        metrics['dominant_eigenvector_centrality'] = compute_dominant_eigenvector_centrality(G)
    except Exception as e:
        print(f"Warning: Centrality metrics failed: {e}")
    
    # Eigenvalue stats
    try:
        ev_stats = compute_eigenvalue_stats(G)
        metrics.update(ev_stats)
    except Exception as e:
        print(f"Warning: Eigenvalue stats failed: {e}")
    
    # QBioCode complexity
    if config['complexity'].get('qbiocode', {}).get('enabled', True):
        try:
            qbc = compute_qbc_complexity(
                G,
                normalized=config['complexity']['qbiocode'].get('normalized', True),
                mean_center=config['complexity']['qbiocode'].get('mean_center', True)
            )
            metrics['qbc_complexity'] = qbc
        except Exception as e:
            print(f"Warning: QBioCode complexity failed: {e}")
    
    print(f"Computed {len(metrics)} complexity metrics")
    return metrics


def generate_parameter_combinations(config: Dict) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations based on experiment design.
    
    Returns list of parameter dictionaries.
    """
    design = config['experiment_design']
    mode = design.get('mode', 'factorial')
    
    # Get parameter ranges
    walks_config = config['embeddings']['walks']
    views_config = config['embeddings']['views']
    train_config = config['embeddings']['train']
    
    # Define parameter space
    param_space = {
        'num_walks': walks_config['num_walks_range'],
        'walk_length': walks_config['walk_length_range'],
        'restart_prob': walks_config['restart_prob_range'],
        'dtqw_steps': walks_config['steps_range'],
        'ctqw_time': walks_config['time_range'],
        'coin': walks_config['coin_options'],
        'num_views': views_config['num_views_range'],
        'max_degree': views_config['max_degree_range'],
        'degree_alpha': views_config['degree_alpha_range'],
        'embedding_dim': train_config['embedding_dim_range'],
        'window': train_config['window_range'],
        'negative': train_config['negative_range'],
        'epochs': train_config['epochs_range']
    }
    
    if mode == 'factorial':
        # Generate all combinations (can be huge!)
        max_combinations = design['factorial'].get('max_combinations_per_graph', 50)
        
        # Prioritize certain parameters
        priority_params = design['factorial'].get('prioritize_parameters', [])
        
        # Generate combinations for priority parameters
        priority_combinations = []
        for param in priority_params:
            if param in param_space:
                priority_combinations.append(param_space[param])
        
        # Generate limited factorial design
        if priority_combinations:
            combinations = list(itertools.product(*priority_combinations))[:max_combinations]
        else:
            # Use all parameters but limit
            all_combinations = list(itertools.product(*param_space.values()))
            combinations = all_combinations[:max_combinations]
        
        # Convert to parameter dicts
        param_dicts = []
        for combo in combinations:
            param_dict = {}
            for i, param_name in enumerate(priority_params if priority_combinations else param_space.keys()):
                if i < len(combo):
                    param_dict[param_name] = combo[i]
            
            # Fill in missing parameters with defaults (first value in range)
            for param_name, values in param_space.items():
                if param_name not in param_dict:
                    param_dict[param_name] = values[0]
            
            param_dicts.append(param_dict)
    
    elif mode == 'random_sample':
        # Random sampling
        num_samples = design['random_sample']['num_samples_per_graph']
        seed = design['random_sample'].get('seed', 42)
        rng = np.random.RandomState(seed)
        
        param_dicts = []
        for _ in range(num_samples):
            param_dict = {}
            for param_name, values in param_space.items():
                param_dict[param_name] = rng.choice(values)
            param_dicts.append(param_dict)
    
    else:
        raise ValueError(f"Unknown experiment design mode: {mode}")
    
    print(f"Generated {len(param_dicts)} parameter combinations")
    return param_dicts


def run_embedding_pipeline(
    G: nx.Graph,
    seeds: List[int],
    targets: List[int],
    params: Dict[str, Any],
    method: str,
    config: Dict
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Run embedding pipeline for a single method and parameter combination.
    
    Returns:
        embeddings, evaluation_metrics
    """
    try:
        if method == 'rwr':
            # Generate views
            view_builder = ViewBuilder(
                G,
                num_views=params['num_views'],
                constrained=True,
                max_degree=params['max_degree'],
                max_nodes=config['embeddings']['views']['max_nodes_range'][0],
                max_edges=config['embeddings']['views']['max_edges_range'][0],
                degree_norm=True,
                degree_alpha=params['degree_alpha']
            )
            views = view_builder.generate_views(seeds)
            
            # Generate walks
            walker = RandomWalkWithRestart(
                restart_prob=params['restart_prob'],
                max_iter=config['embeddings']['walks']['max_iter']
            )
            corpus = []
            for view in views:
                walks = walker.generate_walks(
                    view,
                    seeds,
                    num_walks=params['num_walks'],
                    walk_length=params['walk_length']
                )
                corpus.extend(walks)
            
            # Train embeddings
            embeddings = train_word2vec(
                corpus,
                embedding_dim=params['embedding_dim'],
                window=params['window'],
                sg=1,
                negative=params['negative'],
                min_count=1,
                workers=4,
                epochs=params['epochs']
            )
        
        elif method == 'dtqw':
            # Similar to RWR but with DTQW
            view_builder = ViewBuilder(
                G,
                num_views=params['num_views'],
                constrained=True,
                max_degree=params['max_degree'],
                max_nodes=config['embeddings']['views']['max_nodes_range'][0],
                max_edges=config['embeddings']['views']['max_edges_range'][0],
                degree_norm=True,
                degree_alpha=params['degree_alpha']
            )
            views = view_builder.generate_views(seeds)
            
            walker = DiscreteTimeQuantumWalk(
                steps=params['dtqw_steps'],
                coin=params['coin']
            )
            corpus = []
            for view in views:
                walks = walker.generate_walks(
                    view,
                    seeds,
                    num_walks=params['num_walks'],
                    walk_length=params['walk_length']
                )
                corpus.extend(walks)
            
            embeddings = train_word2vec(
                corpus,
                embedding_dim=params['embedding_dim'],
                window=params['window'],
                sg=1,
                negative=params['negative'],
                min_count=1,
                workers=4,
                epochs=params['epochs']
            )
        
        elif method == 'ctqw':
            # Similar to RWR but with CTQW
            view_builder = ViewBuilder(
                G,
                num_views=params['num_views'],
                constrained=True,
                max_degree=params['max_degree'],
                max_nodes=config['embeddings']['views']['max_nodes_range'][0],
                max_edges=config['embeddings']['views']['max_edges_range'][0],
                degree_norm=True,
                degree_alpha=params['degree_alpha']
            )
            views = view_builder.generate_views(seeds)
            
            walker = ContinuousTimeQuantumWalk(time=params['ctqw_time'])
            corpus = []
            for view in views:
                walks = walker.generate_walks(
                    view,
                    seeds,
                    num_walks=params['num_walks'],
                    walk_length=params['walk_length']
                )
                corpus.extend(walks)
            
            embeddings = train_word2vec(
                corpus,
                embedding_dim=params['embedding_dim'],
                window=params['window'],
                sg=1,
                negative=params['negative'],
                min_count=1,
                workers=4,
                epochs=params['epochs']
            )
        
        elif method == 'node2vec':
            n2v = Node2VecEmbedding(
                dimensions=params['embedding_dim'],
                walk_length=params['walk_length'],
                num_walks=params['num_walks'],
                p=config['embeddings']['node2vec']['p_range'][0],
                q=config['embeddings']['node2vec']['q_range'][0],
                window=params['window'],
                min_count=1,
                workers=4
            )
            embeddings = n2v.fit_transform(G)
        
        elif method == 'netmf':
            netmf = NetMFEmbedding(
                dimensions=params['embedding_dim'],
                window_size=params['window'],
                negative=params['negative'],
                use_svd=True
            )
            embeddings = netmf.fit_transform(G)
        
        elif method == 'fused':
            # Run RWR, DTQW, CTQW and fuse
            methods_embeddings = []
            
            for sub_method in ['rwr', 'dtqw', 'ctqw']:
                emb, _ = run_embedding_pipeline(G, seeds, targets, params, sub_method, config)
                methods_embeddings.append(emb)
            
            # Fuse embeddings
            fusion_method = config['embeddings']['fusion']['method_options'][0]
            embeddings = fuse_embeddings(
                methods_embeddings,
                method=fusion_method,
                k=params['embedding_dim'] if fusion_method == 'svd' else None
            )
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Evaluate
        metrics = evaluate_embeddings(embeddings, seeds, targets, config)
        
        return embeddings, metrics
    
    except Exception as e:
        print(f"Error in {method} pipeline: {e}")
        traceback.print_exc()
        return None, {}


def evaluate_embeddings(
    embeddings: np.ndarray,
    seeds: List[int],
    targets: List[int],
    config: Dict
) -> Dict[str, float]:
    """Evaluate embeddings using seed-target ranking."""
    from sklearn.metrics.pairwise import cosine_similarity
    
    metrics = {}
    
    # Compute similarities
    seed_embeddings = embeddings[seeds]
    target_embeddings = embeddings[targets]
    all_embeddings = embeddings
    
    # Average seed embedding
    avg_seed = np.mean(seed_embeddings, axis=0, keepdims=True)
    
    # Compute similarities to all nodes
    similarities = cosine_similarity(avg_seed, all_embeddings)[0]
    
    # Rank nodes
    ranked_nodes = np.argsort(similarities)[::-1]
    
    # Evaluate at different k values
    for k in config['evaluation']['k_values']:
        top_k = ranked_nodes[:k]
        num_targets_found = len(set(top_k) & set(targets))
        
        precision = num_targets_found / k if k > 0 else 0
        recall = num_targets_found / len(targets) if len(targets) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[f'precision@{k}'] = precision
        metrics[f'recall@{k}'] = recall
        metrics[f'f1@{k}'] = f1
        metrics[f'num_targets_found@{k}'] = num_targets_found
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Run single graph pipeline')
    parser.add_argument('--job_id', type=int, help='HPC job array ID')
    parser.add_argument('--graph_id', type=str, help='Graph ID to process')
    parser.add_argument('--config', type=str, required=True, help='Path to experiment config')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    
    args = parser.parse_args()
    
    if args.job_id is None and args.graph_id is None:
        parser.error('Either --job_id or --graph_id must be specified')
    
    # Load config
    config = load_config(args.config)
    
    # Setup paths
    experiment_dir = Path(args.output_dir) if args.output_dir else Path(config['experiment']['output_dir'])
    
    # Load graph metadata
    graph_id, graph_path, metadata_path = load_graph_metadata(
        experiment_dir,
        graph_id=args.graph_id,
        job_id=args.job_id
    )
    
    print(f"{'='*60}")
    print(f"Processing graph: {graph_id}")
    print(f"{'='*60}")
    
    # Load graph
    print(f"Loading graph from {graph_path}")
    G = nx.read_graphml(graph_path)
    G = nx.convert_node_labels_to_integers(G)
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    seeds = metadata['seeds']
    targets = metadata['targets']
    
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Seeds: {len(seeds)}, Targets: {len(targets)}")
    
    # Create output directories
    results_dir = experiment_dir / 'results' / graph_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute complexity metrics
    print("\n" + "="*60)
    print("Computing complexity metrics")
    print("="*60)
    complexity_metrics = compute_all_complexity_metrics(G, config)
    
    # Save complexity metrics
    complexity_path = results_dir / 'complexity_metrics.json'
    with open(complexity_path, 'w') as f:
        json.dump(complexity_metrics, f, indent=2)
    print(f"Saved complexity metrics to {complexity_path}")
    
    # Generate parameter combinations
    print("\n" + "="*60)
    print("Generating parameter combinations")
    print("="*60)
    param_combinations = generate_parameter_combinations(config)
    
    # Save parameter combinations
    params_path = results_dir / 'parameter_combinations.json'
    with open(params_path, 'w') as f:
        json.dump(param_combinations, f, indent=2)
    print(f"Saved {len(param_combinations)} parameter combinations to {params_path}")
    
    # Run embeddings for each method and parameter combination
    all_results = []
    
    for method in config['embeddings']['methods']:
        print(f"\n{'='*60}")
        print(f"Method: {method}")
        print(f"{'='*60}")
        
        for param_idx, params in enumerate(param_combinations):
            print(f"\nParameter combination {param_idx + 1}/{len(param_combinations)}")
            print(f"Parameters: {params}")
            
            try:
                embeddings, eval_metrics = run_embedding_pipeline(
                    G, seeds, targets, params, method, config
                )
                
                if embeddings is not None:
                    # Save embeddings
                    emb_path = results_dir / f'{method}_params{param_idx:04d}_embeddings.npz'
                    np.savez_compressed(emb_path, embeddings=embeddings)
                    
                    # Store results
                    result = {
                        'graph_id': graph_id,
                        'method': method,
                        'param_idx': param_idx,
                        'parameters': params,
                        'evaluation': eval_metrics,
                        'embedding_path': str(emb_path.relative_to(experiment_dir)),
                        'timestamp': datetime.now().isoformat()
                    }
                    all_results.append(result)
                    
                    print(f"F1@10: {eval_metrics.get('f1@10', 0):.4f}")
                
            except Exception as e:
                print(f"Failed: {e}")
                traceback.print_exc()
                continue
    
    # Save all results
    results_path = results_dir / 'all_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Completed processing {graph_id}")
    print(f"Total results: {len(all_results)}")
    print(f"Results saved to {results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

