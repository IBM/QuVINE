"""
Comprehensive Embedding Analysis Pipeline (PARALLELIZED)

This script generates multiple random networks (scale-free and modular),
computes complexity metrics, runs all embedding methods, and analyzes
the correlation between complexity and embedding performance.

Methods compared:
- QuVINE-fused (fusion of quantum walks)
- QuVINE with RWR (Random Walk with Restart)
- QuVINE with CTQW (Continuous-Time Quantum Walk)
- QuVINE with DTQW (Discrete-Time Quantum Walk)
- NetMF (Network Embedding as Matrix Factorization)
- Node2Vec (classical baseline)

PARALLELIZATION:
- Networks are processed in parallel
- Each network runs all methods independently
- Results are collected and aggregated
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from omegaconf import DictConfig, OmegaConf
from joblib import Parallel, delayed
import multiprocessing

from quvine.data.random_graphs import (
    generate_barabasi_albert,
    generate_modular_network,
    get_graph_statistics
)
from quvine.complexity.graph import compute_graph_complexity_metrics
from quvine.baselines import run_netmf, run_node2vec
from quvine.embedding.word2vec import corpus_to_embedding
from quvine.corpus.builder import CorpusBuilder
from quvine.walks.base import BaseWalker
from quvine.views.generator import ViewBuilder
from quvine.fusion.fuse import fuse_embeddings
from quvine.evaluation.ranking import (
    seed_centroid_scores,
    max_seed_cosine_scores,
    evaluate_embeddings_ranking
)
from quvine.evaluation.classification import (
    evaluate_all_label_strategies,
    summarize_classification_results
)
from quvine.evaluation.link_prediction import (
    evaluate_link_prediction_cv,
    evaluate_all_edge_feature_methods,
    summarize_link_prediction_results,
    split_edges
)
from quvine.embedding.registry import EmbeddingStore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComprehensiveEmbeddingAnalysis:
    """
    Comprehensive analysis of embedding methods across different network types
    and complexity levels.
    """
    
    def __init__(
        self,
        output_dir: str = "outputs/comprehensive_analysis",
        n_networks_per_type: int = 20,
        n_nodes: int = 200,
        num_seeds: int = 15,
        num_targets: int = 25,
        embedding_dim: int = 128,
        seed: int = 42,
        n_jobs: int = -1  # -1 means use all available cores
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_networks_per_type = n_networks_per_type
        self.n_nodes = n_nodes
        self.num_seeds = num_seeds
        self.num_targets = num_targets
        self.embedding_dim = embedding_dim
        self.base_seed = seed
        
        # Determine number of parallel jobs
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        logger.info(f"Using {self.n_jobs} parallel workers")
        
        # Results storage
        self.complexity_results = []
        self.embedding_results = []
        self.performance_results = []
        self.classification_results = []
        self.link_prediction_results = []
        
    def _select_seeds_targets(self, G: nx.Graph) -> Tuple[List[int], List[int]]:
        """
        Select seed and target nodes ensuring no overlap.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
            
        Returns
        -------
        tuple
            (seeds, targets) lists
        """
        nodes = list(G.nodes())
        n_nodes = len(nodes)
        
        # Adjust counts if graph is too small
        num_seeds = min(self.num_seeds, n_nodes // 3)
        num_targets = min(self.num_targets, n_nodes // 3)
        
        if num_seeds + num_targets > n_nodes:
            num_seeds = n_nodes // 3
            num_targets = n_nodes // 3
        
        rng = np.random.default_rng(self.base_seed)
        selected_indices = rng.choice(n_nodes, size=num_seeds + num_targets, replace=False)
        
        seeds = [nodes[i] for i in selected_indices[:num_seeds]]
        targets = [nodes[i] for i in selected_indices[num_seeds:num_seeds + num_targets]]
        
        return seeds, targets
    
    def load_benchmark_networks(self) -> List[Tuple[str, nx.Graph, List[int], List[int]]]:
        """
        Load real-world benchmark networks for testing.
        
        Returns
        -------
        list of tuples
            Each tuple contains (network_id, graph, seeds, targets)
        """
        logger.info("Loading benchmark networks...")
        networks = []
        
        # 1. Karate Club (34 nodes)
        try:
            G = nx.karate_club_graph()
            seeds, targets = self._select_seeds_targets(G)
            networks.append(("benchmark_karate_club", G, seeds, targets))
            logger.info(f"  Loaded Karate Club: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to load Karate Club: {e}")
        
        # 2. Dolphins social network (62 nodes)
        try:
            G = nx.read_gml("data/dolphins.gml", label='id')
            seeds, targets = self._select_seeds_targets(G)
            networks.append(("benchmark_dolphins", G, seeds, targets))
            logger.info(f"  Loaded Dolphins: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except:
            logger.info("  Dolphins network not available (optional)")
        
        # 3. Les Miserables (77 nodes)
        try:
            G = nx.les_miserables_graph()
            seeds, targets = self._select_seeds_targets(G)
            networks.append(("benchmark_les_miserables", G, seeds, targets))
            logger.info(f"  Loaded Les Miserables: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to load Les Miserables: {e}")
        
        # 4. Davis Southern Women (32 nodes, bipartite)
        try:
            G = nx.davis_southern_women_graph()
            # Convert to simple graph (project one mode)
            women = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
            G_proj = nx.bipartite.projected_graph(G, women)
            seeds, targets = self._select_seeds_targets(G_proj)
            networks.append(("benchmark_davis_women", G_proj, seeds, targets))
            logger.info(f"  Loaded Davis Women: {G_proj.number_of_nodes()} nodes, {G_proj.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to load Davis Women: {e}")
        
        # 5. Florentine Families (15 nodes)
        try:
            G = nx.florentine_families_graph()
            seeds, targets = self._select_seeds_targets(G)
            networks.append(("benchmark_florentine", G, seeds, targets))
            logger.info(f"  Loaded Florentine Families: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to load Florentine Families: {e}")
        
        logger.info(f"Loaded {len(networks)} benchmark networks")
        return networks
    
    def generate_networks(self, include_benchmarks: bool = True) -> List[Tuple[str, nx.Graph, List[int], List[int]]]:
        """
        Generate scale-free and modular networks with seeds and targets.
        
        Parameters
        ----------
        include_benchmarks : bool, default=True
            If True, include real-world benchmark networks
        
        Returns
        -------
        list of tuples
            Each tuple contains (network_id, graph, seeds, targets)
        """
        logger.info(f"Generating {self.n_networks_per_type} networks of each type...")
        networks = []
        
        # Add benchmark networks first
        if include_benchmarks:
            networks.extend(self.load_benchmark_networks())
        
        # Generate scale-free networks (Barabási-Albert)
        for i in range(self.n_networks_per_type):
            seed = self.base_seed + i
            rng = np.random.default_rng(seed)
            
            # Vary m parameter for diversity
            m = rng.integers(2, 6)
            G = generate_barabasi_albert(self.n_nodes, m, seed=seed)
            
            # Ensure connected
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            # Select seeds and targets (no overlap)
            seeds, targets = self._select_seeds_targets(G)
            
            network_id = f"scale_free_{i:02d}"
            networks.append((network_id, G, seeds, targets))
            logger.info(f"Generated {network_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Generate modular networks
        for i in range(self.n_networks_per_type):
            seed = self.base_seed + 1000 + i
            rng = np.random.default_rng(seed)
            
            # Vary modularity parameters
            num_communities = rng.integers(3, 8)
            nodes_per_community = self.n_nodes // num_communities
            p_intra = rng.uniform(0.2, 0.5)
            p_inter = rng.uniform(0.005, 0.03)
            
            G, communities = generate_modular_network(
                num_communities, nodes_per_community, p_intra, p_inter, seed=seed
            )
            
            # Ensure connected
            if not nx.is_connected(G):
                largest_cc = max(nx.connected_components(G), key=len)
                G = G.subgraph(largest_cc).copy()
            
            # Select seeds and targets (no overlap)
            seeds, targets = self._select_seeds_targets(G)
            
            network_id = f"modular_{i:02d}"
            networks.append((network_id, G, seeds, targets))
            logger.info(f"Generated {network_id}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return networks
    
    def _compute_complexity_single(
        self,
        network_tuple: Tuple[str, nx.Graph, List[int], List[int]]
    ) -> Dict:
        """
        Compute complexity for a single network (for parallel execution).
        
        Parameters
        ----------
        network_tuple : tuple
            (network_id, G, seeds, targets)
            
        Returns
        -------
        dict
            Complexity metrics with network_id and network_type
        """
        network_id, G, seeds, targets = network_tuple
        
        logger.info(f"Computing complexity for {network_id}...")
        metrics = compute_graph_complexity_metrics(G)
        metrics['network_id'] = network_id
        
        # Determine network type
        if 'scale_free' in network_id:
            metrics['network_type'] = 'scale_free'
        elif 'modular' in network_id:
            metrics['network_type'] = 'modular'
        elif 'benchmark' in network_id:
            metrics['network_type'] = 'benchmark'
        else:
            metrics['network_type'] = 'other'
        
        return metrics
    
    def compute_complexity_for_all(
        self,
        networks: List[Tuple[str, nx.Graph, List[int], List[int]]]
    ):
        """Compute complexity metrics for all networks (PARALLELIZED)."""
        logger.info(f"Computing complexity metrics for {len(networks)} networks in parallel...")
        logger.info(f"Using {self.n_jobs} parallel workers")
        
        # Compute complexity in parallel
        parallel = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=10)
        
        self.complexity_results = parallel(
            delayed(self._compute_complexity_single)(network_tuple)
            for network_tuple in networks
        )
        
        # Save complexity results
        complexity_df = pd.DataFrame(self.complexity_results)
        complexity_path = self.output_dir / "complexity_metrics.csv"
        complexity_df.to_csv(complexity_path, index=False)
        logger.info(f"Complexity metrics saved to {complexity_path}")
        
        return complexity_df
    
    def run_embedding_method(
        self,
        method_name: str,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        cfg: Optional[DictConfig] = None
    ) -> np.ndarray:
        """
        Run a specific embedding method.
        
        Parameters
        ----------
        method_name : str
            One of: 'rwr', 'ctqw', 'dtqw', 'fused', 'netmf', 'node2vec'
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes
        targets : list
            Target nodes
        cfg : DictConfig, optional
            Configuration for QuVINE methods
            
        Returns
        -------
        np.ndarray
            Embedding matrix
        """
        nodes = list(G.nodes())
        
        if method_name == 'netmf':
            return run_netmf(
                graph=G,
                nodes=nodes,
                dimensions=self.embedding_dim,
                window_size=10,
                negative=1,
                seed=self.base_seed
            )
        
        elif method_name == 'node2vec':
            return run_node2vec(
                graph=G,
                nodes=nodes,
                dimensions=self.embedding_dim,
                walk_length=10,
                num_walks=10,
                p=1.0,
                q=0.5,
                window=5,
                min_count=1,
                workers=4,
                seed=self.base_seed
            )
        
        elif method_name in ['rwr', 'ctqw', 'dtqw', 'fused']:
            # QuVINE methods
            if cfg is None:
                cfg = self._get_default_quvine_config()
            
            # Set walk type
            if method_name == 'fused':
                cfg.walks.kinds = ['rwr', 'ctqw', 'dtqw']
            else:
                cfg.walks.kinds = [method_name]
            
            # Run QuVINE pipeline for this method
            embeddings = self._run_quvine_walks(G, cfg)
            
            if method_name == 'fused' and len(embeddings) > 1:
                # Fuse embeddings
                store = EmbeddingStore()
                for name, Z in embeddings.items():
                    store.add(name, Z)
                
                L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray().astype(np.float32)
                fused_list, _ = fuse_embeddings(store, method='concatenate', k=3, L=L)
                return fused_list[0]
            else:
                # Return single embedding
                return list(embeddings.values())[0]
        
        else:
            raise ValueError(f"Unknown method: {method_name}")
    
    def _get_default_quvine_config(self) -> DictConfig:
        """Get default QuVINE configuration."""
        cfg = OmegaConf.create({
            'walks': {
                'kinds': ['rwr'],
                'num_walks_per_root': 10,
                'walk_length': 10,
                'rwr': {'alpha': 0.15},
                'ctqw': {'t': 1.0},
                'dtqw': {'steps': 10}
            },
            'views': {
                'enabled': True,
                'num_views': 3,
                'view_size': 50,
                'strategy': 'random'
            },
            'train': {
                'embedding_dim': self.embedding_dim,
                'window': 5,
                'sg': 1,
                'negative': 5,
                'workers': 4,
                'epochs': 5
            },
            'min_count': 1,
            'seed': self.base_seed
        })
        return cfg
    
    def _run_quvine_walks(
        self,
        G: nx.Graph,
        cfg: DictConfig
    ) -> Dict[str, np.ndarray]:
        """Run QuVINE walks and generate embeddings."""
        nodes = list(G.nodes())
        rng = np.random.default_rng(cfg.seed)
        
        # Build corpus for each walk type
        corpus_builders = {kind: CorpusBuilder() for kind in cfg.walks.kinds}
        
        # Generate walks for each root
        for root in nodes:
            # Build views
            view_gen = ViewBuilder(cfg=cfg, rng=rng)
            views = view_gen.build(G, root)
            
            # Run walks
            walker = BaseWalker(cfg=cfg, rng=rng)
            
            for view in views:
                view_g = G.subgraph(view)
                view_nodes = list(view_g.nodes())
                
                walk_outputs = walker.run(G, root, view_nodes)
                
                for walk_kind, walks in walk_outputs.items():
                    if len(walks) > 0:
                        corpus_builders[walk_kind].add(root, walks)
        
        # Build corpora and train embeddings
        embeddings = {}
        for kind, builder in corpus_builders.items():
            corpus = builder.build()
            if len(corpus) > 0:
                Z = corpus_to_embedding(
                    corpus=corpus,
                    nodes=nodes,
                    vector_size=cfg.train.embedding_dim,
                    window=cfg.train.window,
                    sg=cfg.train.sg,
                    negative=cfg.train.negative,
                    min_count=cfg.min_count,
                    workers=cfg.train.workers,
                    epochs=cfg.train.epochs
                )
                embeddings[kind] = Z
        
        return embeddings
    
    def evaluate_embedding(
        self,
        embedding: np.ndarray,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        method_name: str,
        network_id: str
    ) -> Dict:
        """
        Evaluate embedding performance on seed-target ranking task.
        
        Returns
        -------
        dict
            Performance metrics
        """
        nodes = list(G.nodes())
        seed_indices = [nodes.index(s) for s in seeds if s in nodes]
        
        # Compute scores
        centroid_scores = seed_centroid_scores(embedding, seed_indices)
        max_scores = max_seed_cosine_scores(embedding, seed_indices)
        
        # Evaluate ranking
        k_values = [10, 20, 50, 100]
        
        results = {
            'network_id': network_id,
            'method': method_name,
        }
        
        # Compute precision and recall at different k
        for k in k_values:
            if k > len(nodes):
                continue
            
            # Centroid-based
            top_k_centroid = np.argsort(centroid_scores)[-k:]
            top_k_nodes_centroid = [nodes[i] for i in top_k_centroid]
            
            hits_centroid = len(set(top_k_nodes_centroid) & set(targets))
            precision_centroid = hits_centroid / k
            recall_centroid = hits_centroid / len(targets) if len(targets) > 0 else 0
            
            results[f'precision@{k}_centroid'] = precision_centroid
            results[f'recall@{k}_centroid'] = recall_centroid
            
            # Max-based
            top_k_max = np.argsort(max_scores)[-k:]
            top_k_nodes_max = [nodes[i] for i in top_k_max]
            
            hits_max = len(set(top_k_nodes_max) & set(targets))
            precision_max = hits_max / k
            recall_max = hits_max / len(targets) if len(targets) > 0 else 0
            
            results[f'precision@{k}_max'] = precision_max
            results[f'recall@{k}_max'] = recall_max
        
    def evaluate_embedding_classification(
        self,
        embedding: np.ndarray,
        G: nx.Graph,
        method_name: str,
        network_id: str
    ) -> Dict:
        """
        Evaluate embedding performance on node classification task.
        
        Returns
        -------
        dict
            Classification performance metrics
        """
        nodes = list(G.nodes())
        
        try:
            # Evaluate all label strategies
            all_results = evaluate_all_label_strategies(
                G=G,
                embeddings=embedding,
                node_list=nodes,
                test_size=0.3,
                random_state=self.base_seed
            )
            
            # Summarize results
            summary = summarize_classification_results(all_results)
            
            # Add metadata
            summary['network_id'] = network_id
            summary['method'] = method_name
            
            # Add individual strategy results
            for strategy, metrics in all_results.items():
                if 'error' not in metrics:
                    summary[f'f1_{strategy}'] = metrics.get('f1_macro', 0.0)
                    summary[f'accuracy_{strategy}'] = metrics.get('accuracy', 0.0)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Classification evaluation failed for {method_name} on {network_id}: {e}")
            return {
                'network_id': network_id,
                'method': method_name,
                'mean_f1_macro': 0.0,
                'mean_accuracy': 0.0,
                'error': str(e)
            }
    
    def evaluate_embedding_link_prediction(
        self,
        embedding: np.ndarray,
        G: nx.Graph,
        method_name: str,
        network_id: str
    ) -> Dict:
        """
        Evaluate embedding performance on link prediction task.
        
        Returns
        -------
        dict
            Link prediction performance metrics
        """
        nodes = list(G.nodes())
        
        try:
            # Split edges for link prediction
            train_graph, _, test_edges, negative_edges = split_edges(
                G, test_ratio=0.2, val_ratio=0.0, seed=self.base_seed
            )
            
            # Evaluate all edge feature methods
            all_results = evaluate_all_edge_feature_methods(
                embeddings=embedding,
                node_list=nodes,
                positive_edges=test_edges,
                negative_edges=negative_edges,
                k_values=[10, 50, 100],
                random_state=self.base_seed
            )
            
            # Summarize results
            summary = summarize_link_prediction_results(all_results)
            
            # Add metadata
            summary['network_id'] = network_id
            summary['method'] = method_name
            
            # Add individual method results
            for edge_method, metrics in all_results.items():
                if 'error' not in metrics:
                    summary[f'auc_roc_{edge_method}'] = metrics.get('auc_roc', 0.0)
                    summary[f'auc_pr_{edge_method}'] = metrics.get('auc_pr', 0.0)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Link prediction evaluation failed for {method_name} on {network_id}: {e}")
            return {
                'network_id': network_id,
                'method': method_name,
                'mean_auc_roc': 0.0,
                'mean_auc_pr': 0.0,
                'error': str(e)
            }
    
        return results
    
    def _process_single_network(
        self,
        network_tuple: Tuple[str, nx.Graph, List[int], List[int]],
        include_classification: bool = True,
        include_link_prediction: bool = True
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Process a single network with all methods (for parallel execution).
        
        Parameters
        ----------
        network_tuple : tuple
            (network_id, G, seeds, targets)
        include_classification : bool
            Whether to include node classification evaluation
        include_link_prediction : bool
            Whether to include link prediction evaluation
            
        Returns
        -------
        tuple of lists
            (ranking_results, classification_results, link_prediction_results)
        """
        network_id, G, seeds, targets = network_tuple
        methods = ['rwr', 'ctqw', 'dtqw', 'fused', 'netmf', 'node2vec']
        
        ranking_results = []
        classification_results = []
        link_prediction_results = []
        
        logger.info(f"Processing network: {network_id}")
        
        for method in methods:
            try:
                # Generate embedding
                embedding = self.run_embedding_method(method, G, seeds, targets)
                
                # 1. Evaluate ranking (original task)
                ranking_result = self.evaluate_embedding(
                    embedding, G, seeds, targets, method, network_id
                )
                ranking_results.append(ranking_result)
                
                # 2. Evaluate node classification
                if include_classification:
                    classification_result = self.evaluate_embedding_classification(
                        embedding, G, method, network_id
                    )
                    classification_results.append(classification_result)
                
                # 3. Evaluate link prediction
                if include_link_prediction:
                    link_pred_result = self.evaluate_embedding_link_prediction(
                        embedding, G, method, network_id
                    )
                    link_prediction_results.append(link_pred_result)
                
                logger.info(f"  ✓ {method} completed on {network_id}")
                
            except Exception as e:
                logger.error(f"  ✗ {method} failed on {network_id}: {e}")
                continue
        
        return ranking_results, classification_results, link_prediction_results
    
    def run_all_methods_on_networks(
        self,
        networks: List[Tuple[str, nx.Graph, List[int], List[int]]],
        include_classification: bool = True,
        include_link_prediction: bool = True
    ):
        """
        Run all embedding methods on all networks and evaluate (PARALLELIZED).
        
        Networks are processed in parallel, with each network running all methods.
        Evaluates multiple downstream tasks: ranking, classification, link prediction.
        """
        methods = ['rwr', 'ctqw', 'dtqw', 'fused', 'netmf', 'node2vec']
        
        logger.info(f"Running {len(methods)} methods on {len(networks)} networks in parallel...")
        logger.info(f"Downstream tasks: ranking=True, classification={include_classification}, link_prediction={include_link_prediction}")
        logger.info(f"Using {self.n_jobs} parallel workers")
        
        # Process networks in parallel
        parallel = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=10)
        
        all_results = parallel(
            delayed(self._process_single_network)(
                network_tuple,
                include_classification=include_classification,
                include_link_prediction=include_link_prediction
            )
            for network_tuple in networks
        )
        
        # Flatten results
        for ranking_results, classification_results, link_pred_results in all_results:
            self.performance_results.extend(ranking_results)
            self.classification_results.extend(classification_results)
            self.link_prediction_results.extend(link_pred_results)
        
        # Save all results
        performance_df = pd.DataFrame(self.performance_results)
        performance_path = self.output_dir / "embedding_performance_ranking.csv"
        performance_df.to_csv(performance_path, index=False)
        logger.info(f"\nRanking performance results saved to {performance_path}")
        
        if include_classification and self.classification_results:
            classification_df = pd.DataFrame(self.classification_results)
            classification_path = self.output_dir / "embedding_performance_classification.csv"
            classification_df.to_csv(classification_path, index=False)
            logger.info(f"Classification performance results saved to {classification_path}")
        
        if include_link_prediction and self.link_prediction_results:
            link_pred_df = pd.DataFrame(self.link_prediction_results)
            link_pred_path = self.output_dir / "embedding_performance_link_prediction.csv"
            link_pred_df.to_csv(link_pred_path, index=False)
            logger.info(f"Link prediction performance results saved to {link_pred_path}")
        
        return performance_df
    
    def analyze_correlations(
        self,
        complexity_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        task_name: str = 'ranking'
    ):
        """
        Analyze correlations between complexity metrics and embedding performance.
        
        Parameters
        ----------
        complexity_df : pd.DataFrame
            Complexity metrics dataframe
        performance_df : pd.DataFrame
            Performance metrics dataframe
        task_name : str
            Name of the downstream task ('ranking', 'classification', 'link_prediction')
        """
        logger.info(f"Analyzing correlations between complexity and {task_name} performance...")
        
        # Merge dataframes
        merged_df = performance_df.merge(complexity_df, on='network_id')
        
        # Complexity metrics to analyze (including new topological metrics)
        complexity_metrics = [
            # Spectral metrics
            'spectral_gap', 'algebraic_connectivity', 'spectral_entropy',
            'von_neumann_entropy', 'quantum_complexity', 'estrada_index',
            'inverse_participation_ratio', 'participation_ratio',
            # Centrality metrics
            'centrality_entropy', 'centrality_variance',
            # Topological metrics (NEW)
            'orc_mean', 'orc_negative_fraction', 'cyclomatic_number',
            'kirchhoff_index', 'kirchhoff_per_pair', 'betti_0', 'betti_1',
            'persistence_entropy_0', 'persistence_entropy_1',
            # Quantum advantage scores
            'quantum_advantage_arithmetic', 'quantum_advantage_geometric', 'quantum_advantage_harmonic'
        ]
        
        # Performance metrics to analyze (task-specific)
        if task_name == 'ranking':
            performance_metrics = [col for col in performance_df.columns
                                 if 'precision@' in col or 'recall@' in col]
        elif task_name == 'classification':
            performance_metrics = [col for col in performance_df.columns
                                 if 'f1_' in col or 'accuracy_' in col or col in ['mean_f1_macro', 'mean_accuracy']]
        elif task_name == 'link_prediction':
            performance_metrics = [col for col in performance_df.columns
                                 if 'auc_' in col or 'mrr' in col or col in ['mean_auc_roc', 'mean_auc_pr', 'mean_mrr']]
        else:
            performance_metrics = [col for col in performance_df.columns
                                 if col not in ['network_id', 'method', 'network_type']]
        
        # Compute correlations for each method
        correlation_results = []
        
        for method in merged_df['method'].unique():
            method_df = merged_df[merged_df['method'] == method]
            
            for complexity_metric in complexity_metrics:
                for perf_metric in performance_metrics:
                    if complexity_metric in method_df.columns and perf_metric in method_df.columns:
                        # Remove NaN values
                        valid_data = method_df[[complexity_metric, perf_metric]].dropna()
                        
                        if len(valid_data) > 3:
                            try:
                                pearson_corr, pearson_p = pearsonr(
                                    valid_data[complexity_metric],
                                    valid_data[perf_metric]
                                )
                                spearman_corr, spearman_p = spearmanr(
                                    valid_data[complexity_metric],
                                    valid_data[perf_metric]
                                )
                                
                                correlation_results.append({
                                    'task': task_name,
                                    'method': method,
                                    'complexity_metric': complexity_metric,
                                    'performance_metric': perf_metric,
                                    'pearson_correlation': pearson_corr,
                                    'pearson_pvalue': pearson_p,
                                    'spearman_correlation': spearman_corr,
                                    'spearman_pvalue': spearman_p,
                                    'n_samples': len(valid_data)
                                })
                            except Exception as e:
                                logger.warning(f"Correlation computation failed for {complexity_metric} vs {perf_metric}: {e}")
                                continue
        
        correlation_df = pd.DataFrame(correlation_results)
        correlation_path = self.output_dir / f"complexity_performance_correlations_{task_name}.csv"
        correlation_df.to_csv(correlation_path, index=False)
        logger.info(f"Correlation analysis for {task_name} saved to {correlation_path}")
        
        return correlation_df, merged_df
    
    def create_visualizations(
        self,
        complexity_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        correlation_df: pd.DataFrame,
        merged_df: pd.DataFrame
    ):
        """Create comprehensive visualizations."""
        logger.info("Creating visualizations...")
        
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Complexity distribution by network type
        self._plot_complexity_distributions(complexity_df, viz_dir)
        
        # 2. Performance comparison across methods
        self._plot_performance_comparison(performance_df, viz_dir)
        
        # 3. Correlation heatmaps
        self._plot_correlation_heatmaps(correlation_df, viz_dir)
        
        # 4. Scatter plots for significant correlations
        self._plot_significant_correlations(correlation_df, merged_df, viz_dir)
        
        logger.info(f"Visualizations saved to {viz_dir}")
    
    def _plot_complexity_distributions(self, complexity_df: pd.DataFrame, viz_dir: Path):
        """Plot complexity metric distributions by network type."""
        metrics = ['quantum_complexity', 'von_neumann_entropy', 'spectral_gap',
                  'inverse_participation_ratio', 'participation_ratio']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in complexity_df.columns:
                ax = axes[idx]
                for net_type in ['scale_free', 'modular']:
                    data = complexity_df[complexity_df['network_type'] == net_type][metric]
                    ax.hist(data, alpha=0.6, label=net_type, bins=15)
                ax.set_xlabel(metric.replace('_', ' ').title())
                ax.set_ylabel('Count')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "complexity_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self, performance_df: pd.DataFrame, viz_dir: Path):
        """Plot performance comparison across methods."""
        metrics = ['precision@20_centroid', 'recall@20_centroid',
                  'precision@50_centroid', 'recall@50_centroid']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            if metric in performance_df.columns:
                ax = axes[idx]
                data = performance_df.groupby('method')[metric].agg(['mean', 'std'])
                data = data.sort_values('mean', ascending=False)
                
                ax.bar(range(len(data)), data['mean'], yerr=data['std'],
                      capsize=5, alpha=0.7)
                ax.set_xticks(range(len(data)))
                ax.set_xticklabels(data.index, rotation=45, ha='right')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(viz_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmaps(self, correlation_df: pd.DataFrame, viz_dir: Path):
        """Plot correlation heatmaps."""
        # Filter significant correlations
        sig_corr = correlation_df[correlation_df['pearson_pvalue'] < 0.05]
        
        if len(sig_corr) == 0:
            logger.warning("No significant correlations found")
            return
        
        # Create pivot table for heatmap
        for method in sig_corr['method'].unique():
            method_corr = sig_corr[sig_corr['method'] == method]
            
            pivot = method_corr.pivot_table(
                values='pearson_correlation',
                index='complexity_metric',
                columns='performance_metric',
                aggfunc='mean'
            )
            
            if pivot.empty:
                continue
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                       center=0, vmin=-1, vmax=1, cbar_kws={'label': 'Pearson Correlation'})
            plt.title(f'Complexity-Performance Correlations: {method.upper()}')
            plt.tight_layout()
            plt.savefig(viz_dir / f"correlation_heatmap_{method}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_significant_correlations(
        self,
        correlation_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        viz_dir: Path
    ):
        """Plot scatter plots for significant correlations."""
        # Get top correlations
        sig_corr = correlation_df[
            (correlation_df['pearson_pvalue'] < 0.05) &
            (abs(correlation_df['pearson_correlation']) > 0.3)
        ].sort_values('pearson_correlation', key=abs, ascending=False).head(12)
        
        if len(sig_corr) == 0:
            logger.warning("No significant correlations to plot")
            return
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for idx, (_, row) in enumerate(sig_corr.iterrows()):
            if idx >= 12:
                break
            
            ax = axes[idx]
            method = row['method']
            complexity_metric = row['complexity_metric']
            perf_metric = row['performance_metric']
            
            method_df = merged_df[merged_df['method'] == method]
            
            ax.scatter(method_df[complexity_metric], method_df[perf_metric],
                      alpha=0.6, s=50)
            ax.set_xlabel(complexity_metric.replace('_', ' ').title())
            ax.set_ylabel(perf_metric.replace('_', ' ').title())
            ax.set_title(f"{method.upper()}\nr={row['pearson_correlation']:.2f}, p={row['pearson_pvalue']:.3f}")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(viz_dir / "significant_correlations.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_recommendations(
        self,
        correlation_df: pd.DataFrame,
        complexity_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        task_name: str = 'ranking',
        performance_metric: str = None
    ):
        """
        Generate method recommendations based on complexity metrics.
        
        Parameters
        ----------
        correlation_df : pd.DataFrame
            Correlation analysis results
        complexity_df : pd.DataFrame
            Complexity metrics
        performance_df : pd.DataFrame
            Performance metrics
        task_name : str
            Downstream task name
        performance_metric : str, optional
            Specific performance metric to optimize (auto-selected if None)
        """
        logger.info(f"Generating recommendations for {task_name} task...")
        
        # Analyze which methods perform best under different complexity conditions
        merged_df = performance_df.merge(complexity_df, on='network_id')
        
        # Auto-select performance metric if not specified
        if performance_metric is None:
            if task_name == 'ranking':
                performance_metric = 'recall@50_centroid'
            elif task_name == 'classification':
                performance_metric = 'mean_f1_macro'
            elif task_name == 'link_prediction':
                performance_metric = 'mean_auc_roc'
        
        if performance_metric not in merged_df.columns:
            logger.warning(f"Performance metric {performance_metric} not found. Using first available metric.")
            perf_cols = [col for col in merged_df.columns if col not in ['network_id', 'method', 'network_type']]
            if perf_cols:
                performance_metric = perf_cols[0]
            else:
                logger.error("No performance metrics found!")
                return
        
        recommendations = []
        
        # Define complexity thresholds (based on quartiles) - including new topological metrics
        complexity_metrics = [
            'quantum_complexity', 'von_neumann_entropy', 'spectral_gap',
            'inverse_participation_ratio', 'orc_mean', 'cyclomatic_number',
            'kirchhoff_index', 'quantum_advantage_arithmetic',
            'quantum_advantage_geometric', 'quantum_advantage_harmonic'
        ]
        
        for metric in complexity_metrics:
            if metric not in complexity_df.columns:
                continue
            
            q25 = complexity_df[metric].quantile(0.25)
            q75 = complexity_df[metric].quantile(0.75)
            
            # Low complexity
            low_df = merged_df[merged_df[metric] <= q25]
            if len(low_df) > 0 and performance_metric in low_df.columns:
                try:
                    best_method_low = low_df.groupby('method')[performance_metric].mean().idxmax()
                    best_score_low = low_df.groupby('method')[performance_metric].mean().max()
                    
                    recommendations.append({
                        'task': task_name,
                        'complexity_metric': metric,
                        'condition': 'low',
                        'threshold': f'<= {q25:.3f}',
                        'recommended_method': best_method_low,
                        'performance_metric': performance_metric,
                        'avg_performance': best_score_low
                    })
                except Exception as e:
                    logger.warning(f"Failed to compute recommendation for low {metric}: {e}")
            
            # High complexity
            high_df = merged_df[merged_df[metric] >= q75]
            if len(high_df) > 0 and performance_metric in high_df.columns:
                try:
                    best_method_high = high_df.groupby('method')[performance_metric].mean().idxmax()
                    best_score_high = high_df.groupby('method')[performance_metric].mean().max()
                    
                    recommendations.append({
                        'task': task_name,
                        'complexity_metric': metric,
                        'condition': 'high',
                        'threshold': f'>= {q75:.3f}',
                        'recommended_method': best_method_high,
                        'performance_metric': performance_metric,
                        'avg_performance': best_score_high
                    })
                except Exception as e:
                    logger.warning(f"Failed to compute recommendation for high {metric}: {e}")
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_path = self.output_dir / f"method_recommendations_{task_name}.csv"
        recommendations_df.to_csv(recommendations_path, index=False)
        logger.info(f"Recommendations for {task_name} saved to {recommendations_path}")
        
        return recommendations_df
        
        # Create recommendation table
        self._create_recommendation_table(recommendations_df)
        
        return recommendations_df
    
    def _create_recommendation_table(self, recommendations_df: pd.DataFrame):
        """Create a formatted recommendation table."""
        report_path = self.output_dir / "recommendations_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("EMBEDDING METHOD RECOMMENDATIONS BASED ON NETWORK COMPLEXITY\n")
            f.write("="*80 + "\n\n")
            
            f.write("This guide suggests which embedding method to use based on\n")
            f.write("specific complexity characteristics of your network.\n\n")
            
            for metric in recommendations_df['complexity_metric'].unique():
                f.write(f"\n{metric.replace('_', ' ').upper()}\n")
                f.write("-" * 60 + "\n")
                
                metric_recs = recommendations_df[recommendations_df['complexity_metric'] == metric]
                
                for _, row in metric_recs.iterrows():
                    f.write(f"\n  {row['condition'].upper()} {metric} ({row['threshold']}):\n")
                    f.write(f"    → Recommended: {row['recommended_method'].upper()}\n")
                    f.write(f"    → Avg Recall@50: {row['avg_recall@50']:.3f}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("GENERAL GUIDELINES\n")
            f.write("="*80 + "\n\n")
            
            # Add general insights
            f.write("• QuVINE-fused: Best for complex networks with high modularity\n")
            f.write("• CTQW: Effective for networks with low spectral gap (bottlenecks)\n")
            f.write("• DTQW: Good for structured networks with clear communities\n")
            f.write("• RWR: Reliable baseline for most network types\n")
            f.write("• NetMF: Fast and effective for large-scale networks\n")
            f.write("• Node2Vec: Classical baseline, good for comparison\n\n")
        
        logger.info(f"Recommendation report saved to {report_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        logger.info("="*80)
        logger.info("STARTING COMPREHENSIVE EMBEDDING ANALYSIS")
        logger.info("="*80)
        
        # Step 1: Generate networks
        networks = self.generate_networks()
        
        # Step 2: Compute complexity
        complexity_df = self.compute_complexity_for_all(networks)
        
        # Step 3: Run all methods and evaluate
        performance_df = self.run_all_methods_on_networks(networks)
        
        # Step 4: Analyze correlations
        correlation_df, merged_df = self.analyze_correlations(complexity_df, performance_df)
        
        # Step 5: Create visualizations
        self.create_visualizations(complexity_df, performance_df, correlation_df, merged_df)
        
        # Step 6: Generate recommendations
        recommendations_df = self.generate_recommendations(correlation_df, complexity_df, performance_df)
        
        logger.info("="*80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info("="*80)
        
        return {
            'complexity': complexity_df,
            'performance': performance_df,
            'correlations': correlation_df,
            'recommendations': recommendations_df,
            'merged': merged_df
        }

def run_single_network_analysis(
    G: nx.Graph,
    network_id: str,
    network_metadata: Dict,
    output_dir: str,
    embedding_methods: List[str] = None,
    embedding_dim: int = 128,
    num_seeds: int = 15,
    num_targets: int = 25,
    verbose: bool = True
) -> Dict:
    """
    Run complete analysis for a single network (for HPC parallelization).
    
    This function:
    1. Computes complexity metrics
    2. Generates embeddings for all specified methods
    3. Evaluates all downstream tasks (ranking, classification, link prediction)
    4. Saves all results to network-specific subdirectory
    
    Parameters
    ----------
    G : nx.Graph
        Input network
    network_id : str
        Unique identifier for the network
    network_metadata : dict
        Metadata about the network (type, parameters, etc.)
    output_dir : str
        Output directory for this network's results
    embedding_methods : list of str, optional
        Methods to run. Default: ['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec']
    embedding_dim : int
        Embedding dimension
    num_seeds : int
        Number of seed nodes
    num_targets : int
        Number of target nodes
    verbose : bool
        Print progress messages
        
    Returns
    -------
    dict
        Summary of results with paths to saved files
    """
    import os
    from pathlib import Path
    
    if embedding_methods is None:
        embedding_methods = ['quvine_fused', 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'netmf', 'node2vec']
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        logger.info(f"="*80)
        logger.info(f"Processing network: {network_id}")
        logger.info(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"="*80)
    
    # Initialize temporary analysis object
    temp_analysis = ComprehensiveEmbeddingAnalysis(
        output_dir=output_dir,
        n_networks_per_type=1,
        n_nodes=G.number_of_nodes(),
        num_seeds=num_seeds,
        num_targets=num_targets,
        embedding_dim=embedding_dim,
        seed=42,
        n_jobs=1  # Single network, no parallelization needed
    )
    
    # Select seeds and targets
    seeds, targets = temp_analysis._select_seeds_targets(G)
    
    # Step 1: Compute complexity metrics
    if verbose:
        logger.info("Step 1/4: Computing complexity metrics...")
    
    complexity_metrics = compute_graph_complexity_metrics(G)
    complexity_metrics['network_id'] = network_id
    complexity_metrics['network_type'] = network_metadata.get('type', 'unknown')
    complexity_metrics['n_nodes'] = G.number_of_nodes()
    complexity_metrics['n_edges'] = G.number_of_edges()
    
    # Save complexity metrics
    complexity_df = pd.DataFrame([complexity_metrics])
    complexity_path = output_path / f"{network_id}_complexity.csv"
    complexity_df.to_csv(complexity_path, index=False)
    
    if verbose:
        logger.info(f"  ✓ Complexity metrics saved to {complexity_path}")
    
    # Step 2: Generate embeddings and evaluate all tasks
    all_results = {
        'ranking': [],
        'classification': [],
        'link_prediction': []
    }
    
    for method in embedding_methods:
        if verbose:
            logger.info(f"Step 2/4: Processing method: {method}")
        
        try:
            # Generate embedding
            embedding = temp_analysis.run_embedding_method(method, G, seeds, targets)
            
            # Save embedding
            embedding_path = output_path / f"{network_id}_{method}_embedding.npy"
            np.save(embedding_path, embedding)
            
            # Task 1: Ranking (node prioritization)
            if verbose:
                logger.info(f"  - Evaluating ranking task...")
            ranking_result = temp_analysis.evaluate_embedding(
                embedding, G, seeds, targets, method, network_id
            )
            all_results['ranking'].append(ranking_result)
            
            # Task 2: Node classification
            if verbose:
                logger.info(f"  - Evaluating classification task...")
            classification_result = temp_analysis.evaluate_embedding_classification(
                embedding, G, method, network_id
            )
            all_results['classification'].append(classification_result)
            
            # Task 3: Link prediction
            if verbose:
                logger.info(f"  - Evaluating link prediction task...")
            link_pred_result = temp_analysis.evaluate_embedding_link_prediction(
                embedding, G, method, network_id
            )
            all_results['link_prediction'].append(link_pred_result)
            
            if verbose:
                logger.info(f"  ✓ {method} completed successfully")
                
        except Exception as e:
            logger.error(f"  ✗ {method} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Save all task results
    if verbose:
        logger.info("Step 3/4: Saving task results...")
    
    # Save ranking results
    if all_results['ranking']:
        ranking_df = pd.DataFrame(all_results['ranking'])
        ranking_path = output_path / f"{network_id}_ranking_results.csv"
        ranking_df.to_csv(ranking_path, index=False)
        if verbose:
            logger.info(f"  ✓ Ranking results saved to {ranking_path}")
    
    # Save classification results
    if all_results['classification']:
        classification_df = pd.DataFrame(all_results['classification'])
        classification_path = output_path / f"{network_id}_classification_results.csv"
        classification_df.to_csv(classification_path, index=False)
        if verbose:
            logger.info(f"  ✓ Classification results saved to {classification_path}")
    
    # Save link prediction results
    if all_results['link_prediction']:
        link_pred_df = pd.DataFrame(all_results['link_prediction'])
        link_pred_path = output_path / f"{network_id}_link_prediction_results.csv"
        link_pred_df.to_csv(link_pred_path, index=False)
        if verbose:
            logger.info(f"  ✓ Link prediction results saved to {link_pred_path}")
    
    # Step 4: Create summary
    if verbose:
        logger.info("Step 4/4: Creating summary...")
    
    summary = {
        'network_id': network_id,
        'network_type': network_metadata.get('type', 'unknown'),
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'methods_completed': len(all_results['ranking']),
        'methods_requested': len(embedding_methods),
        'output_dir': str(output_path),
        'complexity_file': str(complexity_path),
        'ranking_file': str(ranking_path) if all_results['ranking'] else None,
        'classification_file': str(classification_path) if all_results['classification'] else None,
        'link_prediction_file': str(link_pred_path) if all_results['link_prediction'] else None
    }
    
    # Save summary
    summary_path = output_path / f"{network_id}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        logger.info(f"="*80)
        logger.info(f"✓ Analysis complete for {network_id}")
        logger.info(f"  Methods completed: {summary['methods_completed']}/{summary['methods_requested']}")
        logger.info(f"  Results saved to: {output_dir}")
        logger.info(f"="*80)
    
    return summary


def collect_and_aggregate_results(
    results_dir: str,
    output_file: str = "comprehensive_results.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Collect and aggregate results from all network analyses into a single CSV.
    
    This function:
    1. Scans the results directory for all network subdirectories
    2. Loads complexity metrics and task results for each network
    3. Merges all data into a single comprehensive DataFrame
    4. Each row = one network × one method combination
    5. Columns = complexity metrics + all task performance metrics
    
    Parameters
    ----------
    results_dir : str
        Directory containing network subdirectories with results
    output_file : str
        Name of output CSV file (saved in results_dir)
    verbose : bool
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Comprehensive results with all networks, methods, complexity, and performance metrics
    """
    from pathlib import Path
    import glob
    
    results_path = Path(results_dir)
    
    if verbose:
        logger.info("="*80)
        logger.info("COLLECTING AND AGGREGATING RESULTS")
        logger.info(f"Results directory: {results_dir}")
        logger.info("="*80)
    
    # Find all network subdirectories
    network_dirs = [d for d in results_path.iterdir() if d.is_dir()]
    
    if verbose:
        logger.info(f"Found {len(network_dirs)} network directories")
    
    all_data = []
    
    for network_dir in network_dirs:
        network_id = network_dir.name
        
        if verbose:
            logger.info(f"Processing: {network_id}")
        
        try:
            # Load complexity metrics
            complexity_files = list(network_dir.glob("*_complexity.csv"))
            if not complexity_files:
                logger.warning(f"  No complexity file found for {network_id}")
                continue
            
            complexity_df = pd.read_csv(complexity_files[0])
            complexity_dict = complexity_df.iloc[0].to_dict()
            
            # Load ranking results
            ranking_files = list(network_dir.glob("*_ranking_results.csv"))
            ranking_df = pd.read_csv(ranking_files[0]) if ranking_files else pd.DataFrame()
            
            # Load classification results
            classification_files = list(network_dir.glob("*_classification_results.csv"))
            classification_df = pd.read_csv(classification_files[0]) if classification_files else pd.DataFrame()
            
            # Load link prediction results
            link_pred_files = list(network_dir.glob("*_link_prediction_results.csv"))
            link_pred_df = pd.read_csv(link_pred_files[0]) if link_pred_files else pd.DataFrame()
            
            # Get unique methods
            methods = set()
            if not ranking_df.empty:
                methods.update(ranking_df['method'].unique())
            if not classification_df.empty:
                methods.update(classification_df['method'].unique())
            if not link_pred_df.empty:
                methods.update(link_pred_df['method'].unique())
            
            # Create one row per method
            for method in methods:
                row_data = complexity_dict.copy()
                row_data['method'] = method
                
                # Add ranking metrics
                if not ranking_df.empty:
                    method_ranking = ranking_df[ranking_df['method'] == method]
                    if not method_ranking.empty:
                        for col in method_ranking.columns:
                            if col not in ['network_id', 'method', 'network_type']:
                                row_data[f'ranking_{col}'] = method_ranking.iloc[0][col]
                
                # Add classification metrics
                if not classification_df.empty:
                    method_classification = classification_df[classification_df['method'] == method]
                    if not method_classification.empty:
                        for col in method_classification.columns:
                            if col not in ['network_id', 'method', 'network_type']:
                                row_data[f'classification_{col}'] = method_classification.iloc[0][col]
                
                # Add link prediction metrics
                if not link_pred_df.empty:
                    method_link_pred = link_pred_df[link_pred_df['method'] == method]
                    if not method_link_pred.empty:
                        for col in method_link_pred.columns:
                            if col not in ['network_id', 'method', 'network_type']:
                                row_data[f'link_prediction_{col}'] = method_link_pred.iloc[0][col]
                
                all_data.append(row_data)
            
            if verbose:
                logger.info(f"  ✓ Loaded {len(methods)} methods")
                
        except Exception as e:
            logger.error(f"  ✗ Failed to process {network_id}: {e}")
            continue
    
    # Create comprehensive DataFrame
    comprehensive_df = pd.DataFrame(all_data)
    
    # Save to CSV
    output_path = results_path / output_file
    comprehensive_df.to_csv(output_path, index=False)
    
    if verbose:
        logger.info("="*80)
        logger.info("AGGREGATION COMPLETE")
        logger.info(f"Total rows: {len(comprehensive_df)}")
        logger.info(f"Total columns: {len(comprehensive_df.columns)}")
        logger.info(f"Networks: {comprehensive_df['network_id'].nunique()}")
        logger.info(f"Methods: {comprehensive_df['method'].nunique()}")
        logger.info(f"Saved to: {output_path}")
        logger.info("="*80)
    
    return comprehensive_df



def main():
    """Main entry point."""
    analysis = ComprehensiveEmbeddingAnalysis(
        output_dir="outputs/comprehensive_analysis",
        n_networks_per_type=20,
        n_nodes=200,
        num_seeds=15,
        num_targets=25,
        embedding_dim=128,
        seed=42
    )
    
    results = analysis.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()

# Made with Bob
