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
        
        return results
    
    def _process_single_network(
        self,
        network_tuple: Tuple[str, nx.Graph, List[int], List[int]]
    ) -> List[Dict]:
        """
        Process a single network with all methods (for parallel execution).
        
        Parameters
        ----------
        network_tuple : tuple
            (network_id, G, seeds, targets)
            
        Returns
        -------
        list of dict
            Performance results for all methods on this network
        """
        network_id, G, seeds, targets = network_tuple
        methods = ['rwr', 'ctqw', 'dtqw', 'fused', 'netmf', 'node2vec']
        
        results = []
        logger.info(f"Processing network: {network_id}")
        
        for method in methods:
            try:
                # Generate embedding
                embedding = self.run_embedding_method(method, G, seeds, targets)
                
                # Evaluate
                result = self.evaluate_embedding(
                    embedding, G, seeds, targets, method, network_id
                )
                
                results.append(result)
                logger.info(f"  ✓ {method} completed on {network_id}")
                
            except Exception as e:
                logger.error(f"  ✗ {method} failed on {network_id}: {e}")
                continue
        
        return results
    
    def run_all_methods_on_networks(
        self,
        networks: List[Tuple[str, nx.Graph, List[int], List[int]]]
    ):
        """
        Run all embedding methods on all networks and evaluate (PARALLELIZED).
        
        Networks are processed in parallel, with each network running all methods.
        """
        methods = ['rwr', 'ctqw', 'dtqw', 'fused', 'netmf', 'node2vec']
        
        logger.info(f"Running {len(methods)} methods on {len(networks)} networks in parallel...")
        logger.info(f"Using {self.n_jobs} parallel workers")
        
        # Process networks in parallel
        parallel = Parallel(n_jobs=self.n_jobs, backend='loky', verbose=10)
        
        all_results = parallel(
            delayed(self._process_single_network)(network_tuple)
            for network_tuple in networks
        )
        
        # Flatten results
        for network_results in all_results:
            self.performance_results.extend(network_results)
        
        # Save performance results
        performance_df = pd.DataFrame(self.performance_results)
        performance_path = self.output_dir / "embedding_performance.csv"
        performance_df.to_csv(performance_path, index=False)
        logger.info(f"\nPerformance results saved to {performance_path}")
        
        return performance_df
    
    def analyze_correlations(
        self,
        complexity_df: pd.DataFrame,
        performance_df: pd.DataFrame
    ):
        """
        Analyze correlations between complexity metrics and embedding performance.
        """
        logger.info("Analyzing correlations between complexity and performance...")
        
        # Merge dataframes
        merged_df = performance_df.merge(complexity_df, on='network_id')
        
        # Complexity metrics to analyze
        complexity_metrics = [
            'spectral_gap', 'algebraic_connectivity', 'spectral_entropy',
            'von_neumann_entropy', 'quantum_complexity', 'estrada_index',
            'inverse_participation_ratio', 'participation_ratio',
            'centrality_entropy', 'centrality_variance'
        ]
        
        # Performance metrics to analyze
        performance_metrics = [col for col in performance_df.columns 
                             if 'precision@' in col or 'recall@' in col]
        
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
                            pearson_corr, pearson_p = pearsonr(
                                valid_data[complexity_metric],
                                valid_data[perf_metric]
                            )
                            spearman_corr, spearman_p = spearmanr(
                                valid_data[complexity_metric],
                                valid_data[perf_metric]
                            )
                            
                            correlation_results.append({
                                'method': method,
                                'complexity_metric': complexity_metric,
                                'performance_metric': perf_metric,
                                'pearson_correlation': pearson_corr,
                                'pearson_pvalue': pearson_p,
                                'spearman_correlation': spearman_corr,
                                'spearman_pvalue': spearman_p,
                                'n_samples': len(valid_data)
                            })
        
        correlation_df = pd.DataFrame(correlation_results)
        correlation_path = self.output_dir / "complexity_performance_correlations.csv"
        correlation_df.to_csv(correlation_path, index=False)
        logger.info(f"Correlation analysis saved to {correlation_path}")
        
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
        performance_df: pd.DataFrame
    ):
        """Generate method recommendations based on complexity metrics."""
        logger.info("Generating recommendations...")
        
        # Analyze which methods perform best under different complexity conditions
        merged_df = performance_df.merge(complexity_df, on='network_id')
        
        recommendations = []
        
        # Define complexity thresholds (based on quartiles)
        complexity_metrics = ['quantum_complexity', 'von_neumann_entropy',
                            'spectral_gap', 'inverse_participation_ratio']
        
        for metric in complexity_metrics:
            if metric not in complexity_df.columns:
                continue
            
            q25 = complexity_df[metric].quantile(0.25)
            q75 = complexity_df[metric].quantile(0.75)
            
            # Low complexity
            low_df = merged_df[merged_df[metric] <= q25]
            if len(low_df) > 0:
                best_method_low = low_df.groupby('method')['recall@50_centroid'].mean().idxmax()
                best_score_low = low_df.groupby('method')['recall@50_centroid'].mean().max()
                
                recommendations.append({
                    'complexity_metric': metric,
                    'condition': 'low',
                    'threshold': f'<= {q25:.3f}',
                    'recommended_method': best_method_low,
                    'avg_recall@50': best_score_low
                })
            
            # High complexity
            high_df = merged_df[merged_df[metric] >= q75]
            if len(high_df) > 0:
                best_method_high = high_df.groupby('method')['recall@50_centroid'].mean().idxmax()
                best_score_high = high_df.groupby('method')['recall@50_centroid'].mean().max()
                
                recommendations.append({
                    'complexity_metric': metric,
                    'condition': 'high',
                    'threshold': f'>= {q75:.3f}',
                    'recommended_method': best_method_high,
                    'avg_recall@50': best_score_high
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_path = self.output_dir / "method_recommendations.csv"
        recommendations_df.to_csv(recommendations_path, index=False)
        logger.info(f"Recommendations saved to {recommendations_path}")
        
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
