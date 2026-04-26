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
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from omegaconf import DictConfig, OmegaConf
from joblib import Parallel, delayed
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optuna = None
import multiprocessing

from quvine.data.random_graphs import (
    generate_barabasi_albert,
    generate_modular_network,
    get_graph_statistics
)
from quvine.complexity.graph_enhanced import compute_enhanced_complexity_metrics, ComplexityConfig
from quvine.complexity.qbc import compute_qbc_metrics

# Import baselines with fallback for missing dependencies
try:
    from quvine.baselines import run_appnp
except ImportError:
    run_appnp = None

try:
    from quvine.baselines import run_netmf
except ImportError:
    run_netmf = None

try:
    from quvine.baselines import run_node2vec
except ImportError:
    run_node2vec = None

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
    summarize_classification_results,
    flatten_classification_results,
    evaluate_nc_stratified,
)
from quvine.evaluation.link_prediction import (
    evaluate_link_prediction_cv,
    evaluate_all_edge_feature_methods,
    summarize_link_prediction_results,
    split_edges,
    sample_negative_edges,
    _make_bidi,
)
from quvine.embedding.quantum_filters import (
    generate_quvine_heat_embedding,
    generate_quvine_poly_embedding,
    generate_baseline_filter_embedding,
    generate_baseline_heat_embedding,
    generate_baseline_poly_embedding,
    generate_rwr_heat_embedding,
    generate_rwr_poly_embedding,
)

from quvine.baselines.gat import (
    generate_gat_embedding,
    generate_gat_embedding_by_method_name,
    GATConfig,
    TrainConfig as GATTrainConfig,
)
try:
    from quvine.baselines.graphgps import (
        generate_graphgps_embedding,
        generate_graphgps_embedding_by_method_name,
        GraphGPSConfig,
        TrainConfig as GraphGPSTrainConfig,
    )
except ImportError:
    generate_graphgps_embedding = None
    generate_graphgps_embedding_by_method_name = None
    GraphGPSConfig = None
    GraphGPSTrainConfig = None
from quvine.embedding.registry import EmbeddingStore



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metric columns written into each tidy link-prediction row.
_LP_METRIC_KEYS = ['auc_roc', 'auc_pr', 'f1', 'mrr',
                   'n_positive', 'n_negative', 'n_train', 'n_test']


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
        
        # Hyperparameter tuning storage keyed by encountered network type.
        # Shape: {network_type: {method_key: params_dict}}
        self.tuned_hyperparameters: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
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
        Load synthetic benchmark networks for testing.
        
        Includes synthetic benchmarks from random_graphs.py:
        - Watts-Strogatz (small-world)
        - Powerlaw Cluster (scale-free with clustering)
        - Hierarchical Network
        - Core-Periphery
        - Erdos-Renyi (random)
        
        Returns
        -------
        list of tuples
            Each tuple contains (network_id, graph, seeds, targets)
        """
        from quvine.data.random_graphs import (
            generate_watts_strogatz,
            generate_powerlaw_cluster,
            generate_hierarchical_network,
            generate_core_periphery,
            generate_erdos_renyi
        )
        
        logger.info("Loading synthetic benchmark networks...")
        networks = []
        
        # ===== Synthetic Benchmarks from random_graphs.py =====
        
        # 6. Watts-Strogatz Small-World (200 nodes)
        try:
            G = generate_watts_strogatz(n=self.n_nodes, k=6, p=0.1, seed=self.base_seed)
            if nx.is_connected(G):
                seeds, targets = self._select_seeds_targets(G)
                networks.append(("benchmark_watts_strogatz", G, seeds, targets))
                logger.info(f"  Generated Watts-Strogatz: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to generate Watts-Strogatz: {e}")
        
        # 7. Powerlaw Cluster (200 nodes)
        try:
            G = generate_powerlaw_cluster(n=self.n_nodes, m=3, p=0.1, seed=self.base_seed)
            if nx.is_connected(G):
                seeds, targets = self._select_seeds_targets(G)
                networks.append(("benchmark_powerlaw_cluster", G, seeds, targets))
                logger.info(f"  Generated Powerlaw Cluster: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to generate Powerlaw Cluster: {e}")
        
        # 8. Hierarchical Network (200 nodes)
        try:
            G, _ = generate_hierarchical_network(levels=3, branching_factor=4, seed=self.base_seed)
            if nx.is_connected(G):
                seeds, targets = self._select_seeds_targets(G)
                networks.append(("benchmark_hierarchical", G, seeds, targets))
                logger.info(f"  Generated Hierarchical: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to generate Hierarchical: {e}")
        
        # 9. Core-Periphery (200 nodes)
        try:
            G, _, _ = generate_core_periphery(n_core=50, n_periphery=150, p_core=0.3, p_core_periphery=0.1, p_periphery=0.05, seed=self.base_seed)
            if nx.is_connected(G):
                seeds, targets = self._select_seeds_targets(G)
                networks.append(("benchmark_core_periphery", G, seeds, targets))
                logger.info(f"  Generated Core-Periphery: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to generate Core-Periphery: {e}")
        
        # 10. Erdos-Renyi (200 nodes)
        try:
            G = generate_erdos_renyi(n=self.n_nodes, p=0.02, seed=self.base_seed)
            if nx.is_connected(G):
                seeds, targets = self._select_seeds_targets(G)
                networks.append(("benchmark_erdos_renyi", G, seeds, targets))
                logger.info(f"  Generated Erdos-Renyi: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        except Exception as e:
            logger.warning(f"  Failed to generate Erdos-Renyi: {e}")
        
        logger.info(f"Loaded {len(networks)} benchmark networks (real + synthetic)")
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
        
        Uses graph_enhanced metrics (36 comprehensive metrics) and QBC metrics.
        
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
        
        # Use enhanced complexity metrics with default config
        config = ComplexityConfig(
            spectral_k=64,
            path_num_sources=64,
            betweenness_k=256,
            random_state=self.base_seed
        )
        metrics = compute_enhanced_complexity_metrics(G, config=config)
        
        # Add QBC metrics
        try:
            metrics.update(compute_qbc_metrics(G))
        except Exception as _qbc_exc:
            warnings.warn(f"QBC metrics failed for {network_id}: {_qbc_exc}")
        
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
    
        
    def tune_gcnmf_hyperparameters(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        diffusion_type: str = 'heat',
        n_trials: int = 50,
        timeout: int = 3600,
        n_jobs_optuna: int = 1
    ) -> Dict:
        """
        Tune GCN-MF hyperparameters using Bayesian optimization (Optuna).
        
        This function performs hyperparameter tuning for GCN-MF methods by:
        1. Creating a validation split (80% train, 20% val)
        2. Using Optuna's TPE sampler for Bayesian optimization
        3. Optimizing for validation ranking performance (recall@50)
        4. Returning best hyperparameters and study results
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes for evaluation
        targets : list
            Target nodes for evaluation
        diffusion_type : str
            'heat' or 'poly' for diffusion type
        n_trials : int
            Number of optimization trials (default: 50)
        timeout : int
            Maximum time in seconds for optimization (default: 3600 = 1 hour)
        n_jobs_optuna : int
            Number of parallel jobs for Optuna (default: 1, set to -1 for all cores)
            
        Returns
        -------
        dict
            Dictionary containing:
            - 'best_params': Best hyperparameters found
            - 'best_value': Best validation score
            - 'study': Optuna study object
            - 'trials_df': DataFrame with all trials
        
        Example
        -------
        >>> analyzer = ComprehensiveEmbeddingAnalysis()
        >>> result = analyzer.tune_gcnmf_hyperparameters(
        ...     G, seeds, targets, diffusion_type='heat', n_trials=30
        ... )
        >>> print(f"Best params: {result['best_params']}")
        >>> print(f"Best recall@50: {result['best_value']:.3f}")
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError(
                "Optuna is required for hyperparameter tuning. "
                "Install with: pip install optuna"
            )
        
        logger.info(f"Starting hyperparameter tuning for GCN-MF ({diffusion_type})")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create train/val split for seeds (80/20) BEFORE generating quantum targets
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)  # 20% for validation
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        
        logger.info(f"Train seeds: {len(train_seeds)}, Val seeds: {len(val_seeds)}")
        
        # Generate quantum targets for calibration using ONLY train seeds (no data leakage)
        q_targets = self._generate_quantum_targets(G, train_seeds)
        
        def objective(trial):
            """Optuna objective function."""
            # Suggest hyperparameters
            n_layers = trial.suggest_int('n_layers', 1, 3)
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            mf_dim = trial.suggest_categorical('mf_dim', [32, 64, 128])
            epochs = trial.suggest_int('epochs', 100, 500, step=100)
            lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            
            # Additional params for polynomial diffusion
            if diffusion_type == 'poly':
                K = trial.suggest_int('K', 2, 6)
                ridge = trial.suggest_float('ridge', 1e-7, 1e-5, log=True)
            else:
                K = 4
                ridge = 1e-6
            
            try:
                # Generate embedding with suggested hyperparameters
                embeddings, _ = generate_quvine_gcnmf_embedding(
                    G=G,
                    q_targets=q_targets,
                    embedding_dim=self.embedding_dim,
                    diffusion_type=diffusion_type,
                    K=K,
                    ridge=ridge,
                    hidden_dim=hidden_dim,
                    mf_dim=mf_dim,
                    n_layers=n_layers,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    normalize_laplacian=True,
                    random_state=self.base_seed
                )
                
                # Evaluate on validation set
                # Use centroid-based ranking
                scores_centroid = seed_centroid_scores(embeddings, val_seeds)
                
                # Compute recall@50 on validation seeds
                k = min(50, len(G) - 1)
                top_k_indices = np.argsort(scores_centroid)[-k:]
                
                # Count how many validation seeds are in top-k
                hits = sum(1 for seed in val_seeds if seed in top_k_indices)
                recall_at_k = hits / len(val_seeds) if len(val_seeds) > 0 else 0.0
                
                return recall_at_k
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                # Return a poor score for failed trials
                return 0.0
        
        # Create Optuna study
        sampler = TPESampler(seed=self.base_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'gcnmf_{diffusion_type}_tuning'
        )
        
        # Optimize
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs_optuna,
            show_progress_bar=True
        )
        
        # Get results
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Tuning complete!")
        logger.info(f"Best validation recall@50: {best_value:.4f}")
        logger.info(f"Best hyperparameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'study': study,
            'trials_df': study.trials_dataframe(),
            'method': f'{diffusion_type}gcnmf'
        }
    
    def tune_node2vec_hyperparameters(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        n_trials: int = 50,
        timeout: int = 1800,
        n_jobs_optuna: int = 1
    ) -> Dict:
        """
        Tune Node2Vec hyperparameters using Bayesian optimization (Optuna).
        
        Optimizes walk parameters (p, q, walk_length, num_walks, window) for
        best validation ranking performance.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes for evaluation
        targets : list
            Target nodes for evaluation
        n_trials : int
            Number of optimization trials (default: 50)
        timeout : int
            Maximum time in seconds (default: 1800 = 30 minutes)
        n_jobs_optuna : int
            Number of parallel jobs for Optuna (default: 1)
            
        Returns
        -------
        dict
            Dictionary with best_params, best_value, study, trials_df
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        logger.info("Starting hyperparameter tuning for Node2Vec")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create train/val split
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        
        logger.info(f"Train seeds: {len(train_seeds)}, Val seeds: {len(val_seeds)}")
        
        nodes = list(G.nodes())
        
        def objective(trial):
            """Optuna objective for Node2Vec."""
            # Suggest hyperparameters
            p = trial.suggest_float('p', 0.25, 4.0)
            q = trial.suggest_float('q', 0.25, 4.0)
            walk_length = trial.suggest_int('walk_length', 10, 80, step=10)
            num_walks = trial.suggest_int('num_walks', 10, 100, step=10)
            window = trial.suggest_int('window', 3, 10)
            
            try:
                # Generate embedding
                embedding = run_node2vec(
                    graph=G,
                    nodes=nodes,
                    dimensions=self.embedding_dim,
                    walk_length=walk_length,
                    num_walks=num_walks,
                    p=p,
                    q=q,
                    window=window,
                    min_count=1,
                    workers=1,
                    seed=self.base_seed
                )
                
                # Evaluate on validation set
                scores_centroid = seed_centroid_scores(embedding, val_seeds)
                k = min(50, len(G) - 1)
                top_k_indices = np.argsort(scores_centroid)[-k:]
                hits = sum(1 for seed in val_seeds if seed in top_k_indices)
                recall_at_k = hits / len(val_seeds) if len(val_seeds) > 0 else 0.0
                
                return recall_at_k
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        sampler = TPESampler(seed=self.base_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='node2vec_tuning'
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs_optuna,
            show_progress_bar=True
        )
        
        logger.info(f"Tuning complete!")
        logger.info(f"Best validation recall@50: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'trials_df': study.trials_dataframe(),
            'method': 'node2vec'
        }
    
    def tune_netmf_hyperparameters(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        n_trials: int = 30,
        timeout: int = 1800,
        n_jobs_optuna: int = 1
    ) -> Dict:
        """
        Tune NetMF hyperparameters using Bayesian optimization (Optuna).
        
        Optimizes window_size, negative samples, and rank for best
        validation ranking performance.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes for evaluation
        targets : list
            Target nodes for evaluation
        n_trials : int
            Number of optimization trials (default: 30)
        timeout : int
            Maximum time in seconds (default: 1800 = 30 minutes)
        n_jobs_optuna : int
            Number of parallel jobs for Optuna (default: 1)
            
        Returns
        -------
        dict
            Dictionary with best_params, best_value, study, trials_df
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is required. Install with: pip install optuna")
        
        logger.info("Starting hyperparameter tuning for NetMF")
        logger.info(f"Trials: {n_trials}, Timeout: {timeout}s")
        
        # Create train/val split
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        
        logger.info(f"Train seeds: {len(train_seeds)}, Val seeds: {len(val_seeds)}")
        
        nodes = list(G.nodes())
        
        def objective(trial):
            """Optuna objective for NetMF."""
            # Suggest hyperparameters
            window_size = trial.suggest_int('window_size', 5, 20)
            negative = trial.suggest_int('negative', 1, 10)
            # rank can be None (auto) or a specific value
            use_auto_rank = trial.suggest_categorical('use_auto_rank', [True, False])
            if use_auto_rank:
                rank = None
            else:
                rank = trial.suggest_int('rank', 64, 512, step=64)
            
            try:
                # Generate embedding
                embedding = run_netmf(
                    graph=G,
                    nodes=nodes,
                    dimensions=self.embedding_dim,
                    window_size=window_size,
                    negative=negative,
                    rank=rank,
                    use_svd=True,
                    seed=self.base_seed
                )
                
                # Evaluate on validation set
                scores_centroid = seed_centroid_scores(embedding, val_seeds)
                k = min(50, len(G) - 1)
                top_k_indices = np.argsort(scores_centroid)[-k:]
                hits = sum(1 for seed in val_seeds if seed in top_k_indices)
                recall_at_k = hits / len(val_seeds) if len(val_seeds) > 0 else 0.0
                
                return recall_at_k
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create and run study
        sampler = TPESampler(seed=self.base_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name='netmf_tuning'
        )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs_optuna,
            show_progress_bar=True
        )
        
        logger.info(f"Tuning complete!")
        logger.info(f"Best validation recall@50: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'trials_df': study.trials_dataframe(),
            'method': 'netmf'
        }
    
        
    def tune_qcaliber_gcnmf_hyperparameters(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        diffusion_type: str = 'heat',  # 'heat' or 'poly'
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Tune hyperparameters for Q-Caliber GCN-MF methods using Bayesian optimization.
        
        Args:
            G: NetworkX graph
            seeds: Seed nodes
            targets: Target nodes for evaluation
            diffusion_type: 'heat' for hgcnmf or 'poly' for pgcnmf
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and study results
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping hyperparameter tuning.")
            return None
        
        logger.info(f"Tuning Q-Caliber GCN-MF ({diffusion_type}) hyperparameters with {n_trials} trials...")
        
        # Create train/val split for seeds (80/20) BEFORE generating quantum targets
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)  # 20% for validation
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        
        logger.info(f"Train seeds: {len(train_seeds)}, Val seeds: {len(val_seeds)}")
        
        # Generate quantum targets using ONLY train seeds (no data leakage)
        q_targets = self._generate_quantum_targets(G, train_seeds)
        
        def objective(trial):
            # Suggest hyperparameters
            if diffusion_type == 'poly':
                K = trial.suggest_int('K', 2, 6)
                ridge = trial.suggest_float('ridge', 1e-8, 1e-4, log=True)
            else:
                K = None
                ridge = None
            
            hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
            mf_dim = trial.suggest_categorical('mf_dim', [32, 64, 128])
            n_layers = trial.suggest_int('n_layers', 1, 3)
            epochs = trial.suggest_int('epochs', 100, 300, step=50)
            lr = trial.suggest_float('lr', 1e-3, 1e-1, log=True)
            weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
            
            try:
                # Generate embedding
                from quvine.baselines.gcn_mf import generate_quvine_gcnmf_embedding
                
                embeddings, _ = generate_quvine_gcnmf_embedding(
                    G=G,
                    q_targets=q_targets,
                    embedding_dim=self.embedding_dim,
                    diffusion_type=diffusion_type,
                    K=K,
                    ridge=ridge,
                    hidden_dim=hidden_dim,
                    mf_dim=mf_dim,
                    n_layers=n_layers,
                    epochs=epochs,
                    lr=lr,
                    weight_decay=weight_decay,
                    normalize_laplacian=True,
                    random_state=self.base_seed
                )
                
                # Evaluate on validation set only (no data leakage)
                # Use centroid-based ranking with validation seeds
                scores_centroid = seed_centroid_scores(embeddings, val_seeds)
                
                # Compute recall@50 on validation seeds
                k = min(50, len(G) - 1)
                top_k_indices = np.argsort(scores_centroid)[-k:]
                
                # Count how many validation seeds are in top-k
                hits = sum(1 for seed in val_seeds if seed in top_k_indices)
                recall_at_k = hits / len(val_seeds) if len(val_seeds) > 0 else 0.0
                
                return recall_at_k
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        sampler = TPESampler(seed=self.base_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'qcaliber_gcnmf_{diffusion_type}_tuning'
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        logger.info(f"Tuning complete!")
        logger.info(f"Best validation recall@50: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'trials_df': study.trials_dataframe(),
            'method': f'{diffusion_type}gcnmf'
        }
    
    
    def tune_quantum_walk_hyperparameters(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        walk_type: str = 'rwr',  # 'rwr', 'ctqw', or 'dtqw'
        n_trials: int = 50,
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Tune hyperparameters for quantum walk methods using Bayesian optimization.
        
        Args:
            G: NetworkX graph
            seeds: Seed nodes
            targets: Target nodes for evaluation
            walk_type: Type of quantum walk ('rwr', 'ctqw', or 'dtqw')
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with best parameters and study results
        """
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available. Skipping hyperparameter tuning.")
            return None
        
        logger.info(f"Tuning {walk_type.upper()} hyperparameters with {n_trials} trials...")
        
        # Create train/val split for seeds (80/20) to prevent data leakage
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)  # 20% for validation
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        
        logger.info(f"Train seeds: {len(train_seeds)}, Val seeds: {len(val_seeds)}")
        
        def objective(trial):
            # Common parameters for all walk types
            num_walks = trial.suggest_int('num_walks', 5, 20)
            walk_length = trial.suggest_int('walk_length', 5, 15)
            num_views = trial.suggest_int('num_views', 2, 5)
            
            # Walk-specific parameters
            if walk_type == 'rwr':
                restart_prob = trial.suggest_float('restart_prob', 0.1, 0.3)
                max_iter = trial.suggest_int('max_iter', 500, 1500, step=250)
                walk_params = {
                    'restart_prob': restart_prob,
                    'max_iter': max_iter
                }
            elif walk_type == 'ctqw':
                time = trial.suggest_float('time', 0.5, 2.0)
                walk_params = {
                    'time': time
                }
            elif walk_type == 'dtqw':
                steps = trial.suggest_int('steps', 5, 25, step=5)
                coin = trial.suggest_categorical('coin', ['grover', 'hadamard'])
                walk_params = {
                    'steps': steps,
                    'coin': coin
                }
            else:
                raise ValueError(f"Unknown walk type: {walk_type}")
            
            try:
                # Create config
                cfg = self._get_default_quvine_config()
                cfg.walks.kinds = [walk_type]
                cfg.walks.num_walks = num_walks
                cfg.walks.walk_length = walk_length
                cfg.views.num_views = num_views
                
                # Update walk-specific parameters
                for key, value in walk_params.items():
                    OmegaConf.update(cfg.walks, key, value)
                
                # Generate embedding
                embeddings = self._run_quvine_walks(G, cfg)
                
                if walk_type not in embeddings or embeddings[walk_type] is None:
                    return 0.0
                
                embedding = embeddings[walk_type]
                
                # Evaluate on validation set only (no data leakage)
                # Use centroid-based ranking with validation seeds
                scores_centroid = seed_centroid_scores(embedding, val_seeds)
                
                # Compute recall@50 on validation seeds
                k = min(50, len(G) - 1)
                top_k_indices = np.argsort(scores_centroid)[-k:]
                
                # Count how many validation seeds are in top-k
                hits = sum(1 for seed in val_seeds if seed in top_k_indices)
                recall_at_k = hits / len(val_seeds) if len(val_seeds) > 0 else 0.0
                
                return recall_at_k
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return 0.0
        
        # Create study
        sampler = TPESampler(seed=self.base_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'{walk_type}_tuning'
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        logger.info(f"Tuning complete!")
        logger.info(f"Best validation recall@50: {study.best_value:.4f}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'study': study,
            'trials_df': study.trials_dataframe(),
            'method': walk_type
        }
    
    def run_embedding_method(
        self,
        method_name: str,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        cfg: Optional[DictConfig] = None,
        network_id: Optional[str] = None,
        method_hyperparams: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        Run a specific embedding method.
        
        Parameters
        ----------
        method_name : str
            One of: 'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw', 'quvine_fused',
                    'quvine_heat', 'quvine_poly', 'quvine_hgcnmf', 'quvine_pgcnmf',
                    'netmf', 'node2vec', 'baseline_gcnmf', 'baseline_filter'
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes
        targets : list
            Target nodes
        cfg : DictConfig, optional
            Configuration for QuVINE methods
        network_id : str, optional
            Network identifier for accessing tuned hyperparameters
            
        Returns
        -------
        np.ndarray
            Embedding matrix
        """
        nodes = list(G.nodes())
        
        if method_name == 'netmf':
            # Priority: method_hyperparams > tuned_hyperparameters > defaults
            if method_hyperparams and 'netmf' in method_hyperparams:
                hp = method_hyperparams['netmf']
                logger.info(f"Using best-tuned NetMF hyperparameters: {hp}")
                return run_netmf(
                    graph=G,
                    nodes=nodes,
                    dimensions=hp.get('dimensions', self.embedding_dim),
                    window_size=hp.get('window_size', 10),
                    negative=hp.get('negative', 1),
                    seed=self.base_seed
                )
            else:
                tuned_params = self._get_method_tuned_params('netmf', network_type=network_id) if network_id else None
                if tuned_params is not None:
                    logger.info(f"Using tuned NetMF hyperparameters for {network_id}: {tuned_params}")
                    return run_netmf(
                        graph=G,
                        nodes=nodes,
                        dimensions=self.embedding_dim,
                        window_size=tuned_params['window_size'],
                        negative=tuned_params['negative'],
                        rank=tuned_params.get('rank'),
                        seed=self.base_seed
                    )
                return run_netmf(
                    graph=G,
                    nodes=nodes,
                    dimensions=self.embedding_dim,
                    window_size=10,
                    negative=1,
                    seed=self.base_seed
                )
        
        elif method_name == 'node2vec':
            # Priority: method_hyperparams > tuned_hyperparameters > defaults
            if method_hyperparams and 'node2vec' in method_hyperparams:
                hp = method_hyperparams['node2vec']
                logger.info(f"Using best-tuned Node2Vec hyperparameters: {hp}")
                return run_node2vec(
                    graph=G,
                    nodes=nodes,
                    dimensions=hp.get('dimensions', self.embedding_dim),
                    walk_length=hp.get('walk_length', 10),
                    num_walks=hp.get('num_walks', 10),
                    p=hp.get('p', 1.0),
                    q=hp.get('q', 0.5),
                    window=hp.get('window', 5),
                    min_count=1,
                    workers=4,
                    seed=self.base_seed
                )
            else:
                tuned_params = self._get_method_tuned_params('node2vec', network_type=network_id) if network_id else None
                if tuned_params is not None:
                    logger.info(f"Using tuned Node2Vec hyperparameters for {network_id}: {tuned_params}")
                    return run_node2vec(
                        graph=G,
                        nodes=nodes,
                        dimensions=self.embedding_dim,
                        walk_length=tuned_params['walk_length'],
                        num_walks=tuned_params['num_walks'],
                        p=tuned_params['p'],
                        q=tuned_params['q'],
                        window=tuned_params['window'],
                        min_count=1,
                        workers=4,
                        seed=self.base_seed
                    )
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
        
        elif method_name == 'graphsage':
            from quvine.baselines.graphsage import run_graphsage
            if method_hyperparams and 'graphsage' in method_hyperparams:
                hp = method_hyperparams['graphsage']
                logger.info(f"Using best-tuned GraphSAGE hyperparameters: {hp}")
                return run_graphsage(
                    graph=G,
                    nodes=nodes,
                    dimensions=hp.get('dimensions', self.embedding_dim),
                    hidden_dim=hp.get('hidden_dim', min(256, self.embedding_dim * 2)),
                    n_layers=hp.get('n_layers', 2),
                    epochs=hp.get('epochs', 50),
                    lr=hp.get('lr', 0.01),
                    neg_samples=hp.get('neg_samples', 5),
                    seed=self.base_seed,
                )
            else:
                hp = self._get_method_tuned_params('graphsage', network_type=network_id) if network_id else None
                if hp is not None:
                    logger.info(f"Using tuned GraphSAGE hyperparameters for {network_id}: {hp}")
                    return run_graphsage(
                        graph=G,
                        nodes=nodes,
                        dimensions=hp.get('dimensions', self.embedding_dim),
                        hidden_dim=hp.get('hidden_dim', min(256, self.embedding_dim * 2)),
                        n_layers=hp.get('n_layers', 2),
                        epochs=hp.get('epochs', 50),
                        lr=hp.get('lr', 0.01),
                        neg_samples=hp.get('neg_samples', 5),
                        seed=self.base_seed,
                    )
                return run_graphsage(
                    graph=G,
                    nodes=nodes,
                    dimensions=self.embedding_dim,
                    hidden_dim=min(256, self.embedding_dim * 2),
                    n_layers=2,
                    epochs=50,
                    lr=0.01,
                    neg_samples=5,
                    seed=self.base_seed,
                )

        elif method_name == 'appnp':
            if method_hyperparams and 'appnp' in method_hyperparams:
                hp = method_hyperparams['appnp']
                logger.info(f"Using best-tuned APPNP hyperparameters: {hp}")
                return run_appnp(
                    graph=G,
                    nodes=nodes,
                    dimensions=hp.get('dimensions', self.embedding_dim),
                    hidden_dim=hp.get('hidden_dim', 64),
                    n_layers=hp.get('n_layers', 2),
                    alpha=hp.get('alpha', 0.1),
                    K=hp.get('K', 10),
                    dropout=hp.get('dropout', 0.5),
                    lr=hp.get('lr', 0.01),
                    weight_decay=hp.get('weight_decay', 5e-4),
                    epochs=hp.get('epochs', 200),
                    seed=self.base_seed
                )
            else:
                hp = self._get_method_tuned_params('appnp', network_type=network_id) if network_id else None
                if hp is not None:
                    logger.info(f"Using tuned APPNP hyperparameters for {network_id}: {hp}")
                    return run_appnp(
                        graph=G,
                        nodes=nodes,
                        dimensions=hp.get('dimensions', self.embedding_dim),
                        hidden_dim=hp.get('hidden_dim', 64),
                        n_layers=hp.get('n_layers', 2),
                        alpha=hp.get('alpha', 0.1),
                        K=hp.get('K', 10),
                        dropout=hp.get('dropout', 0.5),
                        lr=hp.get('lr', 0.01),
                        weight_decay=hp.get('weight_decay', 5e-4),
                        epochs=hp.get('epochs', 200),
                        seed=self.base_seed
                    )
                return run_appnp(
                    graph=G,
                    nodes=nodes,
                    dimensions=self.embedding_dim,
                    hidden_dim=64,
                    n_layers=2,
                    alpha=0.1,
                    K=10,
                    dropout=0.5,
                    lr=0.01,
                    weight_decay=5e-4,
                    epochs=200,
                    seed=self.base_seed
                )

        elif method_name == 'baseline_gcnmf':
            if method_hyperparams and 'baseline_gcnmf' in method_hyperparams:
                hp = method_hyperparams['baseline_gcnmf']
                logger.info(f"Using best-tuned baseline_gcnmf hyperparameters: {hp}")
                return generate_baseline_gcnmf_embedding(
                    G=G,
                    embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                    hidden_dim=hp.get('hidden_dim', 64),
                    mf_dim=hp.get('mf_dim', 64),
                    n_layers=hp.get('n_layers', 2),
                    epochs=hp.get('epochs', 200),
                    lr=hp.get('lr', 0.01),
                    weight_decay=hp.get('weight_decay', 5e-4),
                    random_state=self.base_seed
                )
            else:
                hp = self._get_method_tuned_params('baseline_gcnmf', network_type=network_id) if network_id else None
                if hp is not None:
                    logger.info(f"Using tuned baseline_gcnmf hyperparameters for {network_id}: {hp}")
                    return generate_baseline_gcnmf_embedding(
                        G=G,
                        embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                        hidden_dim=hp.get('hidden_dim', 64),
                        mf_dim=hp.get('mf_dim', 64),
                        n_layers=hp.get('n_layers', 2),
                        epochs=hp.get('epochs', 200),
                        lr=hp.get('lr', 0.01),
                        weight_decay=hp.get('weight_decay', 5e-4),
                        random_state=self.base_seed
                    )
                return generate_baseline_gcnmf_embedding(
                    G=G,
                    embedding_dim=self.embedding_dim,
                    hidden_dim=64,
                    mf_dim=64,
                    n_layers=2,
                    epochs=200,
                    lr=0.01,
                    weight_decay=5e-4,
                    random_state=self.base_seed
                )

        elif method_name == 'baseline_filter':
            if method_hyperparams and 'baseline_filter_heat' in method_hyperparams:
                hp = method_hyperparams['baseline_filter_heat']
                logger.info(f"Using best-tuned baseline_filter hyperparameters: {hp}")
                return generate_baseline_filter_embedding(
                    G=G,
                    filter_type=hp.get('filter_type', 'heat'),
                    t=hp.get('t', 1.0),
                    embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                    normalize=hp.get('normalize', True),
                    random_state=self.base_seed
                )
            else:
                hp = self._get_method_tuned_params('baseline_filter', network_type=network_id) if network_id else None
                if hp is not None:
                    logger.info(f"Using tuned baseline_filter hyperparameters for {network_id}: {hp}")
                    return generate_baseline_filter_embedding(
                        G=G,
                        filter_type=hp.get('filter_type', 'heat'),
                        t=hp.get('t', 1.0),
                        embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                        normalize=hp.get('normalize', True),
                        random_state=self.base_seed
                    )
                return generate_baseline_filter_embedding(
                    G=G,
                    filter_type='heat',
                    t=1.0,
                    embedding_dim=self.embedding_dim,
                    normalize=True,
                    random_state=self.base_seed
                )
        
        elif method_name in [f'quvine_{x}' for x in ['rwr', 'ctqw', 'dtqw']]:
            # QuVINE walk-based methods
            if cfg is None:
                cfg = self._get_default_quvine_config()
            hp = None
            if method_hyperparams and 'quvine_walks' in method_hyperparams:
                hp = method_hyperparams['quvine_walks']
                logger.info(f"Using best-tuned quvine_walks hyperparameters: {hp}")
            elif network_id:
                hp = self._get_method_tuned_params(method_name, network_type=network_id)
                if hp is not None:
                    logger.info(f"Using tuned {method_name} hyperparameters for {network_id}: {hp}")

            if hp is not None:
                cfg.walks.num_walks = hp.get('num_walks', cfg.walks.num_walks)
                cfg.walks.num_walks_per_root = hp.get('num_walks', cfg.walks.num_walks_per_root)
                cfg.walks.walk_length = hp.get('walk_length', cfg.walks.walk_length)
                cfg.walks.restart_prob = hp.get('restart_prob', cfg.walks.restart_prob)
                cfg.walks.steps = hp.get('steps', cfg.walks.steps)
                cfg.walks.time = hp.get('time', cfg.walks.time)
                cfg.views.num_views = hp.get('num_views', cfg.views.num_views)
                cfg.views.max_nodes = hp.get('max_nodes', cfg.views.max_nodes)
                cfg.views.max_edges = hp.get('max_edges', cfg.views.max_edges)
                cfg.views.max_degree = hp.get('max_degree', cfg.views.max_degree)
                cfg.views.degree_alpha = hp.get('degree_alpha', cfg.views.degree_alpha)
                cfg.train.embedding_dim = hp.get('embedding_dim', cfg.train.embedding_dim)
                cfg.train.window = hp.get('window', cfg.train.window)
                cfg.train.negative = hp.get('negative', cfg.train.negative)
                cfg.train.epochs = hp.get('epochs', cfg.train.epochs)
            # Set walk type
            cfg.walks.kinds = [method_name.split("_")[1]]

            # Run QuVINE pipeline for this method
            embeddings = self._run_quvine_walks(G, cfg)

            # Return single embedding
            return list(embeddings.values())[0]
        
        elif method_name.startswith('quvine_fused'):
            # ---------------------------------------------------------------
            # Named semantic fusion variants (highest priority, checked first)
            # ---------------------------------------------------------------
            # quvine_fused-walk:
            #   RWR + CTQW + DTQW fused with attention.  All three are
            #   proximity-based quantum walks; attention weights each by
            #   embedding quality so no single walk dominates.
            #
            # quvine_fused-filt:
            #   Heat + Poly fused with SVD shared/private (attention gate).
            #   Both are spectral filters calibrated by the same quantum
            #   targets; SVD extracts the common spectral consensus while
            #   the gate controls how much filter-specific detail to retain.
            #
            # quvine_fused-gcnmf:
            #   HGCNMF + PGCNMF fused with SVD shared/private (MoE gate).
            #   Same GCN-MF architecture but different quantum supervision
            #   (heat vs. poly diffusion); MoE selects per node which
            #   variant captures the more informative structural pattern.
            # ---------------------------------------------------------------
            _NAMED_FUSED = {
                'quvine_fused-walk':  ('attention',       ['rwr', 'ctqw', 'dtqw'],  'attention'),
                'quvine_fused-filt':  ('svd_shared_priv', ['heat', 'poly'],          'attention'),
                'quvine_fused-gcnmf': ('svd_shared_priv', ['hgcnmf', 'pgcnmf'],     'moe'),
                'quvine_fused_all':   ('all',             ['ctqw', 'dtqw', 'rwr', 'heat', 'poly', 'hgcnmf', 'pgcnmf'], 'attention'),
            }

            if method_name in _NAMED_FUSED:
                fusion_method, methods_to_fuse, gate_type = _NAMED_FUSED[method_name]
            else:
                # Generic _-delimited parsing
                # Format options:
                #   quvine_fused                          -> all methods, svd fusion
                #   quvine_fused_svd                      -> all methods, svd fusion
                #   quvine_fused_graphreg                 -> all methods, graphreg fusion
                #   quvine_fused_attention                -> all methods, attention fusion
                #   quvine_fused_hybrid                   -> all methods, hybrid fusion
                #   quvine_fused_svd_shared_priv          -> heat+poly, svd shared/private (attention gate)
                #   quvine_fused_svd_shared_priv_moe      -> heat+poly, svd shared/private (moe gate)
                #   quvine_fused_svd_ctqw_heat            -> ctqw+heat, svd fusion
                #   quvine_fused_attention_rwr_poly_hgcnmf -> rwr+poly+hgcnmf, attention fusion
                parts = method_name.split('_')

                fusion_methods = ['svd', 'graphreg', 'attention', 'hybrid', 'svd_shared_priv']
                fusion_method = 'svd'
                gate_type = 'attention'
                methods_to_fuse = []

                if len(parts) == 2:  # quvine_fused
                    methods_to_fuse = ['ctqw', 'dtqw', 'rwr', 'heat', 'poly', 'hgcnmf', 'pgcnmf']
                elif len(parts) >= 3:
                    if parts[2] == 'svd' and len(parts) >= 4 and parts[3] == 'shared':
                        fusion_method = 'svd_shared_priv'
                        if len(parts) >= 5:
                            if parts[4] == 'priv':
                                if len(parts) >= 6:
                                    if parts[5] == 'moe':
                                        gate_type = 'moe'
                                        methods_to_fuse = parts[6:] if len(parts) > 6 else ['heat', 'poly']
                                    else:
                                        methods_to_fuse = parts[5:]
                                else:
                                    methods_to_fuse = ['heat', 'poly']
                            else:
                                methods_to_fuse = parts[4:]
                        else:
                            methods_to_fuse = ['heat', 'poly']
                    elif parts[2] in fusion_methods:
                        fusion_method = parts[2]
                        methods_to_fuse = parts[3:] if len(parts) > 3 else \
                            ['ctqw', 'dtqw', 'rwr', 'heat', 'poly', 'hgcnmf', 'pgcnmf']
                    else:
                        methods_to_fuse = parts[2:]
            
            logger.info(f"Fusing embeddings from methods: {methods_to_fuse} using {fusion_method} fusion")
            
            # Generate embeddings for each method
            store = EmbeddingStore()
            for method in methods_to_fuse:
                try:
                    emb = self.run_embedding_method(
                        method_name=f'quvine_{method}',
                        G=G,
                        seeds=seeds,
                        targets=targets,
                        cfg=cfg,
                        network_id=network_id,
                        method_hyperparams=method_hyperparams,
                    )
                    store.add(method, emb)
                    logger.info(f"  Added {method} embedding: shape {emb.shape}")
                except Exception as e:
                    logger.warning(f"  Failed to generate {method} embedding: {e}")
            
            if len(store.names()) == 0:
                raise ValueError("No embeddings were successfully generated for fusion")
            
            # Fuse embeddings using specified method
            L = nx.normalized_laplacian_matrix(G, nodelist=nodes).toarray().astype(np.float32)
            
            # Pass gate_type for svd_shared_priv fusion
            if fusion_method == 'svd_shared_priv':
                svd_rank = self.embedding_dim // 4  # k = d // 4 as suggested
                fused_list, _ = fuse_embeddings(
                    store, method=fusion_method, k=self.embedding_dim, L=L,
                    svd_rank=svd_rank, gate_type=gate_type
                )
            else:
                fused_list, _ = fuse_embeddings(store, method=fusion_method, k=self.embedding_dim, L=L)
            
            logger.info(f"Fused embedding ({fusion_method}) shape: {fused_list[0].shape}")
            return fused_list[0]
        
        elif method_name in ['quvine_'+x for x in ['heat', 'poly']]:
            # QuVINE quantum-calibrated filter embeddings
            q_targets = self._generate_quantum_targets(G, seeds)

            if method_name == 'quvine_heat':
                hp = (method_hyperparams or {}).get('baseline_filter_heat', None)
                if hp is None and network_id:
                    hp = self._get_method_tuned_params('quvine_heat', network_type=network_id)
                hp = hp or {}
                if hp:
                    logger.info(f"Using best-tuned quvine_heat hyperparameters: {hp}")
                return generate_quvine_heat_embedding(
                    G=G,
                    q_targets=q_targets,
                    embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                    normalize=hp.get('normalize', True),
                    random_state=self.base_seed
                )
            elif method_name == 'quvine_poly':
                hp = (method_hyperparams or {}).get('baseline_filter_poly', None)
                if hp is None and network_id:
                    hp = self._get_method_tuned_params('quvine_poly', network_type=network_id)
                hp = hp or {}
                if hp:
                    logger.info(f"Using best-tuned quvine_poly hyperparameters: {hp}")
                return generate_quvine_poly_embedding(
                    G=G,
                    q_targets=q_targets,
                    K=hp.get('K', 4),
                    ridge=1e-6,
                    embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                    normalize=hp.get('normalize', True),
                    random_state=self.base_seed
                )

        elif method_name in ['quvine_'+x for x in ['hgcnmf', 'pgcnmf']]:
            # QuVINE quantum-calibrated GCN-MF embeddings
            q_targets = self._generate_quantum_targets(G, seeds)

            # method_hyperparams['baseline_gcnmf'] overrides tuned_hyperparameters
            _bp_gcnmf = (method_hyperparams or {}).get('baseline_gcnmf', None)

            if method_name == 'quvine_hgcnmf':
                if _bp_gcnmf:
                    params = _bp_gcnmf
                    logger.info(f"Using best-tuned hyperparameters for hgcnmf: {params}")
                else:
                    params = self._get_method_tuned_params('quvine_hgcnmf', network_type=network_id) if network_id else None
                    if params is not None:
                        logger.info(f"Using tuned hyperparameters for hgcnmf ({network_id}): {params}")
                    else:
                        params = {
                            'hidden_dim': self.embedding_dim,
                            'mf_dim': self.embedding_dim // 2,
                            'n_layers': 2,
                            'epochs': 200,
                            'lr': 0.01,
                            'weight_decay': 5e-4
                        }

                embeddings, _ = generate_quvine_gcnmf_embedding(
                    G=G,
                    q_targets=q_targets,
                    embedding_dim=params.get('embedding_dim', self.embedding_dim),
                    diffusion_type='heat',
                    hidden_dim=params.get('hidden_dim', self.embedding_dim),
                    mf_dim=params.get('mf_dim', self.embedding_dim // 2),
                    n_layers=params.get('n_layers', 2),
                    epochs=params.get('epochs', 200),
                    lr=params.get('lr', 0.01),
                    weight_decay=params.get('weight_decay', 5e-4),
                    normalize_laplacian=True,
                    random_state=self.base_seed
                )
                return embeddings

            elif method_name == 'quvine_pgcnmf':
                if _bp_gcnmf:
                    params = _bp_gcnmf
                    logger.info(f"Using best-tuned hyperparameters for pgcnmf: {params}")
                else:
                    params = self._get_method_tuned_params('quvine_pgcnmf', network_type=network_id) if network_id else None
                    if params is not None:
                        logger.info(f"Using tuned hyperparameters for pgcnmf ({network_id}): {params}")
                    else:
                        params = {
                            'K': 4,
                            'ridge': 1e-6,
                            'hidden_dim': self.embedding_dim,
                            'mf_dim': self.embedding_dim // 2,
                            'n_layers': 2,
                            'epochs': 200,
                            'lr': 0.01,
                            'weight_decay': 5e-4
                        }

                embeddings, _ = generate_quvine_gcnmf_embedding(
                    G=G,
                    q_targets=q_targets,
                    embedding_dim=params.get('embedding_dim', self.embedding_dim),
                    diffusion_type='poly',
                    K=params.get('K', 4),
                    ridge=params.get('ridge', 1e-6),
                    hidden_dim=params.get('hidden_dim', self.embedding_dim),
                    mf_dim=params.get('mf_dim', self.embedding_dim // 2),
                    n_layers=params.get('n_layers', 2),
                    epochs=params.get('epochs', 200),
                    lr=params.get('lr', 0.01),
                    weight_decay=params.get('weight_decay', 5e-4),
                    normalize_laplacian=True,
                    random_state=self.base_seed
                )
                return embeddings

        elif method_name in [
            'baseline_graphgps',
            'graphgps_rwr',
            'graphgps_ctqw_heat',
            'graphgps_ctqw_poly',
            'graphgps_rwr_heat',
            'graphgps_rwr_poly',
            'graphgps_dtqw_heat',
            'graphgps_dtqw_poly',
        ]:
            if generate_graphgps_embedding is None or GraphGPSConfig is None or GraphGPSTrainConfig is None:
                raise ImportError("GraphGPS baseline requires quvine.baselines.graphgps and PyG dependencies")

            nodes = list(G.nodes())
            ctqw_targets = self._generate_quantum_targets(G, seeds)
            dtqw_targets = self._generate_quantum_targets(G, seeds)
            hp = (method_hyperparams or {}).get('graphgps', None)
            if hp is None and network_id:
                hp = self._get_method_tuned_params(method_name, network_type=network_id)
            hp = hp or {}

            gps_config = GraphGPSConfig(
                hidden_dim=hp.get('hidden_dim', 64),
                output_dim=hp.get('embedding_dim', self.embedding_dim),
                num_layers=hp.get('num_layers', 2),
                heads=hp.get('heads', 4),
                dropout=hp.get('dropout', 0.2),
                attn_dropout=hp.get('attn_dropout', 0.2),
                local_gnn=hp.get('local_gnn', 'gcn'),
                attn_type=hp.get('attn_type', 'multihead'),
                use_layer_norm=hp.get('use_layer_norm', True),
                activation=hp.get('activation', 'relu'),
                lap_pe_dim=hp.get('lap_pe_dim', 0),
                standardize_features=hp.get('standardize_features', True),
            )
            train_config = GraphGPSTrainConfig(
                task='link_reconstruction',
                epochs=hp.get('epochs', 200),
                lr=hp.get('lr', 5e-3),
                weight_decay=hp.get('weight_decay', 5e-4),
                patience=hp.get('patience', 30),
                edge_batch_size=hp.get('edge_batch_size', 8192),
                val_edge_fraction=hp.get('val_edge_fraction', 0.1),
                device=hp.get('device', 'cpu'),
                random_state=self.base_seed,
                verbose=hp.get('verbose', False),
            )

            variant_map = {
                'baseline_graphgps': 'raw',
                'graphgps_rwr': 'rwr',
                'graphgps_ctqw_heat': 'heat_qcal_ctqw',
                'graphgps_ctqw_poly': 'poly_qcal_ctqw',
                'graphgps_rwr_heat': 'heat_qcal_rwr',
                'graphgps_rwr_poly': 'poly_qcal_rwr',
                'graphgps_dtqw_heat': 'heat_qcal_dtqw',
                'graphgps_dtqw_poly': 'poly_qcal_dtqw',
            }
            variant = variant_map[method_name]

            kwargs = {
                'G': G,
                'variant': variant,
                'nodelist': nodes,
                'embedding_dim': hp.get('embedding_dim', self.embedding_dim),
                'gps_config': gps_config,
                'train_config': train_config,
            }
            if '_ctqw' in variant:
                kwargs['ctqw_targets'] = ctqw_targets
            elif '_dtqw' in variant:
                kwargs['dtqw_targets'] = dtqw_targets
            elif '_rwr' in variant:
                kwargs['ctqw_targets'] = ctqw_targets

            embeddings, _ = generate_graphgps_embedding(**kwargs)
            return embeddings
        
        # ========== New Filter Methods (Phase 2) ==========
        elif method_name == 'quvine_baseline_heat':
            hp = (method_hyperparams or {}).get('quvine_baseline_heat', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params('quvine_baseline_heat', network_type=network_id) or {}
            return generate_baseline_heat_embedding(
                G=G,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                scale=hp.get('scale', 1.0),
                normalize=hp.get('normalize', True),
                random_state=self.base_seed
            )
        
        elif method_name == 'quvine_baseline_poly':
            hp = (method_hyperparams or {}).get('quvine_baseline_poly', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params('quvine_baseline_poly', network_type=network_id) or {}
            return generate_baseline_poly_embedding(
                G=G,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                order=hp.get('order', 4),
                normalize=hp.get('normalize', True),
                random_state=self.base_seed
            )
        
        elif method_name == 'quvine_rwr_heat':
            hp = (method_hyperparams or {}).get('quvine_rwr_heat', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params('quvine_rwr_heat', network_type=network_id) or {}
            return generate_rwr_heat_embedding(
                G=G,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                restart_prob=hp.get('restart_prob', 0.15),
                scale=hp.get('scale', 1.0),
                normalize=hp.get('normalize', True),
                random_state=self.base_seed
            )
        
        elif method_name == 'quvine_rwr_poly':
            hp = (method_hyperparams or {}).get('quvine_rwr_poly', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params('quvine_rwr_poly', network_type=network_id) or {}
            return generate_rwr_poly_embedding(
                G=G,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                restart_prob=hp.get('restart_prob', 0.15),
                order=hp.get('order', 4),
                normalize=hp.get('normalize', True),
                random_state=self.base_seed
            )
        
        # ========== GAT Methods (12 variants) ==========
        elif method_name.startswith('gat_'):
            if generate_gat_embedding_by_method_name is None:
                raise ImportError("GAT methods require quvine.baselines.gat")
            
            # Generate quantum targets if needed
            ctqw_targets = None
            dtqw_targets = None
            if 'ctqw' in method_name or 'dtqw' in method_name:
                ctqw_targets = self._generate_quantum_targets(G, seeds)
                dtqw_targets = ctqw_targets  # Use same targets for both
            
            hp = (method_hyperparams or {}).get('gat', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params(method_name, network_type=network_id) or {}
            
            return generate_gat_embedding_by_method_name(
                G=G,
                method_name=method_name,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                ctqw_targets=ctqw_targets,
                dtqw_targets=dtqw_targets,
                heat_t=hp.get('heat_t', 1.0),
                poly_K=hp.get('poly_K', 4),
                rwr_alpha=hp.get('rwr_alpha', 0.15),
                gat_config=None,  # Will use defaults
                train_config=None,  # Will use defaults
            )
        
        # ========== GraphGPS Methods (remaining 4 variants) ==========
        elif method_name.startswith('graphgps_') and method_name not in [
            'graphgps_rwr', 'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
            'graphgps_rwr_heat', 'graphgps_rwr_poly', 'graphgps_dtqw_heat', 'graphgps_dtqw_poly'
        ]:
            # Handle: graphgps_baseline, graphgps_heat, graphgps_poly, graphgps_ctqw, graphgps_dtqw
            if generate_graphgps_embedding_by_method_name is None:
                raise ImportError("GraphGPS methods require quvine.baselines.graphgps")
            
            # Generate quantum targets if needed
            ctqw_targets = None
            dtqw_targets = None
            direct_features = None
            if 'ctqw' in method_name or 'dtqw' in method_name:
                ctqw_targets = self._generate_quantum_targets(G, seeds)
                dtqw_targets = ctqw_targets
                # For direct variants, we'd need pre-computed walk features
                # For now, use targets for calibration
            
            hp = (method_hyperparams or {}).get('graphgps', {})
            if not hp and network_id:
                hp = self._get_method_tuned_params(method_name, network_type=network_id) or {}
            
            return generate_graphgps_embedding_by_method_name(
                G=G,
                method_name=method_name,
                embedding_dim=hp.get('embedding_dim', self.embedding_dim),
                ctqw_targets=ctqw_targets,
                dtqw_targets=dtqw_targets,
                direct_features=direct_features,
                heat_t=hp.get('heat_t', 1.0),
                poly_K=hp.get('poly_K', 4),
                rwr_alpha=hp.get('rwr_alpha', 0.15),
                gps_config=None,  # Will use defaults
                train_config=None,  # Will use defaults
            )
        
        else:
            raise ValueError(f"Unknown method: {method_name}")

    def _get_network_type_key(self, network_type: Optional[str]) -> str:
        """Normalize a caller-provided network type into a stable cache key."""
        if network_type is None:
            return "unknown"
        key = str(network_type).strip()
        return key if key else "unknown"

    def _get_method_tuned_params(
        self,
        method_name: str,
        network_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return tuned hyperparameters for one method and network type, if present."""
        type_key = self._get_network_type_key(network_type)
        by_type = self.tuned_hyperparameters.get(type_key, {})
        return by_type.get(method_name)

    def _store_method_tuned_params(
        self,
        method_name: str,
        params: Optional[Dict[str, Any]],
        network_type: Optional[str] = None,
    ) -> None:
        """Persist tuned hyperparameters for one method and network type."""
        if not params:
            return
        type_key = self._get_network_type_key(network_type)
        if type_key not in self.tuned_hyperparameters:
            self.tuned_hyperparameters[type_key] = {}
        self.tuned_hyperparameters[type_key][method_name] = dict(params)

    def _evaluate_embedding_recall_at_k(
        self,
        embedding: np.ndarray,
        val_seeds: List[int],
        k: int = 50,
    ) -> float:
        """Shared validation metric used for hyperparameter tuning."""
        if embedding is None or len(val_seeds) == 0:
            return 0.0
        scores_centroid = seed_centroid_scores(embedding, val_seeds)
        top_k = min(k, int(embedding.shape[0]) - 1) if embedding.shape[0] > 1 else 1
        top_k_indices = np.argsort(scores_centroid)[-top_k:]
        hits = sum(1 for seed in val_seeds if seed in top_k_indices)
        return hits / len(val_seeds) if len(val_seeds) > 0 else 0.0

    def _split_tuning_seeds(
        self,
        seeds: List[int],
    ) -> Tuple[List[int], List[int]]:
        """Create a deterministic train/validation seed split for tuning."""
        np.random.seed(self.base_seed)
        n_val = max(1, len(seeds) // 5)
        val_indices = np.random.choice(len(seeds), size=n_val, replace=False)
        train_seeds = [s for i, s in enumerate(seeds) if i not in val_indices]
        val_seeds = [s for i, s in enumerate(seeds) if i in val_indices]
        return train_seeds, val_seeds

    def tune_method_hyperparameters(
        self,
        method_name: str,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        network_type: Optional[str] = None,
        n_trials: int = 20,
        timeout: int = 600,
        n_jobs_optuna: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """
        Tune one embedding method for a given network type and cache the result.

        Returns a plain best-params dict, or None if tuning is skipped/unavailable.
        """
        cached = self._get_method_tuned_params(method_name, network_type=network_type)
        if cached is not None:
            logger.info(
                "Using cached tuned hyperparameters for method=%s network_type=%s",
                method_name,
                self._get_network_type_key(network_type),
            )
            return cached

        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna not available; skipping tuning for %s", method_name)
            return None

        if len(seeds) < 2:
            logger.warning("Not enough seeds to tune %s; skipping tuning", method_name)
            return None

        type_key = self._get_network_type_key(network_type)
        logger.info("Tuning method=%s for encountered network_type=%s", method_name, type_key)

        if method_name == 'node2vec':
            result = self.tune_node2vec_hyperparameters(
                G, seeds, targets, n_trials=n_trials, timeout=timeout, n_jobs_optuna=n_jobs_optuna
            )
            params = result['best_params']
        elif method_name == 'netmf':
            result = self.tune_netmf_hyperparameters(
                G, seeds, targets, n_trials=min(n_trials, 30), timeout=timeout, n_jobs_optuna=n_jobs_optuna
            )
            params = result['best_params']
        elif method_name == 'quvine_hgcnmf':
            result = self.tune_qcaliber_gcnmf_hyperparameters(
                G, seeds, targets, diffusion_type='heat', n_trials=n_trials, timeout=timeout
            )
            params = None if result is None else result['best_params']
        elif method_name == 'quvine_pgcnmf':
            result = self.tune_qcaliber_gcnmf_hyperparameters(
                G, seeds, targets, diffusion_type='poly', n_trials=n_trials, timeout=timeout
            )
            params = None if result is None else result['best_params']
        elif method_name == 'quvine_rwr':
            result = self.tune_quantum_walk_hyperparameters(
                G, seeds, targets, walk_type='rwr', n_trials=n_trials, timeout=timeout
            )
            params = None if result is None else result['best_params']
        elif method_name == 'quvine_ctqw':
            result = self.tune_quantum_walk_hyperparameters(
                G, seeds, targets, walk_type='ctqw', n_trials=n_trials, timeout=timeout
            )
            params = None if result is None else result['best_params']
        elif method_name == 'quvine_dtqw':
            result = self.tune_quantum_walk_hyperparameters(
                G, seeds, targets, walk_type='dtqw', n_trials=n_trials, timeout=timeout
            )
            params = None if result is None else result['best_params']
        else:
            train_seeds, val_seeds = self._split_tuning_seeds(seeds)

            def objective(trial):
                try:
                    hp: Dict[str, Dict[str, Any]] = {}

                    if method_name == 'graphsage':
                        hp['graphsage'] = {
                            'dimensions': self.embedding_dim,
                            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                            'n_layers': trial.suggest_int('n_layers', 1, 3),
                            'epochs': trial.suggest_int('epochs', 25, 100, step=25),
                            'lr': trial.suggest_float('lr', 1e-3, 5e-2, log=True),
                            'neg_samples': trial.suggest_int('neg_samples', 2, 8),
                        }
                    elif method_name == 'appnp':
                        hp['appnp'] = {
                            'dimensions': self.embedding_dim,
                            'hidden_dim': trial.suggest_categorical('hidden_dim', [32, 64, 128]),
                            'n_layers': trial.suggest_int('n_layers', 1, 3),
                            'alpha': trial.suggest_float('alpha', 0.05, 0.3),
                            'K': trial.suggest_int('K', 5, 20),
                            'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                            'lr': trial.suggest_float('lr', 1e-3, 5e-2, log=True),
                            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                            'epochs': trial.suggest_int('epochs', 50, 250, step=50),
                        }
                    elif method_name == 'baseline_gcnmf':
                        hp['baseline_gcnmf'] = {
                            'embedding_dim': self.embedding_dim,
                            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                            'mf_dim': trial.suggest_categorical('mf_dim', [32, 64, 128]),
                            'n_layers': trial.suggest_int('n_layers', 1, 3),
                            'epochs': trial.suggest_int('epochs', 100, 300, step=50),
                            'lr': trial.suggest_float('lr', 1e-3, 5e-2, log=True),
                            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                        }
                    elif method_name == 'baseline_filter':
                        hp['baseline_filter_heat'] = {
                            'filter_type': 'heat',
                            't': trial.suggest_float('t', 1e-2, 10.0, log=True),
                            'embedding_dim': self.embedding_dim,
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_heat':
                        hp['baseline_filter_heat'] = {
                            'embedding_dim': self.embedding_dim,
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_poly':
                        hp['baseline_filter_poly'] = {
                            'K': trial.suggest_int('K', 2, 6),
                            'embedding_dim': self.embedding_dim,
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_baseline_heat':
                        hp['quvine_baseline_heat'] = {
                            'embedding_dim': self.embedding_dim,
                            'scale': trial.suggest_float('scale', 0.1, 5.0, log=True),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_baseline_poly':
                        hp['quvine_baseline_poly'] = {
                            'embedding_dim': self.embedding_dim,
                            'order': trial.suggest_int('order', 2, 8),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_rwr_heat':
                        hp['quvine_rwr_heat'] = {
                            'embedding_dim': self.embedding_dim,
                            'restart_prob': trial.suggest_float('restart_prob', 0.1, 0.3),
                            'scale': trial.suggest_float('scale', 0.1, 5.0, log=True),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_rwr_poly':
                        hp['quvine_rwr_poly'] = {
                            'embedding_dim': self.embedding_dim,
                            'restart_prob': trial.suggest_float('restart_prob', 0.1, 0.3),
                            'order': trial.suggest_int('order', 2, 8),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_ctqw_heat':
                        hp['quvine_ctqw_heat'] = {
                            'embedding_dim': self.embedding_dim,
                            'time': trial.suggest_float('time', 0.1, 5.0),
                            'scale': trial.suggest_float('scale', 0.1, 5.0, log=True),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name == 'quvine_ctqw_poly':
                        hp['quvine_ctqw_poly'] = {
                            'embedding_dim': self.embedding_dim,
                            'time': trial.suggest_float('time', 0.1, 5.0),
                            'order': trial.suggest_int('order', 2, 8),
                            'normalize': trial.suggest_categorical('normalize', [True, False]),
                        }
                    elif method_name in ['baseline_gat', 'gat_ctqw_heat', 'gat_ctqw_poly', 'gat_dtqw_heat', 'gat_dtqw_poly', 'gat_rwr_heat', 'gat_rwr_poly']:
                        hp['gat'] = {
                            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                            'embedding_dim': self.embedding_dim,
                            'num_layers': trial.suggest_int('num_layers', 1, 3),
                            'heads': trial.suggest_categorical('heads', [1, 2, 4]),
                            'dropout': trial.suggest_float('dropout', 0.0, 0.6),
                            'attention_dropout': trial.suggest_float('attention_dropout', 0.0, 0.5),
                            'negative_slope': trial.suggest_float('negative_slope', 0.1, 0.3),
                            'residual': trial.suggest_categorical('residual', [True, False]),
                            'epochs': trial.suggest_int('epochs', 50, 250, step=50),
                            'lr': trial.suggest_float('lr', 1e-3, 2e-2, log=True),
                            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                            'patience': trial.suggest_int('patience', 10, 40, step=5),
                            'edge_batch_size': trial.suggest_categorical('edge_batch_size', [2048, 4096, 8192]),
                            'val_edge_fraction': trial.suggest_float('val_edge_fraction', 0.05, 0.2),
                        }
                    elif method_name in ['baseline_graphgps', 'graphgps_rwr', 'graphgps_ctqw_heat', 'graphgps_ctqw_poly', 'graphgps_rwr_heat', 'graphgps_rwr_poly', 'graphgps_dtqw_heat', 'graphgps_dtqw_poly']:
                        hp['graphgps'] = {
                            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
                            'embedding_dim': self.embedding_dim,
                            'num_layers': trial.suggest_int('num_layers', 1, 3),
                            'heads': trial.suggest_categorical('heads', [1, 2, 4]),
                            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
                            'attn_dropout': trial.suggest_float('attn_dropout', 0.0, 0.5),
                            'local_gnn': trial.suggest_categorical('local_gnn', ['gcn', 'sage', 'gat', 'none']),
                            'use_layer_norm': trial.suggest_categorical('use_layer_norm', [True, False]),
                            'lap_pe_dim': trial.suggest_categorical('lap_pe_dim', [0, 4, 8]),
                            'standardize_features': trial.suggest_categorical('standardize_features', [True, False]),
                            'epochs': trial.suggest_int('epochs', 50, 250, step=50),
                            'lr': trial.suggest_float('lr', 1e-3, 2e-2, log=True),
                            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                            'patience': trial.suggest_int('patience', 10, 40, step=5),
                            'edge_batch_size': trial.suggest_categorical('edge_batch_size', [2048, 4096, 8192]),
                            'val_edge_fraction': trial.suggest_float('val_edge_fraction', 0.05, 0.2),
                        }
                    else:
                        logger.info("No tuner defined for %s; skipping", method_name)
                        return 0.0

                    embedding = self.run_embedding_method(
                        method_name=method_name,
                        G=G,
                        seeds=train_seeds,
                        targets=targets,
                        network_id=type_key,
                        method_hyperparams=hp,
                    )
                    return self._evaluate_embedding_recall_at_k(embedding, val_seeds)

                except Exception as e:
                    logger.warning("Tuning trial failed for %s: %s", method_name, e)
                    return 0.0

            sampler = TPESampler(seed=self.base_seed)
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                study_name=f'{method_name}_{type_key}_tuning',
            )
            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs_optuna,
                show_progress_bar=True,
            )
            params = study.best_params if len(study.trials) > 0 else None

            if params is not None:
                if method_name == 'graphsage':
                    params['dimensions'] = self.embedding_dim
                elif method_name == 'appnp':
                    params['dimensions'] = self.embedding_dim
                elif method_name == 'baseline_gcnmf':
                    params['embedding_dim'] = self.embedding_dim
                elif method_name == 'baseline_filter':
                    params['filter_type'] = 'heat'
                    params['embedding_dim'] = self.embedding_dim
                elif method_name == 'quvine_heat':
                    params['embedding_dim'] = self.embedding_dim
                elif method_name == 'quvine_poly':
                    params['embedding_dim'] = self.embedding_dim
                elif method_name in ['baseline_gat', 'gat_ctqw_heat', 'gat_ctqw_poly', 'gat_dtqw_heat', 'gat_dtqw_poly', 'gat_rwr_heat', 'gat_rwr_poly']:
                    params['embedding_dim'] = self.embedding_dim
                elif method_name in ['baseline_graphgps', 'graphgps_rwr', 'graphgps_ctqw_heat', 'graphgps_ctqw_poly', 'graphgps_rwr_heat', 'graphgps_rwr_poly', 'graphgps_dtqw_heat', 'graphgps_dtqw_poly']:
                    params['embedding_dim'] = self.embedding_dim

        self._store_method_tuned_params(method_name, params, network_type=type_key)
        return params

    def ensure_tuned_hyperparameters_for_network_type(
        self,
        G: nx.Graph,
        seeds: List[int],
        targets: List[int],
        network_type: Optional[str],
        methods: List[str],
        n_trials: int = 20,
        timeout: int = 600,
        n_jobs_optuna: int = 1,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Tune each requested method exactly once for a dataset/network type.

        The first encountered repetition/iteration for that dataset type performs
        tuning and stores the best hyperparameters in-memory. All subsequent
        repetitions for the same dataset type reuse the cached values.
        """
        type_key = self._get_network_type_key(network_type)
        if type_key not in self.tuned_hyperparameters:
            self.tuned_hyperparameters[type_key] = {}

        cached_methods = self.tuned_hyperparameters[type_key]
        if methods and all(method_name in cached_methods for method_name in methods):
            logger.info(
                "All requested methods already tuned for network_type=%s; reusing cached hyperparameters across repetitions",
                type_key,
            )
            return cached_methods

        for method_name in methods:
            if self._get_method_tuned_params(method_name, network_type=type_key) is not None:
                logger.info(
                    "Skipping tuning for method=%s network_type=%s because first-iteration hyperparameters already exist",
                    method_name,
                    type_key,
                )
                continue
            try:
                self.tune_method_hyperparameters(
                    method_name=method_name,
                    G=G,
                    seeds=seeds,
                    targets=targets,
                    network_type=type_key,
                    n_trials=n_trials,
                    timeout=timeout,
                    n_jobs_optuna=n_jobs_optuna,
                )
            except Exception as e:
                logger.warning(
                    "Failed to tune method=%s for network_type=%s: %s",
                    method_name,
                    type_key,
                    e,
                )

        return self.tuned_hyperparameters[type_key]
    
    def _generate_quantum_targets(
        self,
        G: nx.Graph,
        seeds: List[int],
        num_subgraphs: int = 5,
        subgraph_size: int = 20,
        ctqw_steps: int = 20
    ) -> List[Dict]:
        """
        Generate quantum walk targets for calibration.
        
        Samples subgraphs around seed nodes and runs CTQW to get probability distributions.
        
        Parameters
        ----------
        G : nx.Graph
            Input graph
        seeds : list
            Seed nodes to sample around
        num_subgraphs : int
            Number of subgraphs to sample
        subgraph_size : int
            Size of each subgraph
        ctqw_steps : int
            Number of CTQW steps
            
        Returns
        -------
        list of dict
            Each dict contains:
            - 'nodes': List of node IDs in subnetwork
            - 'center': Center node ID
            - 'pQ': Quantum walk probability distribution
        """
        from quvine.walks.ctqw import generate_ctqw_hiperwalk_scores
        from quvine.data.subgraph import expand_neighborhood
        
        q_targets = []
        rng = np.random.default_rng(self.base_seed)
        
        # Sample subgraphs around seeds
        sampled_seeds = rng.choice(seeds, size=min(num_subgraphs, len(seeds)), replace=False)
        
        for center in sampled_seeds:
            try:
                # Expand neighborhood to get candidate nodes
                subgraph_nodes = expand_neighborhood(G, {center}, radius=2)

                # Limit size by random sampling when the neighbourhood is large
                if len(subgraph_nodes) > subgraph_size:
                    subgraph_nodes = list(rng.choice(
                        list(subgraph_nodes),
                        size=subgraph_size,
                        replace=False
                    ))
                    # Guarantee center is present
                    if center not in subgraph_nodes:
                        subgraph_nodes[0] = center

                # Induced subgraph — random sampling can disconnect nodes whose
                # only path to centre passed through an unsampled intermediary.
                # These isolated nodes make hiperwalk's internal Hamiltonian
                # non-square and trigger a broadcast error.  Fix: keep only the
                # connected component that contains center.
                H = G.subgraph(subgraph_nodes).copy()
                if not nx.is_connected(H):
                    cc_nodes = nx.node_connected_component(H, center)
                    H = H.subgraph(cc_nodes).copy()

                if H.number_of_nodes() < 3:
                    logger.debug(
                        f"Subgraph around seed {center} too small after "
                        f"connectivity filter ({H.number_of_nodes()} nodes), skipping"
                    )
                    continue

                # Run CTQW to get probability distribution
                scores = generate_ctqw_hiperwalk_scores(
                    H,
                    root=center,
                    steps=ctqw_steps
                )

                # Convert to probability array aligned with subgraph node order
                nodes_list = list(H.nodes())
                pQ = np.array([scores.get(n, 0.0) for n in nodes_list])
                pQ = pQ / pQ.sum() if pQ.sum() > 0 else pQ

                q_targets.append({
                    'nodes': nodes_list,
                    'center': center,
                    'pQ': pQ
                })

            except Exception as e:
                logger.warning(f"Failed to generate quantum target for seed {center}: {e}")
                continue
        
        if len(q_targets) == 0:
            logger.warning("No quantum targets generated, using fallback")
            # Fallback: create a simple target with uniform distribution
            center = seeds[0]
            neighbors = list(G.neighbors(center))[:10]
            nodes_list = [center] + neighbors
            pQ = np.ones(len(nodes_list)) / len(nodes_list)
            q_targets.append({
                'nodes': nodes_list,
                'center': center,
                'pQ': pQ
            })
        
        logger.info(f"Generated {len(q_targets)} quantum targets for calibration")
        return q_targets
    
    def _get_default_quvine_config(self) -> DictConfig:
        """Get default QuVINE configuration."""
        cfg = OmegaConf.create({
            'walks': {
                'kinds': ['rwr'],
                'num_walks': 20,
                'num_walks_per_root': 20,
                'walk_length': 40,
                'restart_prob': 0.15,
                'time': 1.0,
                'steps': 10,
                'max_iter': 1000,
                'coin': 'grover',
                'rwr': {'alpha': 0.15},
                'ctqw': {'t': 1.0},
                'dtqw': {'steps': 10}
            },
            'views': {
                'enabled': True,
                'num_views': 3,
                'view_size': 50,
                'max_nodes': 200,
                'max_edges': 1000,
                'max_degree': 100,
                'degree_norm': False,
                'degree_alpha': 0.5,
                'strategy': 'random'
            },
            'train': {
                'embedding_dim': self.embedding_dim,
                'window': 10,
                'sg': 1,
                'negative': 5,
                'workers': 4,
                'epochs': 10
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
                        # Convert integer node IDs to strings for corpus builder
                        walks_str = [[str(node) for node in walk] for walk in walks]
                        corpus_builders[walk_kind].add(root, walks_str)
        
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
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        seed_indices = [node_to_idx[s] for s in seeds if s in node_to_idx]
        target_set = set(targets) & set(nodes)

        results = {
            'network_id': network_id,
            'method': method_name,
        }

        k_values = [10, 20, 40, 80]
        # Pre-fill all metrics with 0 so the schema is consistent even on failure
        for k in k_values:
            results[f'precision@{k}_centroid'] = 0.0
            results[f'recall@{k}_centroid'] = 0.0
            results[f'precision@{k}_max'] = 0.0
            results[f'recall@{k}_max'] = 0.0

        if not seed_indices or not target_set:
            logger.warning(
                f"Ranking skipped for {method_name}/{network_id}: "
                f"seeds_found={len(seed_indices)}, targets_found={len(target_set)}"
            )
            return results

        try:
            centroid_scores = seed_centroid_scores(embedding, seed_indices)
            max_scores = max_seed_cosine_scores(embedding, seed_indices)
        except Exception as e:
            logger.warning(f"Score computation failed for {method_name}/{network_id}: {e}")
            return results

        # Exclude seeds from the ranked list: seeds have self-similarity ≈ 1.0
        # and would dominate the top-K slots, leaving no room for targets.
        seed_index_set = set(seed_indices)
        rankable_indices = [i for i in range(len(nodes)) if i not in seed_index_set]
        rankable_nodes = [nodes[i] for i in rankable_indices]
        centroid_rankable = centroid_scores[rankable_indices]
        max_rankable = max_scores[rankable_indices]

        for k in k_values:
            if k > len(rankable_nodes):
                continue

            top_k_centroid = set(rankable_nodes[j] for j in np.argsort(centroid_rankable)[-k:])
            hits_centroid = len(top_k_centroid & target_set)
            results[f'precision@{k}_centroid'] = hits_centroid / k
            results[f'recall@{k}_centroid'] = hits_centroid / len(target_set)

            top_k_max = set(rankable_nodes[j] for j in np.argsort(max_rankable)[-k:])
            hits_max = len(top_k_max & target_set)
            results[f'precision@{k}_max'] = hits_max / k
            results[f'recall@{k}_max'] = hits_max / len(target_set)

        return results
    
    def evaluate_embedding_classification(
        self,
        embedding: np.ndarray,
        G: nx.Graph,
        method_name: str,
        network_id: str,
        output_path: Optional[Path] = None,
    ) -> Dict:
        """
        Evaluate embedding performance on node classification task.

        For each of the 7 label strategies, runs classify and records f1_macro
        and accuracy.  Per-strategy columns are always present in the returned
        dict (NaN when a strategy fails).  If *output_path* is given, appends
        this method's rows to ``{network_id}_node_embedding.csv`` so that the
        full method × strategy breakdown is available per network.

        Returns
        -------
        dict
            Summary statistics plus per-strategy f1/accuracy columns.
        """
        nodes = list(G.nodes())

        # Known strategies — ensures columns exist in the summary dict even if
        # evaluate_all_label_strategies returns fewer keys (e.g. on partial failure).
        _STRATEGIES = [
            'community_louvain', 'community_label_propagation', 'degree_based',
            'centrality_betweenness', 'centrality_pagerank', 'core_periphery',
            'homophily_based',
        ]

        try:
            all_results = evaluate_all_label_strategies(
                G=G,
                embeddings=embedding,
                node_list=nodes,
                test_size=0.3,
                random_state=self.base_seed,
            )

            summary = summarize_classification_results(all_results)
            summary['network_id'] = network_id
            summary['method'] = method_name

            # Per-strategy columns — NaN when the strategy errored or is missing
            for strategy in _STRATEGIES:
                metrics = all_results.get(strategy, {})
                summary[f'f1_{strategy}'] = (
                    metrics.get('f1_macro', np.nan)
                    if 'error' not in metrics else np.nan
                )
                summary[f'accuracy_{strategy}'] = (
                    metrics.get('accuracy', np.nan)
                    if 'error' not in metrics else np.nan
                )

            # Write per-strategy detail rows to node_embedding.csv
            if output_path is not None:
                flat_rows = flatten_classification_results(all_results, network_id, method_name)
                emb_csv = Path(output_path) / f'{network_id}_node_embedding.csv'
                flat_df = pd.DataFrame(flat_rows)
                flat_df.to_csv(
                    emb_csv,
                    mode='a',
                    header=not emb_csv.exists(),
                    index=False,
                )

            return summary

        except Exception as e:
            logger.warning(f"Classification evaluation failed for {method_name} on {network_id}: {e}")
            summary = {
                'network_id': network_id,
                'method': method_name,
                'mean_f1_macro': np.nan,
                'mean_accuracy': np.nan,
                'error': str(e),
            }
            for strategy in _STRATEGIES:
                summary[f'f1_{strategy}'] = np.nan
                summary[f'accuracy_{strategy}'] = np.nan
            return summary
    
    def evaluate_embedding_link_prediction(
        self,
        embedding: np.ndarray,
        G: nx.Graph,
        method_name: str,
        network_id: str,
        negative_strategy: str = 'random',
    ) -> Tuple[Dict, List[Dict]]:
        """
        Evaluate embedding performance on link prediction task.

        Uses a proper train / test split to avoid leakage:
        - Train set : 80 % of edges + equal-sized random negatives
        - Test  set : 20 % of edges + equal-sized strategy-specific negatives
        The StandardScaler is fit on train features only.

        Parameters
        ----------
        negative_strategy : str
            Negative edge sampling strategy for the TEST set.
            'random', 'hard_2hop', or 'same_community'.

        Returns
        -------
        tuple (summary_dict, tidy_rows)
            summary_dict  : one dict with mean AUC-ROC/PR across edge methods
                            (backward-compatible with correlation analysis)
            tidy_rows     : list of dicts, one per edge-feature method, with
                            columns: network_id, method, edge_feature_method,
                            negative_strategy, auc_roc, auc_pr, f1, mrr, …
        """
        nodes = list(G.nodes())

        try:
            # Split edges: 80% train, 20% test
            train_graph, _, test_edges, test_neg_edges = split_edges(
                G, test_ratio=0.2, val_ratio=0.0,
                negative_sampling_strategy=negative_strategy,
                seed=self.base_seed,
            )
            train_edges = list(train_graph.edges())

            # Train negatives: random, same count as test negatives.
            # Exclude all existing edges + already-sampled test negatives so
            # the two negative sets are disjoint and neither overlaps real edges.
            excluded = set(G.edges()) | set(test_neg_edges)
            n_train_neg = max(1, len(test_neg_edges))
            train_neg_edges = sample_negative_edges(
                G,
                n_samples=n_train_neg,
                existing_edges=excluded,
                strategy='random',
                seed=self.base_seed + 1,
            )

            # ── Evaluate all edge-feature methods ─────────────────────────────
            all_results = evaluate_all_edge_feature_methods(
                embeddings=embedding,
                node_list=nodes,
                positive_edges=test_edges,
                negative_edges=test_neg_edges,
                train_positive_edges=train_edges,
                train_negative_edges=train_neg_edges,
                k_values=[10, 50, 100],
                random_state=self.base_seed,
            )

            # ── Build tidy rows (one per edge-feature method) ─────────────────
            tidy_rows = []
            for edge_method, metrics in all_results.items():
                row = {
                    'network_id': network_id,
                    'method': method_name,
                    'edge_feature_method': edge_method,
                    'negative_strategy': negative_strategy,
                }
                if 'error' in metrics:
                    row['error'] = metrics['error']
                    row.update({k: np.nan for k in _LP_METRIC_KEYS})
                else:
                    row.update({k: metrics.get(k, np.nan) for k in _LP_METRIC_KEYS})
                    for k_val in [10, 50, 100]:
                        for pfx in ['precision', 'recall', 'hit']:
                            row[f'{pfx}@{k_val}'] = metrics.get(f'{pfx}@{k_val}', np.nan)
                tidy_rows.append(row)

            # ── Build backward-compatible summary dict ─────────────────────────
            summary = summarize_link_prediction_results(all_results)
            summary['network_id'] = network_id
            summary['method'] = method_name
            summary['negative_strategy'] = negative_strategy
            for edge_method, metrics in all_results.items():
                if 'error' not in metrics:
                    summary[f'auc_roc_{edge_method}'] = metrics.get('auc_roc', 0.0)
                    summary[f'auc_pr_{edge_method}'] = metrics.get('auc_pr', 0.0)
                    summary[f'f1_{edge_method}'] = metrics.get('f1', 0.0)

            return summary, tidy_rows

        except Exception as e:
            logger.warning(f"Link prediction evaluation failed for {method_name} on {network_id}: {e}")
            err_summary = {
                'network_id': network_id,
                'method': method_name,
                'mean_auc_roc': 0.0,
                'mean_auc_pr': 0.0,
                'error': str(e),
            }
            return err_summary, []

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
        methods = [
            'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
            'quvine_heat', 'quvine_poly',
            'quvine_hgcnmf', 'quvine_pgcnmf',
            'baseline_gat', 'graphgps_rwr', 'baseline_graphgps',
            'gat_ctqw_heat', 'gat_ctqw_poly',
            'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
            'graphgps_rwr_heat', 'graphgps_rwr_poly',
            'graphgps_dtqw_heat', 'graphgps_dtqw_poly',
            'graphsage', 'netmf', 'node2vec', 'appnp',
            'quvine_fused_svd', 'quvine_fused_graphreg',
            'quvine_fused_attention', 'quvine_fused_hybrid',
            'quvine_fused_svd_shared_priv_heat_poly',
            'quvine_fused_svd_shared_priv_moe_heat_poly',
            'baseline_gcnmf',
        ]
        
        ranking_results = []
        classification_results = []
        link_prediction_results = []
        
        logger.info(f"Processing network: {network_id}")
        
        for method in methods:
            try:
                # Generate embedding
                embedding = self.run_embedding_method(method, G, seeds, targets, network_id=network_id)
                
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
                    link_pred_result, _ = self.evaluate_embedding_link_prediction(
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
        methods = [
            'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
            'quvine_heat', 'quvine_poly',
            'quvine_hgcnmf', 'quvine_pgcnmf',
            'baseline_gat', 'graphgps_rwr', 'baseline_graphgps',
            'gat_ctqw_heat', 'gat_ctqw_poly',
            'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
            'graphgps_rwr_heat', 'graphgps_rwr_poly',
            'graphgps_dtqw_heat', 'graphgps_dtqw_poly',
            'graphsage', 'netmf', 'node2vec', 'appnp',
            'quvine_fused_svd', 'quvine_fused_graphreg',
            'quvine_fused_attention', 'quvine_fused_hybrid',
            'quvine_fused_svd_shared_priv_heat_poly',
            'quvine_fused_svd_shared_priv_moe_heat_poly',
            'baseline_gcnmf',
        ]
        
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
                            # Check for constant values (std == 0) to avoid warnings
                            complexity_std = valid_data[complexity_metric].std()
                            perf_std = valid_data[perf_metric].std()
                            
                            # Skip if either variable is constant
                            if complexity_std == 0 or perf_std == 0:
                                continue
                            
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

def _compute_degree_distance_matched(
    G: nx.Graph,
    method_embeddings: Dict[str, np.ndarray],
    network_id: str,
    network_metadata: Dict,
    seed: int = 42,
    n_neg: int = 500,
    verbose: bool = False,
) -> List[Dict]:
    """
    Compute binned AUC-PR (cosine similarity) for degree-matched and
    distance-matched hard negative pairs.

    For EVERY network the analysis is run with both 'hard_2hop' and
    'same_community' strategies so that aggregate visualisations work
    regardless of which experiment generated the network.

    Returns a flat list of dicts, one per (method, strategy, bin_type, bin_label).
    """
    from sklearn.metrics import average_precision_score as _ap_score

    _DEGREE_N_BINS = 5
    _DIST_MAX_BIN  = 5   # distances ≥ this are merged to "{DIST_MAX_BIN}+"

    node_list = list(G.nodes())
    node_idx  = {n: i for i, n in enumerate(node_list)}
    degrees   = dict(G.degree())

    network_type    = network_metadata.get('type', 'unknown')
    case            = network_metadata.get('case', '')
    expected_winner = network_metadata.get('expected_winner', '')

    # Use the same edge split that was used during evaluation
    try:
        _, _, pos_edges, _ = split_edges(G, test_ratio=0.2, val_ratio=0.0, seed=seed)
    except Exception as e:
        logger.warning(f"  Edge split failed in degree/distance analysis: {e}")
        return []

    if len(pos_edges) == 0:
        return []

    # ── Per-pair scorers ──────────────────────────────────────────────────────
    def _pair_scores(emb, u, v):
        """Return (cosine, hadamard_norm) for pair (u,v), or (None, None)."""
        ui, vi = node_idx.get(u), node_idx.get(v)
        if ui is None or vi is None:
            return None, None
        eu, ev = emb[ui], emb[vi]
        nu, nv = np.linalg.norm(eu), np.linalg.norm(ev)
        cosine = float(np.dot(eu, ev) / (nu * nv)) if nu > 1e-12 and nv > 1e-12 else 0.0
        # Hadamard score: L2 norm of element-wise product captures feature-dimension
        # co-activation — a different quantity from cosine that matches the signal
        # exploited by Hadamard+LR in the main LP evaluation.
        hadamard = float(np.linalg.norm(eu * ev))
        return cosine, hadamard

    # ── AUC-PR within a subset of negative pairs (two scorers) ───────────────
    def _binned_auc(emb, pos_pairs, neg_subset):
        """Returns dict(auc_pr_cosine, auc_pr_hadamard); np.nan when undefined."""
        nan_result = {'auc_pr_cosine': np.nan, 'auc_pr_hadamard': np.nan}
        if len(neg_subset) == 0 or len(pos_pairs) == 0:
            return nan_result
        cos_scores, had_scores, labels = [], [], []
        for u, v in pos_pairs:
            c, h = _pair_scores(emb, u, v)
            if c is not None:
                cos_scores.append(c); had_scores.append(h); labels.append(1)
        for u, v in neg_subset:
            c, h = _pair_scores(emb, u, v)
            if c is not None:
                cos_scores.append(c); had_scores.append(h); labels.append(0)
        if len(set(labels)) < 2:
            return nan_result
        return {
            'auc_pr_cosine':   float(_ap_score(labels, cos_scores)),
            'auc_pr_hadamard': float(_ap_score(labels, had_scores)),
        }

    records = []

    for strategy in ('hard_2hop', 'same_community'):
        try:
            neg_edges = sample_negative_edges(
                G, n_samples=n_neg, strategy=strategy, seed=seed
            )
        except Exception as e:
            if verbose:
                logger.warning(f"  Negative sampling ({strategy}) failed: {e}")
            continue

        if len(neg_edges) == 0:
            continue

        # ── Per-pair features ─────────────────────────���───────────────────────
        neg_max_deg = np.array([
            max(degrees.get(u, 0), degrees.get(v, 0)) for u, v in neg_edges
        ])

        # Degree quintile bins
        deg_pct = np.percentile(neg_max_deg, np.linspace(0, 100, _DEGREE_N_BINS + 1)[1:])
        deg_pct[-1] += 1   # ensure last bin is inclusive
        neg_deg_bin = [
            f"Q{min(int(np.digitize(d, deg_pct)) + 1, _DEGREE_N_BINS)}"
            for d in neg_max_deg
        ]

        # Shortest-path distance bins (capped at _DIST_MAX_BIN)
        # Batch using BFS from each unique source to avoid O(n²) calls
        unique_sources = set(u for u, _ in neg_edges)
        path_cache: Dict = {}
        for src in unique_sources:
            try:
                path_cache[src] = nx.single_source_shortest_path_length(
                    G, src, cutoff=_DIST_MAX_BIN
                )
            except Exception:
                path_cache[src] = {}

        neg_dist_bin = [
            f"{_DIST_MAX_BIN}+" if min(path_cache.get(u, {}).get(v, _DIST_MAX_BIN + 1), _DIST_MAX_BIN) >= _DIST_MAX_BIN
            else str(min(path_cache.get(u, {}).get(v, _DIST_MAX_BIN + 1), _DIST_MAX_BIN))
            for u, v in neg_edges
        ]

        deg_bin_labels  = [f"Q{i+1}" for i in range(_DEGREE_N_BINS)]
        dist_bin_labels = [
            str(d) if d < _DIST_MAX_BIN else f"{_DIST_MAX_BIN}+"
            for d in range(2, _DIST_MAX_BIN + 1)
        ]

        # ── Compute per-method, per-bin AUC-PR ───────────────────────────────
        for method, emb in method_embeddings.items():
            if emb.shape[0] != len(node_list):
                continue

            # Degree-matched
            for bl in deg_bin_labels:
                idx = [i for i, b in enumerate(neg_deg_bin) if b == bl]
                aucs = _binned_auc(emb, pos_edges, [neg_edges[i] for i in idx])
                records.append(dict(
                    network_id=network_id, network_type=network_type,
                    case=case, expected_winner=expected_winner,
                    negative_strategy=strategy,
                    method=method, bin_type='degree', bin_label=bl,
                    bin_n_negs=len(idx), **aucs,
                ))

            # Distance-matched
            for bl in dist_bin_labels:
                idx = [i for i, b in enumerate(neg_dist_bin) if b == bl]
                aucs = _binned_auc(emb, pos_edges, [neg_edges[i] for i in idx])
                records.append(dict(
                    network_id=network_id, network_type=network_type,
                    case=case, expected_winner=expected_winner,
                    negative_strategy=strategy,
                    method=method, bin_type='distance', bin_label=bl,
                    bin_n_negs=len(idx), **aucs,
                ))

    return records


def _select_seeds_targets_structured(
    G: nx.Graph,
    network_metadata: Dict,
    num_seeds: int = 15,
    num_targets: int = 25,
    base_seed: int = 42,
) -> tuple:
    """
    Select seeds and targets aligned with the experiment's negative strategy.

    - same_community : seeds and targets come from the same detected community,
      so the ranking task is "find same-community members given seeds".
    - hard_2hop      : seeds are top-degree nodes; targets are nodes exactly 2
      hops away from the seed set (not direct neighbours), testing whether
      embeddings distinguish 2-hop structural equivalents.
    - random (fallback): original behaviour — random disjoint subsets.
    """
    from collections import defaultdict

    strategy = network_metadata.get('negative_strategy', 'random')
    rng = np.random.default_rng(base_seed)
    nodes = list(G.nodes())
    n = len(nodes)

    # Pre-defined seeds/targets (e.g., from real disease GWAS data)
    if 'seeds' in network_metadata and 'targets' in network_metadata:
        pre_seeds = [v for v in network_metadata['seeds'] if v in G]
        pre_targets = [v for v in network_metadata['targets'] if v in G]
        # Remove overlap — targets must be disjoint from seeds
        seeds_set = set(pre_seeds)
        pre_targets = [v for v in pre_targets if v not in seeds_set]
        if pre_seeds and pre_targets:
            rng.shuffle(pre_seeds)
            rng.shuffle(pre_targets)
            return pre_seeds[:num_seeds], pre_targets[:num_targets]

    num_seeds  = min(num_seeds,  n // 4)
    num_targets = min(num_targets, n // 4)

    if strategy == 'same_community' and num_seeds + num_targets <= n:
        # Detect communities
        partition: Dict[int, int] = {}
        try:
            import community as cl
            partition = cl.best_partition(G)
        except (ImportError, Exception):
            try:
                for label, comm in enumerate(
                    nx.community.label_propagation_communities(G)
                ):
                    for node in comm:
                        partition[node] = label
            except Exception:
                pass

        if partition:
            communities: Dict[int, list] = defaultdict(list)
            for node, cid in partition.items():
                communities[cid].append(node)

            need = num_seeds + num_targets + 2
            large = sorted(
                [(cid, ns) for cid, ns in communities.items() if len(ns) >= need],
                key=lambda x: len(x[1]),
                reverse=True,
            )
            if large:
                _, comm_nodes = large[0]
                comm_arr = np.array(comm_nodes)
                rng.shuffle(comm_arr)
                seeds   = comm_arr[:num_seeds].tolist()
                targets = comm_arr[num_seeds:num_seeds + num_targets].tolist()
                return seeds, targets

    elif strategy == 'hard_2hop' and num_seeds + num_targets <= n:
        # Seeds: top-degree nodes
        sorted_nodes = sorted(nodes, key=lambda v: G.degree(v), reverse=True)
        seeds = sorted_nodes[:num_seeds]
        seeds_set = set(seeds)

        # Collect direct neighbours of seeds (1-hop)
        one_hop: set = set()
        for s in seeds:
            one_hop.update(G.neighbors(s))
        one_hop -= seeds_set

        # 2-hop candidates: neighbours of 1-hop that are NOT seeds or 1-hop
        two_hop: set = set()
        for nbr in one_hop:
            for nbr2 in G.neighbors(nbr):
                if nbr2 not in seeds_set and nbr2 not in one_hop:
                    two_hop.add(nbr2)

        two_hop_list = list(two_hop)
        if len(two_hop_list) >= num_targets:
            rng.shuffle(two_hop_list)
            targets = two_hop_list[:num_targets]
            return seeds, targets

    # Fallback: random disjoint selection
    idx = rng.choice(n, size=min(num_seeds + num_targets, n), replace=False)
    seeds   = [nodes[i] for i in idx[:num_seeds]]
    targets = [nodes[i] for i in idx[num_seeds:num_seeds + num_targets]]
    return seeds, targets


def _strip_list_attrs(G: nx.Graph) -> nx.Graph:
    """Return a copy of G with list/dict/tuple node and edge attributes removed.

    GraphML does not support collection-valued attributes; stripping them allows
    the GraphML save to succeed.
    """
    H = G.copy()
    for n, data in list(H.nodes(data=True)):
        for k, v in list(data.items()):
            if isinstance(v, (list, dict, tuple, set)):
                del H.nodes[n][k]
    for u, v, data in list(H.edges(data=True)):
        for k, val in list(data.items()):
            if isinstance(val, (list, dict, tuple, set)):
                del H.edges[u, v][k]
    return H


def run_single_network_analysis(
    G: nx.Graph,
    network_id: str,
    network_metadata: Dict,
    output_dir: str,
    embedding_methods: List[str] = None,
    embedding_dim: int = 128,
    num_seeds: int = 15,
    num_targets: int = 25,
    verbose: bool = True,
    resume: bool = False,
    method_hyperparams: Optional[Dict] = None,
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
    import time
    from pathlib import Path

    if embedding_methods is None:
        embedding_methods = [
            # SGNS (3)
            'quvine_rwr', 'quvine_ctqw', 'quvine_dtqw',
            # Filters (6)
            'quvine_baseline_heat', 'quvine_baseline_poly',
            'quvine_rwr_heat', 'quvine_rwr_poly',
            'quvine_ctqw_heat', 'quvine_ctqw_poly',
            # GAT (12)
            'gat_baseline', 'gat_heat', 'gat_poly',
            'gat_rwr', 'gat_ctqw', 'gat_dtqw',
            'gat_rwr_heat', 'gat_rwr_poly',
            'gat_ctqw_heat', 'gat_ctqw_poly',
            'gat_dtqw_heat', 'gat_dtqw_poly',
            # GraphGPS (12)
            'graphgps_baseline', 'graphgps_heat', 'graphgps_poly',
            'graphgps_rwr', 'graphgps_ctqw', 'graphgps_dtqw',
            'graphgps_rwr_heat', 'graphgps_rwr_poly',
            'graphgps_ctqw_heat', 'graphgps_ctqw_poly',
            'graphgps_dtqw_heat', 'graphgps_dtqw_poly',
            # Classical Baselines (6)
            'node2vec', 'netmf', 'graphsage', 'appnp',
            'baseline_filter', 'baseline_gcnmf',
        ]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save graph as GraphML for reproducibility and downstream inspection
    # Strip list/dict node+edge attributes first — GraphML does not support them.
    # Skip if already present (resume-safe).
    graphml_path = output_path / f"{network_id}.graphml"
    _graphml_valid = graphml_path.exists() and graphml_path.stat().st_size > 0
    if not _graphml_valid:
        try:
            nx.write_graphml(_strip_list_attrs(G), str(graphml_path))
            if verbose:
                logger.info(f"  GraphML saved: {graphml_path}")
        except Exception as _e:
            logger.warning(f"  GraphML save failed for {network_id}: {_e}")
    elif verbose:
        logger.info(f"  GraphML already exists, skipping save: {graphml_path}")

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
    
    # Select seeds and targets — community/topology-aware when metadata specifies a strategy
    seeds, targets = _select_seeds_targets_structured(
        G=G,
        network_metadata=network_metadata,
        num_seeds=num_seeds,
        num_targets=num_targets,
        base_seed=42,
    )
    
    _network_type = network_metadata.get('type', 'unknown')
    _dataset_key = f"{_network_type}_n{G.number_of_nodes()}"

    temp_analysis.ensure_tuned_hyperparameters_for_network_type(
        G=G,
        seeds=seeds,
        targets=targets,
        network_type=_dataset_key,
        methods=embedding_methods,
        n_trials=20,
        timeout=600,
        n_jobs_optuna=1,
    )

    # Step 1: Compute complexity metrics
    if verbose:
        logger.info("Step 1/4: Computing complexity metrics...")
    
    complexity_metrics = compute_graph_complexity_metrics(G)
    try:
        complexity_metrics.update(compute_qbc_metrics(G))
    except Exception as _qbc_exc:
        warnings.warn(f"QBC metrics failed for {network_id}: {_qbc_exc}")
    complexity_metrics['network_id'] = network_id
    complexity_metrics['network_type'] = _network_type
    complexity_metrics['n_nodes'] = G.number_of_nodes()
    complexity_metrics['n_edges'] = G.number_of_edges()
    
    # Save complexity metrics
    complexity_df = pd.DataFrame([complexity_metrics])
    complexity_path = output_path / f"{network_id}_complexity.csv"
    complexity_df.to_csv(complexity_path, index=False)
    
    if verbose:
        logger.info(f"  ✓ Complexity metrics saved to {complexity_path}")
    
    # Extract the negative strategy for link prediction evaluation
    _negative_strategy = network_metadata.get('negative_strategy', 'random')

    # Step 2: Generate embeddings and evaluate all tasks
    all_results = {
        'ranking': [],
        'classification': [],
        'link_prediction': [],        # summary dicts (one per method, backward-compat)
        'link_prediction_tidy': [],   # tidy rows (one per method × edge_feature_method)
        'nc_stratified': [],          # degree/distance binned NC rows
    }
    timing_records = []
    method_embeddings: Dict[str, np.ndarray] = {}   # kept for degree/distance analysis

    _network_type = network_metadata.get('type', 'unknown')
    _dataset_key = f"{_network_type}_n{G.number_of_nodes()}"

    # ── Resume: load existing results and determine which methods to skip ─────
    methods_done: set = set()
    if resume:
        _task_files = [
            ('ranking',               output_path / f'{network_id}_ranking_results.csv'),
            ('classification',        output_path / f'{network_id}_classification_results.csv'),
            ('link_prediction',       output_path / f'{network_id}_link_prediction_results.csv'),
            ('link_prediction_tidy',  output_path / f'{network_id}_link_prediction.csv'),
            ('nc_stratified',         output_path / f'{network_id}_nc_stratified.csv'),
        ]
        for task, csv_path in _task_files:
            if csv_path.exists():
                try:
                    _df = pd.read_csv(csv_path)
                    if 'method' in _df.columns:
                        all_results[task].extend(_df.to_dict('records'))
                        # Only count methods_done from the summary file (one row per method)
                        if task == 'link_prediction':
                            methods_done.update(_df['method'].unique())
                except Exception as _e:
                    logger.warning(f"  Resume: could not read {csv_path}: {_e}")

        _timing_csv = output_path / f'{network_id}_timing_results.csv'
        if _timing_csv.exists():
            try:
                _tdf = pd.read_csv(_timing_csv)
                timing_records.extend(_tdf.to_dict('records'))
            except Exception as _e:
                logger.warning(f"  Resume: could not read timing CSV: {_e}")

        # Pre-load existing embeddings so degree/distance analysis still has them
        for _m in methods_done:
            _emb_path = output_path / f'{network_id}_{_m}_embedding.npy'
            if _emb_path.exists():
                try:
                    method_embeddings[_m] = np.load(str(_emb_path))
                except Exception:
                    pass

        if verbose and methods_done:
            logger.info(f"  Resume: {len(methods_done)} methods already done, "
                        f"skipping: {sorted(methods_done)}")

    all_nodes = list(G.nodes())

    for method in embedding_methods:
        if resume and method in methods_done:
            if verbose:
                logger.info(f"  Skipping {method} (already done)")
            continue
        if verbose:
            logger.info(f"Step 2/4: Processing method: {method}")

        try:
            # Generate embedding — timed
            t0 = time.perf_counter()
            embedding = temp_analysis.run_embedding_method(
                method,
                G,
                seeds,
                targets,
                network_id=_network_type,
                method_hyperparams=method_hyperparams,
            )
            elapsed_embedding = time.perf_counter() - t0

            method_embeddings[method] = embedding

            timing_records.append({
                'network_id': network_id,
                'method': method,
                'network_type': _network_type,
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'embedding_time_s': elapsed_embedding,
            })

            if verbose:
                logger.info(f"  - Embedding time: {elapsed_embedding:.3f}s")

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
                embedding, G, method, network_id, output_path=output_path,
            )
            all_results['classification'].append(classification_result)
            
            # Task 3: Link prediction — use the experiment's negative strategy
            if verbose:
                logger.info(f"  - Evaluating link prediction task (negatives={_negative_strategy})...")
            link_pred_summary, link_pred_tidy = temp_analysis.evaluate_embedding_link_prediction(
                embedding, G, method, network_id,
                negative_strategy=_negative_strategy,
            )
            all_results['link_prediction'].append(link_pred_summary)
            all_results['link_prediction_tidy'].extend(link_pred_tidy)

            # Task 4: NC degree/distance stratified
            if verbose:
                logger.info(f"  - Evaluating NC stratified task...")
            try:
                nc_strat_rows = evaluate_nc_stratified(
                    G, embedding, all_nodes,
                    random_state=42,
                )
                for row in nc_strat_rows:
                    row['method'] = method
                    row['network_id'] = network_id
                all_results['nc_stratified'].extend(nc_strat_rows)
            except Exception as _e:
                logger.warning(f"  NC stratified failed for {method}: {_e}")
            
            if verbose:
                logger.info(f"  ✓ {method} completed successfully")
                
        except Exception as e:
            logger.error(f"  ✗ {method} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save timing results
    if timing_records:
        timing_df = pd.DataFrame(timing_records)
        timing_path = output_path / f"{network_id}_timing_results.csv"
        timing_df.to_csv(timing_path, index=False)
        if verbose:
            logger.info(f"  ✓ Timing results saved to {timing_path}")

    # Degree- and distance-matched binned link-prediction analysis
    # Runs for every network type using both hard_2hop and same_community strategies
    # so that the aggregate visualisation has data regardless of the experiment variant.
    if method_embeddings:
        if verbose:
            logger.info("  Computing degree/distance-matched binned AUC-PR…")
        try:
            dd_records = _compute_degree_distance_matched(
                G=G,
                method_embeddings=method_embeddings,
                network_id=network_id,
                network_metadata=network_metadata,
                seed=42,
                n_neg=min(600, max(50, G.number_of_edges() // 5)),
                verbose=verbose,
            )
            if dd_records:
                dd_path = output_path / f"{network_id}_degree_distance_matched.csv"
                pd.DataFrame(dd_records).to_csv(dd_path, index=False)
                if verbose:
                    logger.info(f"  ✓ Degree/distance-matched results saved to {dd_path}")
        except Exception as _e:
            logger.warning(f"  Degree/distance-matched analysis failed for {network_id}: {_e}")
            dd_path = None
    else:
        dd_path = None

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
    
    # Save link prediction results (summary — one row per method)
    link_pred_path = None
    if all_results['link_prediction']:
        link_pred_df = pd.DataFrame(all_results['link_prediction'])
        link_pred_path = output_path / f"{network_id}_link_prediction_results.csv"
        link_pred_df.to_csv(link_pred_path, index=False)
        if verbose:
            logger.info(f"  ✓ Link prediction summary saved to {link_pred_path}")

    # Save tidy link prediction results (one row per method × edge_feature_method)
    lp_tidy_path = None
    if all_results['link_prediction_tidy']:
        lp_tidy_df = pd.DataFrame(all_results['link_prediction_tidy'])
        lp_tidy_path = output_path / f"{network_id}_link_prediction.csv"
        lp_tidy_df.to_csv(lp_tidy_path, index=False)
        if verbose:
            logger.info(f"  ✓ Link prediction tidy CSV saved to {lp_tidy_path}")

    # Save NC stratified results
    nc_strat_path = None
    if all_results['nc_stratified']:
        nc_strat_df = pd.DataFrame(all_results['nc_stratified'])
        nc_strat_path = output_path / f"{network_id}_nc_stratified.csv"
        nc_strat_df.to_csv(nc_strat_path, index=False)
        if verbose:
            logger.info(f"  ✓ NC stratified results saved to {nc_strat_path}")
    
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
        'graphml_file': str(graphml_path),
        'timing_file': str(timing_path) if timing_records else None,
        'degree_distance_file': str(dd_path) if dd_path else None,
        'ranking_file': str(ranking_path) if all_results['ranking'] else None,
        'classification_file': str(classification_path) if all_results['classification'] else None,
        'link_prediction_file': str(link_pred_path) if link_pred_path else None,
        'link_prediction_tidy_file': str(lp_tidy_path) if lp_tidy_path else None,
        'nc_stratified_file': str(nc_strat_path) if nc_strat_path else None,
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
    
    if not all_data:
        logger.error("No data collected — check that result directories are non-empty.")
        return pd.DataFrame()

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
        if 'network_id' in comprehensive_df.columns:
            logger.info(f"Networks: {comprehensive_df['network_id'].nunique()}")
        if 'method' in comprehensive_df.columns:
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


