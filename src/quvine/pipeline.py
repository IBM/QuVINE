from __future__ import annotations 

import logging
import os
from pathlib import Path 
from typing import Dict, List 
import json
from omegaconf import OmegaConf,DictConfig
from hydra.core.hydra_config import HydraConfig
import pandas as pd 
import time
from quvine.data.data_loader import load_graph, load_gwas_data
from quvine.data.prepare import PrepareGraphConfig, prepare_graph
from quvine.views.generator import ViewBuilder
from quvine.walks.base import BaseWalker
from quvine.corpus.builder import CorpusBuilder
from quvine.embedding.word2vec import corpus_to_embedding
from quvine.embedding.registry import EmbeddingStore
from quvine.analysis.compare import compare_embeddings
from quvine.analysis.analyze import *
from quvine.baselines import run_appnp, run_node2vec
from quvine.baselines.graphsage import run_graphsage
from quvine.baselines.gat import (
    generate_gat_embedding,
    GATConfig,
    TrainConfig as GATTrainConfig,
)
from quvine.baselines.gcn_mf import (
    generate_baseline_gcnmf_embedding,
    generate_quvine_gcnmf_embedding,
)
from quvine.baselines.graphgps import (
    generate_graphgps_embedding,
    GraphGPSConfig,
    TrainConfig as GraphGPSTrainConfig,
)
from quvine.embedding.quantum_filters import (
    generate_baseline_filter_embedding,
    generate_quvine_heat_embedding,
    generate_quvine_poly_embedding,
)
from quvine.fusion.fuse import fuse_embeddings
from quvine.evaluation.ranking import (
    seed_centroid_scores,
    max_seed_cosine_scores,
    evaluate_embeddings_ranking
    )   
#from quvine.evals.ranking import evaluate_ranking
# from utils.io import save_embeddings, save_metadata 
from quvine.utils.seed import set_global_seed
from quvine.utils.utilities import *
from joblib import Parallel, delayed


class Pipeline: 
    """
    End-to-end quvine pipeline. 
    
    Stages: 
    Graph Loading 
    Seed/Target Loading 
    Preprocessing
    View Building
    Walking
    Embedding Training 
    Evaluation
    
    """
    
    def __init__(self, cfg:DictConfig): 
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)
        self.run_dir = Path(cfg.runtime.output_dir)
        if self.run_dir.exists():
            if self.cfg.verbose: 
                print(f"Directory {self.run_dir} exists")
        else: 
            self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n_iters = cfg.experiment.iterations 
        self.base_seed = cfg.experiment.base_seed
        
    def run(self): 
        self.log.info("Pipeline started (%d iterations)", self.n_iters)
        
        #load graph data once
        graph_data = self._load_graph()
        if self.cfg.verbose: 
            print(get_stats(graph_data))
        
        if self.cfg.gwas_target:
            source, target = self._load_gwas_data(graph_data)
        else:
            source = None
            target = None
        
        ## Preprocess graph 
        graph_data = self._preprocess_graph( 
                                            graph_data, 
                                            source, 
                                            target)
        
        if self.cfg.draw.graph: 
            draw_graph(cfg=self.cfg, 
                    G=graph_data, 
                    source=source, 
                    target=target)
        
        all_results = []

        for it in range(self.n_iters):
            self.log.info("Iteration %d / %d", it + 1, self.n_iters)
            self._set_iteration_seed(it)

            res = self._run_single_iteration(it, graph_data, source, target)
            all_results.append(res)
        
        if self.cfg.evaluation.enabled:
            # process and save evaluation results
            self._save_evaluation_results(all_results=all_results, nodes=list(graph_data.nodes))  
        
        if self.cfg.save_embeddings:
            # save and output embeddings
            self._save_embeddings(all_results=all_results)   
            
        

    #-----------------
    # One iteration
    # ----------------
    
    def _run_single_iteration(self, it, graph_data, source, target):
        
        beg_time = time.time()
        
        roots = list(graph_data.nodes)
        node2idx = {node: i for i, node in enumerate(sorted(roots))}
        corpus_builder = {kind: CorpusBuilder()
                    for kind in self.cfg.walks.kinds}
        
        # decide serial vs parallel 
        n_roots = len(roots)
        if n_roots < 2000 or self.cfg.runtime.n_jobs == 1: 
            chunks = [roots]
            n_jobs = 1 
        else: 
            chunk_size = self.cfg.runtime.chunk_size 
            chunks = list(self._chunkify(roots, chunk_size))
            n_jobs = self.cfg.runtime.n_jobs 
        
        parallel = Parallel(n_jobs=n_jobs, 
                            backend='loky', 
                            batch_size=1, 
                            prefer='processes'
                            )

        valid_roots = 0
        
        for chunk_results in parallel(
            delayed(self._process_root_chunk)(graph_data, chunk, node2idx, it)
            for chunk in chunks
            ):
        
            for root, walk_outputs in chunk_results: 
                if not walk_outputs or all(len(w)==0 for w in walk_outputs.values()):
                    continue 
                valid_roots +=1 
                
                for walk_kind, walks in walk_outputs.items(): 
                    if len(walks) == 0: 
                        continue 
                    corpus_builder[walk_kind].add(root, walks)
    
        assert valid_roots > 0, "No valid roots with walks were found."
        
        all_corpora = {kind: builder.build() 
                    for kind, builder in corpus_builder.items()}
        
        
        embeddings = self._train_embeddings(graph_data, all_corpora)
        
        store = EmbeddingStore()
        for name, Z in embeddings.items():
            store.add(name, Z)
        end_time = time.time() 
        time_taken = end_time - beg_time
        if self.cfg.verbose:
            print(f"Time taken for one QuVINE iteration {time_taken/60} minutes")
        
        ## baselines and quantum-calibrated downstream methods
        q_targets = None
        if source is not None and len(source) > 0:
            q_targets = self._build_quantum_targets(graph_data, source)

        beg_time = time.time()
        if self.cfg.baselines.node2vec.enabled:
            Z_n2v = run_node2vec(
                        graph=graph_data,
                        nodes=graph_data.nodes,
                        dimensions=self.cfg.baselines.node2vec.dimensions,
                        walk_length=self.cfg.baselines.node2vec.walk_length,
                        num_walks=self.cfg.baselines.node2vec.num_walks,
                        p=self.cfg.baselines.node2vec.p,
                        q=self.cfg.baselines.node2vec.q,
                        window=self.cfg.baselines.node2vec.window,
                        min_count=self.cfg.baselines.node2vec.min_count,
                        workers=self.cfg.baselines.node2vec.workers,
                        seed=self.cfg.baselines.node2vec.seed
                        )
            store.add("node2vec", Z_n2v)
            end_time = time.time()
            time_taken = end_time - beg_time
            if self.cfg.verbose:
                print(f"Time taken for one node2vec iteration {time_taken/60} minutes")

        if hasattr(self.cfg.baselines, "appnp") and self.cfg.baselines.appnp.enabled:
            Z_appnp = run_appnp(
                        graph=graph_data,
                        nodes=list(graph_data.nodes),
                        dimensions=self.cfg.baselines.appnp.dimensions,
                        hidden_dim=self.cfg.baselines.appnp.hidden_dim,
                        n_layers=self.cfg.baselines.appnp.n_layers,
                        alpha=self.cfg.baselines.appnp.alpha,
                        K=self.cfg.baselines.appnp.K,
                        dropout=self.cfg.baselines.appnp.dropout,
                        lr=self.cfg.baselines.appnp.lr,
                        weight_decay=self.cfg.baselines.appnp.weight_decay,
                        epochs=self.cfg.baselines.appnp.epochs,
                        seed=self.cfg.baselines.appnp.seed
                        )
            store.add("appnp", Z_appnp)
            end_time = time.time()
            time_taken = end_time - beg_time
            if self.cfg.verbose:
                print(f"Time taken for one APPNP iteration {time_taken/60} minutes")

        if hasattr(self.cfg.baselines, "baseline_filter") and self.cfg.baselines.baseline_filter.enabled:
            Z_baseline_filter = generate_baseline_filter_embedding(
                        G=graph_data,
                        filter_type=getattr(self.cfg.baselines.baseline_filter, "filter_type", "heat"),
                        t=getattr(self.cfg.baselines.baseline_filter, "t", 1.0),
                        K=getattr(self.cfg.baselines.baseline_filter, "K", 4),
                        embedding_dim=getattr(self.cfg.baselines.baseline_filter, "embedding_dim", self.cfg.train.embedding_dim),
                        use_features=False,
                        features=None,
                        normalize=getattr(self.cfg.baselines.baseline_filter, "normalize", True),
                        random_state=getattr(self.cfg.baselines.baseline_filter, "random_state", self.base_seed),
                        )
            store.add("baseline_filter", Z_baseline_filter)

        if hasattr(self.cfg.baselines, "baseline_gcnmf") and self.cfg.baselines.baseline_gcnmf.enabled:
            Z_baseline_gcnmf = generate_baseline_gcnmf_embedding(
                        G=graph_data,
                        embedding_dim=getattr(self.cfg.baselines.baseline_gcnmf, "embedding_dim", self.cfg.train.embedding_dim),
                        hidden_dim=getattr(self.cfg.baselines.baseline_gcnmf, "hidden_dim", 64),
                        mf_dim=getattr(self.cfg.baselines.baseline_gcnmf, "mf_dim", 64),
                        n_layers=getattr(self.cfg.baselines.baseline_gcnmf, "n_layers", 2),
                        epochs=getattr(self.cfg.baselines.baseline_gcnmf, "epochs", 200),
                        lr=getattr(self.cfg.baselines.baseline_gcnmf, "lr", 0.01),
                        weight_decay=getattr(self.cfg.baselines.baseline_gcnmf, "weight_decay", 5e-4),
                        random_state=getattr(self.cfg.baselines.baseline_gcnmf, "random_state", self.base_seed),
                        )
            store.add("baseline_gcnmf", Z_baseline_gcnmf)

        if hasattr(self.cfg.baselines, "baseline_gat") and self.cfg.baselines.baseline_gat.enabled:
            baseline_gat_dim = getattr(self.cfg.baselines.baseline_gat, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.baseline_gat, "hidden_dim", 64),
                        output_dim=baseline_gat_dim,
                        num_layers=getattr(self.cfg.baselines.baseline_gat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.baseline_gat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.baseline_gat, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.baseline_gat, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.baseline_gat, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.baseline_gat, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.baseline_gat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.baseline_gat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.baseline_gat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.baseline_gat, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.baseline_gat, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.baseline_gat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.baseline_gat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.baseline_gat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.baseline_gat, "verbose", False),
                    )
            Z_baseline_gat, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.baseline_gat, "variant", "raw"),
                        nodelist=list(graph_data.nodes),
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("baseline_gat", Z_baseline_gat)

        if hasattr(self.cfg.baselines, "baseline_graphgps") and self.cfg.baselines.baseline_graphgps.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.baseline_graphgps, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.baseline_graphgps, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.baseline_graphgps, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.baseline_graphgps, "heads", 4),
                        dropout=getattr(self.cfg.baselines.baseline_graphgps, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.baseline_graphgps, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.baseline_graphgps, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.baseline_graphgps, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.baseline_graphgps, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.baseline_graphgps, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.baseline_graphgps, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.baseline_graphgps, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.baseline_graphgps, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.baseline_graphgps, "epochs", 200),
                        lr=getattr(self.cfg.baselines.baseline_graphgps, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.baseline_graphgps, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.baseline_graphgps, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.baseline_graphgps, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.baseline_graphgps, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.baseline_graphgps, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.baseline_graphgps, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.baseline_graphgps, "verbose", False),
                        )
            Z_baseline_graphgps, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.baseline_graphgps, "variant", "raw"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.baseline_graphgps, "embedding_dim", self.cfg.train.embedding_dim),
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("baseline_graphgps", Z_baseline_graphgps)

        if hasattr(self.cfg.baselines, "graphsage") and self.cfg.baselines.graphsage.enabled:
            Z_graphsage = run_graphsage(
                        graph=graph_data,
                        nodes=list(graph_data.nodes),
                        dimensions=getattr(self.cfg.baselines.graphsage, "dimensions", self.cfg.train.embedding_dim),
                        hidden_dim=getattr(self.cfg.baselines.graphsage, "hidden_dim", min(256, self.cfg.train.embedding_dim * 2)),
                        n_layers=getattr(self.cfg.baselines.graphsage, "n_layers", 2),
                        epochs=getattr(self.cfg.baselines.graphsage, "epochs", 50),
                        lr=getattr(self.cfg.baselines.graphsage, "lr", 0.01),
                        neg_samples=getattr(self.cfg.baselines.graphsage, "neg_samples", 5),
                        seed=getattr(self.cfg.baselines.graphsage, "seed", self.base_seed),
                        )
            store.add("graphsage", Z_graphsage)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_heat") and self.cfg.baselines.quvine_heat.enabled:
            Z_quvine_heat = generate_quvine_heat_embedding(
                        G=graph_data,
                        q_targets=q_targets,
                        embedding_dim=getattr(self.cfg.baselines.quvine_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        use_features=False,
                        features=None,
                        normalize=getattr(self.cfg.baselines.quvine_heat, "normalize", True),
                        random_state=getattr(self.cfg.baselines.quvine_heat, "random_state", self.base_seed),
                        )
            store.add("quvine_heat", Z_quvine_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_poly") and self.cfg.baselines.quvine_poly.enabled:
            Z_quvine_poly = generate_quvine_poly_embedding(
                        G=graph_data,
                        q_targets=q_targets,
                        K=getattr(self.cfg.baselines.quvine_poly, "K", 4),
                        ridge=getattr(self.cfg.baselines.quvine_poly, "ridge", 1e-6),
                        embedding_dim=getattr(self.cfg.baselines.quvine_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        use_features=False,
                        features=None,
                        normalize=getattr(self.cfg.baselines.quvine_poly, "normalize", True),
                        random_state=getattr(self.cfg.baselines.quvine_poly, "random_state", self.base_seed),
                        )
            store.add("quvine_poly", Z_quvine_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_hgcnmf") and self.cfg.baselines.quvine_hgcnmf.enabled:
            Z_quvine_hgcnmf, _ = generate_quvine_gcnmf_embedding(
                        G=graph_data,
                        q_targets=q_targets,
                        embedding_dim=getattr(self.cfg.baselines.quvine_hgcnmf, "embedding_dim", self.cfg.train.embedding_dim),
                        diffusion_type="heat",
                        hidden_dim=getattr(self.cfg.baselines.quvine_hgcnmf, "hidden_dim", 64),
                        mf_dim=getattr(self.cfg.baselines.quvine_hgcnmf, "mf_dim", 64),
                        n_layers=getattr(self.cfg.baselines.quvine_hgcnmf, "n_layers", 2),
                        epochs=getattr(self.cfg.baselines.quvine_hgcnmf, "epochs", 200),
                        lr=getattr(self.cfg.baselines.quvine_hgcnmf, "lr", 0.01),
                        weight_decay=getattr(self.cfg.baselines.quvine_hgcnmf, "weight_decay", 5e-4),
                        normalize_laplacian=getattr(self.cfg.baselines.quvine_hgcnmf, "normalize_laplacian", True),
                        random_state=getattr(self.cfg.baselines.quvine_hgcnmf, "random_state", self.base_seed),
                        )
            store.add("quvine_hgcnmf", Z_quvine_hgcnmf)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_pgcnmf") and self.cfg.baselines.quvine_pgcnmf.enabled:
            Z_quvine_pgcnmf, _ = generate_quvine_gcnmf_embedding(
                        G=graph_data,
                        q_targets=q_targets,
                        embedding_dim=getattr(self.cfg.baselines.quvine_pgcnmf, "embedding_dim", self.cfg.train.embedding_dim),
                        diffusion_type="poly",
                        K=getattr(self.cfg.baselines.quvine_pgcnmf, "K", 4),
                        ridge=getattr(self.cfg.baselines.quvine_pgcnmf, "ridge", 1e-6),
                        hidden_dim=getattr(self.cfg.baselines.quvine_pgcnmf, "hidden_dim", 64),
                        mf_dim=getattr(self.cfg.baselines.quvine_pgcnmf, "mf_dim", 64),
                        n_layers=getattr(self.cfg.baselines.quvine_pgcnmf, "n_layers", 2),
                        epochs=getattr(self.cfg.baselines.quvine_pgcnmf, "epochs", 200),
                        lr=getattr(self.cfg.baselines.quvine_pgcnmf, "lr", 0.01),
                        weight_decay=getattr(self.cfg.baselines.quvine_pgcnmf, "weight_decay", 5e-4),
                        normalize_laplacian=getattr(self.cfg.baselines.quvine_pgcnmf, "normalize_laplacian", True),
                        random_state=getattr(self.cfg.baselines.quvine_pgcnmf, "random_state", self.base_seed),
                        )
            store.add("quvine_pgcnmf", Z_quvine_pgcnmf)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_ctqw_heat") and self.cfg.baselines.gat_ctqw_heat.enabled:
            gat_ctqw_heat_dim = getattr(self.cfg.baselines.gat_ctqw_heat, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_ctqw_heat, "hidden_dim", 64),
                        output_dim=gat_ctqw_heat_dim,
                        num_layers=getattr(self.cfg.baselines.gat_ctqw_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_ctqw_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_ctqw_heat, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_ctqw_heat, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_ctqw_heat, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_ctqw_heat, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_ctqw_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_ctqw_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_ctqw_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_ctqw_heat, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_ctqw_heat, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_ctqw_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_ctqw_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_ctqw_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_ctqw_heat, "verbose", False),
                    )
            Z_gat_ctqw_heat, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_ctqw_heat, "variant", "heat_qcal_ctqw"),
                        nodelist=list(graph_data.nodes),
                        ctqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_ctqw_heat", Z_gat_ctqw_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_ctqw_poly") and self.cfg.baselines.gat_ctqw_poly.enabled:
            gat_ctqw_poly_dim = getattr(self.cfg.baselines.gat_ctqw_poly, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_ctqw_poly, "hidden_dim", 64),
                        output_dim=gat_ctqw_poly_dim,
                        num_layers=getattr(self.cfg.baselines.gat_ctqw_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_ctqw_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_ctqw_poly, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_ctqw_poly, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_ctqw_poly, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_ctqw_poly, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_ctqw_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_ctqw_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_ctqw_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_ctqw_poly, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_ctqw_poly, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_ctqw_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_ctqw_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_ctqw_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_ctqw_poly, "verbose", False),
                    )
            Z_gat_ctqw_poly, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_ctqw_poly, "variant", "poly_qcal_ctqw"),
                        nodelist=list(graph_data.nodes),
                        ctqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_ctqw_poly", Z_gat_ctqw_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_dtqw_heat") and self.cfg.baselines.gat_dtqw_heat.enabled:
            gat_dtqw_heat_dim = getattr(self.cfg.baselines.gat_dtqw_heat, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_dtqw_heat, "hidden_dim", 64),
                        output_dim=gat_dtqw_heat_dim,
                        num_layers=getattr(self.cfg.baselines.gat_dtqw_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_dtqw_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_dtqw_heat, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_dtqw_heat, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_dtqw_heat, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_dtqw_heat, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_dtqw_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_dtqw_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_dtqw_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_dtqw_heat, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_dtqw_heat, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_dtqw_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_dtqw_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_dtqw_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_dtqw_heat, "verbose", False),
                    )
            Z_gat_dtqw_heat, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_dtqw_heat, "variant", "heat_qcal_dtqw"),
                        nodelist=list(graph_data.nodes),
                        dtqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_dtqw_heat", Z_gat_dtqw_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_dtqw_poly") and self.cfg.baselines.gat_dtqw_poly.enabled:
            gat_dtqw_poly_dim = getattr(self.cfg.baselines.gat_dtqw_poly, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_dtqw_poly, "hidden_dim", 64),
                        output_dim=gat_dtqw_poly_dim,
                        num_layers=getattr(self.cfg.baselines.gat_dtqw_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_dtqw_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_dtqw_poly, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_dtqw_poly, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_dtqw_poly, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_dtqw_poly, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_dtqw_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_dtqw_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_dtqw_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_dtqw_poly, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_dtqw_poly, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_dtqw_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_dtqw_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_dtqw_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_dtqw_poly, "verbose", False),
                    )
            Z_gat_dtqw_poly, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_dtqw_poly, "variant", "poly_qcal_dtqw"),
                        nodelist=list(graph_data.nodes),
                        dtqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_dtqw_poly", Z_gat_dtqw_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_rwr_heat") and self.cfg.baselines.gat_rwr_heat.enabled:
            gat_rwr_heat_dim = getattr(self.cfg.baselines.gat_rwr_heat, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_rwr_heat, "hidden_dim", 64),
                        output_dim=gat_rwr_heat_dim,
                        num_layers=getattr(self.cfg.baselines.gat_rwr_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_rwr_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_rwr_heat, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_rwr_heat, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_rwr_heat, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_rwr_heat, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_rwr_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_rwr_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_rwr_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_rwr_heat, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_rwr_heat, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_rwr_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_rwr_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_rwr_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_rwr_heat, "verbose", False),
                    )
            Z_gat_rwr_heat, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_rwr_heat, "variant", "heat_qcal_rwr"),
                        nodelist=list(graph_data.nodes),
                        ctqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_rwr_heat", Z_gat_rwr_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "gat_rwr_poly") and self.cfg.baselines.gat_rwr_poly.enabled:
            gat_rwr_poly_dim = getattr(self.cfg.baselines.gat_rwr_poly, "embedding_dim", self.cfg.train.embedding_dim)
            gat_cfg = GATConfig(
                        hidden_dim=getattr(self.cfg.baselines.gat_rwr_poly, "hidden_dim", 64),
                        output_dim=gat_rwr_poly_dim,
                        num_layers=getattr(self.cfg.baselines.gat_rwr_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.gat_rwr_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.gat_rwr_poly, "dropout", 0.2),
                        attention_dropout=getattr(self.cfg.baselines.gat_rwr_poly, "attention_dropout", 0.2),
                        negative_slope=getattr(self.cfg.baselines.gat_rwr_poly, "negative_slope", 0.2),
                        residual=getattr(self.cfg.baselines.gat_rwr_poly, "residual", True),
                    )
            gat_train_cfg = GATTrainConfig(
                        epochs=getattr(self.cfg.baselines.gat_rwr_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.gat_rwr_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.gat_rwr_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.gat_rwr_poly, "patience", 25),
                        edge_batch_size=getattr(self.cfg.baselines.gat_rwr_poly, "edge_batch_size", 4096),
                        val_edge_fraction=getattr(self.cfg.baselines.gat_rwr_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.gat_rwr_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.gat_rwr_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.gat_rwr_poly, "verbose", False),
                    )
            Z_gat_rwr_poly, _ = generate_gat_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.gat_rwr_poly, "variant", "poly_qcal_rwr"),
                        nodelist=list(graph_data.nodes),
                        ctqw_targets=q_targets,
                        gat_config=gat_cfg,
                        train_config=gat_train_cfg,
                    )
            store.add("gat_rwr_poly", Z_gat_rwr_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_graphgps_heat") and self.cfg.baselines.quvine_graphgps_heat.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.quvine_graphgps_heat, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.quvine_graphgps_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.quvine_graphgps_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.quvine_graphgps_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.quvine_graphgps_heat, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.quvine_graphgps_heat, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.quvine_graphgps_heat, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.quvine_graphgps_heat, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.quvine_graphgps_heat, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.quvine_graphgps_heat, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.quvine_graphgps_heat, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.quvine_graphgps_heat, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.quvine_graphgps_heat, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.quvine_graphgps_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.quvine_graphgps_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.quvine_graphgps_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.quvine_graphgps_heat, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.quvine_graphgps_heat, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.quvine_graphgps_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.quvine_graphgps_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.quvine_graphgps_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.quvine_graphgps_heat, "verbose", False),
                        )
            Z_graphgps_ctqw_heat, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.quvine_graphgps_heat, "variant", "heat_qcal_ctqw"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.quvine_graphgps_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        ctqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_ctqw_heat", Z_graphgps_ctqw_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "quvine_graphgps_poly") and self.cfg.baselines.quvine_graphgps_poly.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.quvine_graphgps_poly, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.quvine_graphgps_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.quvine_graphgps_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.quvine_graphgps_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.quvine_graphgps_poly, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.quvine_graphgps_poly, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.quvine_graphgps_poly, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.quvine_graphgps_poly, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.quvine_graphgps_poly, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.quvine_graphgps_poly, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.quvine_graphgps_poly, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.quvine_graphgps_poly, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.quvine_graphgps_poly, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.quvine_graphgps_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.quvine_graphgps_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.quvine_graphgps_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.quvine_graphgps_poly, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.quvine_graphgps_poly, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.quvine_graphgps_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.quvine_graphgps_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.quvine_graphgps_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.quvine_graphgps_poly, "verbose", False),
                        )
            Z_graphgps_ctqw_poly, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.quvine_graphgps_poly, "variant", "poly_qcal_ctqw"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.quvine_graphgps_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        ctqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_ctqw_poly", Z_graphgps_ctqw_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "graphgps_dtqw_heat") and self.cfg.baselines.graphgps_dtqw_heat.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.graphgps_dtqw_heat, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.graphgps_dtqw_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.graphgps_dtqw_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.graphgps_dtqw_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.graphgps_dtqw_heat, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.graphgps_dtqw_heat, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.graphgps_dtqw_heat, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.graphgps_dtqw_heat, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.graphgps_dtqw_heat, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.graphgps_dtqw_heat, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.graphgps_dtqw_heat, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.graphgps_dtqw_heat, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.graphgps_dtqw_heat, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.graphgps_dtqw_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.graphgps_dtqw_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.graphgps_dtqw_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.graphgps_dtqw_heat, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.graphgps_dtqw_heat, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.graphgps_dtqw_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.graphgps_dtqw_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.graphgps_dtqw_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.graphgps_dtqw_heat, "verbose", False),
                        )
            Z_graphgps_dtqw_heat, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.graphgps_dtqw_heat, "variant", "heat_qcal_dtqw"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.graphgps_dtqw_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        dtqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_dtqw_heat", Z_graphgps_dtqw_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "graphgps_dtqw_poly") and self.cfg.baselines.graphgps_dtqw_poly.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.graphgps_dtqw_poly, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.graphgps_dtqw_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.graphgps_dtqw_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.graphgps_dtqw_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.graphgps_dtqw_poly, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.graphgps_dtqw_poly, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.graphgps_dtqw_poly, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.graphgps_dtqw_poly, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.graphgps_dtqw_poly, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.graphgps_dtqw_poly, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.graphgps_dtqw_poly, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.graphgps_dtqw_poly, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.graphgps_dtqw_poly, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.graphgps_dtqw_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.graphgps_dtqw_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.graphgps_dtqw_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.graphgps_dtqw_poly, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.graphgps_dtqw_poly, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.graphgps_dtqw_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.graphgps_dtqw_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.graphgps_dtqw_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.graphgps_dtqw_poly, "verbose", False),
                        )
            Z_graphgps_dtqw_poly, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.graphgps_dtqw_poly, "variant", "poly_qcal_dtqw"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.graphgps_dtqw_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        dtqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_dtqw_poly", Z_graphgps_dtqw_poly)

        if q_targets is not None and hasattr(self.cfg.baselines, "graphgps_rwr_heat") and self.cfg.baselines.graphgps_rwr_heat.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.graphgps_rwr_heat, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.graphgps_rwr_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.graphgps_rwr_heat, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.graphgps_rwr_heat, "heads", 4),
                        dropout=getattr(self.cfg.baselines.graphgps_rwr_heat, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.graphgps_rwr_heat, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.graphgps_rwr_heat, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.graphgps_rwr_heat, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.graphgps_rwr_heat, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.graphgps_rwr_heat, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.graphgps_rwr_heat, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.graphgps_rwr_heat, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.graphgps_rwr_heat, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.graphgps_rwr_heat, "epochs", 200),
                        lr=getattr(self.cfg.baselines.graphgps_rwr_heat, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.graphgps_rwr_heat, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.graphgps_rwr_heat, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.graphgps_rwr_heat, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.graphgps_rwr_heat, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.graphgps_rwr_heat, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.graphgps_rwr_heat, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.graphgps_rwr_heat, "verbose", False),
                        )
            Z_graphgps_rwr_heat, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.graphgps_rwr_heat, "variant", "heat_qcal_rwr"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.graphgps_rwr_heat, "embedding_dim", self.cfg.train.embedding_dim),
                        ctqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_rwr_heat", Z_graphgps_rwr_heat)

        if q_targets is not None and hasattr(self.cfg.baselines, "graphgps_rwr_poly") and self.cfg.baselines.graphgps_rwr_poly.enabled:
            gps_cfg = GraphGPSConfig(
                        hidden_dim=getattr(self.cfg.baselines.graphgps_rwr_poly, "hidden_dim", 64),
                        output_dim=getattr(self.cfg.baselines.graphgps_rwr_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        num_layers=getattr(self.cfg.baselines.graphgps_rwr_poly, "num_layers", 2),
                        heads=getattr(self.cfg.baselines.graphgps_rwr_poly, "heads", 4),
                        dropout=getattr(self.cfg.baselines.graphgps_rwr_poly, "dropout", 0.2),
                        attn_dropout=getattr(self.cfg.baselines.graphgps_rwr_poly, "attn_dropout", 0.2),
                        local_gnn=getattr(self.cfg.baselines.graphgps_rwr_poly, "local_gnn", "gcn"),
                        attn_type=getattr(self.cfg.baselines.graphgps_rwr_poly, "attn_type", "multihead"),
                        use_layer_norm=getattr(self.cfg.baselines.graphgps_rwr_poly, "use_layer_norm", True),
                        activation=getattr(self.cfg.baselines.graphgps_rwr_poly, "activation", "relu"),
                        lap_pe_dim=getattr(self.cfg.baselines.graphgps_rwr_poly, "lap_pe_dim", 0),
                        standardize_features=getattr(self.cfg.baselines.graphgps_rwr_poly, "standardize_features", True),
                        )
            gps_train_cfg = GraphGPSTrainConfig(
                        task=getattr(self.cfg.baselines.graphgps_rwr_poly, "task", "link_reconstruction"),
                        epochs=getattr(self.cfg.baselines.graphgps_rwr_poly, "epochs", 200),
                        lr=getattr(self.cfg.baselines.graphgps_rwr_poly, "lr", 5e-3),
                        weight_decay=getattr(self.cfg.baselines.graphgps_rwr_poly, "weight_decay", 5e-4),
                        patience=getattr(self.cfg.baselines.graphgps_rwr_poly, "patience", 30),
                        edge_batch_size=getattr(self.cfg.baselines.graphgps_rwr_poly, "edge_batch_size", 8192),
                        val_edge_fraction=getattr(self.cfg.baselines.graphgps_rwr_poly, "val_edge_fraction", 0.1),
                        device=getattr(self.cfg.baselines.graphgps_rwr_poly, "device", "cpu"),
                        random_state=getattr(self.cfg.baselines.graphgps_rwr_poly, "random_state", self.base_seed),
                        verbose=getattr(self.cfg.baselines.graphgps_rwr_poly, "verbose", False),
                        )
            Z_graphgps_rwr_poly, _ = generate_graphgps_embedding(
                        G=graph_data,
                        variant=getattr(self.cfg.baselines.graphgps_rwr_poly, "variant", "poly_qcal_rwr"),
                        nodelist=list(graph_data.nodes),
                        embedding_dim=getattr(self.cfg.baselines.graphgps_rwr_poly, "embedding_dim", self.cfg.train.embedding_dim),
                        ctqw_targets=q_targets,
                        gps_config=gps_cfg,
                        train_config=gps_train_cfg,
                        )
            store.add("graphgps_rwr_poly", Z_graphgps_rwr_poly)
        
        if self.cfg.compare_embeddings:
            # compare embeddings 
            comparison_metrics = compare_embeddings(
                                            store,
                                            cca_components=self.cfg.analysis.cca_components,
                                            knn_k=self.cfg.analysis.knn_k,
                                            )
        else: 
            comparison_metrics = None
        
        ## fuse embeddings
        if self.cfg.fusion.enabled:
            
            beg_time = time.time() 
            
            L = nx.normalized_laplacian_matrix(G=graph_data, 
                                        nodelist=graph_data.nodes).toarray().astype(np.float32)
            
            fused_list, fuse_metric = fuse_embeddings(
                                        store,
                                        method=self.cfg.fusion.method,
                                        k=self.cfg.fusion.k,
                                        L=L
                                    )

            for i, Z_fused in enumerate(fused_list): 
                store.add(fuse_metric[i], Z_fused)

            end_time = time.time() 
            time_taken = end_time - beg_time
            if self.cfg.verbose:
                print(f"Time taken for fusion {time_taken/60} minutes")
        
        
        ## target prioritization evaluation 
        if self.cfg.evaluation.enabled: 
            
            seed_indices = [
                i for i, node in enumerate(graph_data.nodes)
                if node in source
            ]
            scores_by_method = {}
            for name, Z in store.items():
                if self.cfg.eval.centroid:
                    scores_by_method[f"{name}_centroid"] = seed_centroid_scores(
                        Z, seed_indices
                    )
                if self.cfg.eval.max_seed:
                    scores_by_method[f"{name}_max"] = max_seed_cosine_scores(
                        Z, seed_indices
                    )
            
            ranking_df = evaluate_embeddings_ranking(
                scores_by_method=scores_by_method,
                subgraph=graph_data,
                seeds=source,
                targets=target,
                nodes=graph_data.nodes,
                k_values=self.cfg.eval.k_values,
                n_repeats=self.cfg.eval.n_repeats,
                deg_tol=self.cfg.eval.deg_tol,
                iteration=it,
            )
            # standard metadata for analysis 
            
            return {
                    "iteration": it,
                    "ranking_df": ranking_df,
                    "comparison": comparison_metrics
                }
        else: 
            return {
                "iteration": it, 
                "embeddings": store, 
                "nodes": list(graph_data.nodes), 
                "comparison": comparison_metrics
            }

    #-----------------
    # Preprocess
    # ----------------
    
    def _preprocess_graph(self, graph_data, source, target):
        cfg_pg = PrepareGraphConfig(
                            subsample_nodes=self.cfg.preprocess.subsample.enabled, 
                            max_nodes=self.cfg.preprocess.subsample.max_nodes, 
                            radius=self.cfg.preprocess.subsample.radius,
                            sparsify_edges=self.cfg.preprocess.sparsify.enabled,
                            retain_ratio=self.cfg.preprocess.sparsify.retain_ratio,
                            max_degree=self.cfg.preprocess.sparsify.max_degree,
                            scoring=self.cfg.preprocess.sparsify.scoring,
                            verbose=self.cfg.verbose
                            )
        graph_data = prepare_graph(
                            cfg_pg, 
                            graph=graph_data, 
                            seeds=source, 
                            targets=target, 
                            seed=self.cfg.seed
                            )
        return graph_data 
    
    #-----------------
    # Data Loading
    # ----------------
    
    def _load_graph(self):
        self.log.info("Loading graph: %s", self.cfg.graph.name)
        
        return load_graph(self.cfg)
    
    def _load_gwas_data(self, graph_data):
        self.log.info("Loading gwas data: %s", self.cfg.disease.name)
        return load_gwas_data(self.cfg, graph_data)
    
    def _set_iteration_seed(self, it):
        seed = self.base_seed + it
        set_global_seed(seed)
        self.log.debug("Iteration seed set to %d", seed)
        

    #--------------------------------
    # Build structured, multi-views
    # -------------------------------
    def _chunkify(self, seq, chunk_size): 
        for i in range(0, len(seq), chunk_size): 
            yield seq[i:i + chunk_size]
    
    def _process_root(self, graph_data, root, node2idx, it): 
        
        idx = node2idx[root]
        seed = (self.cfg.experiment.base_seed + 10000 * it + idx)
        rng = np.random.default_rng(seed)
        
        views = self._build_views(graph_data, root, rng) 
        walk_outputs = self._run_walks_for_root(graph_data, root, views, rng) 
        
        if not walk_outputs or all(len(walks) == 0 for walks in walk_outputs.values()):
            return root, {}   # or mark as invalid
        else:
            return root, walk_outputs
    
    def _process_root_chunk(self, graph_data, roots, node2idx, it):
        """
        Process a batch of roots inside a single worker process.
        Returns a list of (root, walk_outputs).
        """
        results = []

        for root in roots:
            root, walk_outputs = self._process_root(graph_data, root, node2idx, it)
            results.append((root, walk_outputs))

        return results
    
    def _build_views(self, graph_data, root, rng):
        
        view_gen = ViewBuilder(cfg=self.cfg, rng=rng)
        return view_gen.build(graph_data, root)
    
    def _run_walks_for_root(self, graph_data, root, views, rng): 
        
        walker = BaseWalker(cfg=self.cfg, rng=rng)
        all_walks = {k: [] for k in self.cfg.walks.kinds}
        
        for view in views: 
            #induce subgraph 
            view_g = graph_data.subgraph(view)
            view_nodes = list(view_g.nodes())
            
            #run walker once per view 
            out = walker.run(graph_data, root, view_nodes)
            
            for walk_kind, walks in out.items(): 
                all_walks[walk_kind].extend(walks)
        
        return all_walks
    

        
    def _train_embeddings(self, graph_data, all_corpora):
        embeddings = {}
        
        for kind, corpus in all_corpora.items():
            Z = corpus_to_embedding(
                                    corpus=corpus,
                                    nodes=graph_data.nodes,
                                    vector_size=self.cfg.train.embedding_dim,
                                    window=self.cfg.train.window,
                                    sg=self.cfg.train.sg,
                                    negative=self.cfg.train.negative,
                                    min_count=self.cfg.min_count,
                                    workers=self.cfg.train.workers,
                                    epochs=self.cfg.train.epochs
                                    )
            
            embeddings[kind] = Z
            
        return embeddings

    def _build_quantum_targets(self, graph_data, source):
        node_order = list(graph_data.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_order)}
        valid_seeds = [node for node in source if node in node_to_idx]
        if not valid_seeds:
            return None

        targets = []
        max_support = getattr(self.cfg.baselines, "quantum_target_max_nodes", 64)
        for center in valid_seeds:
            lengths = nx.single_source_shortest_path_length(graph_data, center, cutoff=2)
            support_nodes = [node for node, dist in lengths.items() if dist <= 2]
            if center not in support_nodes:
                support_nodes.append(center)
            if len(support_nodes) > max_support:
                support_nodes = support_nodes[:max_support]

            center_idx = node_to_idx[center]
            support_idx = np.array([node_to_idx[node] for node in support_nodes], dtype=np.int64)
            dist = np.abs(support_idx - center_idx).astype(np.float64) + 1.0
            p = 1.0 / dist
            p = p / p.sum()
            targets.append({"nodes": support_nodes, "center": center, "pQ": p})

        return targets
    
    def _save_evaluation_results(self, all_results, nodes):
        
        ranking_df = self._post_process_ranking(all_results)
        comparison_df = self._post_process_comparison(all_results)
            
        out_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(out_dir, exist_ok=True)

        self.log.info("Saving outputs to %s", out_dir)
        
        ranking_path = os.path.join(out_dir, "ranking_results.csv")
        ranking_df.to_csv(ranking_path, index=False)

        comparison_path = os.path.join(out_dir, "embedding_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)

        
        cfg_path = os.path.join(out_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
            
        summary = {
                "n_iterations": self.n_iters,
                "n_nodes": len(nodes),
                "walks": OmegaConf.to_container(self.cfg.walks.kinds, resolve=True),
                }

        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        if self.cfg.plots:
            self._plot_all(
                    ranking_df=ranking_df, 
                    out_dir=out_dir
                    )
        self.log.info("All results saved to %s", out_dir)
        
    def _post_process_ranking(self, all_results):
        
        ranking_dfs = [
                        r["ranking_df"] for r in all_results
                        if r["ranking_df"] is not None
                    ]   

        ranking_results_df = pd.concat(
            ranking_dfs,
            ignore_index=True
        )
        
        return ranking_results_df 
    
    def _post_process_comparison(self, all_results): 
        comparison_rows = []

        for r in all_results:
            it = r["iteration"]
            for pair, metrics in r["comparison"].items():
                for name, value in metrics.items():
                    comparison_rows.append({
                        "iteration": it,
                        "pair": pair,
                        "metric": name,
                        "value": value,
                    })

        comparison_df = pd.DataFrame(comparison_rows)

        return comparison_df
                
    def _plot_all(self, ranking_df, out_dir):
        
        plot_metric(cfg=self.cfg, 
                        df=ranking_df, 
                        metric='recall', 
                        file_path=out_dir)
        plot_metric(cfg=self.cfg, 
                    df=ranking_df, 
                    metric='precision', 
                    file_path=out_dir)
        
        plot_precision_recall(df=ranking_df, 
                            control='true', 
                            file_path=out_dir)
        plot_precision_recall(df=ranking_df, 
                            control='degree_matched', 
                            file_path=out_dir)
        plot_precision_recall(df=ranking_df, 
                            control='distance_matched', 
                            file_path=out_dir)
        
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='true',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='true',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='degree_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='degree_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='recall',
                        control='distance_matched',
                        file_path=out_dir)
        plot_metric_vs_k(df=ranking_df, 
                        metric='precision',
                        control='distance_matched',
                        file_path=out_dir)

    def _save_embeddings(self, all_results):

        out_dir = HydraConfig.get().runtime.output_dir
        emb_dir = os.path.join(out_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)

        self.log.info("Saving embeddings to %s", emb_dir)

        for res in all_results:
            iter_num = res["iteration"]
            
            # Build dictionary for np.savez
            npz_payload = {}

            for emb_name, emb in res["embeddings"].items():
                if emb is None:
                    continue
                npz_payload[emb_name] = emb.astype(np.float32, copy=False)

            # Always store node ordering for alignment downstream
            npz_payload["nodes"] = np.asarray(res["nodes"])

            ofname = os.path.join(
                emb_dir, f"embeddings_iter_{iter_num}.npz"
            )

            np.savez_compressed(ofname, **npz_payload)
            
            comparison_df = self._post_process_comparison(all_results=all_results)
            comp_ofname = "embedding_comparison_iter"+str(iter_num)+'.csv'
            comparison_path = os.path.join(emb_dir, comp_ofname)
            comparison_df.to_csv(comparison_path, index=False)
            

            self.log.debug(
                "Saved iteration %d embeddings: %s",
                iter_num,
                list(npz_payload.keys()),
            )
