from __future__ import annotations 

import logging
import os
from pathlib import Path 
from typing import Dict, List 
import json
from omegaconf import OmegaConf,DictConfig
from hydra.core.hydra_config import HydraConfig
import pandas as pd 

from quvine.data.data_loader import load_graph, load_gwas_data
from quvine.data.preprocess import build_subgraph, sparsify_graph
from quvine.views.generator import ViewBuilder
from quvine.walks.base import BaseWalker
from quvine.corpus.builder import CorpusBuilder
from quvine.embedding.word2vec import corpus_to_embedding
from quvine.embedding.registry import EmbeddingStore
from quvine.analysis.compare import compare_embeddings
from quvine.analysis.analyze import *
from quvine.baselines import run_node2vec
from quvine.fusion.fuse import fuse_embeddings
from quvine.fusion.diagnostics import analyze_fusion
from quvine.evaluation.ranking import (
    seed_centroid_scores,
    max_seed_cosine_scores,
    evaluate_embeddings_ranking
    )   
#from quvine.evals.ranking import evaluate_ranking
# from utils.io import save_embeddings, save_metadata 
from quvine.utils.seed import set_global_seed
from quvine.utils.utilities import *

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
        self.run_dir = Path(cfg.runtime.run_dir)
        if self.run_dir.is_dir(): 
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
        source, target = self._load_gwas_data(graph_data)
        graph_data = self._preprocess_graph(graph_data, source, target)
        
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

        ranking_df, comparison_df, fusion_df = self._post_process(all_results)
        
        out_dir = HydraConfig.get().runtime.output_dir
        os.makedirs(out_dir, exist_ok=True)

        self.log.info("Saving outputs to %s", out_dir)
        
        ranking_path = os.path.join(out_dir, "ranking_results.csv")
        ranking_df.to_csv(ranking_path, index=False)

        comparison_path = os.path.join(out_dir, "embedding_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        fusion_path = os.path.join(out_dir, "fusion_diagnostics.csv")
        fusion_df.to_csv(fusion_path, index=False)
        
        cfg_path = os.path.join(out_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            f.write(OmegaConf.to_yaml(self.cfg))
            
        summary = {
                "n_iterations": self.n_iters,
                "n_nodes": len(graph_data.nodes),
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

    #-----------------
    # One iteration
    # ----------------
    
    def _run_single_iteration(self, it, graph_data, source, target):
        
        corpus_builder = {kind: CorpusBuilder() for kind in self.cfg.walks.kinds}
        
        for root in graph_data.nodes: 
            
            views = self._build_views(graph_data, root, it)
            walk_outputs = self._run_walks_for_root(graph_data, root, views)
            
            for walk_kind, walks in walk_outputs.items(): 
                corpus_builder[walk_kind].add(root, walks)
        
        all_corpora = {kind: builder.build() for kind, builder in corpus_builder.items()}
        
        embeddings = self._train_embeddings(graph_data, all_corpora)
        
        store = EmbeddingStore()
        for name, Z in embeddings.items():
            store.add(name, Z)
        
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
        
        
        # compare embeddings 
        comparison_metrics = compare_embeddings(
                                        store,
                                        cca_components=self.cfg.analysis.cca_components,
                                        knn_k=self.cfg.analysis.knn_k,
                                        )
        
        ## fuse embeddings
        fusion_results = {}

        if self.cfg.fusion.enabled:
            Z_fused, _ = fuse_embeddings(
                store,
                method=self.cfg.fusion.method,
                k=self.cfg.fusion.k,
            )

            fusion_results = analyze_fusion(
                Z_fused,
                n_views=len(store.names()),
            )

            store.add("fused", Z_fused)
        
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
                "comparison": comparison_metrics,
                "fusion": fusion_results,
            }

    
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
        
    #-----------------
    # Preprocess
    # ----------------
    
    def _preprocess_graph(self, graph_data, seeds, targets): 
        if self.cfg.preprocess.subgraph.enabled: 
            graph_data, source, target = build_subgraph(self.cfg, 
                                                    graph=graph_data, 
                                                    seeds=seeds, 
                                                    targets=targets, 
                                                    num_nodes=self.cfg.preprocess.subgraph.num_nodes, 
                                                    max_hops=self.cfg.preprocess.subgraph.max_hops, 
                                                    max_nodes=self.cfg.preprocess.subgraph.max_nodes, 
                                                    random_state=self.base_seed)
        
        if self.cfg.preprocess.sparsify.enabled: 
            graph_data = sparsify_graph( 
                                        graph=graph_data, 
                                        target_avg_degree=self.cfg.preprocess.sparsify.avg_degree,
                                        max_degree=self.cfg.preprocess.sparsify.max_degree,
                                        seed=self.base_seed, 
                                        verbose=self.cfg.verbose)
        return graph_data
            
    #--------------------------------
    # Build structured, multi-views
    # -------------------------------
    
    def _build_views(self, graph_data, root, it):
        
        view_gen = ViewBuilder(cfg=self.cfg, iteration_seed=self.base_seed+it)
        return view_gen.build(graph_data, root)
    
    def _run_walks_for_root(self, graph_data, root, views): 
        
        walker = BaseWalker(cfg=self.cfg)
        all_walks = {k: [] for k in self.cfg.walks.kinds}
        
        for view in views: 
            #induce subgraph 
            view_g = graph_data.subgraph(view)
            view_nodes = list(view_g.nodes())
            
            #run walker once per view 
            out = walker.run(graph_data, root, view_nodes)
            
            for walk_kind, walks in out.items(): 
                # self.log.info(
                #     "root=%s kind=%s n_walks=%d",
                #     root, walk_kind, len(walks)
                # )
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
        
    def _post_process(self, all_results):
        
        ranking_dfs = [
                        r["ranking_df"] for r in all_results
                        if r["ranking_df"] is not None
                    ]   

        ranking_results_df = pd.concat(
            ranking_dfs,
            ignore_index=True
        )
        
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
        
        fusion_rows = []

        for r in all_results:
            it = r["iteration"]
            for name, value in r["fusion"].items():
                fusion_rows.append({
                    "iteration": it,
                    "metric": name,
                    "value": value,
                })

        fusion_df = pd.DataFrame(fusion_rows)

        return ranking_results_df, comparison_df, fusion_df
        
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
        
