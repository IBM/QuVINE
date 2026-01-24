# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from quvine.baselines import run_node2vec
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
        source, target = self._load_gwas_data(graph_data)
        #graph_data = self._preprocess_graph(graph_data, source, target)
        
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

        ranking_df, comparison_df = self._post_process(all_results)
        
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
        
        # compare embeddings 
        comparison_metrics = compare_embeddings(
                                        store,
                                        cca_components=self.cfg.analysis.cca_components,
                                        knn_k=self.cfg.analysis.knn_k,
                                        )
        
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
        
        ## baselines
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
        
        
        ## evaluation 
    
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

        return ranking_results_df, comparison_df
        
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
        
