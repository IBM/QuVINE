"""
Complexity-Aware QuVINE Pipeline

This module provides a pipeline that computes graph complexity metrics
and then runs QuVINE embedding with complexity-aware configuration.
"""

import logging
import os
import json
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd
import networkx as nx
from omegaconf import DictConfig, OmegaConf

from quvine.pipeline import Pipeline
from quvine.complexity.graph import (
    compute_graph_complexity_metrics,
    compute_laplacian_spectrum,
)
from quvine.complexity.qbc import (
    compute_qbc_complexity_from_laplacian,
    check_qbc_available,
)
from quvine.data.random_graphs import generate_graph_with_seeds_and_targets


class ComplexityAwarePipeline:
    """
    Pipeline that computes graph complexity before running QuVINE.
    
    This pipeline:
    1. Computes comprehensive complexity metrics (spectral + QBC)
    2. Saves complexity analysis
    3. Optionally adjusts QuVINE parameters based on complexity
    4. Runs standard QuVINE pipeline
    5. Saves results with complexity metadata
    """
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)
        self.run_dir = Path(cfg.runtime.output_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Complexity results
        self.complexity_metrics = None
        self.qbc_metrics = None
        
    def run(self):
        """Run the complete complexity-aware pipeline."""
        self.log.info("Starting Complexity-Aware QuVINE Pipeline")
        
        # Step 1: Load graph
        graph = self._load_graph()
        
        # Step 2: Compute complexity metrics
        self.log.info("Computing graph complexity metrics...")
        self.complexity_metrics = self._compute_complexity(graph)
        
        # Step 3: Save complexity analysis
        self._save_complexity_analysis()
        
        # Step 4: Optionally adjust configuration based on complexity
        if self.cfg.get('complexity_aware_config', False):
            self._adjust_config_by_complexity()
        
        # Step 5: Run standard QuVINE pipeline
        self.log.info("Running QuVINE pipeline...")
        pipeline = Pipeline(self.cfg)
        pipeline.run()
        
        # Step 6: Add complexity metadata to results
        self._add_complexity_to_results()
        
        self.log.info("Complexity-Aware Pipeline completed successfully")
        
    def _load_graph(self) -> nx.Graph:
        """Load graph from configuration."""
        from quvine.data.data_loader import load_graph
        return load_graph(self.cfg)
    
    def _compute_complexity(self, graph: nx.Graph) -> Dict:
        """
        Compute all complexity metrics for the graph.
        
        Returns
        -------
        dict
            Combined complexity metrics
        """
        # Compute standard complexity metrics
        metrics = compute_graph_complexity_metrics(graph)
        
        # Add QBioCode metrics if available
        if check_qbc_available():
            try:
                self.log.info("Computing QBioCode complexity metrics...")
                qbc_metrics = compute_qbc_complexity_from_laplacian(
                    graph,
                    normalized=True,
                    laplacian_method='eigenvectors'
                )
                self.qbc_metrics = qbc_metrics
                
                # Add selected QBC metrics to main metrics
                qbc_keys = [
                    'Intrinsic_Dimension',
                    'Condition number',
                    'Manifold Complexity',
                    'Total Correlations',
                    'Fractal Dimension'
                ]
                for key in qbc_keys:
                    if key in qbc_metrics:
                        metrics[f'qbc_{key.lower().replace(" ", "_")}'] = qbc_metrics[key]
                        
            except Exception as e:
                self.log.warning(f"Failed to compute QBC metrics: {e}")
        else:
            self.log.info("QBioCode not available, skipping QBC metrics")
        
        return metrics
    
    def _save_complexity_analysis(self):
        """Save complexity analysis to files."""
        out_dir = self.run_dir / "complexity_analysis"
        out_dir.mkdir(exist_ok=True)
        
        # Save main complexity metrics
        complexity_path = out_dir / "complexity_metrics.json"
        with open(complexity_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_metrics = {
                k: float(v) if hasattr(v, 'item') else v
                for k, v in self.complexity_metrics.items()
            }
            json.dump(serializable_metrics, f, indent=2)
        
        self.log.info(f"Complexity metrics saved to {complexity_path}")
        
        # Save QBC metrics separately if available
        if self.qbc_metrics:
            qbc_path = out_dir / "qbc_complexity_metrics.json"
            with open(qbc_path, 'w') as f:
                serializable_qbc = {
                    k: float(v) if hasattr(v, 'item') else v
                    for k, v in self.qbc_metrics.items()
                    if not isinstance(v, (list, dict))
                }
                json.dump(serializable_qbc, f, indent=2)
            
            self.log.info(f"QBC metrics saved to {qbc_path}")
        
        # Create summary report
        self._create_complexity_report(out_dir)
    
    def _create_complexity_report(self, out_dir: Path):
        """Create a human-readable complexity report."""
        report_path = out_dir / "complexity_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("Graph Complexity Analysis Report\n")
            f.write("="*60 + "\n\n")
            
            # Basic properties
            f.write("Basic Properties:\n")
            f.write(f"  Nodes: {self.complexity_metrics['num_nodes']}\n")
            f.write(f"  Edges: {self.complexity_metrics['num_edges']}\n\n")
            
            # Spectral properties
            f.write("Spectral Properties:\n")
            f.write(f"  Spectral Gap: {self.complexity_metrics['spectral_gap']:.4f}\n")
            f.write(f"  Algebraic Connectivity: {self.complexity_metrics['algebraic_connectivity']:.4f}\n")
            f.write(f"  Spectral Entropy: {self.complexity_metrics['spectral_entropy']:.4f}\n\n")
            
            # Quantum metrics
            f.write("Quantum-Inspired Metrics:\n")
            f.write(f"  Von Neumann Entropy: {self.complexity_metrics['von_neumann_entropy']:.4f}\n")
            f.write(f"  Quantum Complexity: {self.complexity_metrics['quantum_complexity']:.4f}\n")
            f.write(f"  Estrada Index: {self.complexity_metrics['estrada_index']:.2f}\n\n")
            
            # QBC metrics if available
            if self.qbc_metrics:
                f.write("QBioCode Metrics:\n")
                if 'Intrinsic_Dimension' in self.qbc_metrics:
                    f.write(f"  Intrinsic Dimension: {self.qbc_metrics['Intrinsic_Dimension']:.2f}\n")
                if 'Condition number' in self.qbc_metrics:
                    f.write(f"  Condition Number: {self.qbc_metrics['Condition number']:.2e}\n")
                if 'Manifold Complexity' in self.qbc_metrics:
                    f.write(f"  Manifold Complexity: {self.qbc_metrics['Manifold Complexity']:.4f}\n")
                f.write("\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            qc = self.complexity_metrics['quantum_complexity']
            if qc > 0.5:
                f.write("  ✓ High quantum complexity - quantum walks (CTQW/DTQW) recommended\n")
            elif qc > 0.3:
                f.write("  • Medium complexity - test both quantum and classical walks\n")
            else:
                f.write("  • Low complexity - classical walks (RWR) may suffice\n")
            
            sg = self.complexity_metrics['spectral_gap']
            if sg < 0.1:
                f.write("  ✓ Low spectral gap - quantum tunneling may help with bottlenecks\n")
            
            f.write("\n" + "="*60 + "\n")
        
        self.log.info(f"Complexity report saved to {report_path}")
    
    def _adjust_config_by_complexity(self):
        """Adjust QuVINE configuration based on complexity metrics."""
        qc = self.complexity_metrics['quantum_complexity']
        
        self.log.info(f"Adjusting configuration based on quantum complexity: {qc:.4f}")
        
        # Adjust walk types based on complexity
        if qc > 0.5:
            # High complexity - prioritize quantum walks
            if 'ctqw' not in self.cfg.walks.kinds:
                self.cfg.walks.kinds.append('ctqw')
            self.log.info("Added CTQW due to high quantum complexity")
        
        # Adjust number of walks based on graph size and complexity
        n_nodes = self.complexity_metrics['num_nodes']
        if n_nodes > 1000 and qc > 0.4:
            # Large complex graph - may need more walks
            original_walks = self.cfg.walks.get('num_walks_per_root', 10)
            self.cfg.walks.num_walks_per_root = max(original_walks, 15)
            self.log.info(f"Increased walks per root to {self.cfg.walks.num_walks_per_root}")
    
    def _add_complexity_to_results(self):
        """Add complexity metadata to final results."""
        summary_path = self.run_dir / "summary.json"
        
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
        else:
            summary = {}
        
        # Add complexity metrics
        summary['complexity'] = {
            'quantum_complexity': float(self.complexity_metrics['quantum_complexity']),
            'von_neumann_entropy': float(self.complexity_metrics['von_neumann_entropy']),
            'spectral_gap': float(self.complexity_metrics['spectral_gap']),
            'num_nodes': int(self.complexity_metrics['num_nodes']),
            'num_edges': int(self.complexity_metrics['num_edges']),
        }
        
        if self.qbc_metrics and 'Manifold Complexity' in self.qbc_metrics:
            summary['complexity']['qbc_manifold_complexity'] = float(
                self.qbc_metrics['Manifold Complexity']
            )
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.log.info("Added complexity metadata to summary.json")


def run_complexity_pipeline_on_random_graphs(
    graph_types: List[str],
    n: int = 100,
    cfg_template: Optional[DictConfig] = None,
    output_base_dir: str = "outputs/complexity_benchmark",
    seed: int = 42
) -> pd.DataFrame:
    """
    Run complexity analysis and QuVINE on multiple random graph types.
    
    Parameters
    ----------
    graph_types : list of str
        Graph types to generate: 'erdos_renyi', 'barabasi_albert', 
        'watts_strogatz', 'modular', etc.
    n : int, default=100
        Number of nodes
    cfg_template : DictConfig, optional
        Base configuration template for QuVINE
    output_base_dir : str
        Base directory for outputs
    seed : int
        Random seed
        
    Returns
    -------
    pd.DataFrame
        Summary of complexity metrics for all graphs
    """
    from quvine.data.random_graphs import (
        generate_erdos_renyi,
        generate_barabasi_albert,
        generate_watts_strogatz,
        generate_modular_network,
    )
    
    results = []
    
    for graph_type in graph_types:
        print(f"\n{'='*60}")
        print(f"Processing: {graph_type}")
        print(f"{'='*60}")
        
        # Generate graph
        if graph_type == 'erdos_renyi':
            G, seeds, targets = generate_graph_with_seeds_and_targets(
                n=n, num_seeds=10, num_targets=15,
                graph_type='erdos_renyi', p=0.05, seed=seed
            )
        elif graph_type == 'barabasi_albert':
            G, seeds, targets = generate_graph_with_seeds_and_targets(
                n=n, num_seeds=10, num_targets=15,
                graph_type='barabasi_albert', m=3, seed=seed
            )
        elif graph_type == 'watts_strogatz':
            G, seeds, targets = generate_graph_with_seeds_and_targets(
                n=n, num_seeds=10, num_targets=15,
                graph_type='watts_strogatz', k=6, p=0.3, seed=seed
            )
        elif graph_type == 'modular':
            G, seeds, targets = generate_graph_with_seeds_and_targets(
                n=n, num_seeds=10, num_targets=15,
                graph_type='modular', num_communities=5,
                p_intra=0.3, p_inter=0.01, seed=seed
            )
        else:
            print(f"Unknown graph type: {graph_type}, skipping")
            continue
        
        # Compute complexity
        metrics = compute_graph_complexity_metrics(G)
        metrics['graph_type'] = graph_type
        
        # Add QBC if available
        if check_qbc_available():
            try:
                qbc_metrics = compute_qbc_complexity_from_laplacian(
                    G, normalized=True, laplacian_method='eigenvectors'
                )
                if 'Manifold Complexity' in qbc_metrics:
                    metrics['qbc_manifold_complexity'] = qbc_metrics['Manifold Complexity']
            except (ImportError, ValueError, KeyError) as e:
                print(f"  QBC manifold complexity computation skipped: {e}")
            except Exception as e:
                print(f"  Warning: QBC manifold complexity computation failed: {e}")
        
        results.append(metrics)
        
        # Print summary
        print(f"Quantum Complexity: {metrics['quantum_complexity']:.4f}")
        print(f"Von Neumann Entropy: {metrics['von_neumann_entropy']:.4f}")
        print(f"Spectral Gap: {metrics['spectral_gap']:.4f}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    output_dir = Path(output_base_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / "complexity_comparison.csv"
    df.to_csv(summary_path, index=False)
    print(f"\nResults saved to {summary_path}")
    
    return df


def compute_and_save_complexity(
    graph: nx.Graph,
    output_dir: str,
    graph_name: str = "graph"
) -> Dict:
    """
    Compute all complexity metrics and save to directory.
    
    Parameters
    ----------
    graph : nx.Graph
        Input graph
    output_dir : str
        Output directory
    graph_name : str
        Name for the graph
        
    Returns
    -------
    dict
        Complexity metrics
    """
    # Compute metrics
    metrics = compute_graph_complexity_metrics(graph)
    
    # Add QBC if available
    qbc_metrics = None
    if check_qbc_available():
        try:
            qbc_metrics = compute_qbc_complexity_from_laplacian(
                graph, normalized=True, laplacian_method='eigenvectors',
                dataset_name=graph_name
            )
        except Exception as e:
            print(f"Warning: QBC computation failed: {e}")
    
    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save main metrics
    metrics_path = out_path / f"{graph_name}_complexity.json"
    with open(metrics_path, 'w') as f:
        serializable = {k: float(v) if hasattr(v, 'item') else v 
                       for k, v in metrics.items()}
        json.dump(serializable, f, indent=2)
    
    # Save QBC metrics
    if qbc_metrics:
        qbc_path = out_path / f"{graph_name}_qbc_complexity.json"
        with open(qbc_path, 'w') as f:
            serializable = {k: float(v) if hasattr(v, 'item') else v 
                           for k, v in qbc_metrics.items()
                           if not isinstance(v, (list, dict))}
            json.dump(serializable, f, indent=2)
    
    print(f"Complexity metrics saved to {out_path}")
    
    return metrics

