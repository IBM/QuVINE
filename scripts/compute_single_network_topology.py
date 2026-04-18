#!/usr/bin/env python3
"""
Compute topological complexity metrics for a single network.

This script is called by submit_topological_complexity_jobs.sh for each network.
"""

import sys
import argparse
from pathlib import Path
import networkx as nx
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# Add quvine to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from quvine.complexity.graph import compute_topological_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphml", required=True, help="Path to graphml file")
    parser.add_argument("--network-id", required=True, help="Network ID")
    parser.add_argument("--output-csv", required=True, help="Output complexity CSV file")
    args = parser.parse_args()

    print(f"="*80)
    print(f"Computing topological metrics for: {args.network_id}")
    print(f"="*80)
    
    # Load graph
    print(f"Loading graph from: {args.graphml}")
    G = nx.read_graphml(args.graphml)
    G = nx.convert_node_labels_to_integers(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Connected: {nx.is_connected(G)}")
    
    # Compute topological metrics
    print("\nComputing topological metrics...")
    try:
        metrics = compute_topological_metrics(
            G,
            include_betti=True,
            include_persistence_entropy=True,
            maxdim=2,
            filtration_scale=1.0
        )
        
        # Extract only topological metrics
        topo_metrics = {
            'betti_0': metrics.get('betti_0', 0),
            'betti_1': metrics.get('betti_1', 0),
            'betti_2': metrics.get('betti_2', 0),
            'betti_sum': metrics.get('betti_sum', 0),
            'euler_characteristic': metrics.get('euler_characteristic', 0),
            'persistence_entropy_H0': metrics.get('persistence_entropy_H0', 0.0),
            'persistence_entropy_H1': metrics.get('persistence_entropy_H1', 0.0),
            'persistence_entropy_H2': metrics.get('persistence_entropy_H2', 0.0),
        }
        
        print("\nResults:")
        for key, value in topo_metrics.items():
            print(f"  {key}: {value}")
        
        # Read existing complexity file
        print(f"\nUpdating complexity file: {args.output_csv}")
        comp_df = pd.read_csv(args.output_csv)
        
        # Add topological metrics
        for key, value in topo_metrics.items():
            comp_df[key] = value
        
        # Save updated file
        comp_df.to_csv(args.output_csv, index=False)
        print(f"Saved! New shape: {comp_df.shape}")
        
        print(f"\n{'='*80}")
        print(f"SUCCESS: {args.network_id}")
        print(f"{'='*80}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# Made with Bob
