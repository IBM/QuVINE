#!/usr/bin/env python3
"""
Run a single extended generator network analysis job.

Generates one of the 5 extended synthetic graph families and runs the full
QuVINE embedding pipeline with all specified methods.

Usage
-----
python scripts/run_extended_generator_network.py \
    --network-type random_regular \
    --network-id   random_regular_n500_rep00 \
    --output-dir   outputs/extended_generators/results/random_regular_n500_rep00 \
    --methods      quvine_fused-walk,quvine_ctqw,...,baseline_filter \
    --n-nodes      500 \
    --seed         10000 \
    [--resume] \
    [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

import networkx as nx

# Allow importing from src/ regardless of install state
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# Extended network generators
# ---------------------------------------------------------------------------

def _build_extended_network(network_type: str, n_nodes: int, seed: int):
    """Generate an extended synthetic network.

    Parameters
    ----------
    network_type : str
        One of: random_regular, heterophilic_sbm, degree_corrected_sbm,
        grid_torus, configuration_model
    n_nodes : int
        Target number of nodes
    seed : int
        Random seed

    Returns
    -------
    G : nx.Graph
        Generated graph with metadata
    metadata : dict
        Flat dict of parameters
    """
    from quvine.data import (
        generate_random_regular_expander_like,
        generate_heterophilic_sbm,
        generate_degree_corrected_sbm,
        generate_grid_torus_lattice,
        generate_configuration_model_graph,
    )

    if network_type == "random_regular":
        # Choose degree based on n_nodes to keep density reasonable
        if n_nodes < 100:
            d = 4
        elif n_nodes < 1000:
            d = 6
        else:
            d = 8
        # Ensure n*d is even
        if (n_nodes * d) % 2 != 0:
            d += 1
        G = generate_random_regular_expander_like(n_nodes, d=d, seed=seed)
        metadata = dict(G.graph)

    elif network_type == "heterophilic_sbm":
        # Scale blocks and avg degree with n_nodes
        if n_nodes < 100:
            n_blocks = 2
            target_avg_degree = 4.0
        elif n_nodes < 1000:
            n_blocks = 4
            target_avg_degree = 8.0
        else:
            n_blocks = 6
            target_avg_degree = 12.0
        out_in_ratio = 2.0  # Heterophilic
        G, labels = generate_heterophilic_sbm(
            n=n_nodes,
            n_blocks=n_blocks,
            target_avg_degree=target_avg_degree,
            out_in_ratio=out_in_ratio,
            seed=seed
        )
        metadata = dict(G.graph)

    elif network_type == "degree_corrected_sbm":
        # Scale blocks and avg degree with n_nodes
        if n_nodes < 100:
            n_blocks = 2
            target_avg_degree = 4.0
        elif n_nodes < 1000:
            n_blocks = 4
            target_avg_degree = 8.0
        else:
            n_blocks = 6
            target_avg_degree = 12.0
        out_in_ratio = 0.5  # Assortative
        G, labels = generate_degree_corrected_sbm(
            n=n_nodes,
            n_blocks=n_blocks,
            target_avg_degree=target_avg_degree,
            out_in_ratio=out_in_ratio,
            degree_distribution='powerlaw',
            seed=seed
        )
        metadata = dict(G.graph)

    elif network_type == "grid_torus":
        # Periodic torus with optional diagonals
        periodic = True
        add_diagonals = (seed % 2 == 0)  # Vary diagonals based on seed
        G = generate_grid_torus_lattice(
            n=n_nodes,
            periodic=periodic,
            add_diagonals=add_diagonals,
            seed=seed
        )
        metadata = dict(G.graph)

    elif network_type == "configuration_model":
        # Power-law degree distribution
        if n_nodes < 100:
            target_avg_degree = 4.0
        elif n_nodes < 1000:
            target_avg_degree = 8.0
        else:
            target_avg_degree = 12.0
        G = generate_configuration_model_graph(
            n=n_nodes,
            distribution='powerlaw',
            target_avg_degree=target_avg_degree,
            gamma=2.5,
            seed=seed
        )
        metadata = dict(G.graph)

    else:
        raise ValueError(f"Unknown network_type: {network_type}")

    return G, metadata


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run extended generator network analysis"
    )
    parser.add_argument("--network-type", required=True,
                        choices=["random_regular", "heterophilic_sbm",
                                "degree_corrected_sbm", "grid_torus",
                                "configuration_model"],
                        help="Extended generator type")
    parser.add_argument("--network-id", required=True,
                        help="Unique network identifier")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for this network")
    parser.add_argument("--methods", required=True,
                        help="Comma-separated list of embedding methods")
    parser.add_argument("--n-nodes", type=int, required=True,
                        help="Target number of nodes")
    parser.add_argument("--seed", type=int, required=True,
                        help="Random seed")
    parser.add_argument("--hparam-file", default=None,
                        help="Path to best hyperparameters JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Skip methods already computed")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphml_path = output_dir / f"{args.network_id}.graphml"
    metadata_path = output_dir / f"{args.network_id}_metadata.json"

    # ── Generate or load network ───────────────────────────────────────────
    if graphml_path.exists():
        if args.verbose:
            print(f"Loading existing network from {graphml_path}")
        G = nx.read_graphml(graphml_path)
        G = nx.convert_node_labels_to_integers(G)
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {"type": args.network_type, "seed": args.seed}
    else:
        if args.verbose:
            print(f"Generating {args.network_type} network with {args.n_nodes} nodes, seed={args.seed}")
        
        G, metadata = _build_extended_network(
            network_type=args.network_type,
            n_nodes=args.n_nodes,
            seed=args.seed
        )
        
        # Save network
        nx.write_graphml(G, graphml_path)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if args.verbose:
            print(f"Saved network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Run embedding pipeline ─────────────────────────────────────────────
    from quvine.pipeline import run_comprehensive_pipeline

    methods = [m.strip() for m in args.methods.split(',')]
    
    if args.verbose:
        print(f"\nRunning pipeline with {len(methods)} methods:")
        for m in methods:
            print(f"  - {m}")

    # Load hyperparameters if provided
    hparams = None
    if args.hparam_file and Path(args.hparam_file).exists():
        with open(args.hparam_file, 'r') as f:
            hparams = json.load(f)
        if args.verbose:
            print(f"Loaded hyperparameters from {args.hparam_file}")

    results = run_comprehensive_pipeline(
        graph=G,
        network_name=args.network_id,
        network_metadata=metadata,
        methods=methods,
        output_dir=str(output_dir),
        hyperparams=hparams,
        resume=args.resume,
        verbose=args.verbose
    )

    if args.verbose:
        print(f"\nPipeline complete. Results saved to {output_dir}")
        print(f"  - ranking_results.csv")
        print(f"  - classification_results.csv")
        print(f"  - link_prediction_results.csv")

    return 0


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
