#!/usr/bin/env python3
"""
Run a single hard-negatives network analysis job.

If the network's .graphml already exists in the output directory, it is loaded
directly (no regeneration).  Pass --resume to skip embedding methods that have
already been computed and whose result rows are present in the output CSVs.

Usage
-----
python scripts/run_hard_negative_network.py \
    --case-name   QW1_modular_strong \
    --network-id  QW1_modular_strong_rep00 \
    --config      configs/hard_negatives_cases.json \
    --output-dir  outputs/hard_negatives/results/QW1_modular_strong_rep00 \
    --methods     quvine_fused-walk,quvine_ctqw,...,baseline_filter \
    --n-nodes     300 \
    --seed        1000 \
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
# Network generators
# ---------------------------------------------------------------------------

def _lcc(G: nx.Graph) -> nx.Graph:
    """Extract largest connected component, integer-relabelled."""
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    return nx.convert_node_labels_to_integers(G)


def _build_network(case_cfg: dict, n_nodes: int, seed: int):
    """Generate a network from a case config dict.

    Returns
    -------
    G : nx.Graph
        Largest-connected-component, integer-labelled.
    metadata : dict
        Flat dict of parameters suitable for network_metadata arg.
    """
    from quvine.data.random_graphs import (
        generate_modular_network,
        generate_barabasi_albert,
        generate_watts_strogatz,
        generate_erdos_renyi,
        generate_core_periphery,
        generate_powerlaw_cluster,
        generate_random_geometric,
    )

    generator = case_cfg["generator"]
    params = case_cfg.get("params", {})

    metadata = {
        "type": case_cfg["network_type"],
        "negative_strategy": case_cfg["negative_strategy"],
        "expected_winner": case_cfg["expected_winner"],
        "seed": seed,
    }

    # ── Synthetic generators ────────────────────────────────────────────────

    if generator == "modular":
        num_communities = params.get("num_communities", 5)
        nodes_per_community = n_nodes // num_communities
        G, _ = generate_modular_network(
            num_communities=num_communities,
            nodes_per_community=nodes_per_community,
            p_intra=params["p_intra"],
            p_inter=params["p_inter"],
            seed=seed,
        )
        metadata.update({
            "num_communities": num_communities,
            "p_intra": params["p_intra"],
            "p_inter": params["p_inter"],
        })

    elif generator == "barabasi_albert":
        m = params.get("m", 2)
        G = generate_barabasi_albert(n=n_nodes, m=m, seed=seed)
        metadata["m"] = m

    elif generator == "watts_strogatz":
        k = params.get("k", 6)
        p = params["p_rewire"]
        G = generate_watts_strogatz(n=n_nodes, k=k, p=p, seed=seed)
        metadata.update({"k": k, "p_rewire": p})

    elif generator == "erdos_renyi":
        p = params["p"]
        G = generate_erdos_renyi(n=n_nodes, p=p, seed=seed)
        metadata["p"] = p

    elif generator == "core_periphery":
        frac = params.get("n_core_fraction", 0.125)
        n_core = int(n_nodes * frac)
        n_periphery = n_nodes - n_core
        G, _, _ = generate_core_periphery(
            n_core=n_core,
            n_periphery=n_periphery,
            p_core=params["p_core"],
            p_core_periphery=params.get("p_core_periphery", 0.05),
            seed=seed,
        )
        metadata.update({
            "n_core": n_core,
            "n_periphery": n_periphery,
            "p_core": params["p_core"],
        })

    elif generator == "powerlaw_cluster":
        # Holme-Kim powerlaw cluster graph (scale-free + local triangles)
        m = params.get("m", 3)
        p_tri = params.get("p_triangle", 0.5)
        G = generate_powerlaw_cluster(n=n_nodes, m=m, p=p_tri, seed=seed)
        metadata.update({"m": m, "p_triangle": p_tri})

    elif generator == "random_geometric":
        radius = params.get("radius", 0.14)
        G = generate_random_geometric(n=n_nodes, radius=radius, seed=seed)
        metadata["radius"] = radius

    elif generator == "stochastic_block_model":
        # Uses networkx directly — equal-size blocks
        import numpy as np
        rng = np.random.default_rng(seed)
        n_blocks = params.get("n_blocks", 5)
        p_in = params.get("p_in", 0.25)
        p_out = params.get("p_out", 0.01)
        block_size = n_nodes // n_blocks
        sizes = [block_size] * n_blocks
        sizes[-1] += n_nodes - sum(sizes)  # absorb remainder into last block
        p_matrix = [
            [p_in if i == j else p_out for j in range(n_blocks)]
            for i in range(n_blocks)
        ]
        G = nx.stochastic_block_model(sizes, p_matrix, seed=int(rng.integers(1 << 31)))
        # Remove self-loops if any
        G.remove_edges_from(nx.selfloop_edges(G))
        metadata.update({"n_blocks": n_blocks, "p_in": p_in, "p_out": p_out})

    # ── Real-world networks ──────────────────────────────────────────────────

    elif generator == "real_network":
        name = params.get("name", "karate_club")
        G = _load_real_network(name)
        metadata["real_network_name"] = name
        # Real networks ignore n_nodes — report actual size
        print(f"  Loaded real network '{name}': {G.number_of_nodes()} nodes, "
              f"{G.number_of_edges()} edges")

    else:
        raise ValueError(f"Unknown generator: {generator!r}")

    return _lcc(G), metadata


_REAL_NETWORK_LOADERS = {
    "karate_club":    lambda: nx.karate_club_graph(),
    "les_miserables": lambda: nx.les_miserables_graph(),
    "florentine":     lambda: nx.florentine_families_graph(),
    "davis":          lambda: nx.davis_southern_women_graph(),
    "polbooks":       lambda: _polbooks_graph(),
}


def _polbooks_graph() -> nx.Graph:
    """Political books network surrogate (3 communities, 105 nodes).

    The actual polbooks dataset is not bundled in networkx; we use a modular
    graph with the same block structure (3 political factions, ~35 nodes each)
    as a surrogate for testing community-aware embedding methods.
    """
    from quvine.data.random_graphs import generate_modular_network
    G, _ = generate_modular_network(
        num_communities=3, nodes_per_community=35,
        p_intra=0.40, p_inter=0.01, seed=42,
    )
    return G


def _load_real_network(name: str) -> nx.Graph:
    """Load a named real network, converting to simple undirected integer-labelled graph."""
    if name not in _REAL_NETWORK_LOADERS:
        raise ValueError(
            f"Unknown real network '{name}'. Available: {list(_REAL_NETWORK_LOADERS)}"
        )
    G_raw = _REAL_NETWORK_LOADERS[name]()
    # Convert to simple undirected graph with integer node labels
    G = nx.Graph(G_raw)
    G.remove_edges_from(nx.selfloop_edges(G))
    return nx.convert_node_labels_to_integers(G)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run a single hard-negatives network analysis job"
    )
    parser.add_argument("--case-name",   required=True,
                        help="Case name matching a key in the config JSON (e.g. QW1_modular_strong)")
    parser.add_argument("--network-id",  required=True,
                        help="Unique network ID (e.g. QW1_modular_strong_rep00)")
    parser.add_argument("--config",      required=True,
                        help="Path to hard_negatives_cases.json")
    parser.add_argument("--output-dir",  required=True,
                        help="Output directory for this network's results")
    parser.add_argument("--methods",     required=True,
                        help="Comma-separated embedding methods to run")
    parser.add_argument("--n-nodes",     type=int, default=300,
                        help="Target number of nodes (before LCC extraction)")
    parser.add_argument("--seed",        type=int, required=True,
                        help="Random seed for graph generation")
    parser.add_argument("--hparam-file", default=None,
                        help="Path to best_hyperparams.json; when provided, uses "
                             "per-network-type tuned hyperparameters instead of defaults")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip methods that are already present in output CSVs")
    parser.add_argument("--verbose",     action="store_true", default=True,
                        help="Verbose logging")
    args = parser.parse_args()

    # --- Load config --------------------------------------------------------
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path) as fh:
        all_cases = json.load(fh)
    if args.case_name not in all_cases:
        print(f"ERROR: case '{args.case_name}' not in config. "
              f"Available: {list(all_cases.keys())}", file=sys.stderr)
        sys.exit(1)

    case_cfg = all_cases[args.case_name]
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Load per-network-type hyperparameters --------------------------------
    # Maps each experiment case to its key in best_hyperparams.json best_params
    _CASE_TO_NETWORK_TYPE = {
        "QW1_modular_strong":       "modular_strong",
        "QW2_modular_medium":       "modular_medium",
        "QW3_scale_free_2hop":      "scale_free",
        "QW4_scale_free_comm":      "scale_free",
        "QW5_core_periphery":       "core_periphery",
        "QW6_ws_low_p":             "watts_strogatz_low_p",
        "QW7_modular_many_comm":    "modular_many_communities",
        "QW8_sbm_assortative":      "stochastic_block_model",
        "QW9_powerlaw_cluster":     "powerlaw_cluster",
        "NC1_erdos_renyi_rand":     "erdos_renyi",
        "NC2_erdos_renyi_2hop":     "erdos_renyi",
        "NC3_ws_high_p":            "watts_strogatz_high_p",
        "NC4_random_geometric":     "random_geometric",
        "RN1_karate_club":          "real_karate",
        "RN2_les_miserables":       "real_lesmis",
        "RN3_polbooks":             "real_polbooks",
    }

    method_hyperparams = None
    if args.hparam_file:
        hp_path = Path(args.hparam_file)
        if not hp_path.exists():
            print(f"WARNING: hparam-file not found: {hp_path} — using defaults",
                  file=sys.stderr)
        else:
            with open(hp_path) as fh:
                all_hparams = json.load(fh)
            network_type_key = _CASE_TO_NETWORK_TYPE.get(args.case_name)
            if network_type_key and network_type_key in all_hparams.get("best_params", {}):
                method_hyperparams = all_hparams["best_params"][network_type_key]
                print(f"Loaded hyperparameters for network type '{network_type_key}' "
                      f"from {hp_path}")
            else:
                print(f"WARNING: no hyperparameters found for case '{args.case_name}' "
                      f"(key '{network_type_key}') in {hp_path} — using defaults",
                      file=sys.stderr)

    # --- Load or generate graph ---------------------------------------------
    graphml_path = output_path / f"{args.network_id}.graphml"

    if graphml_path.exists():
        print(f"Loading existing graph: {graphml_path}")
        G = nx.read_graphml(str(graphml_path))
        G = nx.convert_node_labels_to_integers(G)
        # Reconstruct minimal metadata (full params not needed post-generation)
        metadata = {
            "type":              case_cfg["network_type"],
            "case":              args.case_name,
            "negative_strategy": case_cfg["negative_strategy"],
            "expected_winner":   case_cfg["expected_winner"],
            "seed":              args.seed,
            "network_id":        args.network_id,
        }
        print(f"Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    else:
        print(f"Generating network for {args.network_id} (seed={args.seed}) ...")
        G, metadata = _build_network(case_cfg, args.n_nodes, args.seed)
        metadata["case"] = args.case_name
        metadata["network_id"] = args.network_id
        print(f"Generated: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"Case: {args.case_name}  "
          f"neg={case_cfg['negative_strategy']}  "
          f"expected={case_cfg['expected_winner']}")

    # --- Run analysis -------------------------------------------------------
    from quvine.comprehensive_embedding_analysis import run_single_network_analysis

    run_single_network_analysis(
        G=G,
        network_id=args.network_id,
        network_metadata=metadata,
        output_dir=args.output_dir,
        embedding_methods=args.methods.split(","),
        resume=args.resume,
        verbose=args.verbose,
        method_hyperparams=method_hyperparams,
    )
    print(f"Analysis complete for {args.network_id}")


if __name__ == "__main__":
    main()
