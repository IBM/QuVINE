import networkx as nx
import numpy as np 
import random 
from quvine.utils.utilities import get_stats

def confidence_to_distance(w, eps=1e-6):
    return -np.log(w + eps)


def build_seed_neighborhood_subgraph(
    graph,
    seed_nodes,
    max_hops=4,
    max_nodes=500,
    random_state=None
):
    """
    Build ss_nodes = seeds ∪ neighbors up to max_hops,
    stopping once max_nodes is reached.
    
    Parameters
    ----------
    graph : nx.Graph
        Full PPI graph
    seed_nodes : iterable
        Seed genes (must be in graph)
    max_hops : int
        Maximum hop expansion (default=4)
    max_nodes : int
        Cap on total nodes (default=500)
    random_state : int or None
        For reproducibility
    
    Returns
    -------
    ss_nodes : set
        Selected node set
    """

    if random_state is not None:
        random.seed(random_state)

    # Keep only seeds present in the graph
    seeds = set(seed_nodes).intersection(graph.nodes)
    ss_nodes = set(seeds)

    # Frontier for BFS-style expansion
    frontier = set(seeds)

    for hop in range(1, max_hops + 1):
        if len(ss_nodes) >= max_nodes or not frontier:
            break

        next_frontier = set()

        for u in frontier:
            next_frontier.update(graph.neighbors(u))

        # Remove already-seen nodes
        next_frontier -= ss_nodes

        if not next_frontier:
            break

        # If adding all neighbors exceeds cap, sample
        remaining_slots = max_nodes - len(ss_nodes)
        if len(next_frontier) > remaining_slots:
            next_frontier = set(random.sample(list(next_frontier), remaining_slots))

        ss_nodes.update(next_frontier)
        frontier = next_frontier

    return ss_nodes


def build_subgraph(cfg, 
        graph, 
        seeds, 
        targets, 
        num_nodes=25, 
        max_hops=4, 
        max_nodes=500, 
        random_state=None
        ):
    
    subgraph_seeds = random.sample(seeds, min(len(seeds), num_nodes))
    subgraph_nodes = build_seed_neighborhood_subgraph(graph=graph, 
                                                    seed_nodes=subgraph_seeds, 
                                                    max_hops=max_hops, 
                                                    max_nodes=max_nodes,
                                                    random_state=random_state)
    subgraph = graph.subgraph(subgraph_nodes)
    lcc_nodes = max(nx.connected_components(subgraph), key=len)
    subgraph = subgraph.subgraph(lcc_nodes).copy()
    subgraph_source = set(subgraph_nodes).intersection(seeds) 
    subgraph_target = set(subgraph.nodes).intersection(set(targets))
    
    if cfg.verbose: 
        print("-"*10)
        print("Subgraph")
        print(f"subgraph contains {len(subgraph_source)} seed nodes and {len(subgraph_target)} target nodes, with a total of {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
        subgraph_stats = get_stats(subgraph)
        print(subgraph_stats)
        print("-"*10)
        
    return subgraph, subgraph_source, subgraph_target

def sparsify_graph(
    graph,
    target_avg_degree=25,
    max_degree=32,
    seed=42,
    verbose=False,
):
    """
    Sparsify an unweighted PPI graph while preserving connectivity
    and controlling degree (DTQW-safe).

    Parameters
    ----------
    graph : nx.Graph
        Input unweighted graph (assumed undirected, connected or not).
    target_avg_degree : int
        Desired average degree after sparsification.
    max_degree : int
        Hard cap on node degree (important for DTQW).
    seed : int
        Random seed for reproducibility.
    verbose : bool
        Print stats after sparsification.

    Returns
    -------
    sparse_graph : nx.Graph
        Sparsified graph (largest connected component).
    """

    random.seed(seed)    

    n = graph.number_of_nodes()
    target_edges = int(target_avg_degree * n / 2)

    # ---- Step 1: Build BFS spanning tree (connectivity backbone) ----
    root = random.choice(list(graph.nodes()))
    T = nx.bfs_tree(graph, root).to_undirected()

    # ---- Step 2: Candidate edges (excluding backbone) ----
    backbone_edges = set(T.edges())
    remaining_edges = [
        (u, v) for u, v in graph.edges()
        if (u, v) not in backbone_edges and (v, u) not in backbone_edges
    ]

    # ---- Step 3: Degree-aware ordering (prefer low-degree nodes) ----
    remaining_edges.sort(
        key=lambda e: graph.degree(e[0]) + graph.degree(e[1])
    )

    # ---- Step 4: Add edges until target is reached (degree capped) ----
    for u, v in remaining_edges:
        if T.number_of_edges() >= target_edges:
            break

        if T.degree(u) < max_degree and T.degree(v) < max_degree:
            T.add_edge(u, v)

    # ---- Step 5: Final LCC (safety) ----
    lcc_nodes = max(nx.connected_components(T), key=len)
    sparse_graph = T.subgraph(lcc_nodes).copy()

    if verbose:
        avg_deg = 2 * sparse_graph.number_of_edges() / sparse_graph.number_of_nodes()
        max_deg = max(dict(sparse_graph.degree()).values())
        print("-" * 30)
        print("After sparsification")
        print(f"Nodes: {sparse_graph.number_of_nodes()}")
        print(f"Edges: {sparse_graph.number_of_edges()}")
        print(f"Avg degree: {avg_deg:.2f}")
        print(f"Max degree: {max_deg}")
        print("-" * 30)

    return sparse_graph