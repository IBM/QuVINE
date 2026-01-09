import networkx as nx
import numpy as np 
import random 
from quvine.utils.utilities import get_stats

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

def sparsify_with_connectivity(cfg, graph, target_avg_degree=20, seed=42):
    random.seed(seed)

    # Start with a spanning tree (guarantees connectivity)
    T = nx.minimum_spanning_tree(graph)

    n = graph.number_of_nodes()
    target_edges = int(target_avg_degree * n / 2)

    remaining_edges = list(set(graph.edges()) - set(T.edges()))
    needed = target_edges - T.number_of_edges()

    if needed > 0 and remaining_edges:
        T.add_edges_from(
            random.sample(remaining_edges, min(needed, len(remaining_edges)))
        )

    lcc_nodes = max(nx.connected_components(T), key=len)
    sparse_graph = T.subgraph(lcc_nodes).copy()
    
    if cfg.verbose: 
        sparse_graph_stats = get_stats(sparse_graph)
        print("-"*10)
        print("After Sparsification")
        print(sparse_graph_stats)
        print("-"*10)
        
    return sparse_graph