"""
Random Graph Generator for QuVINE

This module provides functions to generate various types of random graphs with known
structures or specific key elements that are suitable for testing embedding algorithms.
Includes both classical graph models and biologically-inspired structures.
"""

import networkx as nx
import numpy as np
from typing import Optional, Dict, List, Tuple, Set
import warnings


def generate_erdos_renyi(
    n: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    seed: Optional[int] = None,
    directed: bool = False
) -> nx.Graph:
    """
    Generate an Erdős-Rényi random graph.
    
    Parameters
    ----------
    n : int
        Number of nodes
    p : float, optional
        Probability of edge creation (G(n,p) model)
    m : int, optional
        Number of edges (G(n,m) model)
    seed : int, optional
        Random seed for reproducibility
    directed : bool, default=False
        If True, generate a directed graph
        
    Returns
    -------
    nx.Graph or nx.DiGraph
        Random graph
    """
    if p is not None and m is not None:
        raise ValueError("Specify either p or m, not both")
    
    if p is not None:
        if directed:
            return nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
        return nx.erdos_renyi_graph(n, p, seed=seed)
    elif m is not None:
        if directed:
            return nx.gnm_random_graph(n, m, seed=seed, directed=True)
        return nx.gnm_random_graph(n, m, seed=seed)
    else:
        raise ValueError("Must specify either p or m")


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: Optional[int] = None,
    initial_graph: Optional[nx.Graph] = None
) -> nx.Graph:
    """
    Generate a Barabási-Albert scale-free network using preferential attachment.
    
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    seed : int, optional
        Random seed for reproducibility
    initial_graph : nx.Graph, optional
        Initial connected graph with at least m nodes
        
    Returns
    -------
    nx.Graph
        Scale-free network
    """
    return nx.barabasi_albert_graph(n, m, seed=seed, initial_graph=initial_graph)


def generate_watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a Watts-Strogatz small-world network.
    
    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors in ring topology
    p : float
        Probability of rewiring each edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    nx.Graph
        Small-world network
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def generate_powerlaw_cluster(
    n: int,
    m: int,
    p: float,
    seed: Optional[int] = None
) -> nx.Graph:
    """
    Generate a random graph with powerlaw degree distribution and clustering.
    
    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of random edges to add for each new node
    p : float
        Probability of adding a triangle after adding a random edge
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    nx.Graph
        Powerlaw cluster graph
    """
    return nx.powerlaw_cluster_graph(n, m, p, seed=seed)


def generate_stochastic_block_model(
    sizes: List[int],
    p_matrix: List[List[float]],
    seed: Optional[int] = None,
    directed: bool = False,
    selfloops: bool = False
) -> nx.Graph:
    """
    Generate a stochastic block model graph with community structure.
    
    Parameters
    ----------
    sizes : list of int
        Sizes of blocks (communities)
    p_matrix : list of list of float
        Matrix of edge probabilities between and within blocks
    seed : int, optional
        Random seed for reproducibility
    directed : bool, default=False
        If True, generate a directed graph
    selfloops : bool, default=False
        If True, allow self-loops
        
    Returns
    -------
    nx.Graph or nx.DiGraph
        Stochastic block model graph with community structure
    """
    return nx.stochastic_block_model(
        sizes, p_matrix, seed=seed, directed=directed, selfloops=selfloops
    )


def generate_random_geometric(
    n: int,
    radius: float,
    dim: int = 2,
    seed: Optional[int] = None,
    pos: Optional[Dict] = None
) -> nx.Graph:
    """
    Generate a random geometric graph in the unit cube.
    
    Parameters
    ----------
    n : int
        Number of nodes
    radius : float
        Distance threshold for edge creation
    dim : int, default=2
        Dimension of the space
    seed : int, optional
        Random seed for reproducibility
    pos : dict, optional
        Dictionary of node positions
        
    Returns
    -------
    nx.Graph
        Random geometric graph with 'pos' node attribute
    """
    return nx.random_geometric_graph(n, radius, dim=dim, seed=seed, pos=pos)


def generate_modular_network(
    num_communities: int,
    nodes_per_community: int,
    p_intra: float,
    p_inter: float,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a modular network with clear community structure.
    
    Parameters
    ----------
    num_communities : int
        Number of communities
    nodes_per_community : int
        Number of nodes in each community
    p_intra : float
        Probability of edges within communities
    p_inter : float
        Probability of edges between communities
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Modular network
    communities : dict
        Mapping from node to community ID
    """
    sizes = [nodes_per_community] * num_communities
    p_matrix = [[p_intra if i == j else p_inter 
                 for j in range(num_communities)] 
                for i in range(num_communities)]
    
    G = generate_stochastic_block_model(sizes, p_matrix, seed=seed)
    
    # Create community mapping
    communities = {}
    node_idx = 0
    for comm_id, size in enumerate(sizes):
        for _ in range(size):
            communities[node_idx] = comm_id
            node_idx += 1
    
    # Add community as node attribute
    nx.set_node_attributes(G, communities, 'community')
    
    return G, communities


def generate_hierarchical_network(
    levels: int,
    branching_factor: int,
    p_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Dict[int, int]]:
    """
    Generate a hierarchical network with tree-like structure plus random edges.
    
    Parameters
    ----------
    levels : int
        Number of hierarchy levels
    branching_factor : int
        Number of children per parent node
    p_level : float, default=0.1
        Probability of random edges within same level
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Hierarchical network
    node_levels : dict
        Mapping from node to hierarchy level
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Create tree structure
    G = nx.balanced_tree(branching_factor, levels - 1)
    
    # Track node levels
    node_levels = {}
    for node in G.nodes():
        # Calculate level based on tree structure
        level = 0
        temp_node = node
        while temp_node > 0:
            temp_node = (temp_node - 1) // branching_factor
            level += 1
        node_levels[node] = level
    
    # Add random edges within levels
    nodes_by_level = {}
    for node, level in node_levels.items():
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    for level, nodes in nodes_by_level.items():
        if len(nodes) > 1:
            for i, u in enumerate(nodes):
                for v in nodes[i+1:]:
                    if rng.random() < p_level:
                        G.add_edge(u, v)
    
    nx.set_node_attributes(G, node_levels, 'level')
    
    return G, node_levels


def generate_core_periphery(
    n_core: int,
    n_periphery: int,
    p_core: float,
    p_core_periphery: float,
    p_periphery: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Set[int], Set[int]]:
    """
    Generate a core-periphery network structure.
    
    Parameters
    ----------
    n_core : int
        Number of core nodes
    n_periphery : int
        Number of periphery nodes
    p_core : float
        Edge probability within core
    p_core_periphery : float
        Edge probability between core and periphery
    p_periphery : float, default=0.01
        Edge probability within periphery
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Core-periphery network
    core_nodes : set
        Set of core node IDs
    periphery_nodes : set
        Set of periphery node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    n_total = n_core + n_periphery
    G = nx.Graph()
    G.add_nodes_from(range(n_total))
    
    core_nodes = set(range(n_core))
    periphery_nodes = set(range(n_core, n_total))
    
    # Core edges
    for i in core_nodes:
        for j in core_nodes:
            if i < j and rng.random() < p_core:
                G.add_edge(i, j)
    
    # Core-periphery edges
    for i in core_nodes:
        for j in periphery_nodes:
            if rng.random() < p_core_periphery:
                G.add_edge(i, j)
    
    # Periphery edges
    for i in periphery_nodes:
        for j in periphery_nodes:
            if i < j and rng.random() < p_periphery:
                G.add_edge(i, j)
    
    # Add node attributes
    node_types = {node: 'core' if node in core_nodes else 'periphery' 
                  for node in G.nodes()}
    nx.set_node_attributes(G, node_types, 'type')
    
    return G, core_nodes, periphery_nodes


def generate_bipartite_random(
    n1: int,
    n2: int,
    p: Optional[float] = None,
    m: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, Set[int], Set[int]]:
    """
    Generate a random bipartite graph.
    
    Parameters
    ----------
    n1 : int
        Number of nodes in first partition
    n2 : int
        Number of nodes in second partition
    p : float, optional
        Probability of edge creation
    m : int, optional
        Number of edges
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Bipartite graph
    set1 : set
        First partition node IDs
    set2 : set
        Second partition node IDs
    """
    if p is not None and m is not None:
        raise ValueError("Specify either p or m, not both")
    
    if p is not None:
        G = nx.bipartite.random_graph(n1, n2, p, seed=seed)
    elif m is not None:
        G = nx.bipartite.gnmk_random_graph(n1, n2, m, seed=seed)
    else:
        raise ValueError("Must specify either p or m")
    
    set1 = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    set2 = set(G.nodes()) - set1
    
    return G, set1, set2


def add_hub_nodes(
    G: nx.Graph,
    num_hubs: int,
    hub_degree: int,
    seed: Optional[int] = None
) -> Tuple[nx.Graph, List[int]]:
    """
    Add hub nodes to an existing graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
    num_hubs : int
        Number of hub nodes to add
    hub_degree : int
        Degree of each hub node
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    G : nx.Graph
        Graph with added hubs
    hub_nodes : list
        List of hub node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    G = G.copy()
    existing_nodes = list(G.nodes())
    n_existing = len(existing_nodes)
    
    hub_nodes = []
    for i in range(num_hubs):
        hub_id = n_existing + i
        G.add_node(hub_id)
        hub_nodes.append(hub_id)
        
        # Connect to random existing nodes
        targets = rng.choice(existing_nodes, size=min(hub_degree, n_existing), replace=False)
        for target in targets:
            G.add_edge(hub_id, target)
    
    return G, hub_nodes


def generate_graph_with_seeds_and_targets(
    n: int,
    num_seeds: int,
    num_targets: int,
    graph_type: str = 'barabasi_albert',
    seed: Optional[int] = None,
    **kwargs
) -> Tuple[nx.Graph, List[int], List[int]]:
    """
    Generate a random graph with designated seed and target nodes for embedding evaluation.
    
    Parameters
    ----------
    n : int
        Total number of nodes
    num_seeds : int
        Number of seed nodes
    num_targets : int
        Number of target nodes
    graph_type : str, default='barabasi_albert'
        Type of graph: 'erdos_renyi', 'barabasi_albert', 'watts_strogatz', 
        'powerlaw_cluster', 'modular'
    seed : int, optional
        Random seed for reproducibility
    **kwargs
        Additional parameters for specific graph types
        
    Returns
    -------
    G : nx.Graph
        Generated graph
    seeds : list
        List of seed node IDs
    targets : list
        List of target node IDs
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate base graph
    if graph_type == 'erdos_renyi':
        p = kwargs.get('p', 0.1)
        G = generate_erdos_renyi(n, p=p, seed=seed)
    elif graph_type == 'barabasi_albert':
        m = kwargs.get('m', 3)
        G = generate_barabasi_albert(n, m, seed=seed)
    elif graph_type == 'watts_strogatz':
        k = kwargs.get('k', 4)
        p = kwargs.get('p', 0.3)
        G = generate_watts_strogatz(n, k, p, seed=seed)
    elif graph_type == 'powerlaw_cluster':
        m = kwargs.get('m', 3)
        p = kwargs.get('p', 0.3)
        G = generate_powerlaw_cluster(n, m, p, seed=seed)
    elif graph_type == 'modular':
        num_communities = kwargs.get('num_communities', 5)
        nodes_per_community = n // num_communities
        p_intra = kwargs.get('p_intra', 0.3)
        p_inter = kwargs.get('p_inter', 0.01)
        G, _ = generate_modular_network(num_communities, nodes_per_community, 
                                       p_intra, p_inter, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Select seeds and targets
    nodes = list(G.nodes())
    selected = rng.choice(nodes, size=num_seeds + num_targets, replace=False)
    seeds = selected[:num_seeds].tolist()
    targets = selected[num_seeds:].tolist()
    
    # Add node attributes
    node_roles = {}
    for node in G.nodes():
        if node in seeds:
            node_roles[node] = 'seed'
        elif node in targets:
            node_roles[node] = 'target'
        else:
            node_roles[node] = 'regular'
    
    nx.set_node_attributes(G, node_roles, 'role')
    
    return G, seeds, targets


def get_graph_statistics(G: nx.Graph) -> Dict:
    """
    Compute comprehensive statistics for a graph.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph
        
    Returns
    -------
    dict
        Dictionary of graph statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G),
        'is_connected': nx.is_connected(G),
    }
    
    if G.number_of_nodes() > 0:
        degrees = [d for n, d in G.degree()]
        stats['avg_degree'] = float(np.mean(degrees))
        stats['max_degree'] = np.max(degrees)
        stats['min_degree'] = np.min(degrees)
        
        if nx.is_connected(G):
            stats['diameter'] = nx.diameter(G)
            stats['avg_shortest_path'] = nx.average_shortest_path_length(G)
        else:
            stats['num_components'] = nx.number_connected_components(G)
            largest_cc = max(nx.connected_components(G), key=len)
            stats['largest_cc_size'] = len(largest_cc)
        
        stats['avg_clustering'] = nx.average_clustering(G)
        stats['transitivity'] = nx.transitivity(G)
    
    return stats
