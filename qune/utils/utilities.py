import pandas as pd 
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt 
import random

def get_stats(graph): 
    
    stats = {}
    stats['num_nodes'] = graph.number_of_nodes() 
    stats['num_edges'] = graph.number_of_edges()
    stats['density'] = nx.density(graph)
    degrees = [d for n,d in graph.degree()]
    stats['average_degree'] = np.round(np.mean(degrees), decimals=3)
    stats['average_clustering_coefficient'] = np.round(nx.average_clustering(graph), decimals=3)
    betweenness = list(nx.betweenness_centrality(graph).values())
    stats['average_betweenness'] = np.round(np.mean(betweenness), decimals=3)
    stats['average_assortavity_coefficient'] = np.round(nx.degree_assortativity_coefficient(graph),
                                                        decimals=3)
    stats['num_connected_components'] = nx.number_connected_components(graph)
    largest_cc = max(nx.connected_components(graph), key=len)
    stats['largest_cc_size'] = len(largest_cc)
    
    return stats 
    
def draw_graph(cfg, G, source=None, target=None, title="Graph", file_path=None):
    fig = plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    default_color = "#185e8d"
    d = dict(G.degree())
    node_colors = [default_color] * len(G.nodes)
    node_to_index = {node:i for i, node in enumerate(G.nodes)}
    if source is None and target is None:
        nx.draw_networkx(G, 
                pos = pos, 
                with_labels=True,
                nodelist=list(d.keys()), 
                node_color=node_colors,
                node_size=[v*6 for v in d.values()],
                width=2)
    else: 
        for s,t in zip(source, target):
            for i, node in enumerate([s,t]): 
                if node in node_to_index:
                    if i == 0:
                        node_colors[node_to_index[node]] = 'red'
                    elif i == 1: 
                        node_colors[node_to_index[node]] = 'green'
                else:
                    if cfg.verbose:
                        print(f"Node {node} not found in graph")
        nx.draw_networkx(G, 
                pos = pos, 
                with_labels=False,
                nodelist=list(d.keys()), 
                node_color=node_colors,
                node_size=[v*6 for v in d.values()],
                width=2)
    plt.title(title)
    if file_path is not None: 
        fig.savefig(file_path, dpi=300, bbox_inches='tight')
        
def sample_rwr_walks_from_scores(
    rwr_scores,
    num_walks=10,
    walk_length=40,
    seed=None
):
    """
    Generate pseudo-walks by sampling from RWR stationary distribution.

    Parameters
    ----------
    rwr_scores : dict
        Node -> probability
    num_walks : int
    walk_length : int
    seed : int or None

    Returns
    -------
    List[List[node]]
    """
    rng = random.Random(seed)

    nodes = list(rwr_scores.keys())
    probs = [rwr_scores[n] for n in nodes]

    walks = []
    for _ in range(num_walks):
        walk = rng.choices(nodes, weights=probs, k=walk_length)
        walks.append(walk)
    
    return walks
