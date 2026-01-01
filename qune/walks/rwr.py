import random
import networkx as nx 
from qune.utils.utilities import sample_rwr_walks_from_scores

def generate_RWR_walks(G, root, num_walks=10, walk_length=5, restart_prob=0.5, view_nodes=None): 
    walks=[]
    for _ in range(num_walks):
        walk = [root]
        current_node = root
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current_node))
            
            if view_nodes is not None: 
                neighbors = [n for n in neighbors if n in view_nodes]
            if len(neighbors) == 0:
                break
            if random.random() < restart_prob: 
                walk.append(root)
            else:
                walk.append(random.choice(neighbors))
            next_node = random.choice(neighbors)
            
        walks.append(walk)
    return walks

def get_RWR_pagerank_scores(
    G,
    root,
    restart_prob=0.15,
    view_nodes=None,
    weight=None,
    tol=1e-6,
    max_iter=100
):
    """
    Random Walk with Restart using PageRank.

    Parameters
    ----------
    G : networkx.Graph
    root : node
        Restart node
    restart_prob : float
        Alpha (restart probability)
    view_nodes : set or None
        Optional constraint: restrict graph to these nodes
    weight : str or None
        Edge weight attribute
    tol : float
        Convergence tolerance
    max_iter : int
        Max iterations

    Returns
    -------
    dict
        Node -> RWR stationary probability
    """
    if view_nodes is not None:
        G = G.subgraph(view_nodes)

    if root not in G:
        raise ValueError("Root node not in graph or view")

    # Personalization vector (restart distribution)
    personalization = {n: 0.0 for n in G.nodes()}
    personalization[root] = 1.0

    pr = nx.pagerank(
        G,
        alpha=1 - restart_prob,
        personalization=personalization,
        weight=weight,
        tol=tol,
        max_iter=max_iter
    )

    return pr


def generate_RWR_pagerank_walks(G, root, view_nodes=None, num_walks=10, walk_length=6, restart_prob=0.5):
    rwr_scores = get_RWR_pagerank_scores(
        G,
        root,
        restart_prob=restart_prob,
        view_nodes=view_nodes
    )
    walks = sample_rwr_walks_from_scores(
        rwr_scores,
        num_walks=num_walks,
        walk_length=walk_length
    )
    
    return walks


    