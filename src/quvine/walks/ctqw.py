import networkx as nx 
import numpy as np
import hiperwalk as hpw
from quvine.utils.utilities import sample_rwr_walks_from_scores

def generate_ctqw_hiperwalk_scores(G, root, view_nodes=None, steps: int=20, time: float | None=None, gamma: float | None=None):
    """
    Return node probabilities from Hiperwalk continuous-time quantum walk (CTQW)
    Args:
        G (_type_): _description_
        root (_type_): _description_
        view_nodes (_type_, optional): _description_. Defaults to None.
        steps (int, optional): _description_. Defaults to 20.
        time (float | None, optional): _description_. Defaults to None.
        gamma (float | None, optional): _description_. Defaults to None.
    """
    if view_nodes is not None: 
        G = G.subgraph(view_nodes)
    if root not in G: 
        raise ValueError("Root node not in graph or view")
    
    nodes = list(G.nodes())
    node2i = {n:i for i,n in enumerate(nodes)}
    i2node = {i:n for n,i in node2i.items()}
    G_int = nx.relabel_nodes(G,node2i, copy=True)
    
    #build hiperwalk graph + ctqw 
    hg = hpw.Graph(G_int)
    qw = hpw.ContinuousTime(graph=hg) 
    
    if gamma is not None:
        qw.set_gamma(gamma)
    if time is not None:
        qw.set_time(time)
    
    root_i = node2i[root]
    state0 = qw.ket(root_i)
    
    #simulate to obtain final state at steps
    final_state = qw.simulate(range=(steps, steps+1), state=state0)
    
    #convert to node probability
    probs = qw.probability_distribution(final_state)
    probs = np.asarray(probs)[0]
    
    #map to original ids
    scores = {i2node[i]: float(probs[i]) for i in range(len(probs))}
    
    return scores


def generate_CTQW_walks(G, root, view_nodes=None, num_walks: int = 10, 
                                walk_length: int = 6, steps: int=20, time: float | None=None, gamma: float | None=None, seed: int | None=None):
    
    scores = generate_ctqw_hiperwalk_scores(G, root=root, view_nodes=view_nodes, 
                                            steps=steps, time=time, gamma=gamma)
    walks = sample_rwr_walks_from_scores(scores, num_walks=num_walks, walk_length=walk_length, seed=seed)
    return walks 