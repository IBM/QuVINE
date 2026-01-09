import networkx as nx 
import numpy as np
import hiperwalk as hpw
from quvine.utils.utilities import sample_rwr_walks_from_scores


def get_coined_hiperwalk_scores(G, root, view_nodes=None, steps: int=25, coin: str="grover"): 
    """
    Return node probabilities from Hiperwalk coined quantum walk (CQW)
    Args:
        G (_type_): _description_
        root (_type_): _description_
        view_nodes (_type_, optional): _description_. Defaults to None.
        steps (int, optional): _description_. Defaults to 25.
        coin (str, optional): _description_. Defaults to "grover".
    """ 
    
    if view_nodes is not None: 
        G = G.subgraph(view_nodes)
    if root not in G: 
        raise ValueError("Root node not in graph or view")
    
    nodes = list(G.nodes())
    node2i = {n:i for i,n in enumerate(nodes)}
    i2node = {i:n for n,i in node2i.items()}
    G_int = nx.relabel_nodes(G,node2i, copy=True)
    
    #build hiperwalk graph + dtqw 
    hg = hpw.Graph(G_int)
    qw = hpw.Coined(graph=hg, coin=coin) 
    
    root_i = node2i[root]
    neighbors = list(G_int.neighbors(root_i))
    
    if len(neighbors) == 0: 
        raise ValueError("Root node has no neighbors in the graph or view")
    
    state0 = qw.ket(root_i) 
    
    final_state = qw.simulate(range=(steps, steps+1), state=state0)
        
    #convert to node probability
    probs = qw.probability_distribution(final_state)
    probs = np.asarray(probs)[0]
    # 9) Map back to original node labels
    scores = {i2node[i]: float(probs[i]) for i in range(len(probs))}
    
    return scores

def generate_DTQW_walks(G, root, view_nodes=None, num_walks: int = 10, 
                                walk_length: int = 6, steps: int=25, coin: str="grover", seed: int | None=None):
    scores = get_coined_hiperwalk_scores(G, root=root, view_nodes=view_nodes, 
                                        steps=steps, coin=coin)
    walks = sample_rwr_walks_from_scores(scores, num_walks=num_walks, walk_length=walk_length, seed=seed)
    return walks