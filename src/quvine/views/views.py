import networkx as nx
import random 

def unconstrained_view(
                    G, 
                    root,
                    num_views, 
                    radius, 
                    max_nodes, 
                    rng
                    ):
    views = [] 
    #unconstrained view
    ego_nodes = list(nx.single_source_shortest_path_length(G, 
                                                        root, 
                                                        cutoff=radius).keys())
    for _ in range(num_views):
        #random subsample
        k = min(len(ego_nodes), max_nodes)
        view = set(rng.sample(ego_nodes, k))
        views.append(view)
        
    return views 

def constrained_view(G, 
                    root, 
                    num_views, 
                    max_nodes,
                    max_edges,
                    max_degree,
                    rng
                    ): 
    neighbors = list(G.neighbors(root))
    views = [] 
    for _ in range(num_views):
        view_nodes = {root}
        
        #randomize expansion order 
        frontier = neighbors[:]
        rng.shuffle(frontier)
        
        while frontier: 
            candidate_node = frontier.pop()
            if candidate_node in view_nodes:
                continue 
            else:
                new_nodes = view_nodes | {candidate_node}
                induced = G.subgraph(new_nodes)
                
                if induced.number_of_edges() > max_edges:
                    continue
                if max(dict(induced.degree()).values()) > max_degree:
                    continue 
                
                #accept 
                view_nodes.add(candidate_node)
                
                #add new neighbors to frontier
                for neighbor in G.neighbors(candidate_node):
                    if neighbor not in view_nodes:
                        frontier.append(neighbor)
                        
                if induced.number_of_edges() >= max_edges or len(view_nodes) >= max_nodes: 
                    break
                
        views.append(view_nodes)
    return views 

