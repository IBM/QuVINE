import networkx as nx


# def unconstrained_view(
#                     G, 
#                     root,
#                     num_views, 
#                     radius, 
#                     max_nodes, 
#                     rng
#                     ):
#     views = [] 
#     #unconstrained view
#     ego_nodes = list(nx.single_source_shortest_path_length(G, 
#                                                         root, 
#                                                         cutoff=radius).keys())
#     for _ in range(num_views):
#         #random subsample
#         k = min(len(ego_nodes), max_nodes)
#         view = set(random.sample(ego_nodes, k))
#         view.add(root)
#         views.append(view)
        
#     return views 

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
        view_edges = 0 
        
        #randomize expansion order 
        frontier = list(neighbors)
        rng.shuffle(frontier)
                
        while frontier: 
            candidate_node = frontier.pop()
            
            if candidate_node in view_nodes:
                continue 
            
            #compute incremental edges only 
            new_edges = sum(1 for v in view_nodes if G.has_edge(candidate_node, v))
            
            if view_edges + new_edges > max_edges: 
                continue 
            
            #local degree check 
            if new_edges > max_degree: 
                continue 
            
            view_nodes.add(candidate_node) 
            view_edges += new_edges 
            
            for neighbor in G.neighbors(candidate_node): 
                if neighbor not in view_nodes: 
                    frontier.append(neighbor)
                
            #enforce sizes 
            if len(view_nodes) >= max_nodes or view_edges >= max_edges: 
                break 
            
        views.append(view_nodes)
        
    return views 

