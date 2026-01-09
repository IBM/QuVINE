import numpy as np  
from quvine.views.views import constrained_view, unconstrained_view

class ViewBuilder:
    def __init__(self, cfg, iteration_seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(iteration_seed)
        
    def build(self, G, root): 
    
        if not self.cfg.views.constrained: 
            views = unconstrained_view(G=G, 
                                    root=root,
                                    num_views=self.cfg.views.num_views, 
                                    radius=self.cfg.views.radius, 
                                    max_nodes=self.cfg.views.max_nodes,
                                    rng=self.rng)
        else: 
            views = constrained_view(G=G, 
                                    root=root, 
                                    num_views=self.cfg.views.num_views, 
                                    max_nodes=self.cfg.views.max_nodes, 
                                    max_edges=self.cfg.views.max_edges,
                                    max_degree=self.cfg.views.max_degree,
                                    rng=self.rng
                                    )    
        
        return views
