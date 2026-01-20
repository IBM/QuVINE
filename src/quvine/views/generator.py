import numpy as np  
import random
from quvine.views.views import constrained_view

class ViewBuilder:
    def __init__(self, cfg, rng):
        self.cfg = cfg
        self.rng = rng
        
    def build(self, G, root): 
    
        views = constrained_view(G=G, 
                                root=root, 
                                num_views=self.cfg.views.num_views, 
                                max_nodes=self.cfg.views.max_nodes, 
                                max_edges=self.cfg.views.max_edges,
                                max_degree=self.cfg.views.max_degree,
                                rng=self.rng
                                )    
    
        return views
