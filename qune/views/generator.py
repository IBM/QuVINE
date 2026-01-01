import numpy as np 
from qune.views.views import generate_views

class ViewBuilder:
    def __init__(self, cfg, iteration_seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(iteration_seed)
        
    def build(self, G, root): 
        return generate_views(G=G, 
                            root=root, 
                            num_views=self.cfg.views.num_views, 
                            constrained=self.cfg.views.constrained, 
                            max_nodes=self.cfg.views.max_nodes, 
                            max_edges=self.cfg.views.max_edges, 
                            max_degree=self.cfg.views.max_degree,
                            radius=self.cfg.views.radius, 
                            rng=self.rng
                            )
