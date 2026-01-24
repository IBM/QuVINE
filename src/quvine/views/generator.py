# This code is part of a Qiskit project.
#
# (C) Copyright IBM 2021, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
                                rng=self.rng, 
                                degree_norm=self.cfg.views.degree_norm,
                                degree_alpha=self.cfg.views.degree_alpha
                                )    
    
        return views
