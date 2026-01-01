from qune.walks.rwr import generate_RWR_pagerank_walks
from qune.walks.ctqw import generate_CTQW_walks
from qune.walks.dtqw import generate_DTQW_walks

WALKS={
    "rwr": generate_RWR_pagerank_walks,
    "ctqw": generate_CTQW_walks,
    "dtqw": generate_DTQW_walks,
}

class BaseWalker:
    """
    Base class for all walk types.

    Subclasses must implement `_generate_walks`.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.kinds = list(cfg.kinds)

        unknown = set(self.kinds) - WALKS.keys() 
        if unknown: 
            raise ValueError(f"Unknown walk kinds: {unknown}")

        
    def run(self, graph, root, view_nodes):
        """Run the walk 

        Args:
            graph (networkx graph): subgraph of the views
            root (string): Root node tied to the views
            view_nodes (list of list): list of nodes in the views
        -------
        Returns: 
            Dict[str, List[walks]]
        """
        
        out = {} 
        for kind in self.kinds: 
            out[kind] = self._run_walk(kind, graph, root, view_nodes)
        
        return out 
    
    def _run_walk(self, kind, graph, root, view_nodes): 
        
        if kind == "rwr": 
            return generate_RWR_pagerank_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            restart_prob=self.cfg.walks.restart_prob
            )
        elif kind == "ctqw": 
            return generate_CTQW_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            steps=self.cfg.walks.steps,
            seed=self.cfg.seed
            )
        elif kind == "dtqw": 
            return generate_DTQW_walks(
            G=graph, 
            root=root, 
            view_nodes=view_nodes,
            num_walks=self.cfg.walks.num_walks,
            walk_length=self.cfg.walks.walk_length, 
            steps=self.cfg.walks.steps,
            coin=self.cfg.walks.coin, 
            seed=self.cfg.seed
            )