__all__ = []

try:
    from quvine.baselines.node2vec import run_node2vec
    __all__.append("run_node2vec")
except ImportError:
    pass

try:
    from quvine.baselines.netmf import run_netmf
    __all__.append("run_netmf")
except ImportError:
    pass

try:
    from quvine.baselines.graphsage import run_graphsage
    __all__.append("run_graphsage")
except ImportError:
    pass

try:
    from quvine.baselines.gcn_mf import (
        GCNMF,
        GCNLayer,
        QuVINEGCNMF,
        normalize_adjacency,
        train_gcn_mf,
        precompute_quantum_diffusion,
        generate_baseline_gcnmf_embedding,
        generate_baseline_filter_embedding_wrapper,
    )
    __all__.extend([
        "GCNMF",
        "GCNLayer",
        "QuVINEGCNMF",
        "normalize_adjacency",
        "train_gcn_mf",
        "precompute_quantum_diffusion",
        "generate_baseline_gcnmf_embedding",
        "generate_baseline_filter_embedding_wrapper",
    ])
except ImportError:
    pass
