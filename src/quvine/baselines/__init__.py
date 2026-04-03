from quvine.baselines.node2vec import run_node2vec
from quvine.baselines.netmf import run_netmf

try:
    from quvine.baselines.gcn_mf import (
        GCNMF,
        GCNLayer,
        QuVINEGCNMF,
        normalize_adjacency,
        train_gcn_mf,
        precompute_quantum_diffusion,
        generate_baseline_gcnmf_embedding,
        generate_baseline_filter_embedding_wrapper
    )
    __all__ = [
        "run_node2vec",
        "run_netmf",
        "GCNMF",
        "GCNLayer",
        "QuVINEGCNMF",
        "normalize_adjacency",
        "train_gcn_mf",
        "precompute_quantum_diffusion",
        "generate_baseline_gcnmf_embedding",
        "generate_baseline_filter_embedding_wrapper"
    ]
except ImportError:
    # PyTorch not available
    __all__ = [
        "run_node2vec",
        "run_netmf"
    ]
