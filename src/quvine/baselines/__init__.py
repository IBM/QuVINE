from quvine.baselines.node2vec import run_node2vec
from quvine.baselines.netmf import run_netmf

try:
    from quvine.baselines.gcn_mf import GCNMF, GCNLayer, QCaliberGCNMF, normalize_adjacency, train_gcn_mf
    __all__ = [
        "run_node2vec",
        "run_netmf",
        "GCNMF",
        "GCNLayer",
        "QCaliberGCNMF",
        "normalize_adjacency",
        "train_gcn_mf"
    ]
except ImportError:
    # PyTorch not available
    __all__ = [
        "run_node2vec",
        "run_netmf"
    ]
