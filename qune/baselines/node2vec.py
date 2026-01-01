import numpy as np
from node2vec import Node2Vec


def run_node2vec(
    graph,
    nodes,
    dimensions=64,
    walk_length=10,
    num_walks=10,
    p=1.0,
    q=0.5,
    window=5,
    min_count=1,
    workers=8,
    seed=None,
):
    """
    Run Node2Vec and return embeddings aligned to `nodes`.

    Parameters
    ----------
    graph : networkx.Graph
    nodes : List[node]
        Canonical node ordering (must match graph_data.nodes)
    """

    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        seed=seed,
    )

    model = node2vec.fit(
        window=window,
        min_count=min_count,
        batch_words=4,
    )

    # --- align embeddings to nodes ---
    Z = np.zeros((len(nodes), dimensions), dtype=float)

    for i, node in enumerate(nodes):
        Z[i] = model.wv[node]

    return Z
