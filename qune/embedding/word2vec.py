import numpy as np
from gensim.models import Word2Vec


def corpus_to_embedding(
    corpus,
    nodes,
    vector_size=64,
    window=5,
    sg=1,
    negative=10,
    min_count=0,
    workers=8,
    epochs=100,
):
    """
    Train Word2Vec embeddings from a compiled corpus.

    corpus: List[List[str]] or List[List[int]]
    nodes:  List[str] or List[int]
    """

    # ensure all nodes appear at least once
    sentences = list(corpus)
    for v in nodes:
        sentences.append([v])

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        sg=sg,
        negative=negative,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
    )

    Z = np.zeros((len(nodes), vector_size))
    for i, v in enumerate(nodes):
        Z[i] = model.wv[v]

    return Z
