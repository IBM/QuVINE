import itertools
from qune.analysis.analyze import procrustes_residual, cca_correlation, rsa_corr, knn_overlap, spectral_info


def compare_embeddings(store, cca_components=10, knn_k=10):
    """
    store: EmbeddingStore
    Returns dict: pair_name -> metrics
    """
    names = store.names()
    combos = itertools.combinations(names, 2)

    results = {}

    for a, b in combos:
        Z1 = store.get(a)
        Z2 = store.get(b)

        results[f"{a}_vs_{b}"] = {
            "procrustes": procrustes_residual(Z1, Z2),
            "cca_corr": cca_correlation(Z1, Z2, n_components=cca_components),
            "rsa_corr": rsa_corr(Z1, Z2),
            "knn_overlap": knn_overlap(Z1, Z2, k=knn_k),
            "spectral": spectral_info(Z1, Z2),
        }

    return results
