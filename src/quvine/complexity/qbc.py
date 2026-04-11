"""
QBC Complexity Metrics for QuVINE

Computes QBioCode-equivalent complexity metrics from graph Laplacian eigenvectors.
All computations are self-contained — no qbiocode or TensorFlow imports.

The 23 qbc_* metrics are computed using scipy / sklearn / skdim / hfda on the
graph's normalized Laplacian eigenvector matrix (nodes × nodes).
"""

import gc
import warnings

import networkx as nx
import numpy as np
import pandas as pd

# ── Optional heavy deps (graceful degradation if missing) ────────────────────
try:
    from skdim.id import lPCA as _lPCA
    _SKDIM_OK = True
except ImportError:
    _SKDIM_OK = False

try:
    import hfda as _hfda
    _HFDA_OK = True
except ImportError:
    _HFDA_OK = False

try:
    from sklearn.feature_selection import mutual_info_classif as _mic
    from sklearn.manifold import Isomap as _Isomap
    from sklearn.neighbors import KernelDensity as _KDE
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False


# ── Laplacian eigenvector representation ─────────────────────────────────────

_QBC_MAX_COMPONENTS = 200   # cap eigenvectors for graphs larger than this

def _eigvec_matrix(G: nx.Graph, normalized: bool = True,
                   max_components: int = _QBC_MAX_COMPONENTS) -> np.ndarray:
    """Return (n_nodes × k) eigenvector matrix of the normalized Laplacian.

    For graphs with n <= max_components: k = n (full decomposition, dense).
    For graphs with n >  max_components: k = max_components smallest eigenvectors
    using sparse eigsh — O(n·k²) instead of O(n³).
    """
    n = G.number_of_nodes()
    if normalized:
        L = nx.normalized_laplacian_matrix(G)
    else:
        L = nx.laplacian_matrix(G)

    if n <= max_components:
        _, vecs = np.linalg.eigh(L.toarray())   # full dense decomp, shape (n, n)
        return vecs

    # Large graph: compute only the max_components smallest eigenpairs (sparse).
    from scipy.sparse.linalg import eigsh as _eigsh
    k = min(max_components, n - 2)              # eigsh needs k < n
    # which='SM' = smallest magnitude, but for Laplacian use LM after shift-invert
    vals, vecs = _eigsh(L, k=k, which='SM', tol=1e-5, maxiter=5 * n)
    idx = np.argsort(vals)
    return vecs[:, idx]                         # shape (n, k)


def _binary_labels(n: int) -> np.ndarray:
    return np.array([i % 2 for i in range(n)], dtype=int)


# ── Per-metric helpers ────────────────────────────────────────────────────────

def _intrinsic_dim(X: np.ndarray) -> float:
    if not _SKDIM_OK:
        return float("nan")
    try:
        pca = _lPCA()
        pca.fit(X)
        d = float(pca.dimension_)
        del pca
        return d
    except Exception:
        return float("nan")


def _condition_number(X: np.ndarray) -> float:
    try:
        return float(np.linalg.cond(X))
    except Exception:
        return float("nan")


def _fisher_discriminant_ratio(X: np.ndarray, y: np.ndarray) -> float:
    try:
        classes = np.unique(y)
        if len(classes) != 2:
            return float("nan")
        m1 = X[y == classes[0]].mean(axis=0)
        m2 = X[y == classes[1]].mean(axis=0)
        S_w = sum(np.cov(X[y == c].T) for c in classes)
        S_b = np.outer(m1 - m2, m1 - m2)
        denom = np.trace(S_w)
        return float(np.trace(S_b) / denom) if denom != 0 else float("nan")
    except Exception:
        return float("nan")


def _total_correlation(X: np.ndarray) -> float:
    try:
        C = pd.DataFrame(X).corr().values.copy()  # copy so fill_diagonal can mutate it
        np.fill_diagonal(C, 0)
        return float(np.nansum(np.abs(C)))
    except Exception:
        return float("nan")


def _mutual_information(X: np.ndarray, y: np.ndarray) -> float:
    if not _SKLEARN_OK:
        return float("nan")
    try:
        return float(np.mean(_mic(X, y)))
    except Exception:
        return float("nan")


def _variance_stats(X: np.ndarray):
    v = X.var(axis=0)
    return float(v.mean()), float(v.std())


def _cv_stats(X: np.ndarray):
    m = np.abs(X.mean(axis=0))
    s = X.std(axis=0)
    cv = np.where(m > 1e-12, (s / m) * 100, np.nan)
    return float(np.nanmean(cv)), float(np.nanstd(cv))


def _low_var_features(X: np.ndarray) -> int:
    try:
        v = X.var(axis=0)
        return int(np.sum(v <= np.percentile(v, 25)))
    except Exception:
        return 0


def _log_kernel_density(X: np.ndarray) -> float:
    if not _SKLEARN_OK:
        return float("nan")
    try:
        kde = _KDE(bandwidth=0.2, kernel="gaussian").fit(X)
        score = float(kde.score_samples(X).mean())
        del kde
        return score
    except Exception:
        return float("nan")


def _isomap_error(X: np.ndarray) -> float:
    if not _SKLEARN_OK:
        return float("nan")
    try:
        n = X.shape[0]
        iso = _Isomap(n_neighbors=min(10, n - 1), n_components=min(2, n - 1))
        iso.fit(X)
        err = float(iso.reconstruction_error())
        del iso
        return err
    except Exception:
        return float("nan")


def _fractal_dim(X: np.ndarray, k_max: int = 5) -> float:
    """Higuchi fractal dimension averaged over eigenvectors (1D time series each)."""
    if not _HFDA_OK:
        return float("nan")
    try:
        fds = []
        for col in X.T:
            try:
                fds.append(float(_hfda.measure(col.astype(float), k_max)))
            except Exception:
                pass
        return float(np.mean(fds)) if fds else float("nan")
    except Exception:
        return float("nan")


def _moments(X: np.ndarray):
    df = pd.DataFrame(X)
    sk = df.skew()
    ku = df.kurtosis()
    return float(sk.mean()), float(sk.std()), float(ku.mean()), float(ku.std())


def _label_entropy(y: np.ndarray):
    try:
        p = np.bincount(y).astype(float)
        p /= p.sum()
        h = float(-np.sum(p * np.log2(p + 1e-12)))
        return h, 0.0
    except Exception:
        return float("nan"), float("nan")


# ── Public API ────────────────────────────────────────────────────────────────

def check_qbc_available() -> bool:
    """Return True — self-contained implementation requires no qbiocode package."""
    return True


def get_qbc_installation_instructions() -> str:
    return "QBC metrics are built-in (no qbiocode package required)."


def compute_qbc_metrics(G: nx.Graph) -> dict:
    """
    Compute QBC-equivalent complexity metrics for graph G.

    Uses the normalized Laplacian eigenvector matrix as a dataset representation
    and computes 23 dataset-complexity metrics using scipy / sklearn / skdim / hfda.
    No qbiocode or TensorFlow dependency.

    For graphs with > _QBC_MAX_COMPONENTS nodes only the smallest
    _QBC_MAX_COMPONENTS eigenvectors are used (sparse eigsh), keeping
    runtime tractable for large PPI networks.

    Returns a flat dict with qbc_* keys.
    """
    X = _eigvec_matrix(G, normalized=True)   # (n_nodes, k) float64
    n = G.number_of_nodes()
    k = X.shape[1]                           # k == n for small graphs, else _QBC_MAX_COMPONENTS
    y = _binary_labels(k)

    avg_var, std_var   = _variance_stats(X)
    avg_cv, std_cv     = _cv_stats(X)
    avg_sk, std_sk, avg_ku, std_ku = _moments(X)
    h, std_h           = _label_entropy(y)

    result = {
        "qbc_n_features":                  k,
        "qbc_n_samples":                   n,
        "qbc_feature_samples_ratio":       k / n if n > 0 else 1.0,
        "qbc_intrinsic_dimension":         _intrinsic_dim(X),
        "qbc_condition_number":            _condition_number(X),
        "qbc_fisher_discriminant_ratio":   _fisher_discriminant_ratio(X, y),
        "qbc_total_correlations":          _total_correlation(X),
        "qbc_mutual_information":          _mutual_information(X, y),
        "qbc_num_non_zero_entries":        int(np.count_nonzero(X)),
        "qbc_num_low_variance_features":   _low_var_features(X),
        "qbc_variation":                   avg_var,
        "qbc_std_var":                     std_var,
        "qbc_coefficient_of_variation":    avg_cv,
        "qbc_std_co_of_v":                std_cv,
        "qbc_skewness":                    avg_sk,
        "qbc_std_skew":                    std_sk,
        "qbc_kurtosis":                    avg_ku,
        "qbc_std_kurt":                    std_ku,
        "qbc_mean_log_kernel_density":     _log_kernel_density(X),
        "qbc_isomap_reconstruction_error": _isomap_error(X),
        "qbc_fractal_dimension":           _fractal_dim(X, k_max=5),
        "qbc_entropy":                     h,
        "qbc_std_entropy":                 std_h,
    }
    del X, y
    gc.collect()
    return result


# ── Compatibility aliases (old API surface) ───────────────────────────────────

def compute_qbc_complexity_from_laplacian(
    G: nx.Graph,
    normalized: bool = True,
    laplacian_method: str = "eigenvectors",
    label_method: str = "binary",
    dataset_name: str = None,
    mean_center: bool = False,
) -> dict:
    """Compatibility wrapper — delegates to compute_qbc_metrics."""
    return compute_qbc_metrics(G)


def compute_qbc_complexity_multimethod(G: nx.Graph, **kwargs) -> dict:
    """Compatibility wrapper — delegates to compute_qbc_metrics."""
    return {"eigenvectors": compute_qbc_metrics(G)}


def compute_comprehensive_complexity(G: nx.Graph, **kwargs) -> dict:
    """Compatibility wrapper — delegates to compute_qbc_metrics."""
    return compute_qbc_metrics(G)


def compare_qbc_complexity(graphs: dict, **kwargs) -> dict:
    """Compatibility wrapper — computes compute_qbc_metrics for each graph."""
    return {name: compute_qbc_metrics(g) for name, g in graphs.items()}
