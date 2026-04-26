"""
Enhanced Graph Complexity Metrics for QuVINE

This module provides advanced complexity metrics for graphs with theory-grade
additions tied to quantum walk vs classical advantage literature. It extends
the base graph.py module with 36 comprehensive metrics (27 original + 9 new).

These metrics are designed to:
- Predict quantum advantage in graph algorithms
- Characterize graph structure at multiple scales
- Support scalable computation on large graphs
- Integrate seamlessly with existing QuVINE complexity metrics

Based on research in quantum walks, spectral graph theory, and network science.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, eigs, expm_multiply


# -----------------------------------------------------------------------------
# Candidate metric lists
# -----------------------------------------------------------------------------

CANDIDATE_27_METRICS: List[str] = [
    # Size / density controls
    "log_num_nodes",
    "log_num_edges",
    "density",
    "avg_degree",

    # Connectivity / mixing
    "normalized_spectral_gap",
    "approx_avg_path_length",
    "approx_conductance",

    # Degree / centrality concentration
    "degree_gini",
    "max_degree_fraction",
    "pagerank_gini",
    "betweenness_gini_approx",

    # Community / cyclic structure
    "modularity",
    "transitivity",
    "cycle_density",
    "nonbacktracking_spectral_radius",

    # Curvature / bottleneck geometry
    "orc_kLB_mean",
    "orc_negative_fraction",

    # Spectral richness / localization
    "laplacian_effective_rank_partial",
    "ipr_low_mean",
    "ipr_high_mean",
    "spectral_degeneracy_fraction",

    # Symmetry / core-periphery
    "wl_compression_ratio",
    "core_number_gini",

    # Task signal
    "label_homophily",
    "feature_dirichlet_energy",

    # Additional controls/structure
    "degree_assortativity",
    "largest_cc_fraction",
]

CANDIDATE_NEW_METRICS: List[str] = [
    # Theory-grade additions tied to QW vs classical advantage literature
    "bipartite_proximity",
    "log_odd_girth",
    "algebraic_connectivity_ratio",
    "spectral_entropy_partial",
    "heat_kernel_trace_t1",
    "heat_kernel_trace_t10",
    "adjacency_ipr_low_mean",
    "adjacency_ipr_high_mean",
    "closeness_gini_approx",
]

CANDIDATE_ALL_METRICS: List[str] = CANDIDATE_27_METRICS + CANDIDATE_NEW_METRICS


@dataclass
class ComplexityConfig:
    """Runtime and approximation settings for scalable metrics."""

    spectral_k: int = 64
    eig_tol: float = 1e-5
    path_num_sources: int = 64
    betweenness_k: int = 256
    wl_iterations: int = 3
    nonbacktracking_max_directed_edges: int = 1_000_000  # raised from 200_000
    random_state: int = 0
    use_largest_cc_for_path: bool = True
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 200
    pagerank_tol: float = 1e-6

    # Heat kernel trace (stochastic Hutchinson estimator)
    heat_kernel_t_values: Tuple[float, ...] = (1.0, 10.0)
    heat_kernel_n_probes: int = 20

    # Odd girth (BFS-based, sampled sources)
    odd_girth_max_sources: int = 32
    odd_girth_min_cycle_break: int = 5  # break early if found cycle <= this


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------

def sanitize_graph(G: nx.Graph, make_undirected: bool = True, remove_selfloops: bool = True) -> nx.Graph:
    """
    Return a simple NetworkX graph suitable for undirected complexity metrics.
    """
    if make_undirected and G.is_directed():
        H = nx.Graph(G)
    else:
        H = nx.Graph(G) if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)) else G.copy()

    if remove_selfloops:
        H.remove_edges_from(nx.selfloop_edges(H))
    return H


def safe_float(x: Any, default: float = np.nan) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except Exception:
        return default


def gini_coefficient(values: Iterable[float]) -> float:
    """Compute Gini coefficient for a nonnegative vector."""
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.maximum(x, 0.0)
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * total) - (n + 1.0) / n)


def get_nodelist(G: nx.Graph) -> List[Hashable]:
    return list(G.nodes())


def get_sparse_laplacian(G: nx.Graph, normalized: bool = True) -> Tuple[sp.csr_matrix, List[Hashable]]:
    """Sparse Laplacian with explicit nodelist for reproducibility."""
    nodelist = get_nodelist(G)
    if normalized:
        L = nx.normalized_laplacian_matrix(G, nodelist=nodelist).astype(float).tocsr()
    else:
        L = nx.laplacian_matrix(G, nodelist=nodelist).astype(float).tocsr()
    return L, nodelist


def get_sparse_adjacency(G: nx.Graph) -> Tuple[sp.csr_matrix, List[Hashable]]:
    """Sparse adjacency with explicit nodelist for reproducibility."""
    nodelist = get_nodelist(G)
    A = nx.adjacency_matrix(G, nodelist=nodelist).astype(float).tocsr()
    return A, nodelist


def safe_eigsh(
    L: sp.spmatrix,
    k: int,
    which: str,
    tol: float = 1e-5,
    return_eigenvectors: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Robust wrapper around scipy.sparse.linalg.eigsh."""
    n = L.shape[0]
    if n < 3:
        return np.array([]), None if return_eigenvectors else None
    k_eff = min(max(1, k), n - 2)
    try:
        vals, vecs = eigsh(L, k=k_eff, which=which, tol=tol, return_eigenvectors=True)
        vals = np.real(vals)
        vecs = np.real(vecs)
        idx = np.argsort(vals)
        return vals[idx], vecs[:, idx]
    except Exception as exc:
        warnings.warn(f"eigsh failed for which={which}, k={k_eff}: {exc}")
        if return_eigenvectors:
            return np.array([]), np.empty((n, 0))
        return np.array([]), None


# -----------------------------------------------------------------------------
# 1-4. Size and density controls
# -----------------------------------------------------------------------------

def compute_size_density_metrics(G: nx.Graph) -> Dict[str, float]:
    """Compute scale and density controls."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    density = nx.density(G) if n > 1 else 0.0
    avg_degree = (2.0 * m / n) if n > 0 else 0.0
    return {
        "log_num_nodes": float(np.log1p(n)),
        "log_num_edges": float(np.log1p(m)),
        "density": float(density),
        "avg_degree": float(avg_degree),
    }


# -----------------------------------------------------------------------------
# 5, 18-21 + new spectral metrics. Sparse Lanczos on the normalized Laplacian.
# -----------------------------------------------------------------------------

def compute_sparse_spectral_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """
    Compute scalable spectral descriptors using sparse Lanczos on the
    normalized Laplacian.

    Existing keys (unchanged semantics):
      * normalized_spectral_gap
      * laplacian_effective_rank_partial
      * ipr_low_mean
      * ipr_high_mean
      * spectral_degeneracy_fraction (BUG-FIXED: within-block gaps only)

    New keys:
      * bipartite_proximity            : max(0, 2 - lambda_n^(L_norm)).
                                          Equals 0 iff a bipartite component exists.
      * algebraic_connectivity_ratio   : lambda_2 / lambda_n^(L_norm).
      * spectral_entropy_partial       : Shannon entropy of normalized partial spectrum,
                                          normalized to [0,1].
    """
    n = G.number_of_nodes()
    if n < 2:
        return {
            "normalized_spectral_gap": 0.0,
            "laplacian_effective_rank_partial": 0.0,
            "ipr_low_mean": 0.0,
            "ipr_high_mean": 0.0,
            "spectral_degeneracy_fraction": 0.0,
            "bipartite_proximity": np.nan,
            "algebraic_connectivity_ratio": np.nan,
            "spectral_entropy_partial": np.nan,
        }

    L, _ = get_sparse_laplacian(G, normalized=True)
    k = min(config.spectral_k, max(2, n - 2))

    # FIX: which="SA" (smallest algebraic) is more stable than "SM" for PSD operators.
    vals_low, vecs_low = safe_eigsh(L, k=k, which="SA", tol=config.eig_tol)
    vals_high, vecs_high = safe_eigsh(L, k=k, which="LA", tol=config.eig_tol)

    out: Dict[str, float] = {}

    # ---- normalized_spectral_gap ----
    if vals_low.size >= 2:
        vals_low_clean = vals_low.copy()
        vals_low_clean[np.abs(vals_low_clean) < 1e-10] = 0.0
        out["normalized_spectral_gap"] = float(max(vals_low_clean[1] - vals_low_clean[0], 0.0))
    else:
        out["normalized_spectral_gap"] = np.nan

    # ---- laplacian_effective_rank_partial ----
    vals_obs = np.concatenate([vals_low, vals_high]) if vals_high.size else vals_low
    vals_pos = vals_obs[np.isfinite(vals_obs) & (vals_obs > 1e-10)]
    if vals_pos.size > 0:
        out["laplacian_effective_rank_partial"] = float((vals_pos.sum() ** 2) / np.sum(vals_pos ** 2))
    else:
        out["laplacian_effective_rank_partial"] = np.nan

    # ---- ipr_low_mean / ipr_high_mean (Laplacian eigenvectors) ----
    if vecs_low is not None and vecs_low.shape[1] > 0:
        out["ipr_low_mean"] = float(np.mean(np.sum(vecs_low ** 4, axis=0)))
    else:
        out["ipr_low_mean"] = np.nan
    if vecs_high is not None and vecs_high.shape[1] > 0:
        out["ipr_high_mean"] = float(np.mean(np.sum(vecs_high ** 4, axis=0)))
    else:
        out["ipr_high_mean"] = np.nan

    # ---- spectral_degeneracy_fraction (BUG FIX) ----
    # Count near-zero adjacent eigenvalue gaps within each contiguous block.
    deg_tol = max(1e-5, 10.0 * config.eig_tol)
    deg_counts, deg_total = 0, 0
    for block in (vals_low, vals_high):
        block_clean = np.sort(block[np.isfinite(block)])
        if block_clean.size >= 2:
            gaps = np.diff(block_clean)
            deg_counts += int(np.sum(np.abs(gaps) < deg_tol))
            deg_total += gaps.size
    out["spectral_degeneracy_fraction"] = float(deg_counts / deg_total) if deg_total > 0 else np.nan

    # ---- NEW: bipartite_proximity ----
    # For a normalized Laplacian, lambda_n in [0, 2], with lambda_n = 2 iff a
    # connected component is bipartite. We report 2 - lambda_n_max as a continuous
    # proximity-to-bipartite measure.
    if vals_high.size > 0:
        lam_max = float(vals_high.max())
        out["bipartite_proximity"] = float(max(0.0, 2.0 - lam_max))
    else:
        out["bipartite_proximity"] = np.nan

    # ---- NEW: algebraic_connectivity_ratio ----
    if vals_low.size >= 2 and vals_high.size > 0:
        lam2 = float(vals_low[1])
        lam_max = float(vals_high.max())
        out["algebraic_connectivity_ratio"] = float(lam2 / lam_max) if lam_max > 1e-10 else np.nan
    else:
        out["algebraic_connectivity_ratio"] = np.nan

    # ---- NEW: spectral_entropy_partial ----
    if vals_pos.size > 1:
        p = vals_pos / vals_pos.sum()
        ent = -float(np.sum(p * np.log(p + 1e-20)))
        # Normalize to [0,1] by dividing by log(k); 1 = uniform spectrum, 0 = single mode.
        out["spectral_entropy_partial"] = float(ent / np.log(vals_pos.size))
    else:
        out["spectral_entropy_partial"] = np.nan

    return out


# -----------------------------------------------------------------------------
# NEW. Adjacency-spectrum localization (band-edge IPR on A).
# -----------------------------------------------------------------------------

def compute_adjacency_spectral_metrics(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute IPR of band-edge eigenvectors of the unsigned adjacency matrix A.

    Theoretically motivated for QW pathways that use H = A (rather than H = L),
    and complementary to Laplacian-IPR because adjacency eigenvectors are not
    degree-normalized; localization signals on hubs survive.
    """
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return {
            "adjacency_ipr_low_mean": np.nan,
            "adjacency_ipr_high_mean": np.nan,
        }

    A, _ = get_sparse_adjacency(G)
    k = min(config.spectral_k, max(2, n - 2))

    # SA = smallest algebraic (most negative for adjacency); LA = largest algebraic.
    vals_low, vecs_low = safe_eigsh(A, k=k, which="SA", tol=config.eig_tol)
    vals_high, vecs_high = safe_eigsh(A, k=k, which="LA", tol=config.eig_tol)

    out: Dict[str, float] = {}
    if vecs_low is not None and vecs_low.shape[1] > 0:
        out["adjacency_ipr_low_mean"] = float(np.mean(np.sum(vecs_low ** 4, axis=0)))
    else:
        out["adjacency_ipr_low_mean"] = np.nan
    if vecs_high is not None and vecs_high.shape[1] > 0:
        out["adjacency_ipr_high_mean"] = float(np.mean(np.sum(vecs_high ** 4, axis=0)))
    else:
        out["adjacency_ipr_high_mean"] = np.nan
    return out


# -----------------------------------------------------------------------------
# NEW. Heat-kernel trace via Hutchinson + scipy expm_multiply.
# -----------------------------------------------------------------------------

def compute_heat_kernel_traces(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute normalized heat kernel traces tr(exp(-t L)) / n via the Hutchinson
    estimator with Rademacher probe vectors and scipy.sparse.linalg.expm_multiply.

    Theoretically motivated: tr(exp(-t L)) = sum_i exp(-t lambda_i) is the smooth
    spectral observable that integrates the diffusion behavior the relevant QW vs
    classical mixing bounds depend on. At small t it is dominated by the bulk
    spectrum; at large t it is dominated by the spectral gap.

    Each call is ~O(n_probes * matvec * scipy_internal_steps).
    For n=5000 with sparse L, this is at most a few seconds.
    """
    n = G.number_of_nodes()
    out: Dict[str, float] = {}
    for t in config.heat_kernel_t_values:
        out[f"heat_kernel_trace_t{int(round(t))}"] = np.nan
    if n < 2:
        return out

    L, _ = get_sparse_laplacian(G, normalized=True)
    rng = np.random.default_rng(config.random_state)
    Z = rng.choice(np.array([-1.0, 1.0]), size=(n, config.heat_kernel_n_probes)).astype(float)

    for t in config.heat_kernel_t_values:
        key = f"heat_kernel_trace_t{int(round(t))}"
        try:
            HZ = expm_multiply(-t * L, Z)
            # Hutchinson: E[z^T A z] = tr(A) for Rademacher z.
            trace_per_probe = np.sum(Z * HZ, axis=0)
            trace_est = float(np.mean(trace_per_probe))
            out[key] = float(trace_est / n)
        except Exception as exc:
            warnings.warn(f"heat kernel trace at t={t} failed: {exc}")
            out[key] = np.nan
    return out


# -----------------------------------------------------------------------------
# NEW. Odd girth (length of shortest odd cycle).
# -----------------------------------------------------------------------------

def compute_odd_girth_metric(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """
    Compute log(1 + shortest_odd_cycle_length).

    Procedure:
      1. If G is bipartite, return NaN (no odd cycle exists).
      2. Fast triangle existence check via shared-neighbor scan; if any triangle
         exists, return log(1 + 3).
      3. Otherwise, BFS from up to `odd_girth_max_sources` sampled sources;
         for each source s, scan all edges and identify same-level closures
         (level u == level v), giving an odd cycle of length 2 * level + 1
         passing through s. Track the minimum.

    Returns NaN if no odd cycle is found within the source budget.
    """
    n = G.number_of_nodes()
    if n < 3 or G.number_of_edges() == 0:
        return {"log_odd_girth": np.nan}

    if nx.is_bipartite(G):
        return {"log_odd_girth": np.nan}

    # Fast triangle existence check (early termination).
    for u, v in G.edges():
        nu = set(G.neighbors(u))
        nv = set(G.neighbors(v))
        if (nu & nv) - {u, v}:
            return {"log_odd_girth": float(np.log1p(3))}

    # No triangles: search via BFS from sampled sources.
    rng = np.random.default_rng(config.random_state)
    nodes = list(G.nodes())
    n_sources = min(config.odd_girth_max_sources, len(nodes))
    sources = rng.choice(nodes, size=n_sources, replace=False)

    best_odd: float = np.inf
    edges_list = list(G.edges())

    for src in sources:
        levels = nx.single_source_shortest_path_length(G, src)
        for u, v in edges_list:
            if u in levels and v in levels and levels[u] == levels[v]:
                cycle_len = 2 * levels[u] + 1
                if cycle_len < best_odd:
                    best_odd = cycle_len
        if best_odd <= config.odd_girth_min_cycle_break:
            break

    if not np.isfinite(best_odd):
        return {"log_odd_girth": np.nan}
    return {"log_odd_girth": float(np.log1p(best_odd))}


# -----------------------------------------------------------------------------
# 6 + new. Approximate path-length AND closeness-Gini (free from same BFS).
# -----------------------------------------------------------------------------

def compute_approx_path_length_metric(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """
    Approximate average shortest-path length using sampled BFS sources.

    Also returns:
      * largest_cc_fraction
      * closeness_gini_approx (NEW; free byproduct of same BFS calls).

    For disconnected graphs, the metric is computed on the largest connected
    component when use_largest_cc_for_path is True.
    """
    n = G.number_of_nodes()
    if n == 0:
        return {
            "approx_avg_path_length": np.nan,
            "largest_cc_fraction": 0.0,
            "closeness_gini_approx": np.nan,
        }

    if G.number_of_edges() == 0:
        return {
            "approx_avg_path_length": np.nan,
            "largest_cc_fraction": 1.0 / n,
            "closeness_gini_approx": np.nan,
        }

    if nx.is_connected(G):
        H = G
        lcc_frac = 1.0
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        lcc_frac = len(largest_cc) / n
        H = G.subgraph(largest_cc).copy() if config.use_largest_cc_for_path else G

    nodes = list(H.nodes())
    if len(nodes) < 2:
        return {
            "approx_avg_path_length": 0.0,
            "largest_cc_fraction": float(lcc_frac),
            "closeness_gini_approx": np.nan,
        }

    rng = np.random.default_rng(config.random_state)
    sources = rng.choice(nodes, size=min(config.path_num_sources, len(nodes)), replace=False)

    distances: List[int] = []
    closeness_values: List[float] = []
    for source in sources:
        d = nx.single_source_shortest_path_length(H, source)
        # Closeness centrality of source within H.
        total_dist = sum(v for v in d.values() if v > 0)
        n_reach = sum(1 for v in d.values() if v > 0)
        if total_dist > 0 and n_reach > 0:
            closeness_values.append(n_reach / total_dist)
        distances.extend(d.values())

    if not distances:
        avg_path = np.nan
    else:
        arr = np.asarray(distances, dtype=float)
        arr = arr[arr > 0]
        avg_path = float(arr.mean()) if arr.size > 0 else 0.0

    closeness_gini = (
        gini_coefficient(closeness_values) if len(closeness_values) >= 2 else np.nan
    )

    return {
        "approx_avg_path_length": avg_path,
        "largest_cc_fraction": float(lcc_frac),
        "closeness_gini_approx": float(closeness_gini) if np.isfinite(closeness_gini) else np.nan,
    }


# -----------------------------------------------------------------------------
# 7, 12. Community modularity and conductance
# -----------------------------------------------------------------------------

def _get_communities(G: nx.Graph, seed: int = 0) -> List[set]:
    """Compute communities using Louvain when available, else greedy modularity."""
    if G.number_of_nodes() == 0:
        return []
    try:
        return [set(c) for c in nx.community.louvain_communities(G, seed=seed)]
    except Exception:
        try:
            return [set(c) for c in nx.community.greedy_modularity_communities(G)]
        except Exception:
            return [set(G.nodes())]


def _conductance_for_set(G: nx.Graph, S: set) -> float:
    n = G.number_of_nodes()
    if not S or len(S) == n:
        return np.nan
    S = set(S)
    vol_S = sum(dict(G.degree(S)).values())
    T = set(G.nodes()) - S
    vol_T = sum(dict(G.degree(T)).values())
    if min(vol_S, vol_T) <= 0:
        return np.nan
    cut = nx.cut_size(G, S, T)
    return float(cut / min(vol_S, vol_T))


def compute_community_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """Compute modularity and approximate conductance from detected communities."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {"modularity": 0.0, "approx_conductance": np.nan}

    communities = _get_communities(G, seed=config.random_state)
    try:
        mod = float(nx.community.modularity(G, communities)) if communities else 0.0
    except Exception:
        mod = np.nan

    conductances = [_conductance_for_set(G, c) for c in communities if 0 < len(c) < G.number_of_nodes()]
    conductances = [c for c in conductances if np.isfinite(c)]
    approx_cond = float(np.min(conductances)) if conductances else np.nan

    return {"modularity": mod, "approx_conductance": approx_cond}


# -----------------------------------------------------------------------------
# 8-11, 26. Degree and centrality concentration
# -----------------------------------------------------------------------------

def compute_degree_metrics(G: nx.Graph) -> Dict[str, float]:
    """Degree heterogeneity, hub dominance, and assortativity."""
    n = G.number_of_nodes()
    if n == 0:
        return {
            "degree_gini": np.nan,
            "max_degree_fraction": np.nan,
            "degree_assortativity": np.nan,
        }

    deg = np.asarray([d for _, d in G.degree()], dtype=float)
    max_possible = max(n - 1, 1)

    try:
        assort = nx.degree_assortativity_coefficient(G) if G.number_of_edges() > 0 else np.nan
    except Exception:
        assort = np.nan

    return {
        "degree_gini": gini_coefficient(deg),
        "max_degree_fraction": float(deg.max() / max_possible) if deg.size else np.nan,
        "degree_assortativity": safe_float(assort, default=np.nan),
    }


def compute_centrality_concentration_metrics(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> Dict[str, float]:
    """Approximate betweenness Gini and PageRank Gini."""
    n = G.number_of_nodes()
    if n == 0:
        return {"pagerank_gini": np.nan, "betweenness_gini_approx": np.nan}

    try:
        pr = nx.pagerank(
            G,
            alpha=config.pagerank_alpha,
            max_iter=config.pagerank_max_iter,
            tol=config.pagerank_tol,
        )
        pagerank_gini = gini_coefficient(pr.values())
    except Exception as exc:
        warnings.warn(f"PageRank failed: {exc}")
        pagerank_gini = np.nan

    try:
        k = min(config.betweenness_k, n)
        btw = nx.betweenness_centrality(G, k=k, seed=config.random_state, normalized=True)
        betweenness_gini = gini_coefficient(btw.values())
    except Exception as exc:
        warnings.warn(f"Approximate betweenness failed: {exc}")
        betweenness_gini = np.nan

    return {
        "pagerank_gini": float(pagerank_gini) if np.isfinite(pagerank_gini) else np.nan,
        "betweenness_gini_approx": float(betweenness_gini) if np.isfinite(betweenness_gini) else np.nan,
    }


# -----------------------------------------------------------------------------
# 13-15. Cycles, transitivity, and non-backtracking structure
# -----------------------------------------------------------------------------

def compute_cycle_metrics(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """Compute transitivity, normalized cycle density, and nonbacktracking spectral radius."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    if n == 0:
        return {
            "transitivity": np.nan,
            "cycle_density": np.nan,
            "nonbacktracking_spectral_radius": np.nan,
        }

    try:
        trans = float(nx.transitivity(G)) if m > 0 else 0.0
    except Exception:
        trans = np.nan

    components = nx.number_connected_components(G) if n > 0 else 0
    cyclomatic = max(0, m - n + components)
    cycle_density = float(cyclomatic / max(m, 1))

    nbr = compute_nonbacktracking_spectral_radius(G, config=config)

    return {
        "transitivity": trans,
        "cycle_density": cycle_density,
        "nonbacktracking_spectral_radius": nbr,
    }


def compute_nonbacktracking_spectral_radius(
    G: nx.Graph,
    config: ComplexityConfig = ComplexityConfig(),
) -> float:
    """Approximate spectral radius of the Hashimoto/non-backtracking matrix."""
    m = G.number_of_edges()
    if m == 0:
        return 0.0

    directed_edges: List[Tuple[Hashable, Hashable]] = []
    for u, v in G.edges():
        directed_edges.append((u, v))
        directed_edges.append((v, u))

    q = len(directed_edges)
    if q > config.nonbacktracking_max_directed_edges:
        warnings.warn(
            f"Skipping nonbacktracking spectral radius: {q} directed edges exceed cap "
            f"{config.nonbacktracking_max_directed_edges}."
        )
        return np.nan

    edge_to_idx = {e: i for i, e in enumerate(directed_edges)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for i, (u, v) in enumerate(directed_edges):
        for w in G.neighbors(v):
            if w == u:
                continue
            j = edge_to_idx.get((v, w))
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1.0)

    if not data:
        return 0.0

    B = sp.csr_matrix((data, (rows, cols)), shape=(q, q), dtype=float)

    try:
        val = eigs(B, k=1, which="LM", return_eigenvectors=False, tol=config.eig_tol)[0]
        return float(abs(val))
    except Exception as exc:
        warnings.warn(f"Nonbacktracking eigs failed: {exc}")
        return np.nan


# -----------------------------------------------------------------------------
# 16-17. Ollivier-Ricci curvature proxies
# -----------------------------------------------------------------------------

def compute_orc_proxy_metrics(G: nx.Graph) -> Dict[str, float]:
    """
    Compute scalable ORC-inspired edge bottleneck proxies.

    Uses the Jost-Liu style lower-bound proxy:
        kappa_LB(u,v) = Delta/max(d_u,d_v) + 1/d_u + 1/d_v - 1
    """
    if G.number_of_edges() == 0:
        return {"orc_kLB_mean": np.nan, "orc_negative_fraction": np.nan}

    neighbor_sets = {u: set(G.neighbors(u)) for u in G.nodes()}
    kappa_vals: List[float] = []

    for u, v in G.edges():
        du = len(neighbor_sets[u])
        dv = len(neighbor_sets[v])
        if du == 0 or dv == 0:
            kappa = -1.0
        else:
            common = (neighbor_sets[u] & neighbor_sets[v]) - {u, v}
            Delta = len(common)
            kappa = Delta / max(du, dv) + 1.0 / du + 1.0 / dv - 1.0
        kappa_vals.append(kappa)

    arr = np.asarray(kappa_vals, dtype=float)
    return {
        "orc_kLB_mean": float(np.mean(arr)),
        "orc_negative_fraction": float(np.mean(arr < 0.0)),
    }


# -----------------------------------------------------------------------------
# 22. Weisfeiler-Lehman compression / symmetry proxy
# -----------------------------------------------------------------------------

def compute_wl_compression_ratio(G: nx.Graph, config: ComplexityConfig = ComplexityConfig()) -> Dict[str, float]:
    """WL color compression ratio after a few 1-WL refinement iterations."""
    n = G.number_of_nodes()
    if n == 0:
        return {"wl_compression_ratio": np.nan}

    colors: Dict[Hashable, int] = {u: int(G.degree(u)) for u in G.nodes()}

    for _ in range(config.wl_iterations):
        signatures = {}
        for u in G.nodes():
            neigh_colors = tuple(sorted(colors[v] for v in G.neighbors(u)))
            signatures[u] = (colors[u], neigh_colors)

        unique = {sig: i for i, sig in enumerate(sorted(set(signatures.values()), key=str))}
        colors = {u: unique[sig] for u, sig in signatures.items()}

    num_colors = len(set(colors.values()))
    return {"wl_compression_ratio": float(num_colors / n)}


# -----------------------------------------------------------------------------
# 23. Core-periphery proxy
# -----------------------------------------------------------------------------

def compute_core_metrics(G: nx.Graph) -> Dict[str, float]:
    """k-core concentration as a scalable core-periphery proxy."""
    n = G.number_of_nodes()
    if n == 0:
        return {"core_number_gini": np.nan}
    try:
        # core_number requires no self-loops, which we already strip in sanitize_graph.
        core = nx.core_number(G)
        vals = list(core.values())
        return {"core_number_gini": gini_coefficient(vals)}
    except Exception as exc:
        warnings.warn(f"core_number failed: {exc}")
        return {"core_number_gini": np.nan}


# -----------------------------------------------------------------------------
# 24. Label homophily
# -----------------------------------------------------------------------------

def _labels_to_dict(
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]],
    nodelist: Sequence[Hashable],
) -> Optional[Dict[Hashable, Any]]:
    if labels is None:
        return None
    if isinstance(labels, Mapping):
        return dict(labels)
    arr = np.asarray(labels)
    if arr.shape[0] != len(nodelist):
        raise ValueError("labels length must match number of nodes when labels is an array/sequence.")
    return {node: arr[i] for i, node in enumerate(nodelist)}


def compute_label_homophily(
    G: nx.Graph,
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]],
) -> Dict[str, float]:
    """Fraction of edges connecting nodes with identical labels."""
    nodelist = get_nodelist(G)
    label_dict = _labels_to_dict(labels, nodelist)
    if label_dict is None:
        return {"label_homophily": np.nan}

    same = 0
    total = 0
    for u, v in G.edges():
        if u in label_dict and v in label_dict:
            if label_dict[u] is None or label_dict[v] is None:
                continue
            same += int(label_dict[u] == label_dict[v])
            total += 1

    return {"label_homophily": float(same / total) if total > 0 else np.nan}


# -----------------------------------------------------------------------------
# 25. Feature Dirichlet energy
# -----------------------------------------------------------------------------

def _features_to_array(
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]],
    nodelist: Sequence[Hashable],
) -> Optional[np.ndarray]:
    if features is None:
        return None
    if isinstance(features, Mapping):
        X = []
        for node in nodelist:
            if node not in features:
                raise ValueError(f"Missing feature for node {node!r}.")
            X.append(features[node])
        return np.asarray(X, dtype=float)
    X = np.asarray(features, dtype=float)
    if X.shape[0] != len(nodelist):
        raise ValueError("features.shape[0] must match number of nodes.")
    if X.ndim == 1:
        X = X[:, None]
    return X


def compute_feature_dirichlet_energy(
    G: nx.Graph,
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]],
    normalized_laplacian: bool = True,
) -> Dict[str, float]:
    """Compute normalized feature Dirichlet energy Tr(X^T L X) / Tr(X^T X)."""
    L, nodelist = get_sparse_laplacian(G, normalized=normalized_laplacian)
    X = _features_to_array(features, nodelist)
    if X is None:
        return {"feature_dirichlet_energy": np.nan}

    denom = float(np.sum(X * X))
    if denom <= 0:
        return {"feature_dirichlet_energy": np.nan}

    LX = L @ X
    energy = float(np.sum(X * LX) / denom)
    return {"feature_dirichlet_energy": energy}


# -----------------------------------------------------------------------------
# Full metric interface (now returns 27 + 9 = 36 metrics).
# -----------------------------------------------------------------------------

def compute_enhanced_complexity_metrics(
    G: nx.Graph,
    labels: Optional[Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]] = None,
    features: Optional[Union[np.ndarray, Mapping[Hashable, Sequence[float]]]] = None,
    config: ComplexityConfig = ComplexityConfig(),
    sanitize: bool = True,
) -> Dict[str, float]:
    """
    Compute the enhanced QuVINE complexity metrics for a single graph.

    This function computes 36 comprehensive metrics (27 original + 9 new theory-grade
    metrics) that characterize graph structure and predict quantum advantage.

    Parameters
    ----------
    G : nx.Graph
        Input graph
    labels : optional
        Node labels for computing label homophily
    features : optional
        Node features for computing feature Dirichlet energy
    config : ComplexityConfig
        Configuration for approximation parameters
    sanitize : bool, default=True
        If True, convert to simple undirected graph and remove self-loops

    Returns
    -------
    dict
        Dictionary containing all 36 complexity metrics
    """
    H = sanitize_graph(G) if sanitize else G.copy()

    metrics: Dict[str, float] = {
        "num_nodes_raw": float(H.number_of_nodes()),
        "num_edges_raw": float(H.number_of_edges()),
    }

    metric_functions = [
        lambda graph: compute_size_density_metrics(graph),
        lambda graph: compute_sparse_spectral_metrics(graph, config=config),
        lambda graph: compute_adjacency_spectral_metrics(graph, config=config),
        lambda graph: compute_heat_kernel_traces(graph, config=config),
        lambda graph: compute_odd_girth_metric(graph, config=config),
        lambda graph: compute_approx_path_length_metric(graph, config=config),
        lambda graph: compute_community_metrics(graph, config=config),
        lambda graph: compute_degree_metrics(graph),
        lambda graph: compute_centrality_concentration_metrics(graph, config=config),
        lambda graph: compute_cycle_metrics(graph, config=config),
        lambda graph: compute_orc_proxy_metrics(graph),
        lambda graph: compute_wl_compression_ratio(graph, config=config),
        lambda graph: compute_core_metrics(graph),
        lambda graph: compute_label_homophily(graph, labels=labels),
        lambda graph: compute_feature_dirichlet_energy(graph, features=features),
    ]

    for fn in metric_functions:
        try:
            metrics.update(fn(H))
        except Exception as exc:
            warnings.warn(f"Metric function {getattr(fn, '__name__', repr(fn))} failed: {exc}")

    for key in CANDIDATE_ALL_METRICS:
        metrics.setdefault(key, np.nan)

    return metrics


def compute_complexity_table(
    graphs: Mapping[str, nx.Graph],
    labels: Optional[Mapping[str, Union[Mapping[Hashable, Any], Sequence[Any], np.ndarray]]] = None,
    features: Optional[Mapping[str, Union[np.ndarray, Mapping[Hashable, Sequence[float]]]]] = None,
    config: ComplexityConfig = ComplexityConfig(),
) -> "Any":
    """
    Compute a pandas DataFrame of complexity metrics for many graphs.

    Parameters
    ----------
    graphs : dict
        Dictionary mapping graph names to NetworkX graphs
    labels : optional
        Dictionary mapping graph names to node labels
    features : optional
        Dictionary mapping graph names to node features
    config : ComplexityConfig
        Configuration for approximation parameters

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per graph and columns for each metric
    """
    import pandas as pd

    rows = []
    for name, G in graphs.items():
        lab = labels.get(name) if labels is not None else None
        feat = features.get(name) if features is not None else None
        row = compute_enhanced_complexity_metrics(G, labels=lab, features=feat, config=config)
        row["graph_name"] = name
        rows.append(row)

    df = pd.DataFrame(rows).set_index("graph_name")
    return df

# Made with Bob
