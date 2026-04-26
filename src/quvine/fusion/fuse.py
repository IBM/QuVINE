# Copyright 2021, IBM Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from numpy.linalg import svd
from quvine.analysis.analyze import normalize

def _row_norm(Z, eps=1e-8):
    nrm = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (nrm + eps)

def _block_standardize(Z, eps=1e-8):
    mu = Z.mean(axis=0, keepdims=True)
    sd = Z.std(axis=0, keepdims=True)
    return (Z - mu) / (sd + eps)

def _prep_blocks(Zs, do_row_norm=True, do_block_standardize=True):
    out = []
    for Z in Zs:
        Zp = Z
        if do_row_norm:
            Zp = _row_norm(Zp)
        if do_block_standardize:
            Zp = _block_standardize(Zp)
        out.append(Zp)
    return out

def fuse_embeddings_svd(Zs, k):
    """
    Fast early fusion:
    concatenate (after per-block normalization) + SVD/PCA to shared k-dim space.
    """
    Zcat = np.concatenate(Zs, axis=1)            # (n, sum d_v)
    U, S, _ = svd(Zcat, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_fused = U[:, :k_eff] * S[:k_eff]           # (n,k_eff), scaled PCs
    if k_eff < k:
        Z_fused = np.pad(Z_fused, ((0, 0), (0, k - k_eff)), mode='constant')
    return Z_fused

# ---------- Optional: graph-regularized shared U, scalable ----------
def _apply_graph_regularization(U0, L, beta, lam, max_cg_iter=200, cg_tol=1e-6):
    """
    Solve (beta*L + (1+lam)*I) u_j = u0_j for each column via sparse CG.
    Returns U0 unchanged if scipy is unavailable or beta <= 0.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    if not sp.issparse(L):
        # DO NOT densify on big graphs; pass sparse L.
        L = sp.csr_matrix(L)

    n, k = U0.shape
    Aop = (beta * L) + (1.0 + lam) * sp.eye(n, format="csr")

    U = np.zeros_like(U0)
    for j in range(k):
        rhs = U0[:, j]
        uj, info = spla.cg(Aop, rhs, maxiter=max_cg_iter, rtol=cg_tol)
        if info != 0:
            uj = rhs
        U[:, j] = uj
    return U


def fuse_embeddings_graphreg(Zs, k, L, beta=1e-2, lam=1e-2, max_cg_iter=200, cg_tol=1e-6):
    """
    Solve: argmin_U ||U - Zbar||_F^2 + beta * tr(U^T L U) + lam ||U||_F^2
    where Zbar is the SVD-fused initialization projected to k dims.

    This is a *much simpler* regularization story than your full multiview + Ws + alpha,
    and it avoids over-parameterization reviewers will question.
    """
    U0 = fuse_embeddings_svd(Zs, k)

    if beta <= 0:
        return U0

    try:
        return _apply_graph_regularization(U0, L, beta, lam, max_cg_iter, cg_tol)
    except ImportError:
        return U0

def fuse_embeddings_attention(Zs, k, temperature=1.0):
    """
    Attention-based fusion: learn attention weights for each embedding view.

    Uses softmax attention over embedding similarities to weight each view.
    """
    n = Zs[0].shape[0]
    
    # Compute mean pairwise similarity for each view as a quality score.
    # Mean of gram matrix = ||Z_norm||_F^2 / n, avoids materializing N×N matrix.
    similarities = []
    for Z in Zs:
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        similarities.append((Z_norm ** 2).sum() / n)

    from scipy.special import softmax
    attention_weights = softmax(np.array(similarities) / temperature)
    
    # Weighted combination
    Z_weighted = sum(w * Z for w, Z in zip(attention_weights, Zs))
    
    # Apply SVD for dimensionality reduction
    U, S, _ = svd(Z_weighted, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_fused = U[:, :k_eff] * S[:k_eff]
    if k_eff < k:
        Z_fused = np.pad(Z_fused, ((0, 0), (0, k - k_eff)), mode='constant')
    return Z_fused


def fuse_embeddings_hybrid(Zs, k, L=None, beta=1e-2, lam=1e-2, temperature=1.0):
    """
    Hybrid fusion: combines SVD and attention-based fusion.
    
    First applies attention-weighted fusion, then applies SVD with optional graph regularization.
    """
    # Step 1: Attention-based weighting
    attention_fused = fuse_embeddings_attention(Zs, k=k, temperature=temperature)
    
    if L is not None and beta > 0:
        try:
            return _apply_graph_regularization(attention_fused, L, beta, lam)
        except ImportError:
            pass

    return attention_fused


def fuse_embeddings_svd_shared_private(Zs, k, gate_type='attention'):
    """
    SVD-based shared/private fusion for N ≥ 2 embedding views.

    Each view is decomposed into a *shared* component (extracted by a joint
    rank-k SVD over all views concatenated) and a *private* residual.  The
    views are then recombined using one of two gating strategies:

    ``attention`` — a per-feature sigmoid gate is applied to each view's
        private component; the result is added to the mean shared component.
        Good when views are complementary and private details matter.

    ``moe`` — a per-node softmax over the V views selects a weighted
        combination of all normalised views.  Good when views are exchangeable
        and the model should pick the most informative one per node.

    Parameters
    ----------
    Zs : list of np.ndarray, each (n, d)
        Embedding views to fuse (V ≥ 2).
    k : int
        Rank for SVD approximation (typically d // 4).
    gate_type : {'attention', 'moe'}

    Returns
    -------
    Z_final : np.ndarray, shape (n, d)
    """
    V = len(Zs)
    if V < 2:
        raise ValueError("fuse_embeddings_svd_shared_private requires at least 2 views.")

    n, d = Zs[0].shape

    # 1. Layer-normalise all views.
    Zn = [_layer_norm(Z) for Z in Zs]

    # 2. Joint SVD over the horizontally concatenated views → rank-k shared reconstruction.
    Z_cat = np.concatenate(Zn, axis=1)          # (n, V*d)
    U, S, Vh = svd(Z_cat, full_matrices=False)
    k_eff = min(k, U.shape[1])
    Z_hat = (U[:, :k_eff] * S[:k_eff]) @ Vh[:k_eff, :]  # (n, V*d)

    # 3. Per-view shared and private components.
    Z_sh   = [Z_hat[:, i*d:(i+1)*d] for i in range(V)]   # each (n, d)
    Z_priv = [Zn[i] - Z_sh[i]        for i in range(V)]   # each (n, d)

    # Mean shared component — the consensus across all views.
    Z_shared_mean = np.mean(np.stack(Z_sh, axis=0), axis=0)   # (n, d)

    if gate_type == 'attention':
        # Gate features: [all shared ‖ all private] → (n, 2V*d)
        # Gate output:   (n, V*d)  — one d-dimensional sigmoid gate per view.
        gate_feat = np.concatenate(Z_sh + Z_priv, axis=1)     # (n, 2V*d)
        gate = _simple_mlp_gate(gate_feat, V * d)              # (n, V*d)

        # Add gated private corrections to the shared mean; average over views.
        private_mix = sum(gate[:, i*d:(i+1)*d] * Z_priv[i] for i in range(V))
        Z_final = Z_shared_mean + private_mix / V

    elif gate_type == 'moe':
        # Gate features: normalised views concatenated → (n, V*d)
        # Gate output:   (n, V) softmax weights over views.
        gate_feat = np.concatenate(Zn, axis=1)                 # (n, V*d)
        raw = _simple_mlp_gate(gate_feat, V)                   # (n, V) sigmoid
        gate_w = raw / (raw.sum(axis=1, keepdims=True) + 1e-8) # (n, V) sum-to-1
        Z_final = sum(gate_w[:, i:i+1] * Zn[i] for i in range(V))  # (n, d)

    else:
        raise ValueError(f"Unknown gate_type '{gate_type}'. Use 'attention' or 'moe'.")

    return Z_final


def _layer_norm(Z, eps=1e-8):
    """Layer normalization."""
    mean = Z.mean(axis=1, keepdims=True)
    std = Z.std(axis=1, keepdims=True)
    return (Z - mean) / (std + eps)


def _simple_mlp_gate(features, output_dim, random_state=0):
    n, input_dim = features.shape
    hidden_dim = max(64, output_dim)
    rng = np.random.default_rng(random_state)
    W1 = rng.standard_normal((input_dim, hidden_dim)) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, output_dim)) * 0.01
    b2 = np.zeros(output_dim)
    h = np.maximum(0, features @ W1 + b1)
    return 1.0 / (1.0 + np.exp(-(h @ W2 + b2)))


def fuse_embeddings(store, k=None, L=None, method="svd",
                beta=1e-2, lam=1e-2, temperature=1.0, svd_rank=None, gate_type='attention'):
    """
    Fuse multiple embeddings using various methods.
    
    Parameters
    ----------
    store : EmbeddingStore
        Store containing multiple embeddings
    k : int, optional
        Target embedding dimension
    L : sparse matrix, optional
        Graph Laplacian (required for 'graphreg' and 'hybrid')
    method : str
        Fusion method:
        - "svd"              : SVD-based fusion (fast, default)
        - "graphreg"         : Graph-regularized fusion (requires L)
        - "attention"        : Attention-weighted fusion
        - "hybrid"           : Attention + graph regularization (requires L)
        - "svd_shared_priv"  : SVD shared/private decomposition with gating
        - "all"              : Compute all methods (requires L)
    beta : float
        Graph regularization strength
    lam : float
        L2 regularization strength
    temperature : float
        Temperature for attention softmax
    svd_rank : int, optional
        Rank for SVD approximation in shared/private decomposition (default: k // 4)
    gate_type : str
        Gate type for shared/private fusion: 'attention' or 'moe'
        
    Returns
    -------
    embeddings : list
        List of fused embeddings
    names : list
        List of method names
    """
    names = store.names()
    Zs_raw = [store.get(name) for name in names]

    # Column-normalize then row-normalize each view once.
    Zs = [normalize(Z) for Z in Zs_raw]
    Zs = _prep_blocks(Zs, do_row_norm=True, do_block_standardize=False)

    if k is None:
        k = min(Z.shape[1] for Z in Zs)

    if method == "svd":
        return [fuse_embeddings_svd(Zs, k)], ['svd']

    if method == "graphreg":
        if L is None:
            raise ValueError("L must be provided for method='graphreg'.")
        return [fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)], ['graphreg']
    
    if method == "attention":
        return [fuse_embeddings_attention(Zs, k, temperature=temperature)], ['attention']
    
    if method == "hybrid":
        if L is None:
            raise ValueError("L must be provided for method='hybrid'.")
        return [fuse_embeddings_hybrid(Zs, k, L=L, beta=beta, lam=lam, temperature=temperature)], ['hybrid']
    
    if method == "svd_shared_priv":
        rank = svd_rank if svd_rank is not None else max(k // 4, 1)
        return [fuse_embeddings_svd_shared_private(Zs, rank, gate_type=gate_type)], [f'svd_shared_priv_{gate_type}']

    if method == "all":
        if L is None:
            raise ValueError("L must be provided for method='graphreg' and 'hybrid'.")
        
        svd_emb = fuse_embeddings_svd(Zs, k)
        graphreg_emb = fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)
        attention_emb = fuse_embeddings_attention(Zs, k, temperature=temperature)
        hybrid_emb = fuse_embeddings_hybrid(Zs, k, L=L, beta=beta, lam=lam, temperature=temperature)
        
        rank = svd_rank if svd_rank is not None else max(k // 4, 1)
        svd_sp_att = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='attention')
        svd_sp_moe = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='moe')
        return [svd_emb, graphreg_emb, attention_emb, hybrid_emb, svd_sp_att, svd_sp_moe], \
               ['svd', 'graphreg', 'attention', 'hybrid', 'svd_shared_priv_attention', 'svd_shared_priv_moe']
        
    raise ValueError(f"Unknown fusion method: {method}")
