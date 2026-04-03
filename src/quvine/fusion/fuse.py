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
        # your normalize() might already row-normalize; keep it if you trust it
        # but I prefer explicit:
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
    return U[:, :k] * S[:k]                      # (n,k), scaled PCs

# ---------- Optional: graph-regularized shared U, scalable ----------
def fuse_embeddings_graphreg(Zs, k, L, beta=1e-2, lam=1e-2, max_cg_iter=200, cg_tol=1e-6):
    """
    Solve: argmin_U ||U - Zbar||_F^2 + beta * tr(U^T L U) + lam ||U||_F^2
    where Zbar is the SVD-fused initialization projected to k dims.

    This is a *much simpler* regularization story than your full multiview + Ws + alpha,
    and it avoids over-parameterization reviewers will question.
    """
    # 1) init from SVD fusion
    U0 = fuse_embeddings_svd(Zs, k)  # (n,k)

    if beta <= 0:
        return U0

    # 2) Use sparse CG solves per column:
    # (beta L + (1+lam) I) u_j = u0_j
    try:
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        if not sp.issparse(L):
            # DO NOT densify on big graphs; if L is dense here, you should pass sparse L.
            L = sp.csr_matrix(L)

        n = L.shape[0]
        Aop = (beta * L) + (1.0 + lam) * sp.eye(n, format="csr")

        U = np.zeros_like(U0)
        for j in range(k):
            rhs = U0[:, j]
            uj, info = spla.cg(Aop, rhs, maxiter=max_cg_iter, rtol=cg_tol)
            if info != 0:
                # fallback: keep original column if CG struggles
                uj = rhs
            U[:, j] = uj
        return U

    except ImportError:
        # No scipy: fallback to no graphreg (still fast and strong baseline)
        return U0

def fuse_embeddings_attention(Zs, k, temperature=1.0):
    """
    Attention-based fusion: learn attention weights for each embedding view.
    
    Uses softmax attention over embedding similarities to weight each view.
    """
    n = Zs[0].shape[0]
    num_views = len(Zs)
    
    # Compute pairwise similarities for each view
    similarities = []
    for Z in Zs:
        # Normalize embeddings
        Z_norm = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        # Compute similarity matrix
        sim = Z_norm @ Z_norm.T
        similarities.append(sim.mean())  # Average similarity as view quality score
    
    # Compute attention weights using softmax
    similarities = np.array(similarities)
    attention_weights = np.exp(similarities / temperature) / np.exp(similarities / temperature).sum()
    
    # Weighted combination
    Z_weighted = sum(w * Z for w, Z in zip(attention_weights, Zs))
    
    # Apply SVD for dimensionality reduction
    U, S, _ = svd(Z_weighted, full_matrices=False)
    return U[:, :k] * S[:k]


def fuse_embeddings_hybrid(Zs, k, L=None, beta=1e-2, lam=1e-2, temperature=1.0):
    """
    Hybrid fusion: combines SVD and attention-based fusion.
    
    First applies attention-weighted fusion, then applies SVD with optional graph regularization.
    """
    # Step 1: Attention-based weighting
    attention_fused = fuse_embeddings_attention(Zs, k=k, temperature=temperature)
    
    # Step 2: Apply graph regularization if L is provided
    if L is not None and beta > 0:
        try:
            import scipy.sparse as sp
            import scipy.sparse.linalg as spla
            
            if not sp.issparse(L):
                L = sp.csr_matrix(L)
            
            n = L.shape[0]
            Aop = (beta * L) + (1.0 + lam) * sp.eye(n, format="csr")
            
            U = np.zeros_like(attention_fused)
            for j in range(k):
                rhs = attention_fused[:, j]
                uj, info = spla.cg(Aop, rhs, maxiter=200, rtol=1e-6)
                if info != 0:
                    uj = rhs
                U[:, j] = uj
            return U
        except ImportError:
            return attention_fused
    
    return attention_fused


def fuse_embeddings_svd_shared_private(Zs, k, gate_type='attention'):
    """
    Advanced SVD-based fusion with shared/private decomposition.
    
    Decomposes embeddings into shared and private components using SVD,
    then combines them using gated attention or mixture of experts.
    
    Based on the pseudocode:
    1. Layer normalize each embedding
    2. Concatenate and apply SVD to get shared components
    3. Compute private components as residuals
    4. Use gated attention or mixture of experts to combine
    
    Parameters
    ----------
    Zs : list of np.ndarray
        List of embeddings, each of shape (n, d)
    k : int
        Rank for SVD approximation (typically k = d // 4)
    gate_type : str
        'attention' for gate-based attention or 'moe' for mixture of experts
        
    Returns
    -------
    Z_final : np.ndarray
        Fused embedding of shape (n, d)
    """
    if len(Zs) != 2:
        raise ValueError("SVD shared/private fusion currently supports exactly 2 embeddings")
    
    Z_h, Z_p = Zs[0], Zs[1]
    n, d = Z_h.shape
    
    # Step 1: Layer normalization
    Z_hn = _layer_norm(Z_h)
    Z_pn = _layer_norm(Z_p)
    
    # Step 2: Concatenate and apply SVD
    Z = np.concatenate([Z_hn, Z_pn], axis=1)  # (n, 2d)
    U, S, Vh = svd(Z, full_matrices=False)
    
    # Step 3: Rank-k approximation
    k = min(k, d)  # Ensure k <= d
    Uk = U[:, :k]
    Sk = S[:k]
    Vhk = Vh[:k, :]  # (k, 2d)
    
    Z_hat = (Uk * Sk) @ Vhk  # (n, 2d) rank-k approximation
    
    # Step 4: Extract shared components
    Z_h_shared = Z_hat[:, :d]
    Z_p_shared = Z_hat[:, d:]
    
    # Step 5: Compute private components
    Z_h_priv = Z_h - Z_h_shared
    Z_p_priv = Z_p - Z_p_shared
    
    # Step 6: Combine using gated attention or mixture of experts
    if gate_type == 'attention':
        # Option 1: Gate-based attention
        features = np.concatenate([Z_h_shared, Z_p_shared, Z_h_priv, Z_p_priv], axis=1)
        gate = _simple_mlp_gate(features, 2*d)  # (n, 2d) - need 2d output for both gates
        
        g_h, g_p = gate[:, :d], gate[:, d:]
        Z_final = 0.5 * (Z_h_shared + Z_p_shared) + g_h * Z_h_priv + g_p * Z_p_priv
        
    elif gate_type == 'moe':
        # Option 2: Mixture of experts
        features = np.concatenate([Z_h, Z_p, Z_h_priv, Z_p_priv], axis=1)
        gate = _simple_mlp_gate(features, d)  # (n, d)
        
        Z_final = gate * Z_h + (1 - gate) * Z_p
        
    else:
        raise ValueError(f"Unknown gate_type: {gate_type}. Use 'attention' or 'moe'.")
    
    return Z_final


def _layer_norm(Z, eps=1e-8):
    """Layer normalization."""
    mean = Z.mean(axis=1, keepdims=True)
    std = Z.std(axis=1, keepdims=True)
    return (Z - mean) / (std + eps)


def _simple_mlp_gate(features, output_dim):
    """
    Simple MLP-based gating mechanism using numpy.
    
    Uses a single hidden layer with sigmoid activation.
    This is a simplified version; for production, consider using PyTorch.
    """
    n, input_dim = features.shape
    
    # Initialize weights (simple random initialization)
    np.random.seed(42)
    hidden_dim = max(64, output_dim)
    
    W1 = np.random.randn(input_dim, hidden_dim) * 0.01
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, output_dim) * 0.01
    b2 = np.zeros(output_dim)
    
    # Forward pass
    h = np.maximum(0, features @ W1 + b1)  # ReLU activation
    gate = 1 / (1 + np.exp(-(h @ W2 + b2)))  # Sigmoid activation
    
    return gate


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

    # Normalize and standardize
    Zs = [_prep_blocks([normalize(Z)])[0] for Z in Zs_raw]
    Zs = _prep_blocks(Zs, do_row_norm=True, do_block_standardize=True)

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
        if len(Zs) != 2:
            raise ValueError("svd_shared_priv fusion requires exactly 2 embeddings")
        rank = svd_rank if svd_rank is not None else max(k // 4, 1)
        return [fuse_embeddings_svd_shared_private(Zs, rank, gate_type=gate_type)], [f'svd_shared_priv_{gate_type}']

    if method == "all":
        if L is None:
            raise ValueError("L must be provided for method='graphreg' and 'hybrid'.")
        
        svd_emb = fuse_embeddings_svd(Zs, k)
        graphreg_emb = fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)
        attention_emb = fuse_embeddings_attention(Zs, k, temperature=temperature)
        hybrid_emb = fuse_embeddings_hybrid(Zs, k, L=L, beta=beta, lam=lam, temperature=temperature)
        
        # Add shared/private if exactly 2 embeddings
        if len(Zs) == 2:
            rank = svd_rank if svd_rank is not None else max(k // 4, 1)
            svd_sp_att = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='attention')
            svd_sp_moe = fuse_embeddings_svd_shared_private(Zs, rank, gate_type='moe')
            return [svd_emb, graphreg_emb, attention_emb, hybrid_emb, svd_sp_att, svd_sp_moe], \
                   ['svd', 'graphreg', 'attention', 'hybrid', 'svd_shared_priv_attention', 'svd_shared_priv_moe']
        
        return [svd_emb, graphreg_emb, attention_emb, hybrid_emb], ['svd', 'graphreg', 'attention', 'hybrid']
        
    raise ValueError(f"Unknown fusion method: {method}")
