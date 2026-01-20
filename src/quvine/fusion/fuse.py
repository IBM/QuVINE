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

def fuse_embeddings(store, k=None, L=None, method="svd",
                beta=1e-2, lam=1e-2):
    """
    method:
        - "svd"      : recommended default (fast, competitive, easy to justify)
        - "graphreg" : optional smoothing post-step (requires L, ideally sparse)
    """
    names = store.names()
    Zs_raw = [store.get(name) for name in names]

    # your existing normalize() is kept but I strongly recommend adding block standardization
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

    if method == "all": 
        if L is None:
            raise ValueError("L must be provided for method='graphreg'.")
        
        svd_emb = fuse_embeddings_svd(Zs, k)
        graphreg_emb = fuse_embeddings_graphreg(Zs, k, L=L, beta=beta, lam=lam)
    
        return [svd_emb, graphreg_emb], ['svd', 'graphreg']
        
    raise ValueError(f"Unknown fusion method: {method}")
