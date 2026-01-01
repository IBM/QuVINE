import numpy as np
from numpy.linalg import svd
from qune.analysis.analyze import normalize

def fuse_embeddings(embedding_list, method='concatenate', k=None):
    """
    Fuse multiple embeddings

    Args:
        embedding_list (_type_): _description_
        method (str, optional): _description_. Defaults to 'concatenate'.
    """
    norm_embeddings = []
    for z in embedding_list:
            norm_embeddings.append(normalize(z))
            
    if k is None: 
        ks = [z.shape[1] for z in embedding_list]
        k = min(ks)
    
    if method == 'concatenate': 
        fused_embedding = np.concatenate(norm_embeddings, axis=1)
        ## svd to factor fused embeddings
        U, S, _ = svd(fused_embedding, full_matrices=False)
        shared_embedding = U[:,:k] @ np.diag(S[:k]) #rank k
        return shared_embedding, S[:k]
    
    elif method == 'projection':
        projections = []
        for z in norm_embeddings:
            U, S, _ = svd(z, full_matrices=False)
            Uk = U[:,:k]
            projections.append(Uk @ Uk.T)
            
        fused_projection = sum(projections) / len(projections)
        #get top k eigenvectors
        eigval, eigvec = np.linalg.eigh(fused_projection)
        idx = np.argsort(eigval)[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:,idx]
        shared_embedding = eigvec[:,:k]
        return shared_embedding, eigval[:k]