
import numpy as np
import matplotlib.pyplot as plt 
from qune.analysis.analyze import normalize 

def analyze_fusion(Z_fused, n_views):
    singular_values = np.linalg.svd(Z_fused, compute_uv=False)
    eigvals = np.linalg.eigvalsh(Z_fused.T @ Z_fused)

    return {
        "singular_values": singular_values,
        "eigvals": eigvals,
    }


def plot_concat_singular_values(singular_values, label='Concat–PCA', filename=None):
    plt.figure()
    x = np.arange(1, len(singular_values) + 1)
    plt.plot(x, np.log(singular_values), marker='o', label=label)

    plt.xlabel('Component index')
    plt.ylabel('log(singular value)')
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_consensus_eigenvalues(eigvals, n_views=3, filename=None):
    plt.figure()
    x = np.arange(1, len(eigvals) + 1)
    plt.plot(x, eigvals, marker='o', label='Consensus eigenvalues')

    # reference lines
    plt.axhline(1.0, linestyle='--', color='gray', label='all views')
    plt.axhline(2/n_views, linestyle='--', color='orange', label='2 views')
    plt.axhline(1/n_views, linestyle='--', color='red', label='1 view')

    plt.xlabel('Component index')
    plt.ylabel('Eigenvalue')
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    
def plot_shared_private_energy(embedding_list, U_shared, view_names=None, filename=None):
    if view_names is None:
        view_names = [f'View {i+1}' for i in range(len(norm_embeddings))]

    shared_fracs = []
    private_fracs = []

    norm_embeddings = []
    for z in embedding_list:
            norm_embeddings.append(normalize(z))
    
    for Z in norm_embeddings:
        Z_shared = U_shared @ (U_shared.T @ Z)
        Z_private = Z - Z_shared

        total_energy = np.linalg.norm(Z, 'fro')**2
        shared_energy = np.linalg.norm(Z_shared, 'fro')**2
        private_energy = np.linalg.norm(Z_private, 'fro')**2

        shared_fracs.append(shared_energy / total_energy)
        private_fracs.append(private_energy / total_energy)

    x = np.arange(len(norm_embeddings))

    plt.figure()
    plt.bar(x, shared_fracs, label='Shared', color='tab:blue')
    plt.bar(x, private_fracs, bottom=shared_fracs, label='Private', color='tab:orange')

    plt.xticks(x, view_names)
    plt.ylabel('Fraction of energy')
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    plt.close()
