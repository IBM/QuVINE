"""
GCN-MF: Graph Convolutional Network with Matrix Factorization

A baseline method for disease gene prioritization that combines:
1. Graph Convolutional Networks (GCN) for learning node representations
2. Matrix Factorization for capturing latent features

Reference:
- Kipf & Welling (2017). Semi-Supervised Classification with Graph Convolutional Networks
- Matrix factorization approaches for biological network analysis
"""

import numpy as np
import scipy.sparse as sp

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer."""
    
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass.
        
        Args:
            x: Node features [N, in_features]
            adj: Normalized adjacency matrix [N, N] (sparse or dense)
        
        Returns:
            Output features [N, out_features]
        """
        # Linear transformation
        support = torch.mm(x, self.weight)
        
        # Graph convolution
        if isinstance(adj, torch.sparse.FloatTensor):
            output = torch.sparse.mm(adj, support)
        else:
            output = torch.mm(adj, support)
        
        if self.bias is not None:
            output = output + self.bias
        
        return output


class GCNMF(nn.Module):
    """
    GCN-MF: Graph Convolutional Network with Matrix Factorization.
    
    Architecture:
    1. GCN layers for learning graph-based representations
    2. Matrix factorization component for latent features
    3. Fusion of GCN and MF representations
    """
    
    def __init__(self, n_nodes, input_dim, hidden_dim, output_dim, 
                 mf_dim=64, n_layers=2, dropout=0.5):
        """
        Initialize GCN-MF model.
        
        Args:
            n_nodes: Number of nodes in the graph
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (number of classes)
            mf_dim: Matrix factorization embedding dimension
            n_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super(GCNMF, self).__init__()
        
        self.n_nodes = n_nodes
        self.mf_dim = mf_dim
        self.dropout = dropout
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(n_layers - 1):
            self.gcn_layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Matrix factorization embeddings
        self.node_embeddings = nn.Parameter(torch.FloatTensor(n_nodes, mf_dim))
        nn.init.xavier_uniform_(self.node_embeddings)
        
        # Fusion layer
        self.fusion = nn.Linear(hidden_dim + mf_dim, hidden_dim)
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x, adj, node_indices=None):
        """
        Forward pass.
        
        Args:
            x: Node features [N, input_dim]
            adj: Normalized adjacency matrix [N, N]
            node_indices: Optional node indices for batch processing
        
        Returns:
            Output predictions [N, output_dim]
        """
        # GCN forward pass
        h = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Get matrix factorization embeddings
        if node_indices is not None:
            mf_emb = self.node_embeddings[node_indices]
        else:
            mf_emb = self.node_embeddings
        
        # Fuse GCN and MF representations
        combined = torch.cat([h, mf_emb], dim=1)
        fused = F.relu(self.fusion(combined))
        fused = F.dropout(fused, p=self.dropout, training=self.training)
        
        # Output
        out = self.output(fused)
        
        return out
    
    def get_embeddings(self, x, adj):
        """
        Get node embeddings (before classification layer).
        
        Args:
            x: Node features
            adj: Normalized adjacency matrix
        
        Returns:
            Node embeddings
        """
        # GCN forward pass
        h = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            h = gcn_layer(h, adj)
            if i < len(self.gcn_layers) - 1:
                h = F.relu(h)
        
        # Fuse with MF embeddings
        combined = torch.cat([h, self.node_embeddings], dim=1)
        fused = F.relu(self.fusion(combined))
        
        return fused


def normalize_adjacency(adj):
    """
    Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
    
    Args:
        adj: Adjacency matrix (scipy sparse or numpy array)
    
    Returns:
        Normalized adjacency matrix
    """
    if isinstance(adj, np.ndarray):
        adj = sp.csr_matrix(adj)
    
    # Add self-loops
    adj = adj + sp.eye(adj.shape[0])
    
    # Compute D^(-1/2)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(rowsum, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
    
    return adj_normalized




class QCaliberGCNMF(GCNMF):
    """
    Q-Caliber GCN-MF: Integrates quantum-calibrated diffusion into GCN-MF.
    
    This model uses calibrated diffusion operators (from quantum walk statistics)
    as a preprocessing step before the GCN layers, combining:
    1. Quantum-calibrated diffusion (heat kernel or polynomial)
    2. GCN layers for learning representations
    3. Matrix factorization for latent features
    """
    
    def __init__(self, n_nodes, input_dim, hidden_dim, output_dim,
                 mf_dim=64, n_layers=2, dropout=0.5,
                 diffusion_type='heat', t_star=None, poly_coeffs=None, L=None):
        """
        Initialize Q-Caliber GCN-MF model.
        
        Args:
            n_nodes: Number of nodes
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            mf_dim: Matrix factorization dimension
            n_layers: Number of GCN layers
            dropout: Dropout rate
            diffusion_type: 'heat' or 'poly'
            t_star: Calibrated heat kernel time parameter
            poly_coeffs: Calibrated polynomial coefficients
            L: Laplacian matrix (for diffusion)
        """
        super().__init__(n_nodes, input_dim, hidden_dim, output_dim,
                        mf_dim, n_layers, dropout)
        
        self.diffusion_type = diffusion_type
        self.t_star = t_star
        self.poly_coeffs = poly_coeffs
        self.L = L
    
    def apply_calibrated_diffusion(self, x):
        """
        Apply quantum-calibrated diffusion to input features.
        
        Args:
            x: Input features [N, input_dim]
        
        Returns:
            Diffused features [N, input_dim]
        """
        if self.L is None:
            return x
        
        import scipy.sparse.linalg as spla
        
        x_np = x.detach().cpu().numpy()
        z_np = np.zeros_like(x_np)
        
        if self.diffusion_type == 'heat' and self.t_star is not None:
            # Heat kernel diffusion: exp(-t*L) X
            for f in range(x_np.shape[1]):
                z_np[:, f] = spla.expm_multiply((-self.t_star) * self.L, x_np[:, f])
        
        elif self.diffusion_type == 'poly' and self.poly_coeffs is not None:
            # Polynomial filter: sum_k a_k L^k X
            for f in range(x_np.shape[1]):
                v = x_np[:, f]
                z_np[:, f] = self.poly_coeffs[0] * v
                for k in range(1, len(self.poly_coeffs)):
                    v = self.L @ v
                    z_np[:, f] += self.poly_coeffs[k] * v
        else:
            z_np = x_np
        
        return torch.from_numpy(z_np).float().to(x.device)
    
    def forward(self, x, adj, node_indices=None):
        """
        Forward pass with quantum-calibrated diffusion preprocessing.
        
        Args:
            x: Node features [N, input_dim]
            adj: Normalized adjacency matrix [N, N]
            node_indices: Optional node indices
        
        Returns:
            Output predictions [N, output_dim]
        """
        # Apply quantum-calibrated diffusion first
        x_diffused = self.apply_calibrated_diffusion(x)
        
        # Then apply standard GCN-MF forward pass
        return super().forward(x_diffused, adj, node_indices)
    
    def get_embeddings(self, x, adj):
        """
        Get node embeddings with calibrated diffusion preprocessing.
        
        Args:
            x: Node features
            adj: Normalized adjacency matrix
        
        Returns:
            Node embeddings
        """
        # Apply quantum-calibrated diffusion first
        x_diffused = self.apply_calibrated_diffusion(x)
        
        # Then get embeddings
        return super().get_embeddings(x_diffused, adj)



def train_gcn_mf(model, x, y, adj, train_mask, val_mask, 
                 epochs=200, lr=0.01, weight_decay=5e-4, patience=20):
    """
    Train GCN-MF model.
    
    Args:
        model: GCN-MF model
        x: Node features
        y: Labels
        adj: Normalized adjacency matrix
        train_mask: Training mask
        val_mask: Validation mask
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for regularization
        patience: Early stopping patience
    
    Returns:
        Dictionary with training history
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': [],
        'best_val_acc': 0.0,
        'best_epoch': 0
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(x, adj)
        loss = criterion(out[train_mask], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(x, adj)
            pred = out.argmax(dim=1)
            
            train_acc = (pred[train_mask] == y[train_mask]).float().mean().item()
            val_acc = (pred[val_mask] == y[val_mask]).float().mean().item()
        
        history['train_loss'].append(loss.item())
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_acc'] = best_val_acc
            history['best_epoch'] = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}: Loss={loss.item():.4f}, '
                  f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')
    
    return history

# Made with Bob
