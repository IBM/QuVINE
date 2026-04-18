#!/usr/bin/env python3
"""Debug topological computation."""
import sys
from pathlib import Path
import networkx as nx
import numpy as np

# Add quvine to path
_QUVINE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_QUVINE_SRC) not in sys.path:
    sys.path.insert(0, str(_QUVINE_SRC))

from quvine.complexity.graph import _hop_distance_matrix

# Load a test graph
graphml_path = Path("/Users/aritrabose/Library/CloudStorage/OneDrive-IBM/Research/Quantum/quvine/ppi_disease_v3/results/BioPlex3_asthma_rep00/BioPlex3_asthma_rep00.graphml")

print(f"Loading graph from: {graphml_path}")
G = nx.read_graphml(graphml_path)
G = nx.convert_node_labels_to_integers(G)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Connected: {nx.is_connected(G)}")

# Get largest connected component
if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    print(f"Number of components: {len(components)}")
    largest_cc = max(components, key=len)
    print(f"Largest component size: {len(largest_cc)}")
    G_cc = G.subgraph(largest_cc).copy()
    G_cc = nx.convert_node_labels_to_integers(G_cc)
    print(f"\nUsing largest connected component: {G_cc.number_of_nodes()} nodes, {G_cc.number_of_edges()} edges")
    G = G_cc

# Compute distance matrix
print("\nComputing distance matrix...")
D = _hop_distance_matrix(G)
print(f"Distance matrix shape: {D.shape}")
print(f"Distance matrix stats: min={D.min():.2f}, max={D.max():.2f}, mean={D.mean():.2f}")
print(f"Unique distances: {np.unique(D)}")

# Try Ripser
print("\nRunning Ripser...")
try:
    from ripser import ripser as _ripser
    result = _ripser(D, maxdim=2, distance_matrix=True)
    dgms = result["dgms"]
    print(f"Number of persistence diagrams: {len(dgms)}")
    for dim, dgm in enumerate(dgms):
        print(f"  H{dim}: {len(dgm)} features")
        if len(dgm) > 0:
            print(f"    Birth range: [{dgm[:, 0].min():.2f}, {dgm[:, 0].max():.2f}]")
            print(f"    Death range: [{dgm[:, 1].min():.2f}, {dgm[:, 1].max():.2f}]")
            # Count features alive at eps=1.0
            births = dgm[:, 0]
            deaths = dgm[:, 1]
            alive_at_1 = np.sum((births <= 1.0) & (deaths > 1.0))
            print(f"    Features alive at ε=1.0: {alive_at_1}")
            
except ImportError:
    print("Ripser not installed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Made with Bob
