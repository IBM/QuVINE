#!/usr/bin/env python3
"""Test topological computation on a single graph to debug."""
import sys
from pathlib import Path
import networkx as nx

# Add quvine to path
_QUVINE_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_QUVINE_SRC) not in sys.path:
    sys.path.insert(0, str(_QUVINE_SRC))

from quvine.complexity.graph import compute_betti_numbers, compute_persistence_entropy

# Load a test graph
graphml_path = Path("/Users/aritrabose/Library/CloudStorage/OneDrive-IBM/Research/Quantum/quvine/ppi_disease_v3/results/BioPlex3_asthma_rep00/BioPlex3_asthma_rep00.graphml")

print(f"Loading graph from: {graphml_path}")
G = nx.read_graphml(graphml_path)
G = nx.convert_node_labels_to_integers(G)

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Connected: {nx.is_connected(G)}")
if not nx.is_connected(G):
    components = list(nx.connected_components(G))
    print(f"Number of components: {len(components)}")
    print(f"Largest component size: {len(max(components, key=len))}")

print("\nComputing Betti numbers...")
try:
    betti_result = compute_betti_numbers(G, maxdim=2, filtration_scale=1.0)
    print(f"Betti numbers result: {betti_result}")
    
    print("\nComputing persistence entropy...")
    entropy_result = compute_persistence_entropy(G, maxdim=2, filtration_scale=1.0)
    print(f"Persistence entropy result: {entropy_result}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Made with Bob
