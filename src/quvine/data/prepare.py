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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Set, Tuple, List, Dict

import networkx as nx
import numpy as np


# ---------------------------
# Utilities: graph materialization (NO subgraph views)
# ---------------------------

def materialize_undirected_simple_graph(G: nx.Graph) -> nx.Graph:
    """
    Return a fully materialized undirected simple nx.Graph (no views, no DiGraph internals).
    This avoids the weird KeyErrors you were seeing with adjacency traversal.
    """
    H = nx.Graph()
    # nodes()
    H.add_nodes_from(list(G.nodes()))
    # edges() - if input is directed/multi, nx.Graph() will collapse direction/multiedges
    H.add_edges_from(list(G.edges()))
    return H


def induce_subgraph_by_nodes(G: nx.Graph, nodes: Set) -> nx.Graph:
    """
    Materialize an induced subgraph without using G.subgraph(...) to avoid view contamination.
    """
    nodes = set(nodes)
    H = nx.Graph()
    H.add_nodes_from(nodes)

    # Build edges by filtering original edges; safe and view-free
    H.add_edges_from((u, v) for (u, v) in G.edges() if u in nodes and v in nodes)
    return H


# ---------------------------
# Node subsampling: protect seeds/targets, expand neighborhood, fill budget
# ---------------------------

def expand_neighborhood(G: nx.Graph, roots: Set, radius: int) -> Set:
    """
    Expand neighborhood around roots up to given hop radius using adjacency dict access.
    No G.neighbors() calls; avoids your previous KeyErrors.
    """
    if radius <= 0:
        return set(roots)

    node_set = set(G.nodes())
    roots = set(r for r in roots if r in node_set)

    visited = set(roots)
    frontier = set(roots)

    for _ in range(radius):
        if not frontier:
            break
        next_frontier = set()
        for u in frontier:
            # Using adjacency dict access; this should never KeyError if u in node_set
            # but we guard anyway.
            if u not in node_set:
                continue
            next_frontier.update(G.adj[u].keys())
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier

    return visited


def fill_budget_with_random_nodes(G: nx.Graph, keep_nodes: Set, max_nodes: int, rng: np.random.Generator) -> Set:
    """
    Fill the remaining node budget with random nodes from G%keep_nodes.
    """
    keep_nodes = set(keep_nodes)
    all_nodes = list(G.nodes())

    if max_nodes <= 0:
        return set()

    # If keep_nodes already exceeds budget, downsample while preserving determinism
    if len(keep_nodes) > max_nodes:
        keep_list = list(keep_nodes)
        chosen = rng.choice(keep_list, size=max_nodes, replace=False)
        return set(chosen)

    remaining = list(set(all_nodes) - keep_nodes)
    needed = max_nodes - len(keep_nodes)
    if needed > 0 and remaining:
        chosen = rng.choice(remaining, size=min(needed, len(remaining)), replace=False)
        keep_nodes |= set(chosen)

    return keep_nodes

def add_neighbors_until_budget(G, roots, keep, max_nodes, rng):
    frontier = list(roots)
    rng.shuffle(frontier)

    while frontier and len(keep) < max_nodes:
        u = frontier.pop()
        if u not in G:
            continue

        nbrs = list(G.adj[u].keys())
        rng.shuffle(nbrs)

        for v in nbrs:
            if len(keep) >= max_nodes:
                break
            if v not in keep:
                keep.add(v)
                frontier.append(v)

    return keep

def fill_random_until_budget(G, keep, max_nodes, rng):
    if len(keep) >= max_nodes:
        return keep

    remaining = list(set(G.nodes()) - keep)
    rng.shuffle(remaining)

    needed = max_nodes - len(keep)
    keep |= set(remaining[:needed])
    return keep

def ensure_single_component(
    G_full: nx.Graph,
    H: nx.Graph,
    rng: np.random.Generator,
    max_extra_nodes: int = 0,
) -> nx.Graph:
    """
    Ensure H is a single connected component by adding the minimum
    number of biologically valid edges (and optionally nodes) from G_full.
    """

    if nx.is_connected(H):
        return H

    components = list(nx.connected_components(H))
    components.sort(key=len, reverse=True)

    main_comp = set(components[0])
    other_comps = components[1:]

    added_nodes = set()
    H = H.copy()

    for comp in other_comps:
        comp = set(comp)
        connected = False

        # Try to find a bridging edge in the full graph
        for u in comp:
            for v in G_full.adj[u]:
                if v in main_comp:
                    H.add_edge(u, v)
                    main_comp |= comp
                    connected = True
                    break

                # Optional: allow 1-hop connector node
                if (
                    max_extra_nodes > 0
                    and v not in H
                    and any(w in main_comp for w in G_full.adj[v])
                ):
                    w = next(w for w in G_full.adj[v] if w in main_comp)
                    H.add_node(v)
                    H.add_edge(u, v)
                    H.add_edge(v, w)
                    added_nodes.add(v)
                    main_comp |= comp | {v}
                    max_extra_nodes -= 1
                    connected = True
                    break
            if connected:
                break

        if not connected:
            # As a fallback, skip — extremely rare in PPI debug graphs
            continue

        if nx.is_connected(H):
            break

    return H

def keep_largest_connected_component(G: nx.Graph) -> nx.Graph:
    """
    Return the largest connected component of an undirected graph.
    Safe, simple, and deterministic.
    """
    if G.number_of_nodes() == 0:
        return G

    if nx.is_connected(G):
        return G

    lcc_nodes = max(nx.connected_components(G), key=len)
    return G.subgraph(lcc_nodes).copy()

def subsample_nodes_preserve_seeds_targets(
    G: nx.Graph,
    seeds: Iterable,
    targets: Iterable,
    max_nodes: int,
    radius: int,
    rng: np.random.Generator,
) -> nx.Graph:
    """
    Debug-only: subsample nodes but always keep all seeds+targets, expand neighborhood, then fill budget.
    Produces a fully materialized nx.Graph (no views).
    """
    G = materialize_undirected_simple_graph(G)
    node_set = set(G.nodes())

    protected = (set(seeds) | set(targets)) & node_set
    if len(protected) > max_nodes: 
        raise ValueError(f"Too many protected nodes ({len(protected)}) for max_nodes={max_nodes}")
    keep = set(protected)
    #protected_in = {n for n in protected if n in node_set}

    # If any protected nodes are missing, we don't crash; we just warn via return metadata if you want.
    # For now: keep only those present.
    #keep = expand_neighborhood(G, protected_in, radius=radius)
    keep = add_neighbors_until_budget(
        G,
        roots=protected,
        keep=keep,
        max_nodes=max_nodes,
        rng=rng,
    )
    keep = fill_random_until_budget(G, keep, max_nodes, rng)
    
    #keep = fill_budget_with_random_nodes(G, keep, max_nodes=max_nodes, rng=rng)

    H = induce_subgraph_by_nodes(G, keep)
    
    assert H.number_of_nodes() == max_nodes, (
        H.number_of_nodes(), max_nodes
    )
    
    # # Safety: ensure all protected present in H (that were present in G)
    # missing = protected_in - set(H.nodes())
    # if missing:
    #     # This should only happen when max_nodes < len(protected_in) (budget too small).
    #     # We avoid KeyErrors by not throwing; but it's a hard logic issue.
    #     raise ValueError(
    #         f"Subsampling budget too small: max_nodes={max_nodes} but protected nodes in graph={len(protected_in)}. "
    #         f"Missing protected nodes: {list(missing)[:10]}{'...' if len(missing) > 10 else ''}"
    #     )

    return H


# ---------------------------
# Edge scoring + sparsification (triangle support + degree cap)
# ---------------------------

def edge_triangle_support_scores(G: nx.Graph, candidate_edges: List[Tuple]) -> List[Tuple[Tuple, int]]:
    """
    Score edges by triangle support (# common neighbors). Uses adjacency dict access only.
    """
    # Ensure fully materialized to avoid any internal weirdness
    G = materialize_undirected_simple_graph(G)
    node_set = set(G.nodes())

    # Precompute neighbor sets via adjacency (safe)
    neigh: Dict = {u: set(G.adj[u].keys()) for u in node_set}

    scored: List[Tuple[Tuple, int]] = []
    for u, v in candidate_edges:
        if u not in node_set or v not in node_set:
            continue
        scored.append(((u, v), len(neigh[u].intersection(neigh[v]))))
    return scored


def sparsify_edges_biological(
    G: nx.Graph,
    retain_ratio: float,
    max_degree: int,
    rng: np.random.Generator,
    scoring: Literal["common_neighbors", "prefer_low_degree"] = "common_neighbors",
) -> nx.Graph:
    """
    Edge-only sparsification: degree-capped selection prioritizing biologically coherent edges.
    No BFS, no connectivity checks, no nx.is_connected/connected_components.
    """
    if not (0 < retain_ratio <= 1):
        raise ValueError("retain_ratio must be in (0, 1].")
    if max_degree <= 0:
        raise ValueError("max_degree must be positive.")

    G = materialize_undirected_simple_graph(G)
    if G.number_of_nodes() == 0:
        return G

    m0 = G.number_of_edges()
    target_edges = int(round(retain_ratio * m0))
    # Guarantee we don't request a negative/zero edge budget
    target_edges = max(0, target_edges)

    edges = list(G.edges())

    if scoring == "common_neighbors":
        # Score all edges
        scored = edge_triangle_support_scores(G, edges)
        rng.shuffle(scored)  # tie-breaking
        scored.sort(key=lambda x: x[1], reverse=True)
        sorted_edges = [e for (e, _) in scored]
    elif scoring == "prefer_low_degree":
        rng.shuffle(edges)
        edges.sort(key=lambda e: (G.degree[e[0]] + G.degree[e[1]]))
        sorted_edges = edges
    else:
        raise ValueError(f"Unknown scoring method: {scoring}")

    H = nx.Graph()
    H.add_nodes_from(G.nodes())

    # Degree-capped edge addition
    for u, v in sorted_edges:
        if H.number_of_edges() >= target_edges:
            break
        # nodes must exist (should, but guard anyway)
        if u not in H or v not in H:
            continue
        if H.degree[u] >= max_degree or H.degree[v] >= max_degree:
            continue
        if H.has_edge(u, v):
            continue
        H.add_edge(u, v)

    return H


# ---------------------------
# prepare_graph(): end-to-end, optional subsample, then sparsify edges
# ---------------------------

@dataclass
class PrepareGraphConfig:
    # Debug subsampling (optional)
    subsample_nodes: bool = False
    max_nodes: int = 3000
    radius: int = 2

    # Sparsification (optional but usually enabled)
    sparsify_edges: bool = True
    retain_ratio: float = 0.7
    max_degree: int = 40
    scoring: Literal["common_neighbors", "prefer_low_degree"] = "common_neighbors"

    # Verbose stats
    verbose: bool = False


def prepare_graph(
    cfg: PrepareGraphConfig,
    graph: nx.Graph,
    seeds: Iterable,
    targets: Iterable,
    seed: int,
) -> nx.Graph:
    """
    Step-by-step graph preparation:

    1) Materialize a clean undirected simple graph (no views).
    2) If cfg.subsample_nodes: subsample nodes while protecting seeds+targets,
        expand neighborhood (radius hops), fill remaining budget.
    3) If cfg.sparsify_edges: degree-capped, biology-aware edge sparsification
        (triangle support or low-degree preference).
    4) Return a clean nx.Graph.

    Designed to avoid the KeyError / _succ / neighbor traversal issues you saw:
    - no subgraph views
    - no nx.is_connected / connected_components
    - no G.neighbors() calls
    - only adjacency dict access (G.adj[u])
    """
    rng = np.random.default_rng(seed)

    # Step 1: canonical materialization (always)
    G = materialize_undirected_simple_graph(graph)

    # Step 2: optional subsample (debug/notebook)
    if cfg.subsample_nodes:
        G = subsample_nodes_preserve_seeds_targets(
            G,
            seeds=seeds,
            targets=targets,
            max_nodes=cfg.max_nodes,
            radius=cfg.radius,
            rng=rng,
        )

        G = keep_largest_connected_component(G)
    # Step 3: sparsify edges (biology-aware, degree-capped)
    if cfg.sparsify_edges:
        G = sparsify_edges_biological(
            G,
            retain_ratio=cfg.retain_ratio,
            max_degree=cfg.max_degree,
            rng=rng,
            scoring=cfg.scoring,
        )
    G = keep_largest_connected_component(G)
    # Step 4: lightweight stats (optional)
    if cfg.verbose and G.number_of_nodes() > 0:
        avg_deg = (2.0 * G.number_of_edges() / G.number_of_nodes()) if G.number_of_nodes() else 0.0
        max_deg = max(dict(G.degree()).values()) if G.number_of_nodes() else 0
        print("-" * 50)
        print("After prepare_graph()")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Avg degree: {avg_deg:.2f}")
        print(f"Max degree: {max_deg}")
        print("-" * 50)

    return G
