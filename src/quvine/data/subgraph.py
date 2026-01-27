from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Set, Tuple, List, Dict

import networkx as nx
import numpy as np

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


def subsample_nodes_with_protected(
    G: nx.Graph,
    protected: Iterable | None,
    max_nodes: int,
    rng: np.random.Generator,
    *,
    expand_radius: int | None = None,
    require_full_budget: bool = True,
) -> nx.Graph:
    """
    Subsample nodes while preserving a protected set (if provided),
    optionally expanding their neighborhood, then filling to budget.

    - protected: nodes that must be kept (can be empty or None)
    - expand_radius: if provided, expand protected nodes by radius
    """
    G = materialize_undirected_simple_graph(G)
    node_set = set(G.nodes())
    if len(node_set) > max_nodes: 
        max_nodes = len(node_set)
    # Normalize protected set
    protected = set(protected or []) & node_set

    if len(protected) > max_nodes:
        raise ValueError(
            f"Too many protected nodes ({len(protected)}) for max_nodes={max_nodes}"
        )

    keep = set(protected)

    # Optional neighborhood expansion
    if expand_radius is not None and expand_radius > 0 and len(keep) > 0:
        keep = expand_neighborhood(G, keep, radius=expand_radius)
        if len(keep) > max_nodes:
            keep = set(list(keep)[:max_nodes])

    # Fill remaining budget
    keep = add_neighbors_until_budget(
        G,
        roots=keep,
        keep=keep,
        max_nodes=max_nodes,
        rng=rng,
    )
    keep = fill_random_until_budget(G, keep, max_nodes, rng)

    H = induce_subgraph_by_nodes(G, keep)

    if require_full_budget:
        assert H.number_of_nodes() == max_nodes, (
            H.number_of_nodes(), max_nodes
        )

    return H

def subsample_nodes(
    G: nx.Graph,
    seeds: Iterable | None,
    targets: Iterable | None,
    max_nodes: int,
    radius: int,
    rng: np.random.Generator,
) -> nx.Graph:
    protected = set(seeds or []) | set(targets or [])
    return subsample_nodes_with_protected(
        G,
        protected=protected,
        max_nodes=max_nodes,
        rng=rng,
        expand_radius=radius,
    )
