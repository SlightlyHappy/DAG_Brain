# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""
Independent reference implementations used only to cross-check backend.causal_core.

They follow the textbook definitions directly and share no code with the
engine:
- d-separation via the ancestral moral graph (Lauritzen et al. 1990).
- The adjustment criterion in its original path-based form (Shpitser,
  VanderWeele & Robins 2010, Definition 3): no element of Z is a descendant of
  a non-X node on a proper causal path, and Z blocks every proper non-causal
  path between X and Y.
- Pearl's back-door criterion (sound, not complete).
"""

from __future__ import annotations

import itertools
import random
from typing import Dict, List, Set, Tuple

import networkx as nx


def dsep_moral(G: nx.DiGraph, xs: Set[str], ys: Set[str], zs: Set[str]) -> bool:
    """x ⊥ y | z  iff x and y are disconnected in the moralised ancestral graph of x∪y∪z after deleting z."""
    keep: Set[str] = set()
    for v in xs | ys | zs:
        keep |= nx.ancestors(G, v) | {v}
    A = G.subgraph(keep)
    M = nx.Graph()
    M.add_nodes_from(A.nodes)
    M.add_edges_from(A.to_undirected().edges)
    for v in A.nodes:
        parents = list(A.predecessors(v))
        M.add_edges_from(itertools.combinations(parents, 2))
    M.remove_nodes_from(zs)
    for a in xs:
        for b in ys:
            if a in M and b in M and nx.has_path(M, a, b):
                return False
    return True


def _path_blocked(G: nx.DiGraph, path: List[str], zs: Set[str]) -> bool:
    an_z: Set[str] = set()
    for z in zs:
        an_z |= nx.ancestors(G, z) | {z}
    for k in range(1, len(path) - 1):
        a, v, b = path[k - 1], path[k], path[k + 1]
        collider = G.has_edge(a, v) and G.has_edge(b, v)
        if collider and v not in an_z:
            return True
        if not collider and v in zs:
            return True
    return False


def adjustment_valid_paths(G: nx.DiGraph, x: str, y: str, zs: Set[str]) -> bool:
    """Shpitser et al. (2010) adjustment criterion, written with explicit paths."""
    UG = G.to_undirected(as_view=True)
    # (a) Z contains no descendant of any W != x on a proper causal path x -> ... -> y.
    for path in nx.all_simple_paths(G, x, y):
        for w in path[1:]:
            if (nx.descendants(G, w) | {w}) & zs:
                return False
    # (b) Z blocks every proper non-causal path (x appears only at the start).
    for path in nx.all_simple_paths(UG, x, y):
        directed = all(G.has_edge(path[k], path[k + 1]) for k in range(len(path) - 1))
        if directed:
            continue
        if not _path_blocked(G, path, zs):
            return False
    return True


def backdoor_valid(G: nx.DiGraph, x: str, y: str, zs: Set[str]) -> bool:
    """Pearl's back-door criterion: no descendant of x in Z, and Z blocks every path into x."""
    if (nx.descendants(G, x)) & zs:
        return False
    H = G.copy()
    H.remove_edges_from(list(G.out_edges(x)))
    return dsep_moral(H, {x}, {y}, zs)


def random_dag(n: int, p: float, rng: random.Random) -> nx.DiGraph:
    nodes = [f"v{i}" for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for i, j in itertools.combinations(range(n), 2):
        if rng.random() < p:
            G.add_edge(nodes[i], nodes[j])
    return G


def total_effect_linear(B: Dict[Tuple[str, str], float], order: List[str], x: str, y: str) -> float:
    """Sum over directed paths of products of path coefficients (Wright's rules)."""
    effect = {v: 0.0 for v in order}
    effect[x] = 1.0
    started = False
    for v in order:
        if v == x:
            started = True
            continue
        if not started:
            continue
        effect[v] = sum(c * effect[u] for (u, w), c in B.items() if w == v)
    return effect[y]
