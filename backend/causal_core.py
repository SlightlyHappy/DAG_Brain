"""
Pure graph-theoretic causal identification on DAGs with latent confounding.

Everything here is deterministic and independent of the app's data model, so it
can be tested against textbook graphs, an independent d-separation
implementation, and simulated data (see tests/ and docs/VALIDATION.md).

References
----------
- Pearl, J. (2009). Causality, 2nd ed. (d-separation, single-door criterion, 5.3.1)
- Shpitser, VanderWeele & Robins (2010). On the validity of covariate adjustment
  for estimating causal effects. UAI.
- Perković, Textor, Kalisch & Maathuis (2018). Complete graphical
  characterization and construction of adjustment sets. JMLR 18(220).
- van der Zander, Liśkiewicz & Textor (2019). Separators and adjustment sets in
  causal graphs: complete criteria and an algorithmic framework. AIJ 270.
- Brito & Pearl (2002); van der Zander, Textor & Liśkiewicz (2015) on
  (conditional) instrumental variables.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import networkx as nx

LATENT_PREFIX = "__U__"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


@dataclass
class CausalGraph:
    """A DAG over observed and latent variables.

    Bidirected edges ``a <-> b`` (an unmeasured common cause) are represented
    explicitly as ``a <- U_ab -> b`` with ``U_ab`` latent, which keeps every
    computation inside ordinary DAG theory.
    """

    G: nx.DiGraph
    latent: Set[str] = field(default_factory=set)

    @property
    def observed(self) -> Set[str]:
        return set(self.G.nodes) - self.latent

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.G)

    def cycles(self) -> List[List[str]]:
        return list(nx.simple_cycles(self.G)) if not self.is_dag() else []

    def de(self, v: str) -> Set[str]:
        """Descendants of v, including v itself."""
        return nx.descendants(self.G, v) | {v}

    def an(self, vs: Iterable[str]) -> Set[str]:
        """Ancestors of a set, including the set itself."""
        out: Set[str] = set()
        for v in vs:
            out |= nx.ancestors(self.G, v) | {v}
        return out


def build_graph(
    nodes: Iterable[str],
    directed: Iterable[Tuple[str, str]],
    bidirected: Iterable[Tuple[str, str]] = (),
    latent: Iterable[str] = (),
) -> CausalGraph:
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    lat = set(latent)
    for a, b in directed:
        if a in G and b in G and a != b:
            G.add_edge(a, b)
    for a, b in bidirected:
        if a in G and b in G and a != b:
            u = f"{LATENT_PREFIX}{min(a, b)}__{max(a, b)}"
            G.add_edge(u, a)
            G.add_edge(u, b)
            lat.add(u)
    return CausalGraph(G=G, latent=lat)


# ---------------------------------------------------------------------------
# d-separation
# ---------------------------------------------------------------------------


def _dsep_moral(G: nx.DiGraph, xs: Set[str], ys: Set[str], zs: Set[str]) -> bool:
    """d-separation via the moralised ancestral graph (Lauritzen et al. 1990).

    Used when the installed networkx predates ``is_d_separator`` (3.3), so the
    engine gives the same answers on any networkx version.
    """
    keep: Set[str] = set()
    for v in xs | ys | zs:
        keep |= nx.ancestors(G, v) | {v}
    A = G.subgraph(keep)
    M = nx.Graph()
    M.add_nodes_from(A.nodes)
    M.add_edges_from((u, v) for u, v in A.edges)
    for v in A.nodes:
        M.add_edges_from(itertools.combinations(list(A.predecessors(v)), 2))
    M.remove_nodes_from(zs)
    for comp in nx.connected_components(M):
        if comp & xs and comp & ys:
            return False
    return True


_NX_DSEP = getattr(nx, "is_d_separator", None)


def d_separated(cg: CausalGraph, xs: Set[str], ys: Set[str], zs: Set[str]) -> bool:
    """True iff every path between xs and ys is blocked by zs (Pearl 2009)."""
    if xs & ys:
        return False
    xs, ys, zs = set(xs) - set(zs), set(ys) - set(zs), set(zs)
    if not xs or not ys:
        return True
    if _NX_DSEP is not None:
        return _NX_DSEP(cg.G, xs, ys, zs)
    return _dsep_moral(cg.G, xs, ys, zs)


def minimal_separator(cg: CausalGraph, a: str, b: str, restricted: Set[str]) -> Optional[Set[str]]:
    """A minimal set within ``restricted`` that d-separates a and b, or None.

    Uses networkx's ``find_minimal_d_separator`` when available (3.3+).
    Otherwise: if any separator inside ``restricted`` exists, the ancestral one
    An({a, b}) ∩ restricted does; it is then pruned one variable at a time
    while separation holds.
    """
    finder = getattr(nx, "find_minimal_d_separator", None)
    if finder is not None:
        try:
            sep = finder(cg.G, {a}, {b}, included=set(), restricted=set(restricted))
        except (nx.NetworkXError, nx.NodeNotFound):
            return None
        return None if sep is None else set(sep)
    z = (cg.an({a, b}) & set(restricted)) - {a, b}
    if not d_separated(cg, {a}, {b}, z):
        return None
    for v in sorted(z):
        if d_separated(cg, {a}, {b}, z - {v}):
            z = z - {v}
    return z


# ---------------------------------------------------------------------------
# Total effect: generalised adjustment criterion (complete)
# ---------------------------------------------------------------------------


def causal_nodes(cg: CausalGraph, x: str, y: str) -> Set[str]:
    """cn(x, y): nodes other than x on proper causal paths from x to y."""
    if y not in cg.de(x):
        return set()
    return (cg.de(x) & cg.an({y})) - {x}


def forbidden(cg: CausalGraph, x: str, y: str) -> Set[str]:
    """Forb(x, y) = De(cn(x, y)) ∪ {x}  (Perković et al. 2018)."""
    forb = {x}
    for w in causal_nodes(cg, x, y):
        forb |= cg.de(w)
    return forb


def proper_backdoor_graph(cg: CausalGraph, x: str, y: str) -> CausalGraph:
    """Remove the first edge of every proper causal path from x to y."""
    G = cg.G.copy()
    cn = causal_nodes(cg, x, y)
    for w in list(G.successors(x)):
        if w in cn:
            G.remove_edge(x, w)
    return CausalGraph(G=G, latent=set(cg.latent))


def is_adjustment_set(cg: CausalGraph, x: str, y: str, z: Set[str]) -> bool:
    """Generalised adjustment criterion for the total effect of x on y.

    Z is valid iff (a) Z contains no node of Forb(x, y), and (b) Z d-separates
    x and y in the proper backdoor graph. Sound and complete for DAGs
    (Shpitser et al. 2010; Perković et al. 2018).
    """
    z = set(z)
    if z & cg.latent or x in z or y in z:
        return False
    if z & forbidden(cg, x, y):
        return False
    return d_separated(proper_backdoor_graph(cg, x, y), {x}, {y}, z)


def canonical_adjustment_set(cg: CausalGraph, x: str, y: str) -> Set[str]:
    """Adjust(x, y) = An({x, y}) ∩ observed − ({x, y} ∪ Forb).

    A valid adjustment set exists iff this set is valid
    (van der Zander, Liśkiewicz & Textor 2019).
    """
    return (cg.an({x, y}) & cg.observed) - ({x, y} | forbidden(cg, x, y))


def adjustment_exists(cg: CausalGraph, x: str, y: str) -> bool:
    return is_adjustment_set(cg, x, y, canonical_adjustment_set(cg, x, y))


# ---------------------------------------------------------------------------
# Direct effect: single-door criterion (linear SEM)
# ---------------------------------------------------------------------------


def forbidden_direct(cg: CausalGraph, y: str) -> Set[str]:
    """Variables the single-door criterion forbids: y and its descendants."""
    return cg.de(y)


def is_single_door_set(cg: CausalGraph, x: str, y: str, z: Set[str]) -> bool:
    """Single-door criterion (Pearl 2009, Thm 5.3.1).

    In a linear SEM the path coefficient of x -> y equals the regression
    coefficient of x in y ~ x + Z iff Z contains no descendant of y and Z
    d-separates x and y in the graph with the edge x -> y removed. If there is
    no edge x -> y the direct effect is zero and the same test applies.
    """
    z = set(z)
    if z & cg.latent or x in z or y in z:
        return False
    if z & cg.de(y):
        return False
    G = cg.G.copy()
    if G.has_edge(x, y):
        G.remove_edge(x, y)
    return d_separated(CausalGraph(G=G, latent=cg.latent), {x}, {y}, z)


# ---------------------------------------------------------------------------
# Minimal set enumeration
# ---------------------------------------------------------------------------


@dataclass
class SetSearch:
    sets: List[FrozenSet[str]]
    complete: bool
    candidates: List[str]


def minimal_sets(
    cg: CausalGraph,
    x: str,
    y: str,
    estimand: str = "total_effect",
    max_size: int = 8,
    max_checks: int = 60000,
) -> SetSearch:
    """Enumerate all inclusion-minimal valid sets, smallest first.

    Minimal separators lie inside An({x, y}) (van der Zander et al. 2019), so
    candidates are restricted to observed ancestors that are allowed by the
    criterion. Reports ``complete=False`` if the search budget ran out.
    """
    if estimand == "direct_effect":
        valid = lambda s: is_single_door_set(cg, x, y, s)  # noqa: E731
        excluded = cg.de(y)
    else:
        valid = lambda s: is_adjustment_set(cg, x, y, s)  # noqa: E731
        excluded = forbidden(cg, x, y)
    candidates = sorted((cg.an({x, y}) & cg.observed) - excluded - {x, y})

    found: List[FrozenSet[str]] = []
    checks = 0
    complete = True
    for r in range(0, min(len(candidates), max_size) + 1):
        for combo in itertools.combinations(candidates, r):
            s = frozenset(combo)
            if any(f <= s for f in found):
                continue
            checks += 1
            if checks > max_checks:
                return SetSearch(found, False, candidates)
            if valid(set(s)):
                found.append(s)
    if len(candidates) > max_size:
        complete = False
    return SetSearch(found, complete, candidates)


# ---------------------------------------------------------------------------
# Instrumental variable candidates
# ---------------------------------------------------------------------------


def _without_x_out_edges(cg: CausalGraph, x: str) -> CausalGraph:
    G = cg.G.copy()
    G.remove_edges_from([(x, w) for w in list(G.successors(x))])
    return CausalGraph(G=G, latent=set(cg.latent))


def instrument_candidates(
    cg: CausalGraph, x: str, y: str, max_cond: int = 2
) -> List[Dict[str, object]]:
    """Graphical (conditional) instrument candidates for x -> y.

    Z qualifies given W when: W contains no descendant of x; Z is d-connected
    to x given W (relevance is graphically possible); and Z is d-separated from
    y given W once x's outgoing edges are removed (exclusion and independence).
    These are *necessary graphical conditions under the assumed DAG*. They do
    not establish instrument strength or that the DAG itself is right.
    """
    if x not in cg.G or y not in cg.G:
        return []
    cut = _without_x_out_edges(cg, x)
    desc_x = cg.de(x)
    pool = sorted((cg.observed - desc_x) - {x, y})
    out: List[Dict[str, object]] = []
    for zv in pool:
        others = [w for w in pool if w != zv]
        for r in range(0, max_cond + 1):
            hit = None
            for w in itertools.combinations(others, r):
                ws = set(w)
                if d_separated(cg, {zv}, {x}, ws):
                    continue
                if d_separated(cut, {zv}, {y}, ws):
                    hit = sorted(ws)
                    break
            if hit is not None:
                out.append({"id": zv, "conditional_on": hit})
                break
    return out


# ---------------------------------------------------------------------------
# Structural roles
# ---------------------------------------------------------------------------


def structural_roles(cg: CausalGraph, x: str, y: str) -> Dict[str, str]:
    """Role of each observed variable relative to (x, y), read off the graph."""
    G_no_x = cg.G.copy()
    G_no_x.remove_node(x)
    anc_y_not_via_x = set(nx.ancestors(G_no_x, y)) if y in G_no_x else set()
    anc_x = nx.ancestors(cg.G, x)
    de_x = nx.descendants(cg.G, x)
    de_y = nx.descendants(cg.G, y)
    anc_y = nx.ancestors(cg.G, y)

    roles: Dict[str, str] = {}
    for v in cg.observed:
        if v == x:
            roles[v] = "exposure"
        elif v == y:
            roles[v] = "outcome"
        elif v in de_x and v in anc_y:
            roles[v] = "mediator"
        elif v in de_y:
            roles[v] = "descendant_of_outcome"
        elif v in de_x:
            roles[v] = "descendant_of_exposure"
        elif v in anc_x and v in anc_y_not_via_x:
            roles[v] = "confounder"
        elif v in anc_x:
            roles[v] = "instrument"
        elif v in anc_y:
            roles[v] = "competing_exposure"
        else:
            roles[v] = "other"
    return roles


# ---------------------------------------------------------------------------
# Testable implications
# ---------------------------------------------------------------------------


def _adjacent(cg: CausalGraph, a: str, b: str) -> bool:
    if cg.G.has_edge(a, b) or cg.G.has_edge(b, a):
        return True
    # Sharing a latent parent counts as adjacency for observed variables.
    pa_a = set(cg.G.predecessors(a)) & cg.latent
    pa_b = set(cg.G.predecessors(b)) & cg.latent
    return bool(pa_a & pa_b)


def implied_independencies(cg: CausalGraph) -> List[Dict[str, object]]:
    """For each non-adjacent pair of observed variables, one minimal observed
    separating set (if any). Every listed statement a ⊥ b | S is implied by
    the DAG and can be tested against data (cf. DAGitty's
    impliedConditionalIndependencies)."""
    obs = sorted(cg.observed)
    out: List[Dict[str, object]] = []
    for a, b in itertools.combinations(obs, 2):
        if _adjacent(cg, a, b):
            continue
        sep = minimal_separator(cg, a, b, cg.observed - {a, b})
        if sep is None:
            continue
        out.append({"a": a, "b": b, "given": sorted(sep)})
    return out


# ---------------------------------------------------------------------------
# Path listing (for explanation only; never used to decide validity)
# ---------------------------------------------------------------------------


def explain_paths(
    cg: CausalGraph, x: str, y: str, z: Set[str], limit: int = 400
) -> Tuple[List[Dict[str, object]], bool]:
    """List paths between x and y with per-path blocking status.

    Latent nodes are collapsed so ``a <- U -> b`` is shown as ``a <-> b``.
    Returns (paths, truncated). Validity decisions use d_separated(), not this.
    """
    UG = cg.G.to_undirected(as_view=True)
    z = set(z)
    an_z = cg.an(z) if z else set()
    out: List[Dict[str, object]] = []
    truncated = False
    for i, path in enumerate(nx.all_simple_paths(UG, x, y)):
        if i >= limit:
            truncated = True
            break
        arrows = ["->" if cg.G.has_edge(path[k], path[k + 1]) else "<-" for k in range(len(path) - 1)]
        colliders, blockers, opened = [], [], []
        for k in range(1, len(path) - 1):
            v = path[k]
            is_col = arrows[k - 1] == "->" and arrows[k] == "<-"
            if is_col:
                colliders.append(v)
                if v in an_z:
                    opened.append(v)
                else:
                    blockers.append(v)
            elif v in z:
                blockers.append(v)
        # Collapse latent forks into bidirected display edges.
        shown, shown_arrows = [path[0]], []
        k = 0
        while k < len(arrows):
            nxt = path[k + 1]
            if nxt in cg.latent and k + 1 < len(arrows):
                shown_arrows.append("<->")
                shown.append(path[k + 2])
                k += 2
            else:
                shown_arrows.append(arrows[k])
                shown.append(nxt)
                k += 1
        out.append(
            {
                "nodes": shown,
                "arrows": shown_arrows,
                "is_directed_causal": all(a == "->" for a in arrows),
                "starts_into_x": arrows[0] == "<-",
                "colliders": [c for c in colliders if c not in cg.latent],
                "opened_colliders": [c for c in opened if c not in cg.latent],
                "blocking_nodes": [b for b in blockers if b not in cg.latent],
                "is_open": not blockers,
            }
        )
    return out, truncated
