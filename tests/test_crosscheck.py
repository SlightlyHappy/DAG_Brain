"""Agreement with independent reference implementations on random DAGs."""

import itertools
import random

import networkx as nx
import pytest

from backend import causal_core as cc
from tests.reference import adjustment_valid_paths, backdoor_valid, dsep_moral, random_dag

N_GRAPHS = 500


def _cg(G):
    return cc.CausalGraph(G=G, latent=set())


@pytest.mark.parametrize("seed", range(5))
def test_d_separation_matches_moral_graph(seed):
    rng = random.Random(seed)
    checks = 0
    for _ in range(N_GRAPHS // 5):
        G = random_dag(rng.randint(4, 9), rng.uniform(0.2, 0.5), rng)
        nodes = list(G.nodes)
        x, y = rng.sample(nodes, 2)
        rest = [v for v in nodes if v not in (x, y)]
        z = set(rng.sample(rest, rng.randint(0, len(rest))))
        assert cc.d_separated(_cg(G), {x}, {y}, z) == dsep_moral(G, {x}, {y}, z), (G.edges, x, y, z)
        checks += 1
    assert checks == N_GRAPHS // 5


@pytest.mark.parametrize("seed", range(5))
def test_adjustment_criterion_matches_path_definition(seed):
    rng = random.Random(100 + seed)
    for _ in range(N_GRAPHS // 5):
        G = random_dag(rng.randint(4, 7), rng.uniform(0.25, 0.55), rng)
        x, y = sorted(rng.sample(list(G.nodes), 2), key=lambda v: int(v[1:]))
        rest = [v for v in G.nodes if v not in (x, y)]
        for r in range(0, min(3, len(rest)) + 1):
            for z in itertools.combinations(rest, r):
                z = set(z)
                engine = cc.is_adjustment_set(_cg(G), x, y, z)
                ref = adjustment_valid_paths(G, x, y, z)
                assert engine == ref, (sorted(G.edges), x, y, z)
                # Back-door is sound: anything it accepts must be valid.
                if backdoor_valid(G, x, y, z):
                    assert engine


@pytest.mark.parametrize("seed", range(3))
def test_minimal_sets_are_valid_minimal_and_existence_is_exact(seed):
    rng = random.Random(200 + seed)
    for _ in range(60):
        G = random_dag(rng.randint(4, 8), rng.uniform(0.25, 0.5), rng)
        x, y = sorted(rng.sample(list(G.nodes), 2), key=lambda v: int(v[1:]))
        cg = _cg(G)
        found = cc.minimal_sets(cg, x, y).sets
        for s in found:
            assert adjustment_valid_paths(G, x, y, set(s))
            for v in s:
                assert not adjustment_valid_paths(G, x, y, set(s) - {v}), "set is not minimal"
        # Existence: brute force over all subsets of non-forbidden nodes.
        rest = [v for v in G.nodes if v not in (x, y)]
        brute = any(
            adjustment_valid_paths(G, x, y, set(z))
            for r in range(len(rest) + 1)
            for z in itertools.combinations(rest, r)
        )
        assert cc.adjustment_exists(cg, x, y) == brute
        assert bool(found) == brute


@pytest.mark.skipif(not hasattr(nx, "find_minimal_d_separator"), reason="needs networkx 3.3+")
def test_networkx_find_minimal_separator_agrees():
    rng = random.Random(7)
    for _ in range(100):
        G = random_dag(rng.randint(4, 9), 0.35, rng)
        a, b = rng.sample(list(G.nodes), 2)
        if G.has_edge(a, b) or G.has_edge(b, a):
            continue
        sep = nx.find_minimal_d_separator(G, {a}, {b})
        assert sep is not None and dsep_moral(G, {a}, {b}, sep)


def test_fallbacks_match_networkx_free_paths(monkeypatch):
    """The engine must give identical answers when networkx lacks the 3.3 d-separation API."""
    rng = random.Random(11)
    monkeypatch.setattr(cc, "_NX_DSEP", None)
    monkeypatch.delattr(nx, "find_minimal_d_separator", raising=False)
    for _ in range(200):
        G = random_dag(rng.randint(4, 9), rng.uniform(0.2, 0.5), rng)
        nodes = list(G.nodes)
        x, y = rng.sample(nodes, 2)
        rest = [v for v in nodes if v not in (x, y)]
        z = set(rng.sample(rest, rng.randint(0, len(rest))))
        cg = _cg(G)
        assert cc.d_separated(cg, {x}, {y}, z) == dsep_moral(G, {x}, {y}, z)
        if not (G.has_edge(x, y) or G.has_edge(y, x)):
            sep = cc.minimal_separator(cg, x, y, set(rest))
            assert sep is not None and dsep_moral(G, {x}, {y}, sep)
            assert all(not dsep_moral(G, {x}, {y}, sep - {v}) for v in sep)
