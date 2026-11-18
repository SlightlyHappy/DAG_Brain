"""
Statistical validation against linear structural causal models with known
coefficients.

1. Exact (population) check: for a linear SEM the covariance matrix is
   Σ = (I - B)^-T Ω (I - B)^-1. The population OLS coefficient of x in
   y ~ x + Z is computed exactly from Σ, with no sampling noise. Every set the
   engine calls valid must reproduce the true total effect to machine
   precision; for random coefficients, sets it calls invalid must not.
2. Finite-sample check: data drawn from the same models; valid sets recover the
   truth within 5 standard errors.
3. Hand-built examples for latent confounding and the single-door criterion.
"""

import random

import networkx as nx
import numpy as np
import pytest

from backend import causal_core as cc
from tests.reference import random_dag, total_effect_linear


def random_sem(G, rng):
    order = list(nx.topological_sort(G))
    idx = {v: i for i, v in enumerate(order)}
    B = np.zeros((len(order), len(order)))
    coefs = {}
    for u, v in G.edges:
        c = rng.choice([-1, 1]) * rng.uniform(0.3, 1.0)
        B[idx[u], idx[v]] = c
        coefs[(u, v)] = c
    omega = np.diag([rng.uniform(0.5, 1.5) for _ in order])
    return order, idx, B, coefs, omega


def population_cov(B, omega):
    inv = np.linalg.inv(np.eye(len(B)) - B)
    return inv.T @ omega @ inv


def population_coef(S, idx, x, y, z):
    w = [idx[x]] + [idx[v] for v in sorted(z)]
    beta = np.linalg.solve(S[np.ix_(w, w)], S[np.ix_(w, [idx[y]])])
    return float(beta[0, 0])


def sample(order, idx, B, omega, n, rng_np):
    E = rng_np.normal(size=(n, len(order))) * np.sqrt(np.diag(omega))
    return E @ np.linalg.inv(np.eye(len(B)) - B)  # rows: X = E (I - B)^-1


def ols(Xmat, idx, x, y, z):
    cols = [idx[x]] + [idx[v] for v in sorted(z)]
    D = np.column_stack([np.ones(len(Xmat))] + [Xmat[:, c] for c in cols])
    beta, *_ = np.linalg.lstsq(D, Xmat[:, idx[y]], rcond=None)
    resid = Xmat[:, idx[y]] - D @ beta
    s2 = resid @ resid / (len(resid) - D.shape[1])
    return beta[1], float(np.sqrt(s2 * np.linalg.inv(D.T @ D)[1, 1]))


def _subsets(rest, rng, k=12):
    out = [set()] + [{v} for v in rest]
    for _ in range(k):
        out.append(set(rng.sample(rest, rng.randint(1, len(rest)))))
    return out


@pytest.mark.parametrize("seed", range(10))
def test_population_valid_sets_exact_invalid_sets_biased(seed):
    rng = random.Random(seed)
    n_valid = n_invalid = biased = 0
    for _ in range(30):
        G = random_dag(rng.randint(4, 8), rng.uniform(0.3, 0.6), rng)
        order, idx, B, coefs, omega = random_sem(G, rng)
        x, y = rng.sample(order, 2)
        if idx[x] > idx[y]:
            x, y = y, x
        truth = total_effect_linear(coefs, order, x, y)
        S = population_cov(B, omega)
        cg = cc.CausalGraph(G=G, latent=set())
        rest = [v for v in order if v not in (x, y)]
        if not rest:
            continue
        for z in _subsets(rest, rng):
            coef = population_coef(S, idx, x, y, z)
            if cc.is_adjustment_set(cg, x, y, z):
                n_valid += 1
                assert abs(coef - truth) < 1e-9, (sorted(G.edges), x, y, z, coef, truth)
            else:
                n_invalid += 1
                if abs(coef - truth) > 1e-6:
                    biased += 1
    assert n_valid > 20
    # Exact cancellations have probability zero for continuous random coefficients.
    assert biased == n_invalid, f"{n_invalid - biased} invalid sets were unbiased"


@pytest.mark.parametrize("seed", range(4))
def test_finite_sample_valid_sets_recover_truth(seed):
    rng = random.Random(1000 + seed)
    rng_np = np.random.default_rng(seed)
    checked = 0
    for _ in range(5):
        G = random_dag(rng.randint(5, 8), 0.4, rng)
        order, idx, B, coefs, omega = random_sem(G, rng)
        x, y = order[1], order[-1]
        truth = total_effect_linear(coefs, order, x, y)
        data = sample(order, idx, B, omega, 30000, rng_np)
        cg = cc.CausalGraph(G=G, latent=set())
        for s in cc.minimal_sets(cg, x, y).sets:
            est, se = ols(data, idx, x, y, set(s))
            assert abs(est - truth) < 5 * se + 1e-3, (sorted(G.edges), x, y, s, est, truth)
            checked += 1
    assert checked > 0


def test_latent_confounder_front_door_naive_regression_biased():
    rng_np = np.random.default_rng(1)
    n = 50000
    U = rng_np.normal(size=n)
    X = 0.9 * U + rng_np.normal(size=n)
    M = 0.8 * X + rng_np.normal(size=n)
    Y = 0.7 * M + 1.1 * U + rng_np.normal(size=n)
    cg = cc.build_graph(["X", "M", "Y"], [("X", "M"), ("M", "Y")], [("X", "Y")])
    assert not cc.adjustment_exists(cg, "X", "Y")
    D = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(D, Y, rcond=None)
    assert abs(beta[1] - 0.56) > 0.2  # the engine is right to refuse


def test_direct_effect_single_door_recovers_coefficient():
    rng_np = np.random.default_rng(2)
    n = 60000
    L = rng_np.normal(size=n)
    X = rng_np.normal(size=n)
    M = 0.8 * X + 0.9 * L + rng_np.normal(size=n)
    Y = 0.5 * X + 0.6 * M + 0.7 * L + rng_np.normal(size=n)
    cg = cc.build_graph(["X", "M", "L", "Y"], [("X", "M"), ("M", "Y"), ("X", "Y"), ("L", "M"), ("L", "Y")])
    data = np.column_stack([X, M, L, Y])
    idx = {"X": 0, "M": 1, "L": 2, "Y": 3}
    assert cc.is_single_door_set(cg, "X", "Y", {"M", "L"})
    est, se = ols(data, idx, "X", "Y", {"M", "L"})
    assert abs(est - 0.5) < 5 * se
    assert not cc.is_single_door_set(cg, "X", "Y", {"M"})
    est_bad, se_bad = ols(data, idx, "X", "Y", {"M"})
    assert abs(est_bad - 0.5) > 5 * se_bad


@pytest.mark.parametrize("seed", range(4))
def test_single_door_population_exact(seed):
    rng = random.Random(500 + seed)
    for _ in range(30):
        G = random_dag(rng.randint(4, 7), 0.45, rng)
        order, idx, B, coefs, omega = random_sem(G, rng)
        x, y = rng.sample(order, 2)
        if idx[x] > idx[y]:
            x, y = y, x
        truth = coefs.get((x, y), 0.0)
        S = population_cov(B, omega)
        cg = cc.CausalGraph(G=G, latent=set())
        rest = [v for v in order if v not in (x, y)]
        for z in _subsets(rest, rng, 6) if rest else [set()]:
            if cc.is_single_door_set(cg, x, y, z):
                assert abs(population_coef(S, idx, x, y, z) - truth) < 1e-9
