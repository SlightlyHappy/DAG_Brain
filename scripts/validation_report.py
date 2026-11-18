# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""
Reproducible validation study for DAG Brain's causal engine and estimators.

Run:  python scripts/validation_report.py            (writes docs/VALIDATION.md)
      python scripts/validation_report.py --quick    (smaller study, for CI)

Every number in docs/VALIDATION.md comes from this script with a fixed seed.
"""

from __future__ import annotations

import argparse
import itertools
import platform
import random
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import causal_core as cc  # noqa: E402
from backend import data_analysis as da  # noqa: E402
from tests.reference import (  # noqa: E402
    adjustment_valid_paths,
    backdoor_valid,
    dsep_moral,
    random_dag,
    total_effect_linear,
)

SEED = 20240613


def section_dsep(n_graphs):
    rng = random.Random(SEED)
    agree_ref = agree_nx = total = 0
    for _ in range(n_graphs):
        G = random_dag(rng.randint(4, 10), rng.uniform(0.15, 0.6), rng)
        nodes = list(G.nodes)
        for _ in range(5):
            x, y = rng.sample(nodes, 2)
            rest = [v for v in nodes if v not in (x, y)]
            z = set(rng.sample(rest, rng.randint(0, len(rest))))
            ours = cc.d_separated(cc.CausalGraph(G=G, latent=set()), {x}, {y}, z)
            agree_ref += ours == dsep_moral(G, {x}, {y}, z)
            agree_nx += ours == (nx.is_d_separator(G, {x}, {y}, z) if hasattr(nx, "is_d_separator") else nx.d_separated(G, {x}, {y}, z))
            total += 1
    return total, agree_ref, agree_nx


def section_adjustment(n_graphs):
    rng = random.Random(SEED + 1)
    total = agree = bd_accepted = bd_sound = 0
    exist_total = exist_agree = 0
    min_checked = min_ok = 0
    for _ in range(n_graphs):
        G = random_dag(rng.randint(4, 7), rng.uniform(0.25, 0.55), rng)
        x, y = sorted(rng.sample(list(G.nodes), 2), key=lambda v: int(v[1:]))
        cg = cc.CausalGraph(G=G, latent=set())
        rest = [v for v in G.nodes if v not in (x, y)]
        brute_exists = False
        for r in range(len(rest) + 1):
            for z in itertools.combinations(rest, r):
                z = set(z)
                ref = adjustment_valid_paths(G, x, y, z)
                brute_exists |= ref
                ours = cc.is_adjustment_set(cg, x, y, z)
                total += 1
                agree += ours == ref
                if backdoor_valid(G, x, y, z):
                    bd_accepted += 1
                    bd_sound += ours
        exist_total += 1
        exist_agree += cc.adjustment_exists(cg, x, y) == brute_exists
        for s in cc.minimal_sets(cg, x, y).sets:
            min_checked += 1
            min_ok += adjustment_valid_paths(G, x, y, set(s)) and all(
                not adjustment_valid_paths(G, x, y, set(s) - {v}) for v in s
            )
    return dict(total=total, agree=agree, bd_accepted=bd_accepted, bd_sound=bd_sound,
                exist_total=exist_total, exist_agree=exist_agree, min_checked=min_checked, min_ok=min_ok)


def _sem(G, rng):
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


def section_population(n_graphs):
    rng = random.Random(SEED + 2)
    valid_err = []
    invalid_bias = []
    for _ in range(n_graphs):
        G = random_dag(rng.randint(4, 8), rng.uniform(0.3, 0.6), rng)
        order, idx, B, coefs, omega = _sem(G, rng)
        x, y = rng.sample(order, 2)
        if idx[x] > idx[y]:
            x, y = y, x
        truth = total_effect_linear(coefs, order, x, y)
        inv = np.linalg.inv(np.eye(len(B)) - B)
        S = inv.T @ omega @ inv
        cg = cc.CausalGraph(G=G, latent=set())
        rest = [v for v in order if v not in (x, y)]
        cands = [set()] + [{v} for v in rest] + [set(rng.sample(rest, rng.randint(1, len(rest)))) for _ in range(8) if rest]
        for z in cands:
            w = [idx[x]] + [idx[v] for v in sorted(z)]
            coef = float(np.linalg.solve(S[np.ix_(w, w)], S[np.ix_(w, [idx[y]])])[0, 0])
            if cc.is_adjustment_set(cg, x, y, z):
                valid_err.append(abs(coef - truth))
            else:
                invalid_bias.append(abs(coef - truth))
    return np.array(valid_err), np.array(invalid_bias)


def section_coverage(reps, n):
    """95% CI coverage of OLS on engine-certified sets, across random graphs."""
    rng = random.Random(SEED + 3)
    rng_np = np.random.default_rng(SEED + 3)
    covered = total = 0
    for _ in range(reps):
        G = random_dag(rng.randint(5, 8), 0.4, rng)
        order, idx, B, coefs, omega = _sem(G, rng)
        x, y = order[1], order[-1]
        truth = total_effect_linear(coefs, order, x, y)
        sets = cc.minimal_sets(cc.CausalGraph(G=G, latent=set()), x, y).sets
        if not sets:
            continue
        z = sorted(sets[0])
        E = rng_np.normal(size=(n, len(order))) * np.sqrt(np.diag(omega))
        X = E @ np.linalg.inv(np.eye(len(B)) - B)
        D = np.column_stack([np.ones(n), X[:, idx[x]]] + [X[:, idx[v]] for v in z])
        beta, *_ = np.linalg.lstsq(D, X[:, idx[y]], rcond=None)
        resid = X[:, idx[y]] - D @ beta
        se = np.sqrt(resid @ resid / (n - D.shape[1]) * np.linalg.inv(D.T @ D)[1, 1])
        covered += abs(beta[1] - truth) <= 1.96 * se
        total += 1
    return covered, total


def section_dml(reps, n):
    rng = np.random.default_rng(SEED + 4)
    import pandas as pd

    covered, bias_dml, bias_naive = 0, [], []
    for _ in range(reps):
        C1 = rng.normal(size=n)
        C2 = rng.uniform(-2, 2, size=n)
        Xv = np.sin(C1) + 0.5 * C2 ** 2 + rng.normal(size=n)
        Yv = 1.5 * Xv + 2 * np.cos(C1) + C2 ** 2 + rng.normal(size=n)
        df = pd.DataFrame({"X": Xv, "Y": Yv, "C1": C1, "C2": C2})
        res = da.estimate_effect(df, {c: c for c in df.columns}, "X", "Y", ["C1", "C2"], bootstrap=0)
        d = res["estimators"]["dml"]
        covered += d["ci_low"] <= 1.5 <= d["ci_high"]
        bias_dml.append(d["estimate"] - 1.5)
        bias_naive.append(np.polyfit(Xv, Yv, 1)[0] - 1.5)
    return covered, reps, float(np.mean(bias_dml)), float(np.mean(bias_naive))


def section_ci_tests(reps, n):
    rng = np.random.default_rng(SEED + 5)
    out = {}
    for name, fn in (("partial correlation", da.partial_correlation_test), ("GCM (boosted trees)", da.gcm_test)):
        t1 = t1_nl = power = 0
        for _ in range(reps):
            Z = rng.normal(size=n)
            # Null, linear: X ⊥ Y | Z
            Xl = 0.8 * Z + rng.normal(size=n)
            Yl = -0.6 * Z + rng.normal(size=n)
            t1 += fn(Xl, Yl, Z.reshape(-1, 1))["p_value"] < 0.05
            # Null, non-linear in Z
            Xn = np.tanh(2 * Z) + rng.normal(scale=0.5, size=n)
            Yn = Z ** 2 + rng.normal(scale=0.5, size=n)
            t1_nl += fn(Xn, Yn, Z.reshape(-1, 1))["p_value"] < 0.05
            # Alternative: a direct link X -> Y of modest size
            Ya = Yl + 0.15 * Xl
            power += fn(Xl, Ya, Z.reshape(-1, 1))["p_value"] < 0.05
        out[name] = (t1 / reps, t1_nl / reps, power / reps)
    return out


def textbook_table():
    def g(edges, bi=()):
        nodes = {v for e in list(edges) + list(bi) for v in e}
        return cc.build_graph(nodes, edges, bi)

    def fmt(ss):
        return " or ".join("{" + ", ".join(sorted(s)) + "}" if s else "{} (no adjustment)" for s in ss) or "none"

    cases = [
        ("Simple confounding", g([("C", "X"), ("C", "Y"), ("X", "Y")]), "{C}"),
        ("M-bias", g([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")]), "{} (no adjustment); adjusting for M alone is invalid"),
        ("Butterfly", g([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("M", "X"), ("M", "Y"), ("X", "Y")]), "{A, M} or {B, M}"),
        ("Mediator", g([("X", "M"), ("M", "Y"), ("X", "Y")]), "{} (no adjustment); M is forbidden"),
        ("Front-door with latent confounding", g([("X", "M"), ("M", "Y")], [("X", "Y")]), "none (not identifiable by adjustment)"),
        ("Instrument Z with latent X–Y confounding", g([("Z", "X"), ("X", "Y")], [("X", "Y")]), "none; Z is an instrument candidate"),
    ]
    rows = []
    for name, cg, expected in cases:
        got = fmt([s for s in cc.minimal_sets(cg, "X", "Y").sets])
        ivs = ", ".join(i["id"] for i in cc.instrument_candidates(cg, "X", "Y")) or "none"
        rows.append(f"| {name} | {expected} | {got} | {ivs} |")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = args.quick
    t0 = time.time()

    n_d = 300 if q else 2000
    total, a_ref, a_nx = section_dsep(n_d)
    adj = section_adjustment(80 if q else 400)
    v_err, i_bias = section_population(100 if q else 600)
    cov, cov_n = section_coverage(150 if q else 1000, 2000)
    dml_cov, dml_n, dml_bias, naive_bias = section_dml(10 if q else 60, 1500)
    ci = section_ci_tests(60 if q else 300, 500)
    elapsed = time.time() - t0

    import scipy
    import sklearn

    lines = [
        "# Validation report",
        "",
        "*DAG Brain, created by SlightlyHappy. https://github.com/SlightlyHappy/DAG_Brain*",
        "",
        f"Generated by `scripts/validation_report.py`{' --quick' if q else ''} with seed {SEED}.  ",
        f"Python {platform.python_version()}, networkx {nx.__version__}, numpy {np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}. Runtime {elapsed:.0f} s.",
        "",
        "This report shows two things: that the engine's graph algorithms give the answers the theory defines, and that the estimators behave as their statistical theory says on data with a known true effect. It cannot show that any particular causal graph is true. That is a scientific judgement, which the app supports with evidence classes, testable implications and sensitivity analysis.",
        "",
        "## 1. d-separation",
        "",
        f"{total:,} random queries (random DAGs with 4–10 variables, random conditioning sets).",
        "",
        "| Compared with | Agreement |",
        "| :-- | --: |",
        f"| Independent ancestral moral-graph implementation (Lauritzen et al. 1990) | {a_ref:,}/{total:,} ({100 * a_ref / total:.2f}%) |",
        f"| `networkx.is_d_separator` | {a_nx:,}/{total:,} ({100 * a_nx / total:.2f}%) |",
        "",
        "## 2. Adjustment sets (total effect)",
        "",
        "The engine uses the generalised adjustment criterion (Shpitser, VanderWeele & Robins 2010; Perković et al. 2018). It is checked against an independent implementation of the criterion's original path-based definition, applied to every subset of variables in each graph.",
        "",
        "| Check | Result |",
        "| :-- | --: |",
        f"| Validity of every candidate set agrees with the path-based definition | {adj['agree']:,}/{adj['total']:,} ({100 * adj['agree'] / adj['total']:.2f}%) |",
        f"| Every set accepted by Pearl's back-door criterion is accepted (soundness) | {adj['bd_sound']:,}/{adj['bd_accepted']:,} |",
        f"| \"An adjustment set exists\" agrees with brute-force search | {adj['exist_agree']}/{adj['exist_total']} |",
        f"| Reported minimal sets are valid and inclusion-minimal | {adj['min_ok']}/{adj['min_checked']} |",
        "",
        "## 3. Exact check on linear structural causal models",
        "",
        "For random linear-Gaussian models the population covariance matrix is known exactly, so the population regression coefficient of the exposure, adjusted for any set, can be computed with no sampling noise. The true total effect is the sum over directed paths of the products of path coefficients.",
        "",
        "| Sets | Count | Result |",
        "| :-- | --: | :-- |",
        f"| Certified valid by the engine | {len(v_err):,} | max abs. error {v_err.max():.1e} (machine precision) |",
        f"| Rejected by the engine | {len(i_bias):,} | biased in {np.mean(i_bias > 1e-6) * 100:.1f}%; median abs. bias {np.median(i_bias):.3f} |",
        "",
        "## 4. Finite-sample coverage",
        "",
        f"Ordinary least squares adjusting for the engine's first minimal set, on {cov_n:,} independent simulated data sets (n = 2,000 each): the nominal 95% interval covered the true effect in **{100 * cov / cov_n:.1f}%** of runs.",
        "",
        "## 5. Double machine learning under non-linear confounding",
        "",
        f"True effect 1.5; confounders enter the exposure and the outcome non-linearly. {dml_n} replications, n = 1,500.",
        "",
        "| Estimator | Mean bias | 95% CI coverage |",
        "| :-- | --: | --: |",
        f"| Unadjusted regression | {naive_bias:+.3f} | not applicable |",
        f"| Partially linear DML, 5-fold cross-fitted gradient boosting | {dml_bias:+.3f} | {100 * dml_cov / dml_n:.1f}% |",
        "",
        "## 6. Conditional-independence tests (used for testable implications)",
        "",
        "Rejection rates at α = 0.05, n = 500. The type I error should be close to 5%; power should be high.",
        "",
        "| Test | Type I error, linear null | Type I error, non-linear null | Power, small direct link |",
        "| :-- | --: | --: | --: |",
    ]
    for name, (t1, t1nl, pw) in ci.items():
        lines.append(f"| {name} | {100 * t1:.1f}% | {100 * t1nl:.1f}% | {100 * pw:.1f}% |")
    lines += [
        "",
        "Partial correlation is exact for linear-Gaussian data but rejects true independencies far too often when the dependence on the conditioning set is non-linear. The generalised covariance measure (Shah & Peters 2020) brings that error close to the nominal level; it stays somewhat above 5% at n = 500 because the boosted regressions are imperfect, and it shrinks as n grows. The app therefore uses the GCM by default for n of at least 500 and corrects for multiple testing; treat a single borderline rejection with caution.",
        "",
        "## 7. Textbook graphs",
        "",
        "| Graph | Known answer | Engine's minimal sets | Instrument candidates |",
        "| :-- | :-- | :-- | :-- |",
        *textbook_table(),
        "",
        "## What this does not show",
        "",
        "- That a researcher's graph is correct. Identification is conditional on the graph, and the app reports the assumptions each result relies on.",
        "- That an instrument candidate is a valid instrument. The graph can rule a variable out, but it cannot confirm instrument strength or the exclusion restriction.",
        "- That a non-rejected implication confirms the graph. Tests can lack power, and different graphs can imply the same independencies.",
        "- The quality of AI-generated summaries or proposals. These are kept separate from the authoritative state and must be reviewed by the researcher.",
        "",
        "The automated test suite (`pytest`) runs smaller versions of every check above on each change.",
    ]
    out = ROOT / "docs" / "VALIDATION.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
