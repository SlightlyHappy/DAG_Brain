# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""Statistical behaviour of the data-analysis layer on simulated data with known truth."""

import math

import numpy as np
import pandas as pd
import pytest

from backend import data_analysis as da
from backend import causal_core as cc


def imp(a, b, given=()):
    return {"a": {"id": a, "label": a}, "b": {"id": b, "label": b}, "given": [{"id": g, "label": g} for g in given]}


@pytest.fixture(scope="module")
def chain_df():
    rng = np.random.default_rng(0)
    n = 3000
    A = rng.normal(size=n)
    B = np.sin(2 * A) + 0.5 * A + rng.normal(scale=0.5, size=n)  # non-linear link
    C = B ** 2 * 0.3 + B + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"A": A, "B": B, "C": C})


def test_true_independence_not_rejected(chain_df):
    mapping = {c: c for c in chain_df.columns}
    res = da.check_implications(chain_df, mapping, [imp("A", "C", ["B"])], method="gcm")
    assert res["tested"] == 1 and res["rejected"] == 0


def test_false_independence_rejected(chain_df):
    mapping = {c: c for c in chain_df.columns}
    res = da.check_implications(chain_df, mapping, [imp("A", "C")], method="partial_correlation")
    assert res["rejected"] == 1


def test_wrong_graph_detected_end_to_end():
    # Truth: A -> B -> C plus A -> C. Researcher's graph omits A -> C, so it implies A ⊥ C | B.
    rng = np.random.default_rng(1)
    n = 4000
    A = rng.normal(size=n)
    B = 0.8 * A + rng.normal(size=n)
    C = 0.7 * B + 0.5 * A + rng.normal(size=n)
    df = pd.DataFrame({"A": A, "B": B, "C": C})
    wrong = cc.build_graph(["A", "B", "C"], [("A", "B"), ("B", "C")])
    implications = [imp(i["a"], i["b"], i["given"]) for i in cc.implied_independencies(wrong)]
    res = da.check_implications(df, {c: c for c in df.columns}, implications)
    assert res["rejected"] == 1


def test_gcm_type_one_error_is_controlled():
    rng = np.random.default_rng(2)
    rejections = 0
    trials = 40
    for _ in range(trials):
        n = 600
        Z = rng.normal(size=n)
        X = np.tanh(Z) + rng.normal(scale=0.6, size=n)
        Y = Z ** 2 + rng.normal(scale=0.6, size=n)
        p = da.gcm_test(X, Y, Z.reshape(-1, 1))["p_value"]
        rejections += p < 0.05
    assert rejections / trials <= 0.15  # nominal 5%, allowing Monte Carlo error


def test_multiple_testing_corrections():
    p = [0.01, 0.04, 0.03, 0.5]
    holm = da.holm(p)
    bh = da.benjamini_hochberg(p)
    assert holm == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert bh == pytest.approx([0.04, 0.0533333, 0.0533333, 0.5], rel=1e-4)


def test_e_value_matches_published_example():
    # VanderWeele & Ding (2017): RR = 3.9 gives E-value 7.26; lower CI 1.8 gives 3.0.
    ev = da.e_value(3.9, 1.8, 8.7)
    assert ev["point"] == pytest.approx(7.26, abs=0.01)
    assert ev["ci_limit"] == pytest.approx(3.0, abs=0.01)
    assert da.e_value(1.2, 0.9, 1.6)["ci_limit"] == 1.0


def test_dml_continuous_exposure_recovers_effect_under_nonlinear_confounding():
    rng = np.random.default_rng(3)
    n = 4000
    C1 = rng.normal(size=n)
    C2 = rng.uniform(-2, 2, size=n)
    X = np.sin(C1) + 0.5 * C2 ** 2 + rng.normal(size=n)
    Y = 1.5 * X + np.cos(C1) * 2 + C2 ** 2 + rng.normal(size=n)
    df = pd.DataFrame({"X": X, "Y": Y, "C1": C1, "C2": C2})
    res = da.estimate_effect(df, {c: c for c in df.columns}, "X", "Y", ["C1", "C2"], bootstrap=50)
    dml = res["estimators"]["dml"]
    assert dml["ci_low"] <= 1.5 <= dml["ci_high"]
    unadjusted = np.polyfit(X, Y, 1)[0]
    assert abs(unadjusted - 1.5) > 0.3  # confounding is real


def test_aipw_binary_exposure_and_outcome():
    rng = np.random.default_rng(4)
    n = 6000
    C = rng.normal(size=n)
    p_t = 1 / (1 + np.exp(-(0.8 * C)))
    T = rng.binomial(1, p_t)
    p_y = 1 / (1 + np.exp(-(-1 + 0.9 * T + 1.2 * C)))
    Y = rng.binomial(1, p_y)
    # True ATE by simulation of potential outcomes.
    ate = np.mean(1 / (1 + np.exp(-(-1 + 0.9 + 1.2 * C))) - 1 / (1 + np.exp(-(-1 + 1.2 * C))))
    df = pd.DataFrame({"T": T, "Y": Y, "C": C})
    res = da.estimate_effect(df, {c: c for c in df.columns}, "T", "Y", ["C"], bootstrap=20)
    dml = res["estimators"]["dml"]
    assert dml["ci_low"] - 0.01 <= ate <= dml["ci_high"] + 0.01
    assert "risk_ratio" in dml and res["e_value"]["point"] > 1
    assert 0 < dml["overlap"]["propensity_min"] < dml["overlap"]["propensity_max"] < 1


def test_auto_map_matches_labels():
    class N:
        def __init__(self, i, l):
            self.id, self.label = i, l
    m = da.auto_map([N("sleep_dep", "Sleep Deprivation"), N("x", "Age (years)")], ["sleep deprivation", "AGE_YEARS"])
    assert m == {"sleep_dep": "sleep deprivation", "x": "AGE_YEARS"}
