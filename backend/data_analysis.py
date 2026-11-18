"""
Data-driven checks of a causal graph and effect estimation.

1. Testable implications. Every conditional independence the DAG implies is
   tested against the data with:
   - partial correlation and Fisher's z (exact under joint normality), and
   - the Generalised Covariance Measure (Shah & Peters 2020): residuals from
     cross-fitted gradient-boosted regressions, so non-linear dependence on
     the conditioning set is handled.
   P-values are corrected with Holm (family-wise) and Benjamini-Hochberg (FDR).
   A non-rejection is consistency with the data, never proof of the graph.

2. Effect estimation, only with an adjustment set the engine has certified:
   - regression adjustment (OLS with HC3 robust errors, or logistic g-computation),
   - double/debiased machine learning (Chernozhukov et al. 2018): the
     partially linear model for a continuous exposure, and the augmented
     inverse-probability-weighted (AIPW) estimator for a binary exposure,
     both with 5-fold cross-fitting.

3. Sensitivity to unmeasured confounding: E-values (VanderWeele & Ding 2017).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SEED = 20240613


# ---------------------------------------------------------------------------
# Loading and profiling
# ---------------------------------------------------------------------------


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")


def column_kind(s: pd.Series) -> str:
    vals = s.dropna().unique()
    if len(vals) <= 2:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        return "continuous"
    return "categorical"


def profile(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [
        {
            "name": c,
            "kind": column_kind(df[c]),
            "missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)),
        }
        for c in df.columns
    ]


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def auto_map(nodes: Sequence[Any], columns: Sequence[str]) -> Dict[str, str]:
    """Match graph variables to columns by id or label, ignoring case and punctuation."""
    by_norm = {_norm(c): c for c in columns}
    mapping: Dict[str, str] = {}
    for n in nodes:
        for key in (n.id, n.label):
            col = by_norm.get(_norm(key))
            if col:
                mapping[n.id] = col
                break
    return mapping


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    s = df[col]
    if column_kind(s) == "binary" and not pd.api.types.is_numeric_dtype(s):
        levels = sorted(s.dropna().unique().tolist(), key=str)
        return (s == levels[-1]).astype(float).to_numpy()
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)


def _design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    """Design matrix for conditioning variables; categoricals are one-hot encoded."""
    if not cols:
        return np.zeros((len(df), 0))
    parts = []
    for c in cols:
        if column_kind(df[c]) == "categorical":
            parts.append(pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True).to_numpy(dtype=float))
        else:
            parts.append(_numeric(df, c).reshape(-1, 1))
    return np.hstack(parts)


# ---------------------------------------------------------------------------
# Conditional independence tests
# ---------------------------------------------------------------------------


def partial_correlation_test(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> Dict[str, float]:
    n = len(a)
    D = np.column_stack([np.ones(n), Z]) if Z.size else np.ones((n, 1))
    ra = a - D @ np.linalg.lstsq(D, a, rcond=None)[0]
    rb = b - D @ np.linalg.lstsq(D, b, rcond=None)[0]
    r = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0
    r = max(min(r, 0.999999), -0.999999)
    dof = n - D.shape[1] - 2
    z = math.atanh(r) * math.sqrt(max(dof, 1))
    p = float(2 * stats.norm.sf(abs(z)))
    return {"statistic": z, "p_value": p, "partial_r": r}


def _crossfit_residuals(y: np.ndarray, Z: np.ndarray, folds: int = 5, seed: int = SEED) -> np.ndarray:
    if Z.size == 0:
        return y - y.mean()
    res = np.zeros_like(y, dtype=float)
    for train, test in KFold(folds, shuffle=True, random_state=seed).split(Z):
        model = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed)
        model.fit(Z[train], y[train])
        res[test] = y[test] - model.predict(Z[test])
    return res


def gcm_test(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> Dict[str, float]:
    """Generalised Covariance Measure (Shah & Peters 2020) with cross-fitted boosting."""
    ra = _crossfit_residuals(a, Z)
    rb = _crossfit_residuals(b, Z)
    R = ra * rb
    sd = R.std(ddof=1)
    T = float(math.sqrt(len(R)) * R.mean() / sd) if sd > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(T)))
    r = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 0 and rb.std() > 0 else 0.0
    return {"statistic": T, "p_value": p, "partial_r": r}


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i], reverse=True)
    adj = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        k = m - rank
        running = min(running, pvals[i] * m / k)
        adj[i] = min(1.0, running)
    return adj


def check_implications(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    implications: List[Dict[str, Any]],
    method: str = "auto",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test each implied independence a ⊥ b | S whose variables are all mapped."""
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for imp in implications:
        a, b = imp["a"]["id"], imp["b"]["id"]
        given = [g["id"] for g in imp["given"]]
        needed = [a, b, *given]
        missing = [v for v in needed if v not in mapping]
        label = f"{imp['a']['label']} ⊥ {imp['b']['label']}" + (
            f" | {', '.join(g['label'] for g in imp['given'])}" if given else ""
        )
        if missing:
            skipped.append({"implication": label, "reason": "unmapped: " + ", ".join(missing)})
            continue
        cols = [mapping[v] for v in needed]
        sub = df[cols].dropna()
        n = len(sub)
        if n < 30:
            skipped.append({"implication": label, "reason": f"only {n} complete rows"})
            continue
        av, bv = _numeric(sub, mapping[a]), _numeric(sub, mapping[b])
        Z = _design(sub, [mapping[g] for g in given])
        use = method if method != "auto" else ("gcm" if n >= 500 and Z.shape[1] > 0 else "partial_correlation")
        res = gcm_test(av, bv, Z) if use == "gcm" else partial_correlation_test(av, bv, Z)
        rows.append({"implication": label, "a": a, "b": b, "given": given, "n": n, "method": use, **res})

    if rows:
        p = [r["p_value"] for r in rows]
        for r, ph, pb in zip(rows, holm(p), benjamini_hochberg(p)):
            r["p_holm"] = ph
            r["p_bh"] = pb
            r["rejected"] = ph < alpha
    n_rej = sum(1 for r in rows if r["rejected"])
    return {
        "alpha": alpha,
        "tested": len(rows),
        "rejected": n_rej,
        "results": sorted(rows, key=lambda r: r["p_value"]),
        "skipped": skipped,
        "verdict": (
            "no tested implication is contradicted by the data"
            if rows and n_rej == 0
            else (f"{n_rej} implication(s) contradicted by the data: the graph is missing an edge or the tests' assumptions fail" if rows else "nothing could be tested")
        ),
        "interpretation": "A rejected implication shows the data disagree with the graph. A non-rejection does not prove the graph: tests can lack power and different graphs can imply the same independencies.",
    }


# ---------------------------------------------------------------------------
# Effect estimation
# ---------------------------------------------------------------------------


def _ols_hc3(y: np.ndarray, X: np.ndarray) -> Dict[str, float]:
    n = len(y)
    D = np.column_stack([np.ones(n), X])
    XtX_inv = np.linalg.pinv(D.T @ D)
    beta = XtX_inv @ D.T @ y
    resid = y - D @ beta
    h = np.einsum("ij,jk,ik->i", D, XtX_inv, D)
    w = (resid / np.clip(1 - h, 1e-8, None)) ** 2
    cov = XtX_inv @ (D.T * w) @ D @ XtX_inv
    se = float(math.sqrt(max(cov[1, 1], 0)))
    est = float(beta[1])
    return {"estimate": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se}


def _dml_plm(y: np.ndarray, x: np.ndarray, Z: np.ndarray, folds: int = 5) -> Dict[str, float]:
    """Partially linear model θ in y = θ x + g(Z) + e, cross-fitted (Robinson / DML)."""
    ry = _crossfit_residuals(y, Z, folds)
    rx = _crossfit_residuals(x, Z, folds)
    theta = float((rx @ ry) / (rx @ rx))
    psi = (ry - theta * rx) * rx
    J = float(np.mean(rx * rx))
    se = float(math.sqrt(np.mean(psi ** 2) / (J ** 2) / len(y)))
    return {"estimate": theta, "se": se, "ci_low": theta - 1.96 * se, "ci_high": theta + 1.96 * se}


def _aipw(y: np.ndarray, t: np.ndarray, Z: np.ndarray, binary_y: bool, folds: int = 5, trim: float = 0.01) -> Dict[str, Any]:
    """AIPW estimate of E[Y(1)] - E[Y(0)] with cross-fitted nuisances."""
    n = len(y)
    Zm = Z if Z.size else np.zeros((n, 1))
    e_hat = np.zeros(n)
    m1 = np.zeros(n)
    m0 = np.zeros(n)
    splitter = StratifiedKFold(folds, shuffle=True, random_state=SEED)
    for train, test in splitter.split(Zm, t):
        ps = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=SEED)
        ps.fit(Zm[train], t[train])
        e_hat[test] = ps.predict_proba(Zm[test])[:, 1]
        for arm, out in ((1, m1), (0, m0)):
            idx = train[t[train] == arm]
            if binary_y:
                if len(np.unique(y[idx])) < 2:
                    out[test] = y[idx].mean()
                    continue
                mdl = GradientBoostingClassifier(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=SEED)
                mdl.fit(Zm[idx], y[idx])
                out[test] = mdl.predict_proba(Zm[test])[:, 1]
            else:
                mdl = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=SEED)
                mdl.fit(Zm[idx], y[idx])
                out[test] = mdl.predict(Zm[test])
    raw_range = (float(e_hat.min()), float(e_hat.max()))
    trimmed = int(np.sum((e_hat < trim) | (e_hat > 1 - trim)))
    e = np.clip(e_hat, trim, 1 - trim)
    phi1 = m1 + t * (y - m1) / e
    phi0 = m0 + (1 - t) * (y - m0) / (1 - e)
    mu1, mu0 = float(phi1.mean()), float(phi0.mean())
    psi = phi1 - phi0
    ate = float(psi.mean())
    se = float(psi.std(ddof=1) / math.sqrt(n))
    out: Dict[str, Any] = {
        "estimate": ate, "se": se, "ci_low": ate - 1.96 * se, "ci_high": ate + 1.96 * se,
        "mean_treated_potential": mu1, "mean_control_potential": mu0,
        "overlap": {"propensity_min": raw_range[0], "propensity_max": raw_range[1], "clipped_units": trimmed, "clip_at": trim},
    }
    if binary_y and mu0 > 0 and mu1 > 0:
        # Delta-method CI for log risk ratio from the influence functions.
        ifl = (phi1 - mu1) / mu1 - (phi0 - mu0) / mu0
        se_log = float(ifl.std(ddof=1) / math.sqrt(n))
        rr = mu1 / mu0
        out["risk_ratio"] = {"estimate": rr, "ci_low": rr * math.exp(-1.96 * se_log), "ci_high": rr * math.exp(1.96 * se_log)}
    return out


def e_value(rr: float, ci_low: Optional[float] = None, ci_high: Optional[float] = None) -> Dict[str, Optional[float]]:
    """E-value for a risk ratio and its confidence limit closest to the null (VanderWeele & Ding 2017)."""

    def ev(r: float) -> float:
        r = 1 / r if r < 1 else r
        return r + math.sqrt(r * (r - 1))

    point = ev(rr) if rr > 0 else None
    limit = None
    if ci_low is not None and ci_high is not None:
        if ci_low <= 1 <= ci_high:
            limit = 1.0
        else:
            limit = ev(ci_low if rr >= 1 else ci_high)
    return {"point": point, "ci_limit": limit}


def smd_to_rr(d: float) -> float:
    """Approximate risk ratio for a standardised mean difference (VanderWeele 2017: RR ≈ exp(0.91 d))."""
    return math.exp(0.91 * d)


def estimate_effect(
    df: pd.DataFrame,
    mapping: Dict[str, str],
    x: str,
    y: str,
    adjustment: List[str],
    bootstrap: int = 200,
) -> Dict[str, Any]:
    cols = [mapping[v] for v in [x, y, *adjustment]]
    sub = df[cols].dropna()
    n = len(sub)
    if n < 50:
        raise ValueError(f"Only {n} complete rows; at least 50 are needed.")
    xv = _numeric(sub, mapping[x])
    yv = _numeric(sub, mapping[y])
    Z = _design(sub, [mapping[a] for a in adjustment])
    x_kind = column_kind(sub[mapping[x]])
    y_kind = column_kind(sub[mapping[y]])
    binary_x = x_kind == "binary"
    binary_y = y_kind == "binary"
    if binary_x:
        xv = (xv == np.nanmax(xv)).astype(float)
    if binary_y:
        yv = (yv == np.nanmax(yv)).astype(float)

    result: Dict[str, Any] = {
        "n": n,
        "exposure_type": x_kind,
        "outcome_type": y_kind,
        "adjusted_for": adjustment,
        "estimators": {},
    }

    # 1. Regression adjustment.
    reg = _ols_hc3(yv, np.column_stack([xv, Z]) if Z.size else xv.reshape(-1, 1))
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(bootstrap):
        idx = rng.integers(0, n, n)
        Xb = np.column_stack([xv[idx], Z[idx]]) if Z.size else xv[idx].reshape(-1, 1)
        Db = np.column_stack([np.ones(n), Xb])
        boots.append(np.linalg.lstsq(Db, yv[idx], rcond=None)[0][1])
    reg["bootstrap_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    reg["label"] = "Linear regression adjustment (HC3 robust SE, percentile bootstrap)"
    reg["meaning"] = (
        "Difference in mean outcome between exposed and unexposed, holding the adjustment set fixed (assumes a linear outcome model)."
        if binary_x
        else "Change in mean outcome per one-unit increase in the exposure (assumes linearity)."
    )
    result["estimators"]["regression"] = reg

    # 2. Double machine learning.
    if binary_x:
        aipw = _aipw(yv, xv.astype(int), Z, binary_y)
        aipw["label"] = "AIPW with cross-fitted gradient boosting (double machine learning)"
        aipw["meaning"] = "Average treatment effect: mean outcome if everyone were exposed minus if no one were."
        result["estimators"]["dml"] = aipw
    else:
        plm = _dml_plm(yv, xv, Z if Z.size else np.zeros((n, 0)))
        plm["label"] = "Partially linear model, cross-fitted gradient boosting (double machine learning)"
        plm["meaning"] = "Average change in outcome per unit of exposure, with confounders entering flexibly."
        result["estimators"]["dml"] = plm

    # 3. Sensitivity analysis.
    dml = result["estimators"]["dml"]
    if binary_y and "risk_ratio" in dml:
        rr = dml["risk_ratio"]
        result["e_value"] = {**e_value(rr["estimate"], rr["ci_low"], rr["ci_high"]), "scale": "risk ratio"}
    else:
        sd_y = float(np.std(yv, ddof=1)) or 1.0
        scale_x = 1.0 if binary_x else float(np.std(xv, ddof=1)) or 1.0
        d = dml["estimate"] * scale_x / sd_y
        lo = dml["ci_low"] * scale_x / sd_y
        hi = dml["ci_high"] * scale_x / sd_y
        result["e_value"] = {
            **e_value(smd_to_rr(d), smd_to_rr(lo), smd_to_rr(hi)),
            "scale": "approximate risk ratio from the standardised effect" + ("" if binary_x else " per SD of exposure"),
        }
    result["e_value"]["meaning"] = (
        "The minimum strength of association, on the risk-ratio scale, that an unmeasured confounder would need "
        "with both exposure and outcome to fully explain away the estimate (point) or its confidence limit."
    )
    result["assumptions"] = [
        "The adjustment set is valid for this DAG, and the DAG is correct.",
        "No unmeasured confounding beyond what the DAG encodes (probe this with the E-value).",
        "Positivity: check the propensity range for binary exposures.",
        "Variables are measured without important error and units do not interfere.",
    ]
    return result
