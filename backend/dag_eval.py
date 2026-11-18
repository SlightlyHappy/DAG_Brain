# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
"""
Evaluation of a candidate causal graph (for example an AI-drafted DAG) against a
reference graph (for example one built by domain experts).

Two kinds of agreement are measured, because they answer different questions:

1. Structural agreement: are the same arrows present?
   - skeleton precision, recall and F1 (adjacencies, ignoring direction),
   - oriented-edge precision, recall and F1,
   - structural Hamming distance (SHD): insertions + deletions + reversals.

2. Inferential agreement: would the candidate lead to the right analysis?
   - adjustment identification distance (AID; Henckel, Würtzen & Weichwald
     2024): over all ordered pairs of variables, the share for which the
     identification strategy read off the candidate (claim of no effect,
     an adjustment set, or "not identifiable") is wrong in the reference,
     computed with parent-based and ancestor-based adjustment sets;
   - for the study's own exposure and outcome: does every minimal adjustment
     set of the candidate remain valid in the reference, and do the two
     graphs agree on identifiability?

Two graphs can differ in many arrows yet support the same analysis, or differ
in one arrow and lead to a biased estimate. Reporting both is the point.

Node correspondence: variables are matched by normalised id or label, then by
string similarity, and the mapping can be overridden. Reference variables the
candidate lacks are added to it as isolated variables (the candidate asserts
nothing about them). Candidate variables the reference lacks are kept as
unobserved, so paths through them still count but they cannot be adjusted for.
"""

from __future__ import annotations

import difflib
import itertools
import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx

from . import causal_core as cc


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", s or "")
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def match_nodes(
    candidate: List[Dict[str, Any]],
    reference: List[Dict[str, Any]],
    manual: Optional[Dict[str, str]] = None,
    threshold: float = 0.82,
) -> Dict[str, str]:
    """Map candidate node ids to reference node ids (one-to-one)."""
    mapping: Dict[str, str] = {}
    used: Set[str] = set()
    ref_ids = {r["id"] for r in reference}
    for c_id, r_id in (manual or {}).items():
        if r_id in ref_ids and r_id not in used:
            mapping[c_id] = r_id
            used.add(r_id)
    keys = {r["id"]: {_norm(r["id"]), _norm(r.get("label", ""))} - {""} for r in reference}
    for c in candidate:
        if c["id"] in mapping:
            continue
        ck = {_norm(c["id"]), _norm(c.get("label", ""))} - {""}
        for r in reference:
            if r["id"] not in used and ck & keys[r["id"]]:
                mapping[c["id"]] = r["id"]
                used.add(r["id"])
                break
    scored = []
    for c in candidate:
        if c["id"] in mapping:
            continue
        for r in reference:
            if r["id"] in used:
                continue
            score = max(
                difflib.SequenceMatcher(None, _norm(c.get("label", c["id"])), _norm(r.get("label", r["id"]))).ratio(),
                difflib.SequenceMatcher(None, _norm(c["id"]), _norm(r["id"])).ratio(),
            )
            if score >= threshold:
                scored.append((score, c["id"], r["id"]))
    for score, c_id, r_id in sorted(scored, reverse=True):
        if c_id not in mapping and r_id not in used:
            mapping[c_id] = r_id
            used.add(r_id)
    return mapping


def _edge_sets(edges: Iterable[Dict[str, Any]], tr: Dict[str, str]) -> Tuple[Set[Tuple[str, str]], Set[frozenset]]:
    directed, bidirected = set(), set()
    for e in edges:
        s, t = tr.get(e["source"], e["source"]), tr.get(e["target"], e["target"])
        if s == t:
            continue
        if e.get("edge_type", "directed") == "bidirected":
            bidirected.add(frozenset((s, t)))
        else:
            directed.add((s, t))
    return directed, bidirected


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def _build(nodes: Iterable[str], directed, bidirected, latent: Iterable[str]) -> cc.CausalGraph:
    return cc.build_graph(list(nodes), list(directed), [tuple(b) for b in bidirected], latent)


def _strategy(g: cc.CausalGraph, x: str, y: str, kind: str):
    """Identification strategy read off a graph for the effect of x on y."""
    if y not in nx.descendants(g.G, x):
        return ("zero", None)
    if kind == "parent":
        z = set(g.G.predecessors(x)) & g.observed
        if cc.is_adjustment_set(g, x, y, z):
            return ("adjust", z)
    z = cc.canonical_adjustment_set(g, x, y)
    if cc.is_adjustment_set(g, x, y, z):
        return ("adjust", z)
    return ("none", None)


def _strategy_correct(true_g: cc.CausalGraph, x: str, y: str, strat) -> bool:
    kind, z = strat
    if kind == "zero":
        return y not in nx.descendants(true_g.G, x)
    if kind == "adjust":
        z = set(z) & true_g.observed
        return cc.is_adjustment_set(true_g, x, y, z)
    return not cc.adjustment_exists(true_g, x, y)


def aid(true_g: cc.CausalGraph, guess_g: cc.CausalGraph, nodes: List[str], kind: str = "ancestor") -> Dict[str, Any]:
    """Adjustment identification distance over ordered pairs of observed nodes."""
    errors, total = 0, 0
    worst: List[Tuple[str, str]] = []
    for x, y in itertools.permutations(nodes, 2):
        total += 1
        if not _strategy_correct(true_g, x, y, _strategy(guess_g, x, y, kind)):
            errors += 1
            if len(worst) < 25:
                worst.append((x, y))
    return {"errors": errors, "pairs": total, "distance": errors / total if total else 0.0, "examples": worst}


def compare(
    candidate: Dict[str, Any],
    reference: Dict[str, Any],
    manual_mapping: Optional[Dict[str, str]] = None,
    exposure: Optional[str] = None,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare two graphs given as {"nodes": [...], "edges": [...]} dicts.

    Exposure and outcome are reference ids; by default they are read from the
    reference nodes' roles.
    """
    c_nodes, r_nodes = candidate["nodes"], reference["nodes"]
    mapping = match_nodes(c_nodes, r_nodes, manual_mapping)
    ref_ids = [r["id"] for r in r_nodes]
    ref_latent = {r["id"] for r in r_nodes if r.get("observed") is False or r.get("role") == "latent"}
    extra = [c["id"] for c in c_nodes if c["id"] not in mapping]
    missing = [r for r in ref_ids if r not in set(mapping.values())]
    tr = {**mapping, **{e: f"extra::{e}" for e in extra}}

    c_dir, c_bi = _edge_sets(candidate["edges"], tr)
    r_dir, r_bi = _edge_sets(reference["edges"], {})

    def skel(d, b):
        return {frozenset(e) for e in d} | set(b)

    cs, rs = skel(c_dir, c_bi), skel(r_dir, r_bi)
    skeleton = _prf(len(cs & rs), len(cs - rs), len(rs - cs))
    oriented_c = {("d",) + e for e in c_dir} | {("b",) + tuple(sorted(b)) for b in c_bi}
    oriented_r = {("d",) + e for e in r_dir} | {("b",) + tuple(sorted(b)) for b in r_bi}
    oriented = _prf(len(oriented_c & oriented_r), len(oriented_c - oriented_r), len(oriented_r - oriented_c))
    reversed_edges = [(s, t) for (s, t) in c_dir if (t, s) in r_dir and (s, t) not in r_dir]
    missing_adj = rs - cs
    extra_adj = cs - rs
    def mark(adj: frozenset, d: Set[Tuple[str, str]], b: Set[frozenset]) -> Tuple[str, ...]:
        u, v = sorted(adj)
        if adj in b and (u, v) not in d and (v, u) not in d:
            return ("<->",)
        return tuple(sorted(m for m in (("->",) if (u, v) in d else ()) + (("<-",) if (v, u) in d else ()))) + (("<->",) if adj in b else ())

    reoriented = [a for a in cs & rs if mark(a, c_dir, c_bi) != mark(a, r_dir, r_bi)]
    # SHD: adjacencies to add or delete, plus shared adjacencies whose orientation differs.
    shd = len(missing_adj) + len(extra_adj) + len(reoriented)

    ref_g = _build(ref_ids, r_dir, r_bi, ref_latent)
    cand_node_ids = ref_ids + [f"extra::{e}" for e in extra]
    cand_latent = {f"extra::{e}" for e in extra} | ref_latent
    cand_g = _build(cand_node_ids, c_dir, c_bi, cand_latent)

    result: Dict[str, Any] = {
        "nodes": {
            "reference": len(r_nodes),
            "candidate": len(c_nodes),
            "matched": len(mapping),
            "missing": missing,
            "extra": extra,
            "mapping": mapping,
        },
        "skeleton": skeleton,
        "oriented": oriented,
        "shd": shd,
        "edges": {
            "correct": sorted(f"{s} -> {t}" for (_, s, t) in [e for e in oriented_c & oriented_r if e[0] == "d"]),
            "reversed": sorted(f"{s} -> {t}" for s, t in reversed_edges),
            "missing": sorted(" - ".join(sorted(a)) for a in missing_adj),
            "extra": sorted(" - ".join(sorted(a)) for a in extra_adj),
        },
        "is_dag": cand_g.is_dag(),
    }

    observed = [n for n in ref_ids if n not in ref_latent]
    if cand_g.is_dag() and ref_g.is_dag():
        result["aid_parent"] = aid(ref_g, cand_g, observed, "parent")
        result["aid_ancestor"] = aid(ref_g, cand_g, observed, "ancestor")

    x = exposure or next((r["id"] for r in r_nodes if r.get("role") == "exposure"), None)
    y = outcome or next((r["id"] for r in r_nodes if r.get("role") == "outcome"), None)
    if x and y and cand_g.is_dag() and ref_g.is_dag():
        ref_sets = [set(s) for s in cc.minimal_sets(ref_g, x, y).sets]
        cand_sets = [set(s) for s in cc.minimal_sets(cand_g, x, y).sets]
        valid_in_ref = [s for s in cand_sets if cc.is_adjustment_set(ref_g, x, y, s & ref_g.observed)]
        ref_ident = cc.adjustment_exists(ref_g, x, y)
        cand_ident = bool(cand_sets) or cc.adjustment_exists(cand_g, x, y)
        first_valid = bool(cand_sets) and cc.is_adjustment_set(ref_g, x, y, cand_sets[0] & ref_g.observed)
        result["target"] = {
            "exposure": x,
            "outcome": y,
            "reference_identifiable": ref_ident,
            "candidate_identifiable": cand_ident,
            "reference_sets": [sorted(s) for s in ref_sets],
            "candidate_sets": [sorted(s) for s in cand_sets],
            "candidate_sets_valid_in_reference": len(valid_in_ref),
            "candidate_sets_total": len(cand_sets),
            "first_set_valid": first_valid,
            # What a researcher using the candidate would do: adjust for its first
            # minimal set, or conclude the effect is not identifiable by adjustment.
            "analysis_correct": first_valid if cand_ident else not ref_ident,
        }
    return result


def summarise(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Mean of the headline metrics over several comparisons."""
    keys = {
        "skeleton_f1": lambda r: r["skeleton"]["f1"],
        "oriented_f1": lambda r: r["oriented"]["f1"],
        "shd": lambda r: r["shd"],
        "aid_ancestor": lambda r: r.get("aid_ancestor", {}).get("distance"),
        "aid_parent": lambda r: r.get("aid_parent", {}).get("distance"),
        "target_analysis_correct": lambda r: float(r["target"]["analysis_correct"]) if "target" in r else None,
    }
    out = {}
    for k, f in keys.items():
        vals = [v for v in (f(r) for r in results) if v is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out
