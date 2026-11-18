"""Evaluation metrics on cases with known answers."""

from backend import dag_eval as ev


def G(nodes, edges, roles=None):
    roles = roles or {}
    return {
        "nodes": [{"id": n, "label": n, "role": roles.get(n, "other")} for n in nodes],
        "edges": [
            {"source": e[0], "target": e[1], "edge_type": "bidirected" if len(e) > 2 and e[2] == "bi" else "directed"}
            for e in edges
        ],
    }


ROLES = {"X": "exposure", "Y": "outcome"}
REF = G(["C", "X", "M", "Y"], [("C", "X"), ("C", "Y"), ("X", "M"), ("M", "Y")], ROLES)


def test_identical_graph_is_perfect():
    r = ev.compare(REF, REF)
    assert r["skeleton"]["f1"] == 1 and r["oriented"]["f1"] == 1 and r["shd"] == 0
    assert r["aid_ancestor"]["errors"] == 0 and r["aid_parent"]["errors"] == 0
    assert r["target"]["analysis_correct"] and r["target"]["first_set_valid"]


def test_reversed_edge_counts_once_in_shd():
    cand = G(["C", "X", "M", "Y"], [("C", "X"), ("C", "Y"), ("M", "X"), ("M", "Y")], ROLES)
    r = ev.compare(cand, REF)
    assert r["shd"] == 1 and r["skeleton"]["f1"] == 1 and r["oriented"]["f1"] < 1
    assert r["edges"]["reversed"] == ["M -> X"]


def test_missing_confounder_is_an_inferential_error():
    # Dropping C -> Y loses one arrow but makes the candidate wrongly say "no adjustment needed".
    cand = G(["C", "X", "M", "Y"], [("C", "X"), ("X", "M"), ("M", "Y")], ROLES)
    r = ev.compare(cand, REF)
    assert r["shd"] == 1
    assert not r["target"]["analysis_correct"]
    assert r["target"]["candidate_sets"] == [[]]
    assert r["target"]["candidate_sets_valid_in_reference"] == 0
    # AID's ancestor strategy adjusts for every ancestor and so happens to include C here,
    # which is why the target-level check is reported alongside AID.


def test_aid_detects_wrong_parent_adjustment():
    # Candidate misses C -> X, so the parent set of X ({}) no longer blocks X <- C -> Y.
    cand = G(["C", "X", "M", "Y"], [("C", "Y"), ("X", "M"), ("M", "Y")], ROLES)
    r = ev.compare(cand, REF)
    assert ["X", "Y"] in [list(p) for p in r["aid_parent"]["examples"]]
    assert r["aid_parent"]["errors"] > 0 and not r["target"]["analysis_correct"]


def test_many_edge_errors_can_still_give_the_right_adjustment():
    # Extra irrelevant structure among outcome descendants does not change X -> Y adjustment.
    ref = G(["C", "X", "Y", "D1", "D2"], [("C", "X"), ("C", "Y"), ("X", "Y"), ("Y", "D1"), ("D1", "D2")], ROLES)
    cand = G(["C", "X", "Y", "D1", "D2"], [("C", "X"), ("C", "Y"), ("X", "Y"), ("Y", "D2"), ("D2", "D1")], ROLES)
    r = ev.compare(cand, ref)
    assert r["shd"] >= 2
    assert r["target"]["analysis_correct"] and r["target"]["first_set_valid"]


def test_node_matching_by_label_and_similarity():
    cand = {
        "nodes": [{"id": "smoking", "label": "Maternal smoking"}, {"id": "lbw", "label": "Low birth weight"}, {"id": "zzz", "label": "Unrelated"}],
        "edges": [],
    }
    ref = {
        "nodes": [{"id": "maternal_smoking", "label": "Maternal Smoking"}, {"id": "birthweight", "label": "Low birthweight"}],
        "edges": [],
    }
    m = ev.match_nodes(cand["nodes"], ref["nodes"])
    assert m == {"smoking": "maternal_smoking", "lbw": "birthweight"}
    r = ev.compare(cand, ref)
    assert r["nodes"]["extra"] == ["zzz"] and r["nodes"]["missing"] == []


def test_missing_and_extra_variables():
    cand = G(["X", "Y", "Q"], [("X", "Y"), ("Q", "Y")], ROLES)
    r = ev.compare(cand, REF)
    assert set(r["nodes"]["missing"]) == {"C", "M"}
    assert r["nodes"]["extra"] == ["Q"]
    assert r["skeleton"]["recall"] < 1


def test_bidirected_edges_compared_as_latent_confounding():
    ref = G(["X", "Y", "M"], [("X", "M"), ("M", "Y"), ("X", "Y", "bi")], ROLES)
    cand = G(["X", "Y", "M"], [("X", "M"), ("M", "Y")], ROLES)
    r = ev.compare(cand, ref)
    assert r["edges"]["missing"] == ["X - Y"]
    assert r["target"]["reference_identifiable"] is False
    assert r["target"]["candidate_identifiable"] is True
    assert not r["target"]["analysis_correct"]


def test_cyclic_candidate_is_reported_not_crashed():
    cand = G(["X", "Y"], [("X", "Y"), ("Y", "X")], ROLES)
    r = ev.compare(cand, G(["X", "Y"], [("X", "Y")], ROLES))
    assert r["is_dag"] is False and "aid_ancestor" not in r


def test_summary_means():
    a = ev.compare(REF, REF)
    b = ev.compare(G(["C", "X", "M", "Y"], [("C", "X"), ("X", "M"), ("M", "Y")], ROLES), REF)
    s = ev.summarise([a, b])
    assert s["target_analysis_correct"] == 0.5 and 0 < s["skeleton_f1"] < 1
