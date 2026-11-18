"""Known answers from the causal inference literature."""

from backend import causal_core as cc


def g(edges, bidirected=(), latent=()):
    nodes = {v for e in list(edges) + list(bidirected) for v in e} | set(latent)
    return cc.build_graph(nodes, edges, bidirected, latent)


def sets(cg, x, y, estimand="total_effect"):
    return {frozenset(s) for s in cc.minimal_sets(cg, x, y, estimand).sets}


def test_simple_confounding():
    cg = g([("C", "X"), ("C", "Y"), ("X", "Y")])
    assert not cc.is_adjustment_set(cg, "X", "Y", set())
    assert cc.is_adjustment_set(cg, "X", "Y", {"C"})
    assert sets(cg, "X", "Y") == {frozenset({"C"})}


def test_m_bias_empty_set_valid_and_collider_invalid():
    # X <- A -> M <- B -> Y ; X -> Y   (Greenland, Pearl & Robins 1999)
    cg = g([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("X", "Y")])
    assert cc.is_adjustment_set(cg, "X", "Y", set())
    assert not cc.is_adjustment_set(cg, "X", "Y", {"M"})
    assert cc.is_adjustment_set(cg, "X", "Y", {"M", "A"})
    assert cc.is_adjustment_set(cg, "X", "Y", {"M", "B"})
    assert sets(cg, "X", "Y") == {frozenset()}


def test_butterfly_bias():
    # M is both a confounder and a collider: adjust for M plus A or B.
    cg = g([("A", "X"), ("A", "M"), ("B", "M"), ("B", "Y"), ("M", "X"), ("M", "Y"), ("X", "Y")])
    assert not cc.is_adjustment_set(cg, "X", "Y", set())
    assert not cc.is_adjustment_set(cg, "X", "Y", {"M"})
    assert sets(cg, "X", "Y") == {frozenset({"M", "A"}), frozenset({"M", "B"})}


def test_mediator_is_forbidden_for_total_effect():
    cg = g([("X", "M"), ("M", "Y"), ("X", "Y")])
    assert cc.is_adjustment_set(cg, "X", "Y", set())
    assert not cc.is_adjustment_set(cg, "X", "Y", {"M"})
    assert "M" in cc.forbidden(cg, "X", "Y")


def test_descendant_of_mediator_is_forbidden():
    cg = g([("X", "M"), ("M", "Y"), ("M", "D")])
    assert not cc.is_adjustment_set(cg, "X", "Y", {"D"})


def test_front_door_has_no_adjustment_set():
    # X <-> Y through a latent U, X -> M -> Y  (Pearl 1995)
    cg = g([("X", "M"), ("M", "Y")], bidirected=[("X", "Y")])
    assert not cc.adjustment_exists(cg, "X", "Y")
    assert sets(cg, "X", "Y") == set()


def test_bidirected_edge_is_latent_confounding_not_causation():
    cg = g([("X", "Y")], bidirected=[("C", "Y")])
    # C <-> Y must not make C a cause of X or a descendant of anything.
    assert cc.is_adjustment_set(cg, "X", "Y", set())
    cg2 = g([("X", "Y")], bidirected=[("X", "Y")])
    assert not cc.adjustment_exists(cg2, "X", "Y")


def test_instrument_candidate_and_role():
    # Z -> X -> Y with X <-> Y: the textbook instrument.
    cg = g([("Z", "X"), ("X", "Y")], bidirected=[("X", "Y")])
    ivs = cc.instrument_candidates(cg, "X", "Y")
    assert [iv["id"] for iv in ivs] == ["Z"]
    assert cc.structural_roles(cg, "X", "Y")["Z"] == "instrument"


def test_instrument_role_not_confounder_when_x_causes_y():
    # Regression: ancestors of X were all labelled "confounder".
    cg = g([("Z", "X"), ("C", "X"), ("C", "Y"), ("X", "Y")])
    roles = cc.structural_roles(cg, "X", "Y")
    assert roles["Z"] == "instrument"
    assert roles["C"] == "confounder"


def test_invalid_instrument_with_direct_path_to_outcome():
    cg = g([("Z", "X"), ("Z", "Y"), ("X", "Y")])
    assert cc.instrument_candidates(cg, "X", "Y") == []


def test_conditional_instrument():
    # Z <- W -> Y ; Z -> X -> Y : Z is an instrument given W.
    cg = g([("W", "Z"), ("W", "Y"), ("Z", "X"), ("X", "Y")])
    ivs = {iv["id"]: iv["conditional_on"] for iv in cc.instrument_candidates(cg, "X", "Y")}
    assert ivs.get("Z") == ["W"]


def test_long_backdoor_path_beyond_old_cutoff():
    # Regression: path search was capped at 10 edges, hiding this open path.
    chain = [f"c{i}" for i in range(12)]
    edges = [("X", "Y"), (chain[0], "X")]
    edges += [(chain[i + 1], chain[i]) for i in range(len(chain) - 1)]
    edges += [(chain[-1], "Y")]
    cg = g(edges)
    assert not cc.is_adjustment_set(cg, "X", "Y", set())
    assert cc.is_adjustment_set(cg, "X", "Y", {"c5"})


def test_direct_effect_single_door():
    # X -> M -> Y, X -> Y : the direct effect needs M; the total effect forbids it.
    cg = g([("X", "M"), ("M", "Y"), ("X", "Y")])
    assert cc.is_single_door_set(cg, "X", "Y", {"M"})
    assert not cc.is_single_door_set(cg, "X", "Y", set())


def test_direct_effect_with_mediator_outcome_confounder():
    # X -> M -> Y, X -> Y, M <- L -> Y : conditioning on M opens X -> M <- L -> Y.
    cg = g([("X", "M"), ("M", "Y"), ("X", "Y"), ("L", "M"), ("L", "Y")])
    assert not cc.is_single_door_set(cg, "X", "Y", {"M"})
    assert cc.is_single_door_set(cg, "X", "Y", {"M", "L"})


def test_implied_independencies_of_a_chain():
    cg = g([("A", "B"), ("B", "C")])
    imp = cc.implied_independencies(cg)
    assert imp == [{"a": "A", "b": "C", "given": ["B"]}]


def test_no_implication_between_latently_confounded_pair():
    cg = g([("A", "B")], bidirected=[("A", "C")])
    pairs = {(i["a"], i["b"]) for i in cc.implied_independencies(cg)}
    assert ("A", "C") not in pairs


def test_cycle_detected():
    cg = g([("A", "B"), ("B", "A")])
    assert not cg.is_dag()
