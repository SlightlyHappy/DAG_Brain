from typing import List, Dict, Any, Set, Tuple, Optional
import re
from .schemas import CausalNode, CausalEdge, ProjectState
from . import causal_core as cc


IDENTIFICATION_ASSUMPTIONS = {
    "total_effect": [
        "The DAG is complete: no unmodelled common cause of any pair of variables beyond the latent nodes and bidirected edges drawn.",
        "Edge directions are correct, and every missing edge is a true absence of direct effect.",
        "Positivity: every exposure level is possible within every stratum of the adjustment set.",
        "Consistency and no interference between units (SUTVA).",
        "Adjustment variables are measured without important error.",
    ],
    "direct_effect": [
        "All assumptions listed for the total effect.",
        "The single-door criterion identifies the x -> y path coefficient in a linear structural equation model. For non-linear models, read it as a controlled direct effect, which additionally requires no exposure-mediator interaction to be summarised by one number.",
    ],
}

WEAK_CLASSES = {
    "ai_hypothesis", "ai_proposed", "author_hypothesis", "researcher_hypothesis",
    "hypothesized", "contested", "empirical_association",
}


class CausalEngine:
    """
    Deterministic causal graph engine.

    Identification uses complete graphical criteria implemented in
    ``causal_core``: the generalised adjustment criterion for total effects and
    the single-door criterion for direct effects. Path listings are produced
    for explanation only and never decide validity.
    """

    def __init__(self, nodes: List[CausalNode], edges: List[CausalEdge]):
        self.nodes_list = nodes
        self.edges_list = [e for e in edges if e.approved_by_researcher]
        self.all_edges_list = edges
        self.node_map: Dict[str, CausalNode] = {n.id: n for n in nodes}

        directed, bidirected = [], []
        for e in self.edges_list:
            if e.source not in self.node_map or e.target not in self.node_map:
                continue
            if self._is_bidirected(e):
                bidirected.append((e.source, e.target))
            else:
                directed.append((e.source, e.target))
        latent = {n.id for n in nodes if not n.observed or n.role == "latent"}
        self.cg = cc.build_graph([n.id for n in nodes], directed, bidirected, latent)
        self.G = self.cg.G

    @staticmethod
    def _is_bidirected(e: CausalEdge) -> bool:
        return e.edge_type == "bidirected"

    # ---------------------------------------------------------------- basics
    def label(self, nid: Optional[str]) -> str:
        n = self.node_map.get(nid) if nid else None
        return n.label if n else (nid or "")

    def get_exposures(self) -> List[str]:
        return [n.id for n in self.nodes_list if n.role == "exposure"]

    def get_outcomes(self) -> List[str]:
        return [n.id for n in self.nodes_list if n.role == "outcome"]

    def get_adjusted_set(self) -> Set[str]:
        return {
            n.id
            for n in self.nodes_list
            if n.adjusted and n.role not in ("exposure", "outcome")
        }

    def get_observable_nodes(self) -> Set[str]:
        return {n.id for n in self.nodes_list if n.observed and n.role != "latent"}

    def check_cycles(self) -> List[List[str]]:
        return self.cg.cycles()

    def check_temporal_violations(self) -> List[Dict[str, Any]]:
        violations = []
        for e in self.edges_list:
            if self._is_bidirected(e):
                continue
            u = self.node_map.get(e.source)
            v = self.node_map.get(e.target)
            if u and v and u.time_index > v.time_index:
                violations.append(
                    {
                        "edge_id": e.id,
                        "source": u.label,
                        "target": v.label,
                        "message": f"'{u.label}' (t{u.time_index}) points backward in time to '{v.label}' (t{v.time_index}). A cause cannot follow its effect.",
                    }
                )
        return violations

    def _fmt_set(self, s) -> List[Dict[str, str]]:
        return [{"id": nid, "label": self.label(nid)} for nid in sorted(s)]

    def _fmt_path(self, p: Dict[str, Any]) -> Dict[str, Any]:
        parts = [self.label(p["nodes"][0])]
        for a, nid in zip(p["arrows"], p["nodes"][1:]):
            parts += [a, self.label(nid)]
        if p["is_directed_causal"]:
            ptype = "causal"
        elif p["colliders"] and not p["opened_colliders"]:
            ptype = "collider_blocked"
        elif p["starts_into_x"]:
            ptype = "backdoor"
        else:
            ptype = "non_causal"
        return {**p, "labels": [self.label(n) for n in p["nodes"]], "pretty": " ".join(parts), "path_type": ptype}

    # ---------------------------------------------------------------- diagnostics
    def run_diagnostics(self, estimand: str = "total_effect") -> Dict[str, Any]:
        cycles = self.check_cycles()
        is_dag = not cycles
        temporal = self.check_temporal_violations()
        exposures, outcomes = self.get_exposures(), self.get_outcomes()
        adjusted = self.get_adjusted_set()
        x = exposures[0] if exposures else None
        y = outcomes[0] if outcomes else None

        warnings: List[Dict[str, str]] = []
        if cycles:
            loop = "; ".join(" -> ".join(self.label(n) for n in c + [c[0]]) for c in cycles[:3])
            warnings.append({
                "severity": "critical",
                "type": "Directed cycle: the graph is not a DAG",
                "message": f"Feedback loop: {loop}. Break it by indexing variables in time (for example X at t0 and X at t2).",
            })
        for tv in temporal:
            warnings.append({"severity": "warning", "type": "Temporal order violation", "message": tv["message"]})
        if len(exposures) > 1 or len(outcomes) > 1:
            warnings.append({
                "severity": "info",
                "type": "Several exposures or outcomes",
                "message": "Identification is computed for the first exposure and the first outcome only.",
            })

        structural = {
            "is_dag": is_dag,
            "cycles": cycles,
            "temporal_violations": temporal,
            "latent_variables": self._fmt_set(n.id for n in self.nodes_list if not n.observed or n.role == "latent"),
            "bidirected_edges": sum(1 for e in self.edges_list if self._is_bidirected(e)),
            "paths_truncated": False,
        }
        base = {
            "is_dag": is_dag,
            "cycles": cycles,
            "estimand": estimand,
            "current_adjusted": self._fmt_set(adjusted),
            "structural": structural,
            "epistemic_summary": self.compute_epistemic_summary(),
        }

        if not x or not y or not is_dag:
            if not x or not y:
                warnings.append({
                    "severity": "info",
                    "type": "No exposure or outcome",
                    "message": "Mark one variable as the exposure and one as the outcome to compute identification.",
                })
            return {
                **base,
                "exposure": {"id": x, "label": self.label(x)} if x else None,
                "outcome": {"id": y, "label": self.label(y)} if y else None,
                "current_adjustment_valid": False,
                "identifiable": False,
                "minimal_adjustment_sets": [],
                "paths": [], "open_backdoor_paths": [], "causal_paths": [],
                "instrumental_variables": [],
                "structural_roles": {},
                "implied_independencies": [],
                "identification": {
                    "status": "not_computed",
                    "reason": "no exposure or outcome" if (not x or not y) else "the graph has a cycle",
                },
                "scientific_support": self.scientific_support(set()),
                "warnings": warnings,
            }

        cg = self.cg
        if estimand == "direct_effect":
            current_valid = cc.is_single_door_set(cg, x, y, adjusted)
        else:
            current_valid = cc.is_adjustment_set(cg, x, y, adjusted)
        search = cc.minimal_sets(cg, x, y, estimand=estimand)
        exists = bool(search.sets) or (estimand == "total_effect" and cc.adjustment_exists(cg, x, y))

        raw_paths, truncated = cc.explain_paths(cg, x, y, adjusted)
        structural["paths_truncated"] = truncated
        paths = [self._fmt_path(p) for p in raw_paths]
        causal_paths = [p for p in paths if p["is_directed_causal"]]
        open_non_causal = [p for p in paths if not p["is_directed_causal"] and p["is_open"]]

        roles = cc.structural_roles(cg, x, y)
        ivs = [
            {
                "id": iv["id"],
                "label": self.label(iv["id"]),
                "conditional_on": self._fmt_set(iv["conditional_on"]),
                "status": "graphical candidate",
            }
            for iv in cc.instrument_candidates(cg, x, y)
        ]
        implications = [
            {"a": self._fmt_set([i["a"]])[0], "b": self._fmt_set([i["b"]])[0], "given": self._fmt_set(i["given"])}
            for i in cc.implied_independencies(cg)
        ]

        # Methodological traps, judged by the criterion rather than by path counts.
        if estimand == "total_effect":
            forb = cc.forbidden(cg, x, y)
            for a in sorted(adjusted & forb):
                kind = roles.get(a, "")
                if kind == "mediator":
                    typ = "Overadjustment: conditioning on a mediator"
                    msg = f"'{self.label(a)}' lies on a causal path from '{self.label(x)}' to '{self.label(y)}'. Adjusting for it removes part of the total effect."
                elif kind == "descendant_of_outcome":
                    typ = "Conditioning on a descendant of the outcome"
                    msg = f"'{self.label(a)}' is caused by the outcome. Adjusting for it can create selection bias."
                else:
                    typ = "Conditioning on a forbidden variable"
                    msg = f"'{self.label(a)}' is a descendant of a variable on the causal path. Adjusting for it biases the total effect."
                warnings.append({"severity": "critical", "type": typ, "message": msg})
        else:
            for a in sorted(adjusted & cc.forbidden_direct(cg, y)):
                warnings.append({
                    "severity": "critical",
                    "type": "Conditioning on a descendant of the outcome",
                    "message": f"'{self.label(a)}' is caused by the outcome. The single-door criterion forbids it.",
                })
        seen_colliders = set()
        for p in open_non_causal:
            key = tuple(p["opened_colliders"])
            if p["opened_colliders"] and key not in seen_colliders:
                seen_colliders.add(key)
                warnings.append({
                    "severity": "critical",
                    "type": "Collider bias: adjustment opens a path",
                    "message": f"Conditioning on {', '.join(self.label(c) for c in p['opened_colliders'])} (or a descendant) opens {p['pretty']}.",
                })
        if not current_valid and open_non_causal:
            n_open = len(open_non_causal)
            warnings.append({
                "severity": "warning",
                "type": f"{n_open} open non-causal path{'s' if n_open != 1 else ''}",
                "message": f"The current adjustment leaves {n_open} non-causal path(s) between '{self.label(x)}' and '{self.label(y)}' open.",
            })
        if not exists:
            warnings.append({
                "severity": "critical",
                "type": "Not identifiable by covariate adjustment",
                "message": "No set of observed variables satisfies the criterion, usually because of an unmeasured confounder. Consider an instrument, a front-door mechanism, or a sensitivity analysis (E-value).",
            })
        if truncated:
            warnings.append({
                "severity": "info",
                "type": "Path list shortened",
                "message": "The graph has many paths, so only the first 400 are listed. Validity is still decided exactly by d-separation.",
            })

        identification = {
            "status": "identifiable_by_adjustment" if exists else "not_identifiable_by_adjustment",
            "criterion": (
                "single-door criterion (Pearl 2009, Theorem 5.3.1)"
                if estimand == "direct_effect"
                else "generalised adjustment criterion (Shpitser et al. 2010; Perkovic et al. 2018)"
            ),
            "current_set_valid": current_valid,
            "search_complete": search.complete,
            "instrument_note": (
                "Instrument candidates meet necessary graphical conditions under this DAG only. "
                "Instrument strength, the exclusion restriction and the absence of instrument-outcome "
                "confounding are substantive assumptions the graph cannot verify."
            ),
            "assumes": IDENTIFICATION_ASSUMPTIONS["direct_effect" if estimand == "direct_effect" else "total_effect"],
        }

        return {
            **base,
            "exposure": {"id": x, "label": self.label(x)},
            "outcome": {"id": y, "label": self.label(y)},
            "current_adjustment_valid": current_valid,
            "identifiable": exists,
            "minimal_adjustment_sets": [self._fmt_set(s) for s in search.sets],
            "paths": paths,
            "open_backdoor_paths": open_non_causal,
            "causal_paths": causal_paths,
            "instrumental_variables": ivs,
            "structural_roles": roles,
            "implied_independencies": implications,
            "identification": identification,
            "scientific_support": self.scientific_support(cg.an({x, y}) & cg.observed),
            "warnings": warnings,
        }

    # ---------------------------------------------------------------- tier 3
    def scientific_support(self, relevant_nodes: Set[str]) -> Dict[str, Any]:
        """Grade the evidence behind every edge that identification relies on.

        Identification depends on the edges among ancestors of the exposure and
        outcome. Each is graded by its evidence class and its researcher-assessed
        evidence links.
        """
        rows = []
        for e in self.edges_list:
            if relevant_nodes and not ({e.source, e.target} <= relevant_nodes):
                continue
            cls = e.evidence_class
            links = list(getattr(e, "evidence_links", []) or [])
            supported = [l for l in links if getattr(l, "researcher_assessment", "not_assessed") in ("supports", "weakly_supports")]
            contradicted = [l for l in links if getattr(l, "researcher_assessment", "") == "contradicts"]
            directional = any(getattr(l, "directionality", "") == "established" for l in links)
            weak_reasons = []
            if cls in WEAK_CLASSES:
                weak_reasons.append(f"evidence class is {cls.replace('_', ' ')}")
            if e.contested:
                weak_reasons.append("marked as contested")
            if not e.citation_ids and not links:
                weak_reasons.append("no linked sources")
            elif links and not supported:
                weak_reasons.append("no source assessed by the researcher as supporting it")
            if contradicted:
                weak_reasons.append(f"{len(contradicted)} source(s) contradict it")
            if not self._is_bidirected(e) and not directional and cls not in ("experimental", "quasi_experimental", "mechanistic"):
                weak_reasons.append("direction not established by any linked study")
            rows.append({
                "edge_id": e.id,
                "edge": f"{self.label(e.source)} {'<->' if self._is_bidirected(e) else '->'} {self.label(e.target)}",
                "evidence_class": cls,
                "citations": len(e.citation_ids),
                "researcher_supported_links": len(supported),
                "directionality_established": directional,
                "weak": bool(weak_reasons),
                "weak_reasons": weak_reasons,
            })
        return {
            "edges": rows,
            "weak_count": sum(1 for r in rows if r["weak"]),
            "total": len(rows),
            "note": "Missing arrows are assumptions too: each absent edge asserts that there is no direct effect.",
        }

    def compute_epistemic_summary(self) -> Dict[str, Any]:
        counts: Dict[str, Any] = {cls: 0 for cls in (
            "empirical_association", "experimental", "quasi_experimental", "mechanistic",
            "author_hypothesis", "ai_hypothesis", "researcher_hypothesis",
        )}
        counts["total_edges"] = len(self.edges_list)
        counts["pending_ai_edges"] = len([e for e in self.all_edges_list if not e.approved_by_researcher])
        counts["contested"] = 0
        counts["cited_edges"] = 0
        counts["assessed_edges"] = 0
        for e in self.edges_list:
            counts[e.evidence_class] = counts.get(e.evidence_class, 0) + 1
            if e.contested:
                counts["contested"] += 1
            if e.citation_ids:
                counts["cited_edges"] += 1
            if any(l.researcher_assessment != "not_assessed" for l in e.evidence_links):
                counts["assessed_edges"] += 1
        return counts

    # =========================================================================
    # EXPORTERS & INTEROPERABILITY (DAGitty, R, Python DoWhy, LaTeX TikZ)
    # =========================================================================

    def to_dagitty(self) -> str:
        lines = ["dag {"]
        for n in self.nodes_list:
            tags = []
            if n.role == "exposure":
                tags.append("exposure")
            elif n.role == "outcome":
                tags.append("outcome")
            elif not n.observed or n.role == "latent":
                tags.append("latent")
            elif n.adjusted:
                tags.append("adjusted")
            rx = round(n.x / 500.0, 3)
            ry = round(n.y / 400.0, 3)
            tags.append(f'pos="{rx},{ry}"')
            tag_str = " [" + ",".join(tags) + "]" if tags else ""
            lines.append(f'  "{n.label}"{tag_str}')
        for e in self.edges_list:
            u = self.node_map.get(e.source)
            v = self.node_map.get(e.target)
            if u and v:
                arrow = "<->" if self._is_bidirected(e) else "->"
                lines.append(f'  "{u.label}" {arrow} "{v.label}"')
        lines.append("}")
        return "\n".join(lines)

    def to_r_script(self, project_title: str = "Causal Model") -> str:
        dagitty_str = self.to_dagitty()
        exposures = self.get_exposures()
        outcomes = self.get_outcomes()
        exp_label = self.node_map[exposures[0]].label if exposures else "Exposure"
        out_label = self.node_map[outcomes[0]].label if outcomes else "Outcome"
        return f"""# ============================================================================
# Reproducible R Causal Analysis Script — Generated by DAG_Brain
# Project: {project_title}
# ============================================================================
library(dagitty)
library(ggdag)
library(ggplot2)

causal_model <- dagitty('{dagitty_str}')

# 1. Print minimal sufficient adjustment sets (Total Effect)
cat("=== Minimal Sufficient Adjustment Sets (Total Effect) ===\\n")
print(adjustmentSets(causal_model, exposure = "{exp_label}", outcome = "{out_label}", effect = "total"))

# 2. Print testable conditional independencies (d-separation implications)
cat("\\n=== Testable Conditional Independencies ===\\n")
print(impliedConditionalIndependencies(causal_model))

# 3. Render publication-quality DAG with ggdag
ggdag_status(causal_model, text = FALSE, use_labels = "name") +
  theme_dag() +
  labs(title = "{project_title}", subtitle = "Exposure: {exp_label} -> Outcome: {out_label}")
"""

    def to_python_dowhy(self, project_title: str = "Causal Model") -> str:
        exposures = self.get_exposures()
        outcomes = self.get_outcomes()
        exp_id = exposures[0] if exposures else "treatment"
        out_id = outcomes[0] if outcomes else "outcome"
        dot = []
        for e in self.edges_list:
            if self._is_bidirected(e):
                u = f"U_{e.source}_{e.target}"
                dot.append(f'"{u}" [observed="no"]; "{u}" -> "{e.source}"; "{u}" -> "{e.target}";')
            else:
                dot.append(f'"{e.source}" -> "{e.target}";')
        for n in self.nodes_list:
            if not n.observed or n.role == "latent":
                dot.append(f'"{n.id}" [observed="no"];')
        gml_edges = " ".join(dot)
        return f'''"""
Reproducible Python DoWhy Causal Inference Script — Generated by DAG_Brain
Project: {project_title}
"""
import pandas as pd
from dowhy import CausalModel

# Define the structural causal graph in DOT format
causal_graph = """
digraph {{
    {gml_edges}
}}
"""

# Load your dataset (columns matching node IDs: {[n.id for n in self.nodes_list]})
# df = pd.read_csv("your_dataset.csv")

def run_causal_estimation(df: pd.DataFrame):
    model = CausalModel(
        data=df,
        treatment="{exp_id}",
        outcome="{out_id}",
        graph=causal_graph
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    print(identified_estimand)

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression"
    )
    print("Causal Estimate:", estimate.value)

    # Refutation checks
    refute_placebo = model.refute_estimate(
        identified_estimand, estimate, method_name="placebo_treatment_refuter"
    )
    print(refute_placebo)
'''

    def to_latex_tikz(self) -> str:
        lines = [
            "% LaTeX TikZ Causal DAG — Generated by DAG_Brain",
            "\\begin{tikzpicture}[>=stealth, node distance=2.5cm, every node/.style={rectangle, rounded corners, draw, align=center, font=\\small, minimum height=0.8cm, minimum width=1.8cm}]",
        ]
        for n in self.nodes_list:
            tx = round(n.x / 90.0, 2)
            ty = round(-n.y / 90.0, 2)
            style = "fill=gray!10"
            if n.role == "exposure":
                style = "fill=green!15, draw=green!60!black, thick"
            elif n.role == "outcome":
                style = "fill=blue!15, draw=blue!60!black, thick"
            elif n.role == "confounder":
                style = "fill=red!12, draw=red!60!black"
            elif n.role == "mediator":
                style = "fill=orange!15, draw=orange!70!black"
            elif n.role == "latent" or not n.observed:
                style = "fill=gray!5, dashed"
            safe_label = n.label.replace("&", "\\&").replace("_", "\\_")
            lines.append(f"  \\node[{style}] ({n.id}) at ({tx},{ty}) {{{safe_label}}};")

        for e in self.edges_list:
            edge_style = "->, thick"
            if self._is_bidirected(e):
                edge_style = "<->, dashed, thick, gray"
            elif e.contested:
                edge_style = "->, dotted, thick, red!70!black"
            elif e.evidence_class not in ("experimental", "quasi_experimental", "mechanistic"):
                edge_style = "->, dashed, thick"
            lines.append(f"  \\draw[{edge_style}] ({e.source}) -- ({e.target});")

        lines.append("\\end{tikzpicture}")
        return "\n".join(lines)

    @staticmethod
    def parse_dagitty(dagitty_text: str) -> Tuple[List[CausalNode], List[CausalEdge]]:
        """Parses standard DAGitty syntax into CausalNode and CausalEdge objects."""
        nodes_by_label: Dict[str, CausalNode] = {}
        edges: List[CausalEdge] = []

        def slugify(text: str) -> str:
            s = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
            return s or "node"

        lines = dagitty_text.splitlines()
        idx = 0
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("dag") or line in ("{", "}"):
                continue
            # Match edge: "A" -> "B" or A -> B
            edge_match = re.match(
                r'^"?([^"\->]+?)"?\s*(->|<->|<-)\s*"?([^"\[\n]+?)"?\s*$', line
            )
            if edge_match:
                u_lbl, arrow, v_lbl = (
                    edge_match.group(1).strip(),
                    edge_match.group(2).strip(),
                    edge_match.group(3).strip(),
                )
                for lbl in (u_lbl, v_lbl):
                    if lbl not in nodes_by_label:
                        idx += 1
                        nodes_by_label[lbl] = CausalNode(
                            id=slugify(lbl),
                            label=lbl,
                            x=180 + (idx % 4) * 190,
                            y=140 + (idx // 4) * 140,
                        )
                u_id = nodes_by_label[u_lbl].id
                v_id = nodes_by_label[v_lbl].id
                if arrow == "<-":
                    u_id, v_id = v_id, u_id
                edges.append(
                    CausalEdge(
                        source=u_id,
                        target=v_id,
                        edge_type="bidirected" if arrow == "<->" else "directed",
                        evidence_class="researcher_hypothesis",
                        origin="import",
                    )
                )
                continue

            # Match node declaration: "Label" [exposure,pos="0.1,0.2"]
            node_match = re.match(r'^"?([^"\[]+?)"?\s*(?:\[(.*?)\])?$', line)
            if node_match:
                lbl = node_match.group(1).strip()
                attrs = (node_match.group(2) or "").lower()
                idx += 1
                role = "other"
                adjusted = False
                observed = True
                if "exposure" in attrs:
                    role = "exposure"
                elif "outcome" in attrs:
                    role = "outcome"
                elif "latent" in attrs:
                    role = "latent"
                    observed = False
                if "adjusted" in attrs:
                    adjusted = True
                nodes_by_label[lbl] = CausalNode(
                    id=slugify(lbl),
                    label=lbl,
                    role=role,
                    adjusted=adjusted,
                    observed=observed,
                    x=180 + (idx % 4) * 190,
                    y=140 + (idx // 4) * 140,
                )

        return list(nodes_by_label.values()), edges
