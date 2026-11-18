from typing import List, Dict, Any, Set, Tuple, Optional
import itertools
import re
import networkx as nx
from .schemas import CausalNode, CausalEdge, ProjectState


class CausalEngine:
    """
    Deterministic Pearlian Causal Graph Engine (DAGitty-equivalent).
    Computes d-separation, backdoor paths, minimal sufficient adjustment sets,
    collider traps, overadjustment warnings, IV candidates, and multi-format code exports.
    """

    def __init__(self, nodes: List[CausalNode], edges: List[CausalEdge]):
        self.nodes_list = nodes
        self.edges_list = [e for e in edges if e.approved_by_researcher]
        self.all_edges_list = edges
        self.node_map: Dict[str, CausalNode] = {n.id: n for n in nodes}

        self.G = nx.DiGraph()
        for n in nodes:
            self.G.add_node(n.id, **n.model_dump())

        for e in self.edges_list:
            if e.source in self.node_map and e.target in self.node_map:
                self.G.add_edge(e.source, e.target, **e.model_dump())
                if e.epistemic_status == "bidirectional_confounded":
                    # Represent latent common cause U between source and target if needed
                    pass

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
        return {
            n.id
            for n in self.nodes_list
            if n.observed and n.role != "latent"
        }

    def check_cycles(self) -> List[List[str]]:
        try:
            cycles = list(nx.simple_cycles(self.G))
            return cycles
        except Exception:
            return []

    def check_temporal_violations(self) -> List[Dict[str, Any]]:
        violations = []
        for e in self.edges_list:
            u = self.node_map.get(e.source)
            v = self.node_map.get(e.target)
            if u and v and u.time_index > v.time_index:
                violations.append(
                    {
                        "edge_id": e.id,
                        "source": u.label,
                        "target": v.label,
                        "source_time": u.time_index,
                        "target_time": v.time_index,
                        "message": f"Temporal violation: '{u.label}' (t={u.time_index}) points backward in time to '{v.label}' (t={v.time_index}).",
                    }
                )
        return violations

    def descendants_of(self, node_id: str) -> Set[str]:
        if node_id not in self.G or not nx.is_directed_acyclic_graph(self.G):
            try:
                return nx.descendants(self.G, node_id)
            except Exception:
                return set()
        return nx.descendants(self.G, node_id)

    def ancestors_of(self, node_id: str) -> Set[str]:
        if node_id not in self.G:
            return set()
        try:
            return nx.ancestors(self.G, node_id)
        except Exception:
            return set()

    def analyze_path(
        self, path: List[str], conditioned_set: Set[str]
    ) -> Dict[str, Any]:
        """
        Analyzes a single undirected path [v_0, v_1, ..., v_k] between X=v_0 and Y=v_k
        using Pearl's exact d-separation criteria under conditioned_set Z.
        """
        k = len(path) - 1
        arrows = []
        for i in range(k):
            u, v = path[i], path[i + 1]
            has_fwd = self.G.has_edge(u, v)
            has_rev = self.G.has_edge(v, u)
            if has_fwd and not has_rev:
                arrows.append("->")
            elif has_rev and not has_fwd:
                arrows.append("<-")
            else:
                arrows.append("<->")

        # Determine if directed causal path X -> ... -> Y
        is_directed_causal = all(a == "->" for a in arrows)
        is_reverse_causal = all(a == "<-" for a in arrows)
        starts_into_x = len(arrows) > 0 and arrows[0] in ("<-", "<->")

        colliders_on_path: List[str] = []
        non_colliders_on_path: List[str] = []

        for i in range(1, k):
            left_arrow = arrows[i - 1]
            right_arrow = arrows[i]
            mid_node = path[i]
            # Collider if -> mid_node <-
            if left_arrow in ("->", "<->") and right_arrow in ("<-", "<->"):
                colliders_on_path.append(mid_node)
            else:
                non_colliders_on_path.append(mid_node)

        # Precompute ancestors of conditioned_set (a collider is opened if it or any of its descendants is in Z)
        conditioned_or_ancestors = set(conditioned_set)
        for z in conditioned_set:
            conditioned_or_ancestors |= self.ancestors_of(z)

        blocking_nodes: List[str] = []
        opened_colliders: List[str] = []

        # 1. Non-colliders block the path if they ARE in conditioned_set
        for nc in non_colliders_on_path:
            if nc in conditioned_set:
                blocking_nodes.append(nc)

        # 2. Colliders block the path UNLESS they (or a descendant) are in conditioned_set
        for col in colliders_on_path:
            if col in conditioned_or_ancestors:
                opened_colliders.append(col)
            else:
                blocking_nodes.append(col)

        is_open = len(blocking_nodes) == 0

        # Format human-readable path string with arrows
        pretty_parts = [self.node_map[path[0]].label]
        for i, a in enumerate(arrows):
            pretty_parts.append(a)
            pretty_parts.append(self.node_map[path[i + 1]].label)
        pretty_str = " ".join(pretty_parts)

        if is_directed_causal:
            path_type = "causal"
        elif is_reverse_causal:
            path_type = "reverse_causal"
        elif len(colliders_on_path) > 0 and not opened_colliders:
            path_type = "collider_blocked"
        elif starts_into_x:
            path_type = "backdoor"
        else:
            path_type = "non_causal"

        return {
            "nodes": path,
            "labels": [self.node_map[n].label for n in path],
            "pretty": pretty_str,
            "path_type": path_type,
            "is_directed_causal": is_directed_causal,
            "starts_into_x": starts_into_x,
            "colliders": colliders_on_path,
            "opened_colliders": opened_colliders,
            "non_colliders": non_colliders_on_path,
            "is_open": is_open,
            "blocking_nodes": blocking_nodes,
        }

    def enumerate_all_paths(
        self, exposure: str, outcome: str, conditioned_set: Set[str]
    ) -> List[Dict[str, Any]]:
        if exposure not in self.G or outcome not in self.G or exposure == outcome:
            return []
        undirected = self.G.to_undirected()
        results = []
        try:
            for simple_path in nx.all_simple_paths(
                undirected, source=exposure, target=outcome, cutoff=10
            ):
                results.append(self.analyze_path(simple_path, conditioned_set))
        except Exception:
            pass
        return results

    def is_valid_adjustment_set(
        self,
        exposure: str,
        outcome: str,
        candidate_set: Set[str],
        estimand: str = "total_effect",
    ) -> bool:
        desc_x = self.descendants_of(exposure)
        # Identify nodes on causal paths X -> ... -> Y
        causal_intermediaries: Set[str] = set()
        for p in self.enumerate_all_paths(exposure, outcome, set()):
            if p["is_directed_causal"]:
                causal_intermediaries.update(p["nodes"][1:-1])

        if estimand == "total_effect":
            # Adjustment set for total effect cannot contain descendants of X
            if candidate_set & desc_x:
                return False
        else:
            # Direct effect: cannot contain descendants of Y
            desc_y = self.descendants_of(outcome)
            if candidate_set & desc_y:
                return False

        paths = self.enumerate_all_paths(exposure, outcome, candidate_set)
        for p in paths:
            if p["is_directed_causal"]:
                if estimand == "direct_effect" and len(p["nodes"]) > 2:
                    # For direct effect, mediated causal paths must be blocked
                    if p["is_open"]:
                        return False
                continue
            # All non-causal paths must be blocked!
            if p["is_open"]:
                return False
        return True

    def find_minimal_adjustment_sets(
        self,
        exposure: str,
        outcome: str,
        estimand: str = "total_effect",
        max_set_size: int = 6,
    ) -> List[List[Dict[str, str]]]:
        observable = self.get_observable_nodes() - {exposure, outcome}
        if estimand == "total_effect":
            desc_x = self.descendants_of(exposure)
            candidates = sorted(list(observable - desc_x))
        else:
            desc_y = self.descendants_of(outcome)
            candidates = sorted(list(observable - desc_y))

        valid_sets: List[Set[str]] = []
        max_k = min(len(candidates), max_set_size)

        for r in range(0, max_k + 1):
            for combo in itertools.combinations(candidates, r):
                cand_set = set(combo)
                # Skip if superset of an already discovered minimal valid set
                if any(vs.issubset(cand_set) for vs in valid_sets):
                    continue
                if self.is_valid_adjustment_set(
                    exposure, outcome, cand_set, estimand=estimand
                ):
                    valid_sets.append(cand_set)

        formatted_sets = []
        for vs in valid_sets:
            formatted_sets.append(
                [
                    {"id": nid, "label": self.node_map[nid].label}
                    for nid in sorted(vs)
                ]
            )
        return formatted_sets

    def find_instrumental_variables(
        self, exposure: str, outcome: str
    ) -> List[Dict[str, str]]:
        """
        Finds structural Instrumental Variable (IV) candidates Z for (X, Y):
        1. Z has a directed path to X.
        2. Z has no directed path to Y except through X.
        3. Z has no open backdoor path to Y when X is removed.
        """
        if exposure not in self.G or outcome not in self.G:
            return []
        anc_x = self.ancestors_of(exposure)
        G_no_x = self.G.copy()
        G_no_x.remove_node(exposure)

        ivs = []
        for nid in anc_x:
            if nid == outcome:
                continue
            node_obj = self.node_map[nid]
            if not node_obj.observed or node_obj.role == "latent":
                continue
            # Check no directed path to Y in G_no_x
            if nx.has_path(G_no_x, nid, outcome):
                continue
            # Check no open undirected path to Y in G_no_x without colliders
            paths_z_y = self.enumerate_all_paths(nid, outcome, set())
            non_x_open = [
                p
                for p in paths_z_y
                if exposure not in p["nodes"] and p["is_open"]
            ]
            if not non_x_open:
                ivs.append({"id": nid, "label": node_obj.label})
        return ivs

    def infer_structural_roles(
        self, exposure: Optional[str], outcome: Optional[str]
    ) -> Dict[str, str]:
        if not exposure or not outcome or exposure not in self.G or outcome not in self.G:
            return {}
        anc_x = self.ancestors_of(exposure)
        desc_x = self.descendants_of(exposure)
        anc_y = self.ancestors_of(outcome)
        desc_y = self.descendants_of(outcome)

        roles: Dict[str, str] = {}
        for nid, node in self.node_map.items():
            if nid == exposure:
                roles[nid] = "exposure"
            elif nid == outcome:
                roles[nid] = "outcome"
            elif not node.observed or node.role == "latent":
                roles[nid] = "latent"
            elif nid in desc_x and nid in anc_y:
                roles[nid] = "mediator"
            elif nid in anc_x and nid in anc_y:
                roles[nid] = "confounder"
            elif nid in desc_x and nid in desc_y:
                roles[nid] = "collider"
            elif nid in anc_y and nid not in anc_x and nid not in desc_x:
                roles[nid] = "competing_exposure"
            elif nid in anc_x and nid not in anc_y:
                roles[nid] = "instrument"
            else:
                roles[nid] = node.role
        return roles

    def run_diagnostics(self, estimand: str = "total_effect") -> Dict[str, Any]:
        cycles = self.check_cycles()
        temporal_violations = self.check_temporal_violations()
        exposures = self.get_exposures()
        outcomes = self.get_outcomes()
        adjusted_set = self.get_adjusted_set()

        exposure = exposures[0] if exposures else None
        outcome = outcomes[0] if outcomes else None

        warnings: List[Dict[str, str]] = []
        if cycles:
            cycle_labels = [
                " -> ".join([self.node_map[n].label for n in c] + [self.node_map[c[0]].label])
                for c in cycles
            ]
            warnings.append(
                {
                    "severity": "critical",
                    "type": "Directed Cycle Detected (Not a DAG)",
                    "message": f"Graph contains a feedback cycle: {'; '.join(cycle_labels)}. Break the cycle by time-indexing variables (e.g., t0 -> t1 -> t2).",
                }
            )

        for tv in temporal_violations:
            warnings.append(
                {
                    "severity": "warning",
                    "type": "Temporal Order Violation",
                    "message": tv["message"],
                }
            )

        if not exposure or not outcome:
            return {
                "is_dag": len(cycles) == 0,
                "cycles": cycles,
                "exposure": exposure,
                "outcome": outcome,
                "estimand": estimand,
                "current_adjusted": [
                    {"id": nid, "label": self.node_map[nid].label}
                    for nid in sorted(adjusted_set)
                ],
                "current_adjustment_valid": False,
                "identifiable": False,
                "minimal_adjustment_sets": [],
                "paths": [],
                "open_backdoor_paths": [],
                "causal_paths": [],
                "instrumental_variables": [],
                "structural_roles": {},
                "warnings": warnings
                + [
                    {
                        "severity": "info",
                        "type": "Missing Exposure or Outcome",
                        "message": "Designate one Exposure (Treatment) node and one Outcome node to compute backdoor paths and adjustment sets.",
                    }
                ],
                "epistemic_summary": self.compute_epistemic_summary(),
            }

        all_paths = self.enumerate_all_paths(exposure, outcome, adjusted_set)
        causal_paths = [p for p in all_paths if p["is_directed_causal"]]
        open_non_causal = [
            p for p in all_paths if not p["is_directed_causal"] and p["is_open"]
        ]
        minimal_sets = (
            self.find_minimal_adjustment_sets(exposure, outcome, estimand=estimand)
            if len(cycles) == 0
            else []
        )
        current_valid = (
            self.is_valid_adjustment_set(
                exposure, outcome, adjusted_set, estimand=estimand
            )
            if len(cycles) == 0
            else False
        )
        structural_roles = self.infer_structural_roles(exposure, outcome)
        ivs = (
            self.find_instrumental_variables(exposure, outcome)
            if len(cycles) == 0
            else []
        )

        # Check specific methodological traps
        desc_x = self.descendants_of(exposure)
        desc_y = self.descendants_of(outcome)

        # 1. Overadjustment on a Mediator
        if estimand == "total_effect":
            for adj_id in adjusted_set:
                if structural_roles.get(adj_id) == "mediator":
                    warnings.append(
                        {
                            "severity": "critical",
                            "type": "Overadjustment Bias (Conditioning on Mediator)",
                            "message": f"You are conditioning on '{self.node_map[adj_id].label}', which lies on the causal pathway from '{self.node_map[exposure].label}' to '{self.node_map[outcome].label}'. This blocks part of the Total Effect.",
                        }
                    )
                elif adj_id in desc_y:
                    warnings.append(
                        {
                            "severity": "critical",
                            "type": "Selection / Berkson's Bias (Descendant of Outcome)",
                            "message": f"You are conditioning on '{self.node_map[adj_id].label}', which is a descendant of the Outcome.",
                        }
                    )

        # 2. Collider Bias check
        for p in all_paths:
            if not p["is_directed_causal"] and p["opened_colliders"] and p["is_open"]:
                col_names = ", ".join(
                    self.node_map[c].label for c in p["opened_colliders"]
                )
                warnings.append(
                    {
                        "severity": "critical",
                        "type": "Collider Bias / M-Bias Trap Opened",
                        "message": f"Conditioning on ({col_names}) or its descendant opened a previously blocked path: {p['pretty']}.",
                    }
                )

        # 3. Open backdoor paths warning
        if open_non_causal:
            warnings.append(
                {
                    "severity": "warning",
                    "type": f"{len(open_non_causal)} Unblocked Non-Causal / Backdoor Path(s)",
                    "message": f"There are {len(open_non_causal)} open confounding path(s) between '{self.node_map[exposure].label}' and '{self.node_map[outcome].label}' under your current adjustment set.",
                }
            )

        # 4. Unidentifiability warning
        if len(minimal_sets) == 0 and len(open_non_causal) > 0:
            warnings.append(
                {
                    "severity": "critical",
                    "type": "Unmeasured Confounding (Not Identifiable via Backdoor Adjustment)",
                    "message": "No observable adjustment set blocks all backdoor paths (likely due to an unobserved/latent confounder). Consider Instrumental Variables (IV), Difference-in-Differences, or E-value sensitivity analysis.",
                }
            )

        # 5. Epistemic / Uncited Edge warning
        uncited_edges = [
            e for e in self.edges_list if len(e.citation_ids) == 0
        ]
        if uncited_edges:
            warnings.append(
                {
                    "severity": "info",
                    "type": f"{len(uncited_edges)} Edge(s) Lack Empirical Citations",
                    "message": f"{len(uncited_edges)} of {len(self.edges_list)} active causal arrows rely solely on hypothesis without attached literature citations.",
                }
            )

        return {
            "is_dag": len(cycles) == 0,
            "cycles": cycles,
            "exposure": {"id": exposure, "label": self.node_map[exposure].label},
            "outcome": {"id": outcome, "label": self.node_map[outcome].label},
            "estimand": estimand,
            "current_adjusted": [
                {"id": nid, "label": self.node_map[nid].label}
                for nid in sorted(adjusted_set)
            ],
            "current_adjustment_valid": current_valid,
            "identifiable": len(minimal_sets) > 0,
            "minimal_adjustment_sets": minimal_sets,
            "paths": all_paths,
            "open_backdoor_paths": open_non_causal,
            "causal_paths": causal_paths,
            "instrumental_variables": ivs,
            "structural_roles": structural_roles,
            "warnings": warnings,
            "epistemic_summary": self.compute_epistemic_summary(),
        }

    def compute_epistemic_summary(self) -> Dict[str, Any]:
        counts = {
            "total_edges": len(self.edges_list),
            "empirical": 0,
            "hypothesized": 0,
            "ai_proposed": len(
                [e for e in self.all_edges_list if not e.approved_by_researcher]
            ),
            "contested": 0,
            "cited_edges": 0,
        }
        for e in self.edges_list:
            status = e.epistemic_status
            if status in counts:
                counts[status] += 1
            if e.citation_ids:
                counts["cited_edges"] += 1
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
                arrow = (
                    "<->"
                    if e.epistemic_status == "bidirectional_confounded"
                    else "->"
                )
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
        gml_edges = " ".join(
            [f'"{e.source}" -> "{e.target}";' for e in self.edges_list]
        )
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
            if e.epistemic_status == "hypothesized":
                edge_style = "->, dashed, thick"
            elif e.epistemic_status == "contested":
                edge_style = "->, dotted, thick, red!70!black"
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
                        epistemic_status=(
                            "bidirectional_confounded"
                            if arrow == "<->"
                            else "hypothesized"
                        ),
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
