"""
Validation and approval layer between AI reasoning and the authoritative
research state.

Flow: AI proposes -> validate_proposal() checks schema, references, acyclicity,
time order and citations, and computes what would change in identification ->
the researcher accepts or rejects -> apply_proposal() re-validates against the
current graph, applies the change, records a graph version and an audit entry.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from .causal_engine import CausalEngine
from .schemas import (
    AuditLogEntry,
    CausalEdge,
    CausalNode,
    GraphPatchProposal,
    ProjectState,
    now_iso,
)

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
AI_ALLOWED_CLASSES = {"ai_hypothesis"}


def _summary(project_nodes: List[CausalNode], edges: List[CausalEdge], estimand: str) -> Dict[str, Any]:
    d = CausalEngine(project_nodes, edges).run_diagnostics(estimand)
    return {
        "is_dag": d["is_dag"],
        "identifiable": d["identifiable"],
        "current_valid": d["current_adjustment_valid"],
        "sets": [[x["label"] for x in s] for s in d["minimal_adjustment_sets"]],
    }


def _known_source_ids(project: ProjectState) -> Set[str]:
    return {s.id for s in project.sources} | {s.id for s in project.staged_sources}


def validate_proposal(project: ProjectState, prop: GraphPatchProposal) -> Dict[str, Any]:
    """Check a proposal against the current authoritative state. Never mutates it."""
    errors: List[str] = []
    warnings: List[str] = []
    node_ids = {n.id for n in project.nodes}
    nodes = [n.model_copy(deep=True) for n in project.nodes]
    edges = [e.model_copy(deep=True) for e in project.edges]

    if prop.action == "add_node":
        n = prop.node
        if not n:
            errors.append("The proposal has no variable.")
        else:
            if not ID_RE.match(n.id):
                errors.append(f"Variable id '{n.id}' must be snake_case letters, digits and underscores.")
            if n.id in node_ids:
                errors.append(f"A variable with id '{n.id}' already exists.")
            if not n.label.strip() or len(n.label) > 120:
                errors.append("The variable needs a label of 1 to 120 characters.")
            if not 0 <= n.time_index <= 3:
                errors.append("Time index must be between 0 and 3.")
            if n.role in ("exposure", "outcome") and any(m.role == n.role for m in project.nodes):
                warnings.append(f"The graph already has an {n.role}; identification uses the first one only.")
            nodes.append(n)

    elif prop.action == "add_edge":
        e = prop.edge
        if not e:
            errors.append("The proposal has no arrow.")
        else:
            for end in (e.source, e.target):
                if end not in node_ids:
                    errors.append(f"Variable '{end}' does not exist. Accept the proposal that adds it first.")
            if e.source == e.target:
                errors.append("An arrow cannot point from a variable to itself.")
            if any(x.source == e.source and x.target == e.target and x.edge_type == e.edge_type for x in project.edges):
                errors.append("This arrow already exists.")
            if prop.provenance.agent_mode and e.evidence_class not in AI_ALLOWED_CLASSES:
                errors.append("Arrows proposed by the assistant must be classed as AI hypotheses until you reclassify them.")
            unknown = [sid for sid in e.citation_ids if sid not in _known_source_ids(project)]
            if unknown:
                errors.append(f"Cited source id(s) not found: {', '.join(unknown)}.")
            if not errors:
                trial = e.model_copy(deep=True)
                trial.approved_by_researcher = True
                edges.append(trial)
                u = next(n for n in project.nodes if n.id == e.source)
                v = next(n for n in project.nodes if n.id == e.target)
                if e.edge_type == "directed" and u.time_index > v.time_index:
                    warnings.append(f"'{u.label}' (t{u.time_index}) would point back in time to '{v.label}' (t{v.time_index}).")

    elif prop.action == "remove_edge":
        e = prop.edge
        if not e or not any(x.id == e.id or (x.source == e.source and x.target == e.target) for x in project.edges):
            errors.append("The arrow to remove does not exist.")
        else:
            edges = [x for x in edges if not (x.id == e.id or (x.source == e.source and x.target == e.target))]

    elif prop.action == "update_node":
        n = prop.node
        if not n or n.id not in node_ids:
            errors.append("The variable to update does not exist.")
        else:
            nodes = [n if m.id == n.id else m for m in nodes]

    elif prop.action == "add_source":
        if not prop.citation:
            errors.append("The proposal has no source.")
        elif prop.citation.injection_flags:
            warnings.append("This source contains instruction-like text: " + "; ".join(prop.citation.injection_flags))

    result: Dict[str, Any] = {"ok": not errors, "errors": errors, "warnings": warnings, "checked_at": now_iso(),
                              "graph_version": project.graph_version}
    if not errors and prop.action in ("add_node", "add_edge", "remove_edge", "update_node"):
        before = _summary(project.nodes, project.edges, project.estimand_type)
        after = _summary(nodes, edges, project.estimand_type)
        if not after["is_dag"]:
            errors.append("Accepting this would create a directed cycle.")
            result.update(ok=False, errors=errors)
        else:
            result["diff"] = {
                "before": before,
                "after": after,
                "identifiability_changes": before["identifiable"] != after["identifiable"],
                "adjustment_sets_change": before["sets"] != after["sets"],
                "current_set_validity_changes": before["current_valid"] != after["current_valid"],
            }
    return result


def apply_proposal(project: ProjectState, prop: GraphPatchProposal, reason: str = "") -> Dict[str, Any]:
    """Re-validate against the current graph and apply. Returns the validation result."""
    result = validate_proposal(project, prop)
    prop.validation = result
    if not result["ok"]:
        return result

    if prop.action == "add_node" and prop.node:
        project.nodes.append(prop.node)
    elif prop.action == "add_edge" and prop.edge:
        edge = prop.edge.model_copy(deep=True)
        edge.approved_by_researcher = True
        # Evidence attached to an accepted proposal enters the library with it.
        staged = {s.id: s for s in project.staged_sources}
        for sid in edge.citation_ids:
            if sid in staged:
                project.sources.append(staged.pop(sid))
        project.staged_sources = list(staged.values())
        project.edges.append(edge)
    elif prop.action == "remove_edge" and prop.edge:
        project.edges = [
            x for x in project.edges
            if not (x.id == prop.edge.id or (x.source == prop.edge.source and x.target == prop.edge.target))
        ]
    elif prop.action == "update_node" and prop.node:
        project.nodes = [prop.node if m.id == prop.node.id else m for m in project.nodes]
    elif prop.action == "add_source" and prop.citation:
        project.sources.append(prop.citation)
        project.staged_sources = [s for s in project.staged_sources if s.id != prop.citation.id]

    prop.status = "accepted"
    prop.decided_at = now_iso()
    prop.decision_reason = reason
    version = project.record_version("researcher (accepted AI proposal)", prop.title)
    project.audit_log.insert(
        0,
        AuditLogEntry(
            actor="ai_accepted",
            action=f"Accepted proposal: {prop.title}",
            details=prop.rationale,
            graph_version=version,
            proposal_id=prop.id,
            provenance=prop.provenance,
        ),
    )
    return result


def reject_proposal(project: ProjectState, prop: GraphPatchProposal, reason: str = "") -> None:
    prop.status = "rejected"
    prop.decided_at = now_iso()
    prop.decision_reason = reason or "Rejected by the researcher."
    project.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action=f"Rejected proposal: {prop.title}",
            details=prop.decision_reason,
            graph_version=project.graph_version,
            proposal_id=prop.id,
            provenance=prop.provenance,
        ),
    )


def intake_proposals(project: ProjectState, raw: List[Dict[str, Any]], provenance: Optional[Dict[str, Any]] = None) -> List[str]:
    """Schema-validate raw AI proposals, attach provenance and validation, queue them.

    Proposals that fail validation are kept with status 'invalid' so the
    researcher can see what the assistant attempted and why it was refused.
    """
    ids: List[str] = []
    for item in raw or []:
        try:
            prop = GraphPatchProposal.model_validate(item)
        except Exception as exc:  # malformed output never reaches state
            project.audit_log.insert(0, AuditLogEntry(
                actor="system", action="Discarded malformed AI proposal", details=str(exc)[:300]))
            continue
        if provenance:
            prop.provenance = prop.provenance.model_validate({**prop.provenance.model_dump(), **provenance})
        if prop.edge is not None:
            prop.edge.origin = "ai"
            prop.edge.approved_by_researcher = False
            prop.edge.evidence_class = "ai_hypothesis"
        prop.validation = validate_proposal(project, prop)
        if not prop.validation["ok"]:
            prop.status = "invalid"
        project.proposals.insert(0, prop)
        ids.append(prop.id)
    return ids
