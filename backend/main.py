import json
import os
import re
import sys
import threading
import webbrowser
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional, List
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, FileResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
import uvicorn

from .config import (
    BASE_DIR,
    DATA_DIR,
    get_runtime_settings,
    save_runtime_settings,
    mask_settings_for_client,
)
from .database import (
    init_db,
    list_projects,
    get_project,
    save_project,
    delete_project,
)
from .schemas import (
    ProjectState,
    CausalNode,
    CausalEdge,
    CitationSource,
    GraphPatchProposal,
    CompetingWorldview,
    ChatMessage,
    AuditLogEntry,
    DatasetMeta,
    Provenance,
    ResearchQuestion,
)
from . import data_analysis as da
from .net import offline_mode
from .untrusted import scan
from .validation import apply_proposal, intake_proposals, reject_proposal, validate_proposal
from .causal_engine import CausalEngine
from .literature_service import (
    search_scholarly_literature,
    summarize_paper_with_ai,
    format_in_text_citation,
    format_reference_entry,
    enrich_source_links,
    extract_pdf,
    find_open_access,
    classify_study_design,
    extract_sample_size,
    _heuristic_paper_summary,
)
from .llm_router import fetch_ollama_models
from .agents import (
    run_agent_turn,
    run_autonomous_deep_research,
    generate_lab_briefing,
)


def _enrich_project_response(project: ProjectState) -> Dict[str, Any]:
    engine = CausalEngine(project.nodes, project.edges)
    diagnostics = engine.run_diagnostics(estimand=project.estimand_type)
    formatted_sources = []
    for idx, src in enumerate(project.sources):
        src = enrich_source_links(src)
        d = src.model_dump()
        d["in_text"] = format_in_text_citation(
            src, style=project.citation_style, index=idx + 1
        )
        d["reference_entry"] = format_reference_entry(
            src, style=project.citation_style, index=idx + 1
        )
        formatted_sources.append(d)

    payload = project.model_dump()
    # Snapshots can be large; send a summary and serve full versions on request.
    payload["versions"] = [
        {"version": v.version, "timestamp": v.timestamp, "actor": v.actor, "reason": v.reason,
         "nodes": len(v.nodes), "edges": len(v.edges)}
        for v in project.versions
    ]
    payload["diagnostics"] = diagnostics
    payload["formatted_sources"] = formatted_sources
    payload["offline_mode"] = offline_mode()
    return payload


def _graph_signature(nodes, edges) -> str:
    return json.dumps(
        [[n.model_dump(exclude={"x", "y"}) for n in nodes], [e.model_dump() for e in edges]],
        sort_keys=True, default=str,
    )


def _not_found(what: str = "Project") -> JSONResponse:
    return JSONResponse({"detail": f"{what} not found"}, status_code=404)


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "DAG Brain", "version": "0.3.0", "offline_mode": offline_mode()}
    )


async def api_get_settings(request: Request) -> JSONResponse:
    settings = get_runtime_settings()
    masked = mask_settings_for_client(settings)
    ollama_models = await fetch_ollama_models(settings.get("ollama_base_url"))
    masked["detected_ollama_models"] = ollama_models
    return JSONResponse(masked)


async def api_update_settings(request: Request) -> JSONResponse:
    updates = await request.json()
    clean_updates = {}
    for k, v in updates.items():
        if isinstance(v, str) and ("..." in v or v == "****"):
            continue
        clean_updates[k] = v
    saved = save_runtime_settings(clean_updates)
    masked = mask_settings_for_client(saved)
    masked["detected_ollama_models"] = await fetch_ollama_models(
        saved.get("ollama_base_url")
    )
    return JSONResponse(masked)


async def api_projects_collection(request: Request) -> JSONResponse:
    init_db()
    if request.method == "GET":
        return JSONResponse(list_projects())

    data = await request.json()
    title = (data.get("title") or "Untitled Causal Study").strip()
    discipline = (data.get("discipline") or "General Research").strip()
    rq = (data.get("research_question") or "").strip()
    exposure_name = (data.get("exposure_label") or "Exposure (X)").strip()
    outcome_name = (data.get("outcome_label") or "Outcome (Y)").strip()

    nodes = [
        CausalNode(
            id="exposure_x",
            label=exposure_name,
            role="exposure",
            time_index=1,
            x=200.0,
            y=220.0,
        ),
        CausalNode(
            id="outcome_y",
            label=outcome_name,
            role="outcome",
            time_index=3,
            x=600.0,
            y=220.0,
        ),
    ]
    edges = [
        CausalEdge(
            source="exposure_x",
            target="outcome_y",
            evidence_class="researcher_hypothesis",
            mechanism="Primary hypothesised causal effect",
            effect_direction="unknown",
        )
    ]

    proj = ProjectState(
        title=title,
        discipline=discipline,
        research_question=rq,
        question=ResearchQuestion(exposure=exposure_name, outcome=outcome_name),
        notebook_markdown=f"# {title}\n\n## Research Question\n{rq or 'Define your primary causal question here.'}\n\n## Variable Operationalization & Identification Notes\n",
        nodes=nodes,
        edges=edges,
        chat_history=[
            ChatMessage(
                role="assistant",
                agent_mode="socratic",
                content=(
                    f"Created new study **{title}** ({discipline}).\n\n"
                    f"I've initialized your canvas with **{exposure_name}** $\\rightarrow$ **{outcome_name}**. "
                    "Use Topic mode to sharpen the question, or Research to search the literature. Anything I find is staged for your review; nothing enters the graph or the library until you accept it."
                ),
                socratic_questions=[
                    f"How is '{exposure_name}' operationalized, and at what time window is it measured relative to '{outcome_name}'?",
                    f"What baseline factors might simultaneously influence both '{exposure_name}' and '{outcome_name}' (common causes / confounders)?",
                ],
            )
        ],
        audit_log=[
            AuditLogEntry(
                actor="researcher",
                action="Created Project",
                details=f"Initialized '{title}' ({discipline})",
            )
        ],
    )
    proj.record_version("researcher", "Created study")
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_project_item(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    if request.method == "GET":
        proj = get_project(project_id)
        if not proj:
            return JSONResponse({"detail": "Project not found"}, status_code=404)
        return JSONResponse(_enrich_project_response(proj))

    if request.method == "DELETE":
        ok = delete_project(project_id)
        if not ok:
            return JSONResponse({"detail": "Project not found"}, status_code=404)
        return JSONResponse({"deleted": True, "id": project_id})

    existing = get_project(project_id)
    if not existing:
        return JSONResponse({"detail": "Project not found"}, status_code=404)
    data = await request.json()
    before_sig = _graph_signature(existing.nodes, existing.edges)

    for field in [
        "title",
        "discipline",
        "research_question",
        "target_population",
        "estimand_type",
        "citation_style",
        "notebook_markdown",
        "lab_briefing_markdown",
    ]:
        if field in data and data[field] is not None:
            setattr(existing, field, data[field])

    if "nodes" in data and isinstance(data["nodes"], list):
        existing.nodes = [CausalNode.model_validate(n) for n in data["nodes"]]
    if "edges" in data and isinstance(data["edges"], list):
        existing.edges = [CausalEdge.model_validate(e) for e in data["edges"]]
    if "sources" in data and isinstance(data["sources"], list):
        existing.sources = [
            enrich_source_links(CitationSource.model_validate(s))
            for s in data["sources"]
        ]
    if isinstance(data.get("question"), dict):
        existing.question = ResearchQuestion.model_validate(data["question"])

    graph_changed = _graph_signature(existing.nodes, existing.edges) != before_sig
    if graph_changed:
        existing.record_version("researcher", data.get("audit_action") or "Edited graph")
    if data.get("audit_note"):
        existing.audit_log.insert(
            0,
            AuditLogEntry(
                actor="researcher",
                action=data.get("audit_action", "Updated Model"),
                details=str(data["audit_note"]),
                graph_version=existing.graph_version,
            ),
        )

    save_project(existing)
    return JSONResponse(_enrich_project_response(existing))


async def api_apply_adjustment_set(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    data = await request.json()
    node_ids = set(data.get("node_ids", []))
    labels = []
    for n in proj.nodes:
        n.adjusted = n.id in node_ids
        if n.adjusted:
            labels.append(n.label)

    version = proj.record_version("researcher", "Applied adjustment set")
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action="Applied adjustment set",
            details=f"Conditioned on: {', '.join(labels) if labels else 'nothing (empty set)'}",
            graph_version=version,
        ),
    )
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_decide_proposal(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    proposal_id = request.path_params["proposal_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    data = await request.json()
    decision = data.get("decision", "accept")
    target_prop: Optional[GraphPatchProposal] = next((p for p in proj.proposals if p.id == proposal_id), None)
    if not target_prop:
        return _not_found("Proposal")
    if target_prop.status not in ("pending", "invalid"):
        return JSONResponse({"detail": f"This proposal was already {target_prop.status}."}, status_code=409)

    if decision == "accept":
        result = apply_proposal(proj, target_prop, reason=data.get("reason", ""))
        if not result["ok"]:
            target_prop.status = "invalid"
            save_project(proj)
            return JSONResponse(
                {"detail": "The proposal no longer passes validation: " + " ".join(result["errors"]),
                 "validation": result},
                status_code=409,
            )
    else:
        reject_proposal(proj, target_prop, data.get("reason", ""))

    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_manage_worldviews(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    data = await request.json()
    action = data.get("action", "create")
    if action == "create":
        wv = CompetingWorldview(
            name=data.get("name") or f"Model Snapshot {len(proj.worldviews) + 1}",
            description=data.get("description") or "",
            nodes=proj.nodes,
            edges=proj.edges,
        )
        proj.worldviews.append(wv)
        proj.audit_log.insert(
            0,
            AuditLogEntry(
                actor="researcher",
                action="Forked Competing Worldview",
                details=f"Saved snapshot '{wv.name}'",
            ),
        )
    elif action == "restore":
        wv_id = data.get("worldview_id")
        found = next((w for w in proj.worldviews if w.id == wv_id), None)
        if not found:
            return JSONResponse({"detail": "Worldview not found"}, status_code=404)
        proj.nodes = found.nodes
        proj.edges = found.edges
        proj.record_version("researcher", f"Loaded worldview '{found.name}'")
        proj.audit_log.insert(
            0,
            AuditLogEntry(
                actor="researcher",
                action="Switched Competing Worldview",
                details=f"Loaded worldview '{found.name}'",
            ),
        )
    elif action == "compare":
        wv_id = data.get("worldview_id")
        found = next((w for w in proj.worldviews if w.id == wv_id), None)
        if not found:
            return JSONResponse({"detail": "Worldview not found"}, status_code=404)
        eng_curr = CausalEngine(proj.nodes, proj.edges).run_diagnostics(
            proj.estimand_type
        )
        eng_other = CausalEngine(found.nodes, found.edges).run_diagnostics(
            proj.estimand_type
        )
        return JSONResponse(
            {
                "current_diagnostics": eng_curr,
                "worldview": found.model_dump(),
                "worldview_diagnostics": eng_other,
            }
        )

    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_chat_turn(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    data = await request.json()
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return JSONResponse({"detail": "Message cannot be empty"}, status_code=400)
    agent_mode = data.get("agent_mode", "socratic")
    provider_override = data.get("provider")
    model_override = data.get("model")

    user_msg_obj = ChatMessage(
        role="user",
        agent_mode=agent_mode,
        content=user_message,
    )
    proj.chat_history.append(user_msg_obj)

    turn_result = await run_agent_turn(
        project=proj,
        user_message=user_message,
        agent_mode=agent_mode,
        provider_override=provider_override,
        model_override=model_override,
    )

    provenance = turn_result.get("provenance") or {}
    _stage_sources(proj, turn_result.get("staged_sources", []))
    new_prop_ids = intake_proposals(proj, turn_result.get("proposals", []), provenance)

    assistant_msg = ChatMessage(
        role="assistant",
        agent_mode=agent_mode,
        content=turn_result.get("reply_markdown", ""),
        socratic_questions=turn_result.get("socratic_questions", []),
        proposal_ids=new_prop_ids,
        provenance=Provenance.model_validate(provenance) if provenance else None,
    )
    proj.chat_history.append(assistant_msg)
    save_project(proj)

    enriched = _enrich_project_response(proj)
    enriched["last_turn"] = turn_result
    return JSONResponse(enriched)


async def api_deep_research(request: Request) -> JSONResponse:
    """Autonomous multi-hop Web (Tavily) + Scholarly (OpenAlex, CORE.ac.uk, PubMed, Scholar) agent."""
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    data = await request.json() if request.headers.get("content-length") else {}
    focus = (data.get("focus") or "").strip()

    res = await run_autonomous_deep_research(proj, custom_focus=focus)

    provenance = res.get("provenance") or {}
    staged = _stage_sources(proj, res.get("staged_sources", []))
    new_prop_ids = intake_proposals(proj, res.get("proposals", []), provenance)

    proj.chat_history.append(
        ChatMessage(
            role="assistant",
            agent_mode="deep_research",
            content=res.get("reply_markdown", ""),
            socratic_questions=res.get("socratic_questions", []),
            proposal_ids=new_prop_ids,
            provenance=Provenance.model_validate(provenance) if provenance else None,
        )
    )
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="ai",
            action="Research run: sources staged for review",
            details=f"Staged {staged} source(s); nothing was added to the library or the graph.",
            graph_version=proj.graph_version,
            provenance=Provenance.model_validate(provenance) if provenance else None,
        ),
    )
    save_project(proj)
    enriched = _enrich_project_response(proj)
    enriched["last_turn"] = res
    return JSONResponse(enriched)


async def api_summarize_source(request: Request) -> JSONResponse:
    """Generates or updates the AI Executive Summary for a specific source in the Vault."""
    project_id = request.path_params["project_id"]
    source_id = request.path_params["source_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    target_src: Optional[CitationSource] = None
    for s in proj.sources:
        if s.id == source_id:
            target_src = s
            break

    if not target_src:
        return JSONResponse({"detail": "Source not found"}, status_code=404)

    target_src.ai_summary = await summarize_paper_with_ai(target_src)
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action="Generated AI Paper Summary",
            details=f"Summarized '{target_src.title}'",
        ),
    )
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_generate_lab_briefing(request: Request) -> JSONResponse:
    """Generates the NotebookLM-style 2-Host Audio/Text Research Lab Briefing."""
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    briefing_md = await generate_lab_briefing(proj)
    proj.lab_briefing_markdown = briefing_md
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action="Generated 2-Host Lab Briefing",
            details="Synthesized NotebookLM-style audio/text briefing script.",
        ),
    )
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_export_project(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    fmt = request.query_params.get("fmt", "dagitty")
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)

    engine = CausalEngine(proj.nodes, proj.edges)
    diagnostics = engine.run_diagnostics(proj.estimand_type)

    if fmt == "dagitty":
        content = engine.to_dagitty()
    elif fmt == "r":
        content = engine.to_r_script(proj.title)
    elif fmt == "python":
        content = engine.to_python_dowhy(proj.title)
    elif fmt == "latex":
        content = engine.to_latex_tikz()
    else:
        refs = [
            format_reference_entry(s, proj.citation_style, i + 1)
            for i, s in enumerate(proj.sources)
        ]
        node_lookup = {n.id: n.label for n in proj.nodes}
        edge_rows = []
        for e in proj.edges:
            u_lbl = node_lookup.get(e.source, e.source)
            v_lbl = node_lookup.get(e.target, e.target)
            cites = [
                format_in_text_citation(s, proj.citation_style, i + 1)
                for i, s in enumerate(proj.sources)
                if s.id in e.citation_ids
            ]
            edge_rows.append(
                f"| {u_lbl} {'↔' if e.edge_type == 'bidirected' else '→'} {v_lbl} | {e.evidence_class.replace('_', ' ')}{' (contested)' if e.contested else ''} | {e.mechanism or e.rationale or '-'} | {', '.join(cites) or 'Uncited'} |"
            )

        min_sets_str = (
            ", ".join(
                "{" + ", ".join(x["label"] for x in ms) + "}"
                for ms in diagnostics.get("minimal_adjustment_sets", [])
            )
            or "None identifiable with observed covariates"
        )

        content = f"""# Causal Modelling Dossier: {proj.title}
**Discipline:** {proj.discipline}  
**Research Question:** {proj.research_question}  
**Target Estimand:** {proj.estimand_type.replace('_', ' ').title()}  
**Citation Format:** {proj.citation_style.upper()}

---

## 1. Structural validity
- **Acyclic (a DAG):** {diagnostics.get('is_dag')}
- **Graph version:** {proj.graph_version}

## 2. Identification (valid only if the graph is correct)
- **Criterion:** {(diagnostics.get('identification') or {}).get('criterion', 'not computed')}
- **Status:** {(diagnostics.get('identification') or {}).get('status', 'not computed')}
- **Minimal sufficient adjustment sets:** {min_sets_str}
- **Open non-causal paths under the current adjustment:** {len(diagnostics.get('open_backdoor_paths', []))}
- **Graphical instrument candidates (validity not established by the graph):** {', '.join(i['label'] for i in diagnostics.get('instrumental_variables', [])) or 'none'}

Assumptions:
{chr(10).join('- ' + a for a in (diagnostics.get('identification') or {}).get('assumes', []))}

## 3. Scientific support for the assumptions
{(diagnostics.get('scientific_support') or {}).get('weak_count', 0)} of {(diagnostics.get('scientific_support') or {}).get('total', 0)} relevant arrows rest on weak support (hypotheses, associations only, or no researcher-assessed source).

| Edge | Evidence class | Mechanism / Rationale | Citations |
| :--- | :--- | :--- | :--- |
{chr(10).join(edge_rows)}

---

## 4. Research Notebook
{proj.notebook_markdown}

---

## 5. Bibliography ({proj.citation_style.upper()})
{chr(10).join('- ' + r for r in refs) if refs else '(No sources in vault)'}
"""

    return JSONResponse({"format": fmt, "content": content, "title": proj.title})


async def api_import_dagitty(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    proj = get_project(project_id)
    if not proj:
        return JSONResponse({"detail": "Project not found"}, status_code=404)
    data = await request.json()
    code = data.get("dagitty_code", "")
    nodes, edges = CausalEngine.parse_dagitty(code)
    if not nodes:
        return JSONResponse(
            {"detail": "Could not parse any nodes from DAGitty syntax."},
            status_code=400,
        )
    proj.nodes = nodes
    proj.edges = edges
    proj.record_version("researcher", "Imported DAGitty code")
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action="Imported DAGitty Code",
            details=f"Parsed {len(nodes)} nodes and {len(edges)} edges.",
        ),
    )
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_search_literature(request: Request) -> JSONResponse:
    q = (request.query_params.get("q") or "").strip()
    limit = int(request.query_params.get("limit", "8"))
    engine = (request.query_params.get("engine") or "all").strip()
    if len(q) < 2:
        return JSONResponse({"query": q, "count": 0, "results": []})
    results = await search_scholarly_literature(q, limit=limit, engine=engine)
    return JSONResponse(
        {
            "query": q,
            "engine": engine,
            "count": len(results),
            "results": [r.model_dump() for r in results],
        }
    )


# ---------------------------------------------------------------------------
# Staged sources (AI-retrieved, awaiting the researcher)
# ---------------------------------------------------------------------------


def _stage_sources(proj: ProjectState, raw: List[Dict[str, Any]]) -> int:
    known = {s.id for s in proj.sources} | {s.id for s in proj.staged_sources}
    count = 0
    for item in raw or []:
        try:
            src = CitationSource.model_validate(item)
        except Exception:
            continue
        if src.id in known:
            continue
        src.added_by = "ai"
        src.injection_flags = scan(" ".join([src.title, src.abstract, src.full_text_snippet, src.key_finding]))
        proj.staged_sources.append(src)
        known.add(src.id)
        count += 1
    return count


async def api_decide_staged(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    sid = request.path_params["source_id"]
    src = next((s for s in proj.staged_sources if s.id == sid), None)
    if not src:
        return _not_found("Staged source")
    data = await request.json()
    proj.staged_sources = [s for s in proj.staged_sources if s.id != sid]
    if data.get("decision") == "accept":
        proj.sources.append(enrich_source_links(src))
        action, details = "Accepted staged source", f"Added '{src.title}' to the library"
    else:
        # Drop citations of the rejected source from pending proposals.
        for p in proj.proposals:
            if p.status in ("pending", "invalid") and p.edge and sid in p.edge.citation_ids:
                p.edge.evidence_links = [l for l in p.edge.evidence_links if l.source_id != sid]
                p.edge.citation_ids = [l.source_id for l in p.edge.evidence_links]
                p.validation = validate_proposal(proj, p)
        action, details = "Rejected staged source", f"Discarded '{src.title}'"
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action=action, details=details, graph_version=proj.graph_version))
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


# ---------------------------------------------------------------------------
# Graph versions
# ---------------------------------------------------------------------------


async def api_versions(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    if request.method == "GET":
        return JSONResponse([v.model_dump() for v in proj.versions])
    data = await request.json()
    v = next((x for x in proj.versions if x.version == int(data.get("version", -1))), None)
    if not v:
        return _not_found("Version")
    proj.nodes = [n.model_copy(deep=True) for n in v.nodes]
    proj.edges = [e.model_copy(deep=True) for e in v.edges]
    new_v = proj.record_version("researcher", f"Restored version {v.version}")
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action="Restored graph version",
                                           details=f"Restored version {v.version} as version {new_v}", graph_version=new_v))
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


# ---------------------------------------------------------------------------
# PDF upload and open-access lookup
# ---------------------------------------------------------------------------


UPLOAD_DIR = DATA_DIR / "uploads"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name or "file")[:120]


async def api_upload_pdf(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"detail": "Attach a PDF in the 'file' field."}, status_code=400)
    raw = await upload.read()
    if not raw.startswith(b"%PDF"):
        return JSONResponse({"detail": "That file is not a PDF."}, status_code=400)
    if len(raw) > da.MAX_UPLOAD_BYTES:
        return JSONResponse({"detail": "PDFs are limited to 50 MB."}, status_code=413)
    folder = UPLOAD_DIR / proj.id / "pdf"
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{os.urandom(4).hex()}_{_safe_name(upload.filename)}"
    path = folder / fname
    path.write_bytes(raw)
    try:
        info = extract_pdf(str(path))
    except Exception as exc:
        return JSONResponse({"detail": f"Could not read the PDF: {exc}"}, status_code=422)
    title = (form.get("title") or info["title"] or upload.filename or "Uploaded PDF").strip()
    abstract = info["abstract"] or info["text"][:1500]
    src = CitationSource(
        title=title[:300],
        authors=[a.strip() for a in re.split(r"[;,]| and ", info["author"]) if a.strip()][:8] or ["Unknown author"],
        doi=info["doi"],
        abstract=abstract,
        full_text_snippet=info["text"][:6000],
        study_design=classify_study_design(title, info["text"][:8000]),
        sample_size=extract_sample_size(info["text"][:8000]),
        key_finding=abstract[:280],
        source_engine="Uploaded PDF",
        local_pdf=fname,
        added_by="researcher",
    )
    src.injection_flags = scan(info["text"][:20000])
    src = enrich_source_links(src)
    src.ai_summary = _heuristic_paper_summary(src)
    proj.sources.append(src)
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action="Uploaded PDF",
                                           details=f"'{src.title}' ({info['pages']} pages){'; instruction-like text flagged' if src.injection_flags else ''}"))
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_source_pdf(request: Request) -> FileResponse | JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    src = next((s for s in proj.sources if s.id == request.path_params["source_id"] and s.local_pdf), None)
    if not src:
        return _not_found("PDF")
    path = UPLOAD_DIR / proj.id / "pdf" / _safe_name(src.local_pdf)
    if not path.exists():
        return _not_found("PDF file")
    return FileResponse(str(path), media_type="application/pdf")


async def api_find_open_access(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    src = next((s for s in proj.sources if s.id == request.path_params["source_id"]), None)
    if not src:
        return _not_found("Source")
    await find_open_access(src)
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


# ---------------------------------------------------------------------------
# Datasets: testing the graph against data, and estimation
# ---------------------------------------------------------------------------


def _dataset(proj: ProjectState, ds_id: str) -> Optional[DatasetMeta]:
    return next((d for d in proj.datasets if d.id == ds_id), None)


async def api_upload_dataset(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        return JSONResponse({"detail": "Attach a CSV in the 'file' field."}, status_code=400)
    raw = await upload.read()
    if len(raw) > da.MAX_UPLOAD_BYTES:
        return JSONResponse({"detail": "Datasets are limited to 50 MB."}, status_code=413)
    folder = UPLOAD_DIR / proj.id / "data"
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{os.urandom(4).hex()}_{_safe_name(upload.filename)}"
    path = folder / fname
    path.write_bytes(raw)
    try:
        df = da.load_csv(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        return JSONResponse({"detail": f"Could not read the CSV: {exc}"}, status_code=422)
    meta = DatasetMeta(
        filename=upload.filename or fname,
        path=fname,
        n_rows=len(df),
        columns=da.profile(df),
        mapping=da.auto_map(proj.nodes, list(df.columns)),
    )
    proj.datasets = [meta]  # one active dataset per study keeps analyses unambiguous
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action="Uploaded dataset",
                                           details=f"{meta.filename}: {meta.n_rows} rows, {len(meta.columns)} columns; {len(meta.mapping)} variables matched"))
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


def _load_df(proj: ProjectState, meta: DatasetMeta):
    return da.load_csv(UPLOAD_DIR / proj.id / "data" / _safe_name(meta.path))


async def api_dataset_item(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    meta = _dataset(proj, request.path_params["dataset_id"])
    if not meta:
        return _not_found("Dataset")
    if request.method == "DELETE":
        (UPLOAD_DIR / proj.id / "data" / _safe_name(meta.path)).unlink(missing_ok=True)
        proj.datasets = [d for d in proj.datasets if d.id != meta.id]
    else:
        data = await request.json()
        cols = {c["name"] for c in meta.columns}
        node_ids = {n.id for n in proj.nodes}
        meta.mapping = {k: v for k, v in (data.get("mapping") or {}).items() if k in node_ids and v in cols}
    save_project(proj)
    return JSONResponse(_enrich_project_response(proj))


async def api_test_implications(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    meta = _dataset(proj, request.path_params["dataset_id"])
    if not meta:
        return _not_found("Dataset")
    data = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    diag = CausalEngine(proj.nodes, proj.edges).run_diagnostics(proj.estimand_type)
    if not diag["is_dag"]:
        return JSONResponse({"detail": "The graph has a cycle, so it implies no independencies to test."}, status_code=409)
    implications = diag.get("implied_independencies") or []
    if not implications:
        from . import causal_core as cc
        eng = CausalEngine(proj.nodes, proj.edges)
        implications = [
            {"a": eng._fmt_set([i["a"]])[0], "b": eng._fmt_set([i["b"]])[0], "given": eng._fmt_set(i["given"])}
            for i in cc.implied_independencies(eng.cg)
        ]
    df = _load_df(proj, meta)
    method = data.get("method", "auto") if data.get("method") in ("auto", "gcm", "partial_correlation") else "auto"
    res = da.check_implications(df, meta.mapping, implications, method=method, alpha=float(data.get("alpha", 0.05)))
    res["graph_version"] = proj.graph_version
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action="Tested graph against data",
                                           details=f"{res['tested']} implications tested, {res['rejected']} rejected (Holm, alpha {res['alpha']})",
                                           graph_version=proj.graph_version))
    save_project(proj)
    return JSONResponse(res)


async def api_estimate(request: Request) -> JSONResponse:
    proj = get_project(request.path_params["project_id"])
    if not proj:
        return _not_found()
    meta = _dataset(proj, request.path_params["dataset_id"])
    if not meta:
        return _not_found("Dataset")
    diag = CausalEngine(proj.nodes, proj.edges).run_diagnostics(proj.estimand_type)
    if proj.estimand_type != "total_effect":
        return JSONResponse({"detail": "Estimation currently supports the total effect only. Switch the estimand to Total."}, status_code=409)
    if not diag.get("exposure") or not diag.get("outcome") or not diag["is_dag"]:
        return JSONResponse({"detail": "Mark an exposure and an outcome in an acyclic graph first."}, status_code=409)
    x, y = diag["exposure"]["id"], diag["outcome"]["id"]
    if diag["current_adjustment_valid"]:
        adj = [a["id"] for a in diag["current_adjusted"]]
        which = "your current adjustment set"
    elif diag["minimal_adjustment_sets"]:
        adj = [a["id"] for a in diag["minimal_adjustment_sets"][0]]
        which = "the first minimal adjustment set (your current set is not valid)"
    else:
        return JSONResponse({"detail": "The effect is not identifiable by adjustment in this graph, so no estimate is produced."}, status_code=409)
    unmapped = [v for v in [x, y, *adj] if v not in meta.mapping]
    if unmapped:
        labels = {n.id: n.label for n in proj.nodes}
        return JSONResponse({"detail": "Map these variables to columns first: " + ", ".join(labels.get(v, v) for v in unmapped)}, status_code=409)
    df = _load_df(proj, meta)
    try:
        res = da.estimate_effect(df, meta.mapping, x, y, adj)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    labels = {n.id: n.label for n in proj.nodes}
    res["adjustment_source"] = which
    res["adjusted_for_labels"] = [labels.get(a, a) for a in adj]
    res["exposure"], res["outcome"] = labels.get(x, x), labels.get(y, y)
    res["graph_version"] = proj.graph_version
    dml = res["estimators"]["dml"]
    proj.audit_log.insert(0, AuditLogEntry(actor="researcher", action="Estimated effect from data",
                                           details=f"DML estimate {dml['estimate']:.4g} (95% CI {dml['ci_low']:.4g} to {dml['ci_high']:.4g}), n={res['n']}",
                                           graph_version=proj.graph_version))
    save_project(proj)
    return JSONResponse(res)


FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIR.mkdir(parents=True, exist_ok=True)


async def serve_spa(request: Request) -> FileResponse:
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@asynccontextmanager
async def lifespan(app: Starlette):
    init_db()
    yield


routes = [
    Route("/api/health", health_check, methods=["GET"]),
    Route("/api/settings", api_get_settings, methods=["GET"]),
    Route("/api/settings", api_update_settings, methods=["POST"]),
    Route("/api/projects", api_projects_collection, methods=["GET", "POST"]),
    Route("/api/projects/{project_id}", api_project_item, methods=["GET", "PUT", "DELETE"]),
    Route("/api/projects/{project_id}/apply-adjustment-set", api_apply_adjustment_set, methods=["POST"]),
    Route("/api/projects/{project_id}/proposals/{proposal_id}/decide", api_decide_proposal, methods=["POST"]),
    Route("/api/projects/{project_id}/worldviews", api_manage_worldviews, methods=["POST"]),
    Route("/api/projects/{project_id}/chat", api_chat_turn, methods=["POST"]),
    Route("/api/projects/{project_id}/deep-research", api_deep_research, methods=["POST"]),
    Route("/api/projects/{project_id}/sources/{source_id}/summarize", api_summarize_source, methods=["POST"]),
    Route("/api/projects/{project_id}/lab-briefing", api_generate_lab_briefing, methods=["POST"]),
    Route("/api/projects/{project_id}/export", api_export_project, methods=["GET"]),
    Route("/api/projects/{project_id}/import-dagitty", api_import_dagitty, methods=["POST"]),
    Route("/api/projects/{project_id}/staged/{source_id}/decide", api_decide_staged, methods=["POST"]),
    Route("/api/projects/{project_id}/versions", api_versions, methods=["GET", "POST"]),
    Route("/api/projects/{project_id}/sources/upload", api_upload_pdf, methods=["POST"]),
    Route("/api/projects/{project_id}/sources/{source_id}/pdf", api_source_pdf, methods=["GET"]),
    Route("/api/projects/{project_id}/sources/{source_id}/open-access", api_find_open_access, methods=["POST"]),
    Route("/api/projects/{project_id}/datasets", api_upload_dataset, methods=["POST"]),
    Route("/api/projects/{project_id}/datasets/{dataset_id}", api_dataset_item, methods=["PUT", "DELETE"]),
    Route("/api/projects/{project_id}/datasets/{dataset_id}/implications", api_test_implications, methods=["POST"]),
    Route("/api/projects/{project_id}/datasets/{dataset_id}/estimate", api_estimate, methods=["POST"]),
    Route("/api/literature/search", api_search_literature, methods=["GET"]),
    Route("/", serve_spa, methods=["GET"]),
    Mount("/static", app=StaticFiles(directory=str(FRONTEND_DIR)), name="static"),
]

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = Starlette(debug=False, routes=routes, middleware=middleware, lifespan=lifespan)


def run() -> None:
    init_db()
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    open_browser = "--open" in sys.argv
    if open_browser:
        url = f"http://127.0.0.1:{port}"
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run("backend.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    run()
