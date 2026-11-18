import os
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
)
from .causal_engine import CausalEngine
from .literature_service import (
    search_scholarly_literature,
    summarize_paper_with_ai,
    format_in_text_citation,
    format_reference_entry,
    enrich_source_links,
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
    payload["diagnostics"] = diagnostics
    payload["formatted_sources"] = formatted_sources
    return payload


async def health_check(request: Request) -> JSONResponse:
    return JSONResponse(
        {"status": "ok", "service": "DAG_Brain Causal Studio", "version": "0.2.0"}
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
            epistemic_status="hypothesized",
            mechanism="Primary hypothesized causal effect",
            effect_direction="unknown",
        )
    ]

    proj = ProjectState(
        title=title,
        discipline=discipline,
        research_question=rq,
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
                    "You can use **🎯 Narrow Topic** to refine your research question, or click **🚀 Autonomous Web & Lit Research** to have me search OpenAlex, CORE.ac.uk, PubMed, Google Scholar, and Tavily automatically!"
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

    if data.get("audit_note"):
        existing.audit_log.insert(
            0,
            AuditLogEntry(
                actor="researcher",
                action=data.get("audit_action", "Updated Model"),
                details=str(data["audit_note"]),
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

    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="researcher",
            action="Applied Minimal Sufficient Adjustment Set",
            details=f"Conditioned on: {', '.join(labels) if labels else 'Empty Set (No adjustment needed)'}",
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
    target_prop: Optional[GraphPatchProposal] = None
    for p in proj.proposals:
        if p.id == proposal_id:
            target_prop = p
            break

    if not target_prop:
        return JSONResponse({"detail": "Proposal not found"}, status_code=404)

    if decision == "accept":
        target_prop.status = "accepted"
        if target_prop.action == "add_node" and target_prop.node:
            existing_ids = {n.id for n in proj.nodes}
            if target_prop.node.id not in existing_ids:
                proj.nodes.append(target_prop.node)
        elif target_prop.action == "add_edge" and target_prop.edge:
            edge = target_prop.edge
            edge.approved_by_researcher = True
            if edge.epistemic_status == "ai_proposed":
                edge.epistemic_status = "hypothesized"
            proj.edges.append(edge)
        proj.audit_log.insert(
            0,
            AuditLogEntry(
                actor="ai_accepted",
                action=f"Accepted AI Proposal: {target_prop.title}",
                details=target_prop.rationale,
            ),
        )
    else:
        target_prop.status = "rejected"
        proj.audit_log.insert(
            0,
            AuditLogEntry(
                actor="researcher",
                action=f"Rejected AI Proposal: {target_prop.title}",
                details=data.get("reason", "Rejected by researcher during review."),
            ),
        )

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

    new_prop_ids = []
    for raw_p in turn_result.get("proposals", []):
        prop_obj = GraphPatchProposal.model_validate(raw_p)
        proj.proposals.insert(0, prop_obj)
        new_prop_ids.append(prop_obj.id)

    assistant_msg = ChatMessage(
        role="assistant",
        agent_mode=agent_mode,
        content=turn_result.get("reply_markdown", ""),
        socratic_questions=turn_result.get("socratic_questions", []),
        proposal_ids=new_prop_ids,
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

    new_prop_ids = []
    for raw_p in res.get("proposals", []):
        prop_obj = GraphPatchProposal.model_validate(raw_p)
        proj.proposals.insert(0, prop_obj)
        new_prop_ids.append(prop_obj.id)

    proj.chat_history.append(
        ChatMessage(
            role="assistant",
            agent_mode="deep_research",
            content=res.get("reply_markdown", ""),
            socratic_questions=res.get("socratic_questions", []),
            proposal_ids=new_prop_ids,
        )
    )
    proj.audit_log.insert(
        0,
        AuditLogEntry(
            actor="ai_accepted",
            action="Completed Autonomous Deep Research",
            details=f"Saved {len(res.get('fetched_papers', []))} new literature sources with AI summaries.",
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
                f"| {u_lbl} → {v_lbl} | `{e.epistemic_status}` | {e.mechanism or e.rationale or '-'} | {', '.join(cites) or 'Uncited'} |"
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

## 1. Causal Identification Summary
- **Valid Acyclic Graph (DAG):** {diagnostics.get('is_dag')}
- **Minimal Sufficient Adjustment Sets:** {min_sets_str}
- **Currently Open Backdoor Paths:** {len(diagnostics.get('open_backdoor_paths', []))}

## 2. Edge-by-Edge Epistemic Audit Table
| Directed Edge | Epistemic Status | Mechanism / Rationale | Citations |
| :--- | :--- | :--- | :--- |
{chr(10).join(edge_rows)}

---

## 3. Research Notebook
{proj.notebook_markdown}

---

## 4. Bibliography ({proj.citation_style.upper()})
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
