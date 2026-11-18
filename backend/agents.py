import re
from typing import Dict, Any, List, Optional
from .schemas import (
    ProjectState,
    CausalNode,
    CausalEdge,
    GraphPatchProposal,
    CitationSource,
)
from .causal_engine import CausalEngine
from .literature_service import (
    search_scholarly_literature,
    search_tavily_and_web,
    summarize_paper_with_ai,
    format_in_text_citation,
    enrich_source_links,
)
from .llm_router import call_llm_json


SYSTEM_PROMPT_TEMPLATE = """You are DAG_Brain, an elite AI Causal Modelling Collaborator, Methodologist, and Socratic Research Assistant.
You combine the mathematical rigor of Judea Pearl's Structural Causal Models (DAGitty) with the source-grounded synthesis of NotebookLM.

ACTIVE AGENT PERSONA: {agent_mode_upper}
- If TOPIC_REFINER: Help the researcher narrow a broad research interest into 2-3 sharp, identifiable causal questions using PICO/PECO (Population, Exposure, Comparator, Outcome, Time-window) and FINER criteria. Propose concrete Exposure (X), Outcome (Y), and Confounder nodes.
- If SOCRATIC: Interview the researcher thoughtfully. Clarify variable operationalizations, temporal ordering (t0 < t1 < t2), target population, and whether proposed variables are confounders, mediators, or colliders before jumping to conclusions.
- If CRITIC (Devil's Advocate / Reviewer 2): Actively challenge the researcher's DAG. Point out open backdoor paths, unmeasured confounding (U), reverse causality, selection bias, overadjustment on mediators, or collider conditioning traps (M-bias / Berkson's bias).
- If SCOUT (Literature & Internet Research): Synthesize empirical evidence from the retrieved scholarly papers (OpenAlex, CORE.ac.uk, PubMed, Google Scholar) and web search (Tavily). Distinguish observational correlation from causal identification (RCT, Mendelian Randomization, DiD, Longitudinal Cohort). Attach citations using {citation_style_upper} style.
- If SYNTHESIS: Draft structured Markdown rationale sections and methodology text suitable for a peer-reviewed journal's Methods / Supplementary section, citing sources in {citation_style_upper} format.

CRITICAL RULES:
1. Never mutate the graph silently. Propose changes via the `proposals` array so the researcher can review and approve them.
2. Explicitly distinguish between Empirical evidence (backed by citations), Researcher hypotheses, and AI-inferred assumptions.
3. Return ONLY valid JSON matching this exact schema:
{{
  "reply_markdown": "Your rich Markdown response (use bolding, bullet points, and inline citations in {citation_style_upper} format)...",
  "socratic_questions": [
    "Question 1 challenging or clarifying a causal assumption?",
    "Question 2 about measurement, time-ordering, or confounding?"
  ],
  "proposals": [
    {{
      "action": "add_node",
      "title": "Add Confounder: [Label]",
      "rationale": "Why this variable is necessary in the causal structure",
      "critic_warning": "Optional methodological caveat",
      "node": {{
        "id": "snake_case_id",
        "label": "Human Readable Name",
        "role": "confounder | mediator | collider | exposure | outcome | instrument | latent",
        "time_index": 0,
        "operationalization": "How it is measured"
      }}
    }},
    {{
      "action": "add_edge",
      "title": "Propose Edge: Source -> Target",
      "rationale": "Mechanism and empirical justification",
      "critic_warning": "Study design limitation or reverse causality risk",
      "edge": {{
        "source": "source_node_id",
        "target": "target_node_id",
        "epistemic_status": "ai_proposed",
        "mechanism": "Biological/economic/behavioral mechanism",
        "effect_direction": "positive | negative | non_linear | unknown",
        "confidence": 0.8
      }}
    }}
  ],
  "notebook_append_suggestion": "Optional Markdown paragraph the researcher can insert into their Research Notebook with 1 click."
}}
"""


async def run_agent_turn(
    project: ProjectState,
    user_message: str,
    agent_mode: str = "socratic",
    provider_override: Optional[str] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    engine = CausalEngine(project.nodes, project.edges)
    diagnostics = engine.run_diagnostics(estimand=project.estimand_type)

    fetched_papers: List[CitationSource] = []
    web_answer = ""
    lower_msg = user_message.lower()
    should_search_lit = (
        agent_mode in ("scout", "topic_refiner")
        or any(
            kw in lower_msg
            for kw in [
                "paper",
                "literature",
                "study",
                "studies",
                "evidence",
                "cite",
                "citation",
                "pubmed",
                "scholar",
                "core",
                "search",
                "research",
                "internet",
                "web",
                "narrow",
                "topic",
            ]
        )
    )
    if should_search_lit:
        search_q = user_message
        if len(search_q) > 120 or len(search_q.split()) < 2:
            exp_lbl = (diagnostics.get("exposure") or {}).get("label", "")
            out_lbl = (diagnostics.get("outcome") or {}).get("label", "")
            search_q = f"{exp_lbl} {out_lbl} {project.research_question}".strip()
        if search_q:
            fetched_papers = await search_scholarly_literature(
                search_q, limit=5, engine="all"
            )
            tavily_res = await search_tavily_and_web(search_q, limit=3)
            web_answer = tavily_res.get("answer") or ""
            for ws in tavily_res.get("sources", []):
                fetched_papers.append(ws)

    nodes_summary = [
        f"- {n.id} ('{n.label}', role={n.role}, adjusted={n.adjusted}, observed={n.observed}, t={n.time_index})"
        for n in project.nodes
    ]
    edges_summary = [
        f"- {e.source} -> {e.target} (status={e.epistemic_status}, mechanism='{e.mechanism}', citations={len(e.citation_ids)})"
        for e in project.edges
    ]
    sources_summary = [
        f"- [{s.id}] {format_in_text_citation(s, project.citation_style, i+1)} '{s.title}' ({s.study_design}, Engine:{s.source_engine}) DOI:{s.doi}"
        for i, s in enumerate(project.sources[:15])
    ]
    fetched_summary = [
        f"- [NEW/{s.source_engine}] '{s.title}' ({s.year}, {s.journal}, Design: {s.study_design}, DOI: {s.doi}): {s.key_finding}"
        for s in fetched_papers
    ]

    user_prompt = f"""PROJECT CONTEXT:
Title: {project.title}
Discipline: {project.discipline}
Research Question: {project.research_question}
Estimand: {project.estimand_type}
Citation Style: {project.citation_style.upper()}

CURRENT NODES ({len(project.nodes)}):
{chr(10).join(nodes_summary) if nodes_summary else "(None yet)"}

CURRENT EDGES ({len(project.edges)}):
{chr(10).join(edges_summary) if edges_summary else "(None yet)"}

CAUSAL ENGINE DIAGNOSTICS:
- Is valid DAG: {diagnostics.get('is_dag')}
- Open Backdoor Paths: {len(diagnostics.get('open_backdoor_paths', []))}
- Minimal Sufficient Adjustment Sets: {diagnostics.get('minimal_adjustment_sets')}
- Active Warnings: {[w.get('type') + ': ' + w.get('message') for w in diagnostics.get('warnings', [])]}

EXISTING CITATION VAULT:
{chr(10).join(sources_summary) if sources_summary else "(Empty)"}

NEWLY RETRIEVED SCHOLARLY & WEB SOURCES:
{chr(10).join(fetched_summary) if fetched_summary else "(None fetched this turn)"}
{f"TAVILY WEB SYNTHESIS: {web_answer}" if web_answer else ""}

RESEARCHER MESSAGE:
{user_message}
"""

    sys_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_mode_upper=agent_mode.upper(),
        citation_style_upper=project.citation_style.upper(),
    )

    llm_data = await call_llm_json(
        system_prompt=sys_prompt,
        user_prompt=user_prompt,
        provider_override=provider_override,
        model_override=model_override,
    )

    if llm_data and isinstance(llm_data, dict) and llm_data.get("reply_markdown"):
        return _normalize_agent_output(
            llm_data, project, fetched_papers, agent_mode, used_llm=True
        )

    return _run_deterministic_copilot(
        project,
        user_message,
        agent_mode,
        diagnostics,
        fetched_papers,
        web_answer=web_answer,
    )


async def run_autonomous_deep_research(
    project: ProjectState, custom_focus: str = ""
) -> Dict[str, Any]:
    """
    Autonomous Multi-Hop Internet & Scholarly Research Agent:
    1. Formulates 3 targeted queries (Direct Causal Effect, Confounders/Bias, Instruments/Mechanisms).
    2. Searches OpenAlex, CORE.ac.uk, PubMed/EuropePMC, Google Scholar/S2, and Tavily Web.
    3. Saves top discovered papers directly into the project's Citation Vault with AI Executive Summaries.
    4. Proposes new evidence-backed Confounder/Mediator nodes and edges.
    """
    engine = CausalEngine(project.nodes, project.edges)
    diagnostics = engine.run_diagnostics(estimand=project.estimand_type)
    exp = diagnostics.get("exposure")
    out = diagnostics.get("outcome")
    exp_label = exp["label"] if isinstance(exp, dict) else project.title
    out_label = out["label"] if isinstance(out, dict) else "Outcome"
    exp_id = exp["id"] if isinstance(exp, dict) else None
    out_id = out["id"] if isinstance(out, dict) else None

    base_focus = custom_focus.strip() or f"{exp_label} {out_label}"
    queries = [
        f"{base_focus} causal effect cohort randomized",
        f"{base_focus} confounding mediator mechanism",
    ]

    discovered_sources: List[CitationSource] = []
    web_syntheses: List[str] = []

    for q in queries:
        papers = await search_scholarly_literature(q, limit=4, engine="all")
        discovered_sources.extend(papers)
        web_res = await search_tavily_and_web(q, limit=2)
        if web_res.get("answer"):
            web_syntheses.append(web_res["answer"])
        discovered_sources.extend(web_res.get("sources", []))

    # Deduplicate against existing vault and save up to 5 new high-value papers
    existing_keys = {
        (s.doi.lower().strip() if s.doi else re.sub(r"[^a-z0-9]", "", s.title.lower())[:50])
        for s in project.sources
    }
    newly_saved: List[CitationSource] = []
    for cand in discovered_sources:
        key = (
            cand.doi.lower().strip()
            if cand.doi
            else re.sub(r"[^a-z0-9]", "", cand.title.lower())[:50]
        )
        if not key or key in existing_keys:
            continue
        existing_keys.add(key)
        cand = enrich_source_links(cand)
        cand.ai_summary = await summarize_paper_with_ai(cand)
        project.sources.append(cand)
        newly_saved.append(cand)
        if len(newly_saved) >= 4:
            break

    # Build executive research report and proposals
    lines = [
        f"### 🌐 Autonomous Deep Research Report: *{base_focus}*",
        f"I autonomously queried **OpenAlex**, **CORE.ac.uk**, **PubMed/EuropePMC**, **Google Scholar**, and **Tavily Web Search**, saving **{len(newly_saved)} new peer-reviewed sources** directly into your Study Vault with AI Executive Summaries:",
    ]

    for idx, src in enumerate(newly_saved):
        cite_str = format_in_text_citation(
            src, project.citation_style, len(project.sources) - len(newly_saved) + idx + 1
        )
        tldr = src.ai_summary.tldr if src.ai_summary else src.key_finding
        grade = src.ai_summary.methodology_grade if src.ai_summary else src.study_design
        lines.append(
            f"{idx + 1}. **{src.title}** {cite_str} — *{src.journal}* (`{src.source_engine}`)\n"
            f"   - **Evidence Grade:** `{grade}` {f'| `{src.sample_size}`' if src.sample_size else ''}\n"
            f"   - **AI Digest:** {tldr}\n"
            f"   - **Full-Text Links:** [Google Scholar]({src.scholar_url}) • [CORE.ac.uk]({src.core_url}) • [Sci-Hub Mirror]({src.scihub_url})"
        )

    if web_syntheses:
        lines.append(f"\n#### 🌍 Live Internet Synthesis (Tavily)\n> {web_syntheses[0]}")

    proposals: List[GraphPatchProposal] = []
    if newly_saved and exp_id and out_id:
        top_paper = newly_saved[0]
        proposals.append(
            GraphPatchProposal(
                action="add_edge",
                title=f"Attach Empirical Evidence ({top_paper.year or 'Study'}) to {exp_label} → {out_label}",
                rationale=f"Backed by '{top_paper.title}' ({top_paper.study_design}): {top_paper.ai_summary.tldr if top_paper.ai_summary else top_paper.key_finding}",
                critic_warning=top_paper.ai_summary.limitations_and_bias if top_paper.ai_summary else "",
                edge=CausalEdge(
                    source=exp_id,
                    target=out_id,
                    epistemic_status="empirical",
                    mechanism=f"Supported by {top_paper.authors[0] if top_paper.authors else 'Literature'} ({top_paper.year or 'n.d.'})",
                    citation_ids=[top_paper.id],
                    approved_by_researcher=False,
                ),
            )
        )

    return {
        "reply_markdown": "\n\n".join(lines),
        "socratic_questions": [
            f"Having reviewed these {len(newly_saved)} newly saved studies, should we upgrade any hypothesized edges to `Empirical`?",
            "Which baseline confounders controlled for in the cohort studies are currently missing from our DAG?",
        ],
        "proposals": [p.model_dump() for p in proposals],
        "fetched_papers": [s.model_dump() for s in newly_saved],
        "agent_mode": "deep_research",
    }


async def generate_lab_briefing(project: ProjectState) -> str:
    """
    Generates a NotebookLM-style 2-Host Audio/Text Briefing Script where
    Dr. Aria Vance (PI) and Prof. Marcus Sterling (Methodologist / Reviewer #2)
    discuss the researcher's DAG, open backdoor paths, and saved literature.
    """
    engine = CausalEngine(project.nodes, project.edges)
    diag = engine.run_diagnostics(project.estimand_type)
    exp_lbl = (diag.get("exposure") or {}).get("label", "the primary exposure")
    out_lbl = (diag.get("outcome") or {}).get("label", "the primary outcome")
    open_bd = diag.get("open_backdoor_paths", [])
    min_sets = diag.get("minimal_adjustment_sets", [])
    ivs = diag.get("instrumental_variables", [])

    sys_prompt = """You are generating a lively, intellectually rigorous 2-Host Research Lab Briefing (inspired by Google NotebookLM Audio Overview, tailored for causal inference scientists).
Host 1: Dr. Aria Vance (Lead Investigator — synthesizes the big picture, mechanisms, and empirical papers).
Host 2: Prof. Marcus Sterling (Causal Methodologist & Reviewer #2 — focuses on d-separation, backdoor paths, collider bias traps, and study design rigor).
Return ONLY valid JSON: {"briefing_markdown": "The full formatted dialogue in Markdown..."}"""

    sources_text = "\n".join(
        f"- {s.title} ({s.year}, {s.study_design}): {s.ai_summary.tldr if s.ai_summary else s.key_finding}"
        for s in project.sources[:8]
    )
    user_prompt = f"""Study: {project.title} ({project.discipline})
Question: {project.research_question}
Exposure: {exp_lbl} -> Outcome: {out_lbl}
Open Backdoor Paths: {len(open_bd)}
Minimal Adjustment Sets: {min_sets}
Valid Instruments: {ivs}
Saved Literature:
{sources_text}
"""
    llm_res = await call_llm_json(sys_prompt, user_prompt)
    if llm_res and isinstance(llm_res, dict) and llm_res.get("briefing_markdown"):
        return str(llm_res["briefing_markdown"])

    # Deterministic Rich Lab Briefing Script
    set_str = (
        ", ".join("{" + ", ".join(x["label"] for x in s) + "}" for s in min_sets[:2])
        if min_sets
        else "no purely observational set without instruments"
    )
    iv_str = ", ".join(i["label"] for i in ivs) if ivs else "none currently modeled"

    script = [
        f"# 🎙️ Causal Studio Lab Briefing: {project.title}",
        f"*A synthesized NotebookLM-style research dialogue reviewing your current DAG ({len(project.nodes)} nodes, {len(project.edges)} edges) and {len(project.sources)} vault sources.*\n",
        f"**👩‍🔬 Dr. Aria Vance (Lead PI):** Welcome back to the lab! Today we're reviewing our structural causal model on **{project.title}** within *{project.discipline}*. Specifically, we're trying to isolate the **{project.estimand_type.replace('_', ' ')}** of **{exp_lbl}** on **{out_lbl}**.",
        f"**👨‍🏫 Prof. Marcus Sterling (Methodologist):** And let's look straight at the graph math before anyone runs a naive regression. Right now, our DAG contains **{len(project.nodes)} variables** across temporal waves $t_0$ through $t_3$. When I run Pearl's $d$-separation check on the current conditioning set, there are **{len(open_bd)} open non-causal backdoor paths**.",
    ]

    if open_bd:
        script.append(
            f"**👩‍🔬 Dr. Aria Vance:** Right—specifically the path `{open_bd[0]['pretty']}` is still unblocked! To close every backdoor path without inducing overadjustment bias, our minimal sufficient adjustment set is **{set_str}**."
        )
    else:
        script.append(
            f"**👩‍🔬 Dr. Aria Vance:** That's the good news: all modeled backdoor paths are currently blocked using **{set_str}**!"
        )

    if project.sources:
        s0 = project.sources[0]
        c0 = format_in_text_citation(s0, project.citation_style, 1)
        script.append(
            f"**👨‍🏫 Prof. Marcus Sterling:** Let's check how well the empirical literature in our Vault supports those arrows. Our anchor source is *{s0.title}* {c0}, classified as **{s0.study_design}**. {s0.ai_summary.tldr if s0.ai_summary else s0.key_finding}"
        )
        if len(project.sources) > 1:
            s1 = project.sources[1]
            c1 = format_in_text_citation(s1, project.citation_style, 2)
            script.append(
                f"**👩‍🔬 Dr. Aria Vance:** And complementing that, we have *{s1.title}* {c1} (`{s1.study_design}`), which helps anchor our upstream confounding structure: {s1.ai_summary.tldr if s1.ai_summary else s1.key_finding}"
            )

    script.append(
        f"**👨‍🏫 Prof. Marcus Sterling:** Two final methodological reminders before we lock this protocol: first, never condition on any $t_2$ mediator when estimating the total effect, and watch out for $t_3$ colliders. Second, if Reviewer #2 argues there is residual unmeasured confounding ($U$), our candidate Instrumental Variable on the canvas is **{iv_str}**."
    )
    return "\n\n".join(script)


def _normalize_agent_output(
    raw: Dict[str, Any],
    project: ProjectState,
    fetched_papers: List[CitationSource],
    agent_mode: str,
    used_llm: bool = True,
) -> Dict[str, Any]:
    proposals: List[GraphPatchProposal] = []
    existing_ids = {n.id for n in project.nodes}

    for idx, p in enumerate(raw.get("proposals", []) or []):
        if not isinstance(p, dict):
            continue
        action = p.get("action", "add_node")
        title = p.get("title") or f"Proposed {action}"
        rationale = p.get("rationale") or "Suggested by causal reasoning assistant."
        critic_warning = p.get("critic_warning") or ""

        node_obj = None
        edge_obj = None

        if action == "add_node" and isinstance(p.get("node"), dict):
            nd = p["node"]
            nid = re.sub(
                r"[^a-zA-Z0-9_]",
                "_",
                str(nd.get("id") or nd.get("label", "var")).lower(),
            ).strip("_")
            label = str(nd.get("label") or nid)
            role = nd.get("role", "confounder")
            if role not in (
                "exposure",
                "outcome",
                "confounder",
                "mediator",
                "collider",
                "instrument",
                "effect_modifier",
                "competing_exposure",
                "latent",
                "other",
            ):
                role = "confounder"
            node_obj = CausalNode(
                id=nid,
                label=label,
                role=role,
                time_index=int(
                    nd.get("time_index", 0 if role == "confounder" else 1)
                ),
                operationalization=str(nd.get("operationalization", "")),
                observed=(role != "latent"),
                x=220.0 + ((len(existing_ids) + idx) % 4) * 180.0,
                y=120.0 + ((len(existing_ids) + idx) // 4) * 130.0,
            )
        elif action == "add_edge" and isinstance(p.get("edge"), dict):
            ed = p["edge"]
            src = str(ed.get("source", "")).strip()
            tgt = str(ed.get("target", "")).strip()
            if src and tgt and src != tgt:
                edge_obj = CausalEdge(
                    source=src,
                    target=tgt,
                    epistemic_status="ai_proposed",
                    mechanism=str(ed.get("mechanism", "")),
                    effect_direction=(
                        ed.get("effect_direction", "unknown")
                        if ed.get("effect_direction")
                        in ("positive", "negative", "non_linear", "unknown")
                        else "unknown"
                    ),
                    confidence=float(ed.get("confidence", 0.78)),
                    rationale=rationale,
                    approved_by_researcher=False,
                )

        if node_obj or edge_obj:
            proposals.append(
                GraphPatchProposal(
                    action=action,
                    title=title,
                    rationale=rationale,
                    critic_warning=critic_warning,
                    node=node_obj,
                    edge=edge_obj,
                )
            )

    return {
        "reply_markdown": raw.get("reply_markdown", ""),
        "socratic_questions": (raw.get("socratic_questions") or [])[:4],
        "proposals": [p.model_dump() for p in proposals],
        "fetched_papers": [fp.model_dump() for fp in fetched_papers],
        "notebook_append_suggestion": raw.get("notebook_append_suggestion", ""),
        "agent_mode": agent_mode,
        "used_llm": used_llm,
    }


def _run_deterministic_copilot(
    project: ProjectState,
    user_message: str,
    agent_mode: str,
    diagnostics: Dict[str, Any],
    fetched_papers: List[CitationSource],
    web_answer: str = "",
) -> Dict[str, Any]:
    exp = diagnostics.get("exposure")
    out = diagnostics.get("outcome")
    exp_label = exp["label"] if isinstance(exp, dict) else "Exposure"
    out_label = out["label"] if isinstance(out, dict) else "Outcome"
    exp_id = exp["id"] if isinstance(exp, dict) else None
    out_id = out["id"] if isinstance(out, dict) else None

    open_backdoors = diagnostics.get("open_backdoor_paths", [])
    min_sets = diagnostics.get("minimal_adjustment_sets", [])
    warnings = diagnostics.get("warnings", [])

    proposals: List[GraphPatchProposal] = []
    questions: List[str] = []
    sections: List[str] = []

    if agent_mode == "topic_refiner":
        sections.append("### 🎯 Research Topic Refiner & PICO / PECO Question Architect")
        sections.append(
            f"Let's narrow down your research interest (**\"{user_message}\"**) into a sharp, identifiable causal model using the **PECO** (*Population, Exposure, Comparator, Outcome*) and **FINER** frameworks:"
        )
        sections.append(
            f"1. **Candidate Formulation A (Total Effect Focus):**\n"
            f"   - **Population ($P$):** Clearly defined target cohort free of outcome at baseline $t_0$.\n"
            f"   - **Exposure ($X, t_1$):** Operationalized dose/duration of **{exp_label}**.\n"
            f"   - **Outcome ($Y, t_3$):** Standardized measure of **{out_label}**.\n"
            f"2. **Candidate Formulation B (Mechanism / Mediation Focus):**\n"
            f"   - Decomposes the pathway from **{exp_label}** to **{out_label}** into direct vs. mediated effects at $t_2$.\n"
            f"3. **Candidate Formulation C (Quasi-Experimental / IV Focus):**\n"
            f"   - Uses an exogenous policy, spatial, or genetic shock ($I, t_0$) to overcome unmeasured confounding ($U$)."
        )
        if fetched_papers:
            sections.append(
                f"\n#### 📚 Relevant Literature Found ({len(fetched_papers)} papers across OpenAlex/CORE/PubMed/Scholar)"
            )
            for idx, fp in enumerate(fetched_papers[:3]):
                sections.append(
                    f"- **{fp.title}** ({fp.year or 'n.d.'}, *{fp.journal}*, `{fp.study_design}`) — [Scholar]({fp.scholar_url}) | [CORE]({fp.core_url}) | [Sci-Hub]({fp.scihub_url})"
                )
        questions = [
            f"Which of the 3 formulations above best matches your available data: Observational Cohort Adjustment, Mediation Analysis, or Quasi-Experimental (IV / DiD)?",
            f"What is the exact time lag between your Exposure ({exp_label}) and Outcome ({out_label})?",
            "Who is the specific target population, and what are the inclusion/exclusion criteria (which could create collider bias)?",
        ]

    elif agent_mode == "critic":
        sections.append(
            f"### ⚖️ Devil's Advocate / Reviewer #2 Critique (`{project.title}`)"
        )
        if warnings:
            sections.append(
                "I inspected your current DAG structure and identified the following **methodological vulnerabilities**:"
            )
            for w in warnings:
                sev_icon = (
                    "🔴"
                    if w["severity"] == "critical"
                    else ("🟠" if w["severity"] == "warning" else "🔵")
                )
                sections.append(f"- {sev_icon} **{w['type']}**: {w['message']}")
        else:
            sections.append(
                f"Your current adjustment strategy blocks all explicitly drawn backdoor paths between **{exp_label}** and **{out_label}**. However, Reviewer #2 will still challenge **omitted variables** and **functional form assumptions**."
            )

        questions = [
            f"Could reverse causality or anticipation effects exist where early subclinical manifestations of '{out_label}' alter '{exp_label}'?",
            f"Is there an unmeasured baseline common cause (e.g., genetic predisposition, institutional selection, or baseline socioeconomic trajectory) affecting both '{exp_label}' and '{out_label}'?",
            "Are any of your measured covariates subject to differential measurement error across exposure groups?",
        ]

        if exp_id and out_id:
            latent_id = "unmeasured_baseline_u"
            existing_ids = {n.id for n in project.nodes}
            if latent_id not in existing_ids:
                proposals.append(
                    GraphPatchProposal(
                        action="add_node",
                        title=f"Stress-Test: Add Unmeasured Confounder (U) between {exp_label} & {out_label}",
                        rationale="Tests whether your identification strategy survives an unobserved common cause (U). Helps evaluate if an Instrumental Variable or E-value sensitivity analysis is needed.",
                        critic_warning="Adding an unobserved confounder will intentionally show if the backdoor path becomes unidentifiable via standard regression.",
                        node=CausalNode(
                            id=latent_id,
                            label="Unmeasured Baseline Confounder (U)",
                            role="latent",
                            observed=False,
                            time_index=0,
                            operationalization="Latent / unmeasured common cause for sensitivity analysis",
                            x=380.0,
                            y=60.0,
                        ),
                    )
                )

    elif agent_mode == "scout":
        sections.append(
            f"### 📚 Federated Literature & Internet Scout ({project.citation_style.upper()} Format)"
        )
        if web_answer:
            sections.append(f"**🌐 Tavily Web Synthesis:** {web_answer}")
        if fetched_papers:
            sections.append(
                f"I searched **OpenAlex**, **CORE.ac.uk**, **PubMed/EuropePMC**, **Google Scholar**, and **Tavily** and retrieved **{len(fetched_papers)} works**:"
            )
            for idx, fp in enumerate(fetched_papers):
                cite_tag = format_in_text_citation(fp, project.citation_style, idx + 1)
                tldr = fp.ai_summary.tldr if fp.ai_summary else fp.key_finding
                sections.append(
                    f"- **{fp.title}** {cite_tag} — *{fp.journal or 'Scholarly Source'}* (`{fp.source_engine}`) | **Design:** `{fp.study_design}`\n"
                    f"  - **AI Digest:** {tldr}\n"
                    f"  - **Access:** [Google Scholar]({fp.scholar_url}) • [CORE.ac.uk]({fp.core_url}) • [Sci-Hub Mirror]({fp.scihub_url})"
                )
            sections.append(
                "\nClick **+ Add to Study Vault** in the Left Pane (or run **🚀 Autonomous Deep Research**) to save and summarize these papers!"
            )
        else:
            sections.append(
                "No external papers matched that exact query string, or you are running offline. Try searching with broader variable terms in the **Left Source Vault**."
            )
        questions = [
            f"Do the empirical studies supporting '{exp_label} -> {out_label}' rely on cross-sectional associations or longitudinal/quasi-experimental designs?",
            "Should we mark edges backed only by cross-sectional studies as `Hypothesized` rather than `Empirical`?",
        ]

    elif agent_mode == "synthesis":
        sections.append(
            f"### 📝 Causal Methodology & Justification Draft ({project.citation_style.upper()})"
        )
        set_strs = []
        for s in min_sets[:3]:
            set_strs.append("{" + ", ".join(item["label"] for item in s) + "}")
        adj_summary = (
            ", or ".join(set_strs)
            if set_strs
            else "No observable minimal set (requires IV/sensitivity analysis)"
        )

        cite_examples = [
            format_in_text_citation(s, project.citation_style, i + 1)
            for i, s in enumerate(project.sources[:3])
        ]
        cite_str = " ".join(cite_examples) if cite_examples else ""

        draft_md = (
            f"#### Causal Identification Strategy\n"
            f"To estimate the **{project.estimand_type.replace('_', ' ')}** of **{exp_label}** on **{out_label}**, "
            f"we constructed a formal Directed Acyclic Graph (DAG) representing our domain assumptions {cite_str}. "
            f"Using Pearl's d-separation criteria, we identified **{len(open_backdoors)}** currently unblocked non-causal path(s) "
            f"and computed the minimal sufficient adjustment set(s): **{adj_summary}**.\n\n"
            f"#### Edge-Level Epistemic Audit\n"
        )
        node_lookup = {n.id: n.label for n in project.nodes}
        for e in project.edges[:8]:
            u_name = node_lookup.get(e.source, e.source)
            v_name = node_lookup.get(e.target, e.target)
            draft_md += f"- **{u_name} $\\rightarrow$ {v_name}** (`{e.epistemic_status}`): {e.mechanism or e.rationale or 'Hypothesized structural link.'}\n"

        sections.append(draft_md)
        questions = [
            "Would you like to append this identification section directly into your Research Notebook?",
            "Should we fork a Competing Worldview DAG to document how the adjustment set changes under Reviewer #2's alternative assumptions?",
        ]

    else:
        sections.append("### 🧠 Socratic Causal Co-Investigator")
        sections.append(
            f"Let's reason carefully through your question: *\"{user_message}\"* within the context of **{exp_label} $\\rightarrow$ {out_label}**."
        )
        if open_backdoors:
            b0 = open_backdoors[0]["pretty"]
            sections.append(
                f"Right now, your graph has **{len(open_backdoors)} open backdoor path(s)**, including:\n- `{b0}`\n\n"
                f"Minimal sufficient adjustment set(s) to block all backdoor paths for the **{project.estimand_type.replace('_', ' ')}**: "
                + (
                    ", ".join(
                        "{" + ", ".join(x["label"] for x in ms) + "}"
                        for ms in min_sets[:3]
                    )
                    if min_sets
                    else "*None currently identifiable with observed variables.*"
                )
            )
        else:
            sections.append(
                f"Currently, all modeled backdoor paths between **{exp_label}** and **{out_label}** are blocked under your conditioning set."
            )

        stop_words = {
            "what", "about", "does", "could", "should", "would", "there", "think",
            "effect", "affect", "impact", "cause", "causes", "between", "variable",
            "confounder", "mediator", "collider", "model", "graph", "also", "with",
        }
        words = [
            w.strip(".,?!:;\"'()[]")
            for w in user_message.split()
            if len(w.strip(".,?!:;\"'()[]")) > 3
            and w.strip(".,?!:;\"'()[]").lower() not in stop_words
        ]
        candidate_concept = (
            " ".join(words[:3]).title() if words else "Baseline Covariate"
        )
        cand_id = (
            re.sub(r"[^a-zA-Z0-9_]", "_", candidate_concept.lower()).strip("_")
            or "candidate_var"
        )

        existing_ids = {n.id for n in project.nodes}
        if cand_id not in existing_ids and len(words) >= 1 and exp_id and out_id:
            proposals.append(
                GraphPatchProposal(
                    action="add_node",
                    title=f"Add Candidate Variable: '{candidate_concept}'",
                    rationale=f"Extracted from your reasoning prompt. Clarify whether '{candidate_concept}' precedes '{exp_label}' (Confounder, t=0) or is caused by '{exp_label}' (Mediator, t=2).",
                    critic_warning="If it is a mediator, conditioning on it will induce overadjustment bias when estimating the total effect.",
                    node=CausalNode(
                        id=cand_id,
                        label=candidate_concept,
                        role="confounder",
                        time_index=0,
                        operationalization=f"Operational measure of {candidate_concept}",
                        x=300.0,
                        y=90.0,
                    ),
                )
            )
            proposals.append(
                GraphPatchProposal(
                    action="add_edge",
                    title=f"Link '{candidate_concept}' -> '{exp_label}'",
                    rationale=f"Hypothesized confounding inflow into {exp_label}.",
                    edge=CausalEdge(
                        source=cand_id,
                        target=exp_id,
                        epistemic_status="ai_proposed",
                        mechanism=f"Direct or upstream influence of {candidate_concept} on {exp_label}",
                        approved_by_researcher=False,
                    ),
                )
            )

        questions = [
            f"Temporal Ordering: Is the variable you're considering measured strictly prior to '{exp_label}' (t0), or can '{exp_label}' influence it over time (t2)?",
            f"Mechanism vs. Direct Effect: Are you interested in the Total Causal Effect of '{exp_label}' on '{out_label}', or the Direct Effect net of mediators?",
            f"Selection / Attrition: Could study inclusion or follow-up completeness depend jointly on '{exp_label}' and '{out_label}' (inducing collider stratification bias)?",
        ]

    notebook_suggestion = (
        f"### Methodological Note ({exp_label} → {out_label})\n"
        f"- **Inquiry:** {user_message}\n"
        f"- **Open Backdoor Paths:** {len(open_backdoors)}\n"
        f"- **Key Socratic Check:** {questions[0] if questions else 'Verify temporal ordering.'}\n"
    )

    return {
        "reply_markdown": "\n\n".join(sections),
        "socratic_questions": questions,
        "proposals": [p.model_dump() for p in proposals],
        "fetched_papers": [fp.model_dump() for fp in fetched_papers],
        "notebook_append_suggestion": notebook_suggestion,
        "agent_mode": agent_mode,
        "used_llm": False,
    }
