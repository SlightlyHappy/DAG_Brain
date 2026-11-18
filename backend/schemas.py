# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, model_validator
import uuid
from datetime import datetime, timezone


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def gen_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


NodeRole = Literal[
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
]

# What kind of evidence stands behind an edge. Only experimental and
# quasi-experimental evidence (and, for a mechanism, mechanistic evidence)
# speaks to causal direction; an association alone does not.
EvidenceClass = Literal[
    "empirical_association",   # observed statistical relationship
    "experimental",            # relationship investigated through an intervention
    "quasi_experimental",      # causal inference from a quasi-experimental design
    "mechanistic",             # evidence supporting a proposed mechanism
    "author_hypothesis",       # proposed by the authors of a paper
    "ai_hypothesis",           # proposed by the AI assistant
    "researcher_hypothesis",   # proposed by the researcher
]

EVIDENCE_CLASSES = list(EvidenceClass.__args__)  # type: ignore[attr-defined]

LEGACY_STATUS_TO_CLASS = {
    "empirical": "empirical_association",
    "hypothesized": "researcher_hypothesis",
    "ai_proposed": "ai_hypothesis",
    "contested": "researcher_hypothesis",
    "bidirectional_confounded": "researcher_hypothesis",
}

StudyDesign = Literal[
    "Meta-Analysis / Systematic Review",
    "Randomized Controlled Trial (RCT)",
    "Quasi-Experimental (IV / DiD / RDD / MR)",
    "Longitudinal Cohort",
    "Case-Control",
    "Cross-Sectional / Observational",
    "Theoretical / Mechanism",
    "Web / Report / Preprint",
    "Unclassified",
]

CitationStyle = Literal["apa", "vancouver", "chicago", "harvard", "ieee", "bibtex"]


class CausalNode(BaseModel):
    id: str
    label: str
    role: NodeRole = "other"
    adjusted: bool = False
    observed: bool = True
    time_index: int = 1  # Temporal wave (0=baseline/pre-exposure, 1=exposure, 2=mediator, 3=outcome)
    operationalization: str = ""
    measurement_scale: str = "continuous"
    x: float = 250.0
    y: float = 200.0
    notes: str = ""


class PaperAISummary(BaseModel):
    """Machine-extracted reading notes. Unverified until the researcher confirms them."""

    tldr: str = ""
    causal_claims: List[str] = Field(default_factory=list)
    confounders_controlled: List[str] = Field(default_factory=list)
    limitations_and_bias: str = ""
    methodology_grade: str = ""
    population: str = ""
    directionality: Literal["established", "not_established", "unclear"] = "unclear"
    extracted_by: Literal["ai", "heuristic"] = "ai"
    verified: bool = False


class CitationSource(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("src"))
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    doi: str = ""
    pmid: str = ""
    url: str = ""
    pdf_url: str = ""
    oa_url: str = ""          # legal open-access copy (Unpaywall, OpenAlex, CORE)
    europepmc_url: str = ""
    proxy_url: str = ""       # institutional EZproxy link, if configured
    scholar_url: str = ""
    core_url: str = ""
    local_pdf: str = ""       # file name of a PDF the researcher uploaded
    source_engine: str = "OpenAlex"
    abstract: str = ""
    full_text_snippet: str = ""
    study_design: StudyDesign = "Unclassified"
    sample_size: str = ""
    key_finding: str = ""
    ai_summary: Optional[PaperAISummary] = None
    stance: Literal["supporting", "conflicting", "context"] = "supporting"
    user_notes: str = ""
    # Retrieved text is untrusted input; flags record instruction-like content.
    injection_flags: List[str] = Field(default_factory=list)
    added_by: Literal["researcher", "ai", "import"] = "researcher"
    added_at: str = Field(default_factory=now_iso)


class EvidenceLink(BaseModel):
    """One study's bearing on one edge, as assessed by the researcher."""

    source_id: str
    population: str = ""
    design: str = ""
    directionality: Literal["established", "not_established", "unclear"] = "unclear"
    limitations: str = ""
    researcher_assessment: Literal[
        "supports", "weakly_supports", "contradicts", "not_assessed"
    ] = "not_assessed"
    extracted_by: Literal["ai", "researcher"] = "researcher"
    notes: str = ""


class CausalEdge(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("edge"))
    source: str
    target: str
    edge_type: Literal["directed", "bidirected"] = "directed"
    evidence_class: EvidenceClass = "researcher_hypothesis"
    contested: bool = False
    mechanism: str = ""
    effect_direction: Literal["positive", "negative", "non_linear", "unknown"] = "unknown"
    confidence: float = 0.75
    evidence_links: List[EvidenceLink] = Field(default_factory=list)
    citation_ids: List[str] = Field(default_factory=list)  # derived from evidence_links
    rationale: str = ""
    origin: Literal["researcher", "ai", "import"] = "researcher"
    approved_by_researcher: bool = True

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy(cls, data: Any) -> Any:
        """Schema v1 used a single `epistemic_status`; map it to class + type."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        legacy = data.pop("epistemic_status", None)
        if legacy and "evidence_class" not in data:
            data["evidence_class"] = LEGACY_STATUS_TO_CLASS.get(legacy, "researcher_hypothesis")
            if legacy == "contested":
                data.setdefault("contested", True)
            if legacy == "bidirectional_confounded":
                data.setdefault("edge_type", "bidirected")
            if legacy == "ai_proposed":
                data.setdefault("origin", "ai")
        return data

    @model_validator(mode="after")
    def _sync_citations(self) -> "CausalEdge":
        linked = {l.source_id for l in self.evidence_links}
        for sid in self.citation_ids:
            if sid not in linked:
                self.evidence_links.append(EvidenceLink(source_id=sid))
                linked.add(sid)
        self.citation_ids = [l.source_id for l in self.evidence_links]
        return self


class Provenance(BaseModel):
    """Where an AI-originated item came from, kept for reproducibility."""

    agent_mode: str = ""
    provider: str = ""
    model: str = ""
    used_llm: bool = False
    prompt_sha256: str = ""
    source_ids: List[str] = Field(default_factory=list)


class GraphPatchProposal(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("prop"))
    action: Literal["add_node", "add_edge", "update_node", "remove_edge", "add_source"]
    title: str
    rationale: str
    critic_warning: str = ""
    node: Optional[CausalNode] = None
    edge: Optional[CausalEdge] = None
    citation: Optional[CitationSource] = None
    status: Literal["pending", "accepted", "rejected", "invalid"] = "pending"
    # Filled by backend.validation: errors, warnings and the identification diff.
    validation: Dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    decided_at: str = ""
    decision_reason: str = ""
    created_at: str = Field(default_factory=now_iso)


class CompetingWorldview(BaseModel):
    """A saved alternative graph: a researcher snapshot, an AI draft, or an expert reference."""

    id: str = Field(default_factory=lambda: gen_id("wv"))
    name: str
    description: str = ""
    kind: Literal["snapshot", "ai_draft", "reference"] = "snapshot"
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    provenance: Optional["Provenance"] = None
    created_at: str = Field(default_factory=now_iso)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("msg"))
    role: Literal["user", "assistant", "system"]
    agent_mode: str = "socratic"
    content: str
    socratic_questions: List[str] = Field(default_factory=list)
    proposal_ids: List[str] = Field(default_factory=list)
    provenance: Optional[Provenance] = None
    created_at: str = Field(default_factory=now_iso)


class AuditLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("log"))
    timestamp: str = Field(default_factory=now_iso)
    actor: Literal["researcher", "ai_accepted", "system", "ai"] = "researcher"
    action: str
    details: str
    graph_version: int = 0
    proposal_id: str = ""
    provenance: Optional[Provenance] = None


class GraphVersion(BaseModel):
    version: int
    timestamp: str = Field(default_factory=now_iso)
    actor: str = "researcher"
    reason: str = ""
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)


class ResearchQuestion(BaseModel):
    """Domain-neutral causal question. Templates only relabel these fields."""

    template: Literal["general", "pico", "peco", "policy", "scm", "longitudinal"] = "general"
    unit: str = ""          # population or units of analysis
    exposure: str = ""      # treatment, exposure or intervention
    comparator: str = ""    # contrast or counterfactual condition
    outcome: str = ""
    time_horizon: str = ""
    estimand: Literal["ate", "att", "cde", "nde", "nie", "other", ""] = ""
    setting: str = ""
    notes: str = ""


class DatasetMeta(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("ds"))
    filename: str
    path: str
    n_rows: int = 0
    columns: List[Dict[str, Any]] = Field(default_factory=list)
    mapping: Dict[str, str] = Field(default_factory=dict)  # node_id -> column name
    uploaded_at: str = Field(default_factory=now_iso)


SCHEMA_VERSION = 2
MAX_VERSIONS = 60


class ProjectState(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: str = Field(default_factory=lambda: gen_id("proj"))
    title: str
    discipline: str = "Epidemiology & Health Sciences"
    research_question: str = ""
    target_population: str = ""
    estimand_type: Literal["total_effect", "direct_effect"] = "total_effect"
    citation_style: CitationStyle = "apa"
    notebook_markdown: str = ""
    lab_briefing_markdown: str = ""
    question: ResearchQuestion = Field(default_factory=ResearchQuestion)
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    sources: List[CitationSource] = Field(default_factory=list)
    staged_sources: List[CitationSource] = Field(default_factory=list)
    datasets: List[DatasetMeta] = Field(default_factory=list)
    graph_version: int = 0
    versions: List[GraphVersion] = Field(default_factory=list)
    worldviews: List[CompetingWorldview] = Field(default_factory=list)
    proposals: List[GraphPatchProposal] = Field(default_factory=list)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    audit_log: List[AuditLogEntry] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)

    def record_version(self, actor: str, reason: str) -> int:
        """Snapshot the authoritative graph after a change. Returns the new version."""
        self.graph_version += 1
        self.versions.append(
            GraphVersion(
                version=self.graph_version,
                actor=actor,
                reason=reason,
                nodes=[n.model_copy(deep=True) for n in self.nodes],
                edges=[e.model_copy(deep=True) for e in self.edges],
            )
        )
        if len(self.versions) > MAX_VERSIONS:
            self.versions = self.versions[-MAX_VERSIONS:]
        return self.graph_version


CompetingWorldview.model_rebuild()
