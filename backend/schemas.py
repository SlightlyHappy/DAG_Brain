from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field
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

EpistemicStatus = Literal[
    "empirical",
    "hypothesized",
    "ai_proposed",
    "contested",
    "bidirectional_confounded",
]

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
    tldr: str = ""
    causal_claims: List[str] = Field(default_factory=list)
    confounders_controlled: List[str] = Field(default_factory=list)
    limitations_and_bias: str = ""
    methodology_grade: str = ""


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
    scholar_url: str = ""
    core_url: str = ""
    scihub_url: str = ""
    source_engine: str = "OpenAlex"
    abstract: str = ""
    full_text_snippet: str = ""
    study_design: StudyDesign = "Unclassified"
    sample_size: str = ""
    key_finding: str = ""
    ai_summary: Optional[PaperAISummary] = None
    stance: Literal["supporting", "conflicting", "context"] = "supporting"
    user_notes: str = ""
    added_at: str = Field(default_factory=now_iso)


class CausalEdge(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("edge"))
    source: str
    target: str
    epistemic_status: EpistemicStatus = "hypothesized"
    mechanism: str = ""
    effect_direction: Literal["positive", "negative", "non_linear", "unknown"] = "unknown"
    confidence: float = 0.75
    citation_ids: List[str] = Field(default_factory=list)
    rationale: str = ""
    approved_by_researcher: bool = True


class GraphPatchProposal(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("prop"))
    action: Literal["add_node", "add_edge", "update_node", "remove_edge"]
    title: str
    rationale: str
    critic_warning: str = ""
    node: Optional[CausalNode] = None
    edge: Optional[CausalEdge] = None
    citation: Optional[CitationSource] = None
    status: Literal["pending", "accepted", "rejected"] = "pending"
    created_at: str = Field(default_factory=now_iso)


class CompetingWorldview(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("wv"))
    name: str
    description: str = ""
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("msg"))
    role: Literal["user", "assistant", "system"]
    agent_mode: str = "socratic"
    content: str
    socratic_questions: List[str] = Field(default_factory=list)
    proposal_ids: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class AuditLogEntry(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("log"))
    timestamp: str = Field(default_factory=now_iso)
    actor: Literal["researcher", "ai_accepted", "system"] = "researcher"
    action: str
    details: str


class ProjectState(BaseModel):
    id: str = Field(default_factory=lambda: gen_id("proj"))
    title: str
    discipline: str = "Epidemiology & Health Sciences"
    research_question: str = ""
    target_population: str = ""
    estimand_type: Literal["total_effect", "direct_effect"] = "total_effect"
    citation_style: CitationStyle = "apa"
    notebook_markdown: str = ""
    lab_briefing_markdown: str = ""
    nodes: List[CausalNode] = Field(default_factory=list)
    edges: List[CausalEdge] = Field(default_factory=list)
    sources: List[CitationSource] = Field(default_factory=list)
    worldviews: List[CompetingWorldview] = Field(default_factory=list)
    proposals: List[GraphPatchProposal] = Field(default_factory=list)
    chat_history: List[ChatMessage] = Field(default_factory=list)
    audit_log: List[AuditLogEntry] = Field(default_factory=list)
    updated_at: str = Field(default_factory=now_iso)
