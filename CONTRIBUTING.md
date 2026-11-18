# Contributing to DAG_Brain (Causal Studio) 🤝

Thank you for your interest in contributing to **DAG_Brain**! Our mission is to build open-source, self-hostable causal research infrastructure that bridges **formal causal inference mathematics** (Judea Pearl's SCMs / DAGitty) with **citation-grounded AI synthesis** (NotebookLM-style RAG) across all scientific disciplines.

We welcome contributions not only from **software engineers and AI developers**, but also from **epidemiologists, economists, psychologists, biostatisticians, and domain researchers**!

---

## 🌟 Ways You Can Contribute

### 1. Domain-Specific Seed Models & Case Studies (`backend/seed_data.py`)
Are you an expert in Epidemiology, Labor Economics, Climate Science, Clinical Trials, Sociology, or Operations Research?
- You can contribute well-documented reference DAGs with real DOIs, operationalizations, and methodological notes to `backend/seed_data.py` so students and researchers can learn from gold-standard causal graphs.

### 2. Causal Graph Algorithms & Identification Methods (`backend/causal_engine.py`)
We want our deterministic Python causal engine to be the most helpful pedagogical and analytical engine in open source:
- **Frontdoor Criterion Solver:** Automated detection of valid frontdoor adjustment sets when backdoor paths are blocked by unmeasured confounders ($U$).
- **SWIGs (Single-World Intervention Graphs):** Rendering Richardson & Robins node-splitting transformations.
- **Time-Varying Treatments & $g$-Methods:** Identifying treatment-confounder feedback loops ($X_{t_0} \to L_{t_1} \to X_{t_2}$) that require Marginal Structural Models (IPTW) or the Parametric $g$-Formula rather than standard regression adjustment.
- **Sensitivity Analysis Calculators:** Built-in E-value guidance for unmeasured confounding edges.

### 3. Scholarly Literature & PDF Connectors (`backend/literature_service.py`)
- **PubMed / NCBI E-Utilities & EuropePMC full-text XML adapters**
- **arXiv & Semantic Scholar adapters**
- **Local PDF Parsing (`Docling` / `PyMuPDF`):** Extracting tables, sample sizes ($N$), and effect sizes from local PDFs dropped into the Left Source Vault.
- **Zotero / BibTeX Sync:** Two-way sync with a researcher's local Zotero library.
- **Additional Citation Styles:** MLA, Nature, Science, Lancet, and custom CSL XML support.

### 4. MCP (Model Context Protocol) Server & Agentic Workflows (`backend/agents.py`)
- Expose `DAG_Brain`'s deterministic graph tools (`check_d_separation`, `propose_causal_edge`, `get_minimal_adjustment_sets`) via an MCP endpoint so Claude Desktop, Cursor, or Gemini CLI can collaborate on a researcher's live canvas.

---

## 🛠️ Local Development Setup

`DAG_Brain` is intentionally designed to run with **zero build-step friction**:

```bash
# 1. Clone your fork
git clone https://github.com/YOUR-USERNAME/DAG_Brain.git
cd DAG_Brain

# 2. Launch with one click (creates .venv if needed and starts on http://127.0.0.1:8000)
./start.sh        # macOS / Linux
.\start.ps1       # Windows PowerShell
start.bat         # Windows CMD
```

### Running the Verification Suite
Before submitting a Pull Request, verify that the causal math engine, citation formatters, and ASGI endpoints pass all checks:

```bash
python -c "
from starlette.testclient import TestClient
from backend.main import app
with TestClient(app) as client:
    assert client.get('/api/health').status_code == 200
    assert len(client.get('/api/projects').json()) >= 2
print('All checks passed!')
"
```

---

## 📜 Architectural Guardrails for PRs

1. **Deterministic Math First:** Never delegate graph-theoretic calculations ($d$-separation, cycle detection, minimal adjustment sets, collider detection) to an LLM prompt. All graph properties must be computed deterministically in `backend/causal_engine.py`.
2. **Human-in-the-Loop Governance:** AI agents must never silently mutate the researcher's graph. All AI suggestions must flow through `GraphPatchProposal` objects so the researcher maintains full epistemic control and auditability.
3. **Zero Mandatory Cloud Lock-In:** Every feature must continue to work in self-hosted environments using local **Ollama** models, **OpenRouter**, or offline mode.
