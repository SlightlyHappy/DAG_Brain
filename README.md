<div align="center">

![DAG Brain](docs/assets/hero-banner.svg)

# DAG Brain

**An open-source, self-hostable workspace for building causal diagrams you can defend.**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-3b82f6.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-10b981.svg)](pyproject.toml)
[![Validated](https://img.shields.io/badge/engine-validated-0f766e.svg)](docs/VALIDATION.md)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-34d399.svg)](CONTRIBUTING.md)

[What it does](#what-it-does) ·
[Three kinds of answer](#three-kinds-of-answer) ·
[Validation](#validation) ·
[Evaluating AI-drafted DAGs](#evaluating-ai-drafted-dags) ·
[Compared with DAGLive](#compared-with-daglive) ·
[How AI is kept in its place](#how-ai-is-kept-in-its-place) ·
[Quickstart](#quickstart) ·
[Privacy and local-only mode](#privacy-and-local-only-mode)

</div>

---

## What it does

A directed acyclic graph (DAG) is a researcher's causal assumptions written down. Get it wrong and the analysis inherits the error: an unmeasured confounder, a mediator adjusted away, a collider that manufactures an association.

DAG Brain puts three things in one place:

1. **A deterministic causal engine** that computes what the graph implies: which paths are open, which adjustment sets identify the effect, which variables could serve as instruments, and which independencies the data should show.
2. **Data checks and estimation**: test the graph's implications against a dataset, then estimate the effect with an adjustment set the engine has certified, using regression and double machine learning, with an E-value for unmeasured confounding.
3. **A literature and reasoning assistant** that helps you formulate the question, find and summarise evidence, and challenge assumptions. It can only *propose*; you decide.

It works for any field that asks causal questions: epidemiology, clinical research, economics, psychology, the social sciences and causal machine learning.

## Three kinds of answer

The app keeps three kinds of statement apart, because they have very different standing.

| | What it tells you | How it is established | How far to trust it |
|:--|:--|:--|:--|
| **Structural validity** | Is the graph acyclic? Do arrows respect time order? | Graph algorithms | Exact |
| **Identification** | Given *this* graph, which adjustment sets identify the effect? Which variables are instrument candidates? What independencies should the data show? | Complete graphical criteria: the generalised adjustment criterion (Shpitser et al. 2010; Perković et al. 2018), the single-door criterion (Pearl 2009), d-separation | Exact, **but only conditional on the graph being right**. Every result lists the assumptions it rests on. |
| **Scientific support** | How well supported is each arrow the identification depends on? | Your evidence: each arrow carries an evidence class and per-study assessments | A judgement. The app grades it and highlights the weakest assumptions. |

Two consequences are made explicit in the interface:

- An **instrument candidate** passes the necessary graphical conditions under your DAG. The graph cannot establish instrument strength or the exclusion restriction, so the app never calls it a "valid instrument".
- A graph is never "proven" by data. A failed implication shows the data disagree with the graph; passing tests show only consistency.

### Evidence classes

Every arrow has an evidence class, and every linked study records its design, population, whether it establishes the direction of the effect, its limitations and your assessment (supports, weakly supports, contradicts, not assessed).

| Class | Meaning |
|:--|:--|
| Experimental evidence | Relationship investigated through an intervention |
| Quasi-experimental evidence | Causal inference from a quasi-experimental design (valid under that design's assumptions) |
| Empirical association | An observed statistical relationship. Does not establish direction. |
| Mechanistic evidence | Supports a proposed mechanism, not an effect size |
| Author hypothesis | Proposed by a paper's authors, not tested there |
| Researcher hypothesis | Your own assumption |
| AI hypothesis | Proposed by the assistant, until you reclassify it |

Arrows on the canvas are drawn by class (solid for experimental and quasi-experimental evidence, dashed for associations, hypotheses and AI proposals), so the weakest links are visible at a glance. Linking a paper to an arrow never upgrades the arrow's class by itself: you decide, after recording whether the study establishes direction. Cross-sectional designs are marked "direction not established" automatically.

## Validation

The engine and estimators are checked against independent implementations and against data with a known true effect. The full report, with every number reproducible from a fixed seed, is in [**docs/VALIDATION.md**](docs/VALIDATION.md). In summary:

- **d-separation** agrees 100% with an independent moral-graph implementation and with `networkx` on thousands of random queries.
- **Adjustment sets** agree 100% with an independent implementation of the criterion's path-based definition, checked on every subset of variables in hundreds of random graphs. Existence of an adjustment set agrees with brute-force search, and every reported minimal set is valid and inclusion-minimal.
- **Exactness**: on random linear structural causal models, every set the engine certifies reproduces the true total effect to machine precision (using the exact population covariance); every set it rejects is biased.
- **Coverage**: regression adjusting for the engine's set gives 95% intervals that cover the truth at close to the nominal rate.
- **Double machine learning** removes non-linear confounding bias that ordinary regression leaves in.
- **Independence tests**: type I error and power are measured and reported for partial correlation and the generalised covariance measure, including where each over-rejects.
- Textbook graphs (M-bias, butterfly bias, mediators, front-door, instruments) give the published answers.

Reproduce it:

```bash
pip install -r requirements-dev.txt
pytest                                  # 90 tests, under a minute
python scripts/validation_report.py     # full study, writes docs/VALIDATION.md
```

## Evaluating AI-drafted DAGs

Language models can draft a whole causal graph, but how good are those drafts? DAG Brain measures it, in the app and as a reproducible benchmark.

**In the app** (Worldviews, drafts and evaluation):

1. **Draft a DAG with AI** from the research question (text-to-DAG) or from the papers and PDFs in your library (PDF-to-DAG). The draft is saved as a separate worldview. It never changes your graph, every arrow in it is an AI hypothesis, and anything the cleanup removed (cycles, unknown variables, a second exposure) is listed.
2. **Import an expert reference graph** as DAGitty code, or mark any snapshot as the reference.
3. **Evaluate** any graph against any other. You get:
   - **structural agreement**: skeleton and oriented-edge precision, recall and F1, structural Hamming distance, and the lists of missing, extra and reversed arrows;
   - **inferential agreement**: the adjustment identification distance (Henckel, Würtzen & Weichwald 2024), which is the share of variable pairs whose analysis the draft would get wrong;
   - **a verdict for your own question**: would adjusting for the draft's recommended set give a valid estimate of your exposure's effect in the reference graph?

Structural and inferential agreement can diverge. A draft can miss several arrows among downstream variables and still lead to the right analysis, or miss a single confounder arrow and lead to a biased one. Edge counts alone cannot show this, so both are reported.

**As a benchmark**: `python scripts/benchmark_extraction.py` drafts a DAG for each case in `benchmarks/cases/`, giving the model only the research question and the names of the exposure and outcome, never the reference. It writes [docs/BENCHMARK.md](docs/BENCHMARK.md). The current file is the no-model baseline: a draft with only the exposure → outcome arrow, which any model should beat. Configure a provider and re-run (with `--repeats` to average over sampling) to benchmark your model. The starter set holds one published reference graph (Hernández-Díaz et al. 2006, the birth weight paradox) and the two worked examples. Add a case by dropping in a JSON file.

## Compared with DAGLive

DAGLive (Srivastava et al., ISPOR Europe 2026, poster MSR298) is a cloud application in which an LLM proposes candidate causal relationships from a research question or study PDFs, and researchers review them, attach evidence to each pathway and version the diagram. We share its premise: the AI proposes, the researcher decides, and the diagram should be a maintained record rather than a one-off drawing. DAG Brain covers the same workflow and adds the parts DAGLive's authors list as not yet done, along with the causal mathematics that turns a diagram into an analysis.

| | DAGLive (as described in the poster) | DAG Brain |
|:--|:--|:--|
| **Accuracy of AI-drafted DAGs** | "Not yet evaluated: the accuracy of LLM extraction against expert-built DAGs" | Evaluated in the app and in a reproducible benchmark, structurally (F1, SHD) and by consequence for the analysis (adjustment identification distance, target-effect verdict) |
| **Starting a DAG** | Research question, study PDFs, Excel/CSV import, manual build | Research question, library sources and uploaded PDFs, DAGitty import, manual build, the assistant's modes |
| **Causal computation** | Colour-coded pathway types | Complete identification: generalised adjustment criterion, single-door criterion, exact test of whether adjustment can identify the effect, all minimal sets, conditional instrument candidates, testable implications, collider and overadjustment warnings |
| **Is the maths right?** | Not reported | Validation study: 10,000 d-separation queries and 5,980 adjustment checks agree 100% with independent implementations; certified sets are exact to 2×10⁻¹⁵ on simulated models; 95% intervals cover at 95.7% ([docs/VALIDATION.md](docs/VALIDATION.md)) |
| **Evidence on each arrow** | Evidence attached to each pathway | An evidence class for every arrow (experimental, quasi-experimental, association, mechanism, hypothesis, AI hypothesis), and per study the design, population, whether direction is established, limitations and your assessment. Linking a paper never upgrades an arrow by itself. |
| **Using data** | Not offered | Test the graph's implied independencies on a CSV (partial correlation or the ML-based generalised covariance measure, with multiple-testing correction); estimate the effect with regression and double machine learning; E-values for unmeasured confounding |
| **AI safeguards** | Human review | Human review plus a validation gate (schema, references, acyclicity, time order, citations must exist), an identification diff shown before you accept, re-validation at accept time, and prompt-injection isolation and detection for retrieved text |
| **Versions** | Version history and comparison | Automatic graph versions with provenance (model, prompt hash, sources) for every accepted AI change, worldviews compared by adjustment sets, and structural evaluation between any two graphs |
| **Deployment** | Cloud, with login; open source planned | Self-hosted and open source now; enforced local-only mode with no outbound requests; data and PDFs stay on your machine |

DAGLive has features we do not have yet: user accounts for teams, grouped-variable views, scenario filters that show only confounding or collider pathways, and spreadsheet import of variable and pathway tables. They are on our roadmap.

## How AI is kept in its place

```mermaid
flowchart TB
    A["AI reasoning<br/>Topic · Interview · Critique · Literature · Methods · Research"] -->|structured proposals only| B
    B["Validation and approval<br/>schema · references · acyclicity · time order<br/>citations must exist · identification diff · human decision"] -->|accepted changes| C
    C["Authoritative research state<br/>graph · variables · evidence links · question<br/>versions · audit log with provenance"] --> D
    D["Deterministic causal engine<br/>d-separation · adjustment · instruments · implications"] --> E
    E["Outputs<br/>diagnostics · data checks · estimates · exports"]
    R["Retrieved papers, web pages, uploaded PDFs"] -. untrusted data, never instructions .-> A
```

- The assistant **cannot change the graph or the library**. It emits proposals that are validated (schema, existing variables, no cycles, citations that resolve to real sources) and annotated with what would change in identification. You accept or reject each one, and acceptance re-validates against the current graph.
- Research runs **stage** sources for review instead of saving them. AI-proposed arrows are always "AI hypothesis".
- Every accepted change creates a **graph version** and an audit entry with the proposal id, model, provider, a hash of the prompt and the sources used.
- **Prompt injection**: retrieved text is wrapped as inert data, delimiter look-alikes are neutralised, and instruction-like content is detected and flagged. Flagged sources are excluded from automatic proposals. Because the model can only produce validated proposals, nothing a document says can change your research state on its own.
- Graph facts (paths, adjustment sets, identifiability) always come from the engine, never from the model.

## Features

- **Workspace**: three resizable panes, as in VS Code. Drag a divider to resize, drag a panel to its edge to collapse it, double-click a divider to reset it. Dividers also work from the keyboard (arrow keys, Enter), and sizes are remembered.
- **Canvas**: drag variables, draw arrows and shared latent causes (↔), arrange by time wave, pan and zoom, keyboard control, undo.
- **Identification strip**: verdict, total or direct effect, minimal adjustment sets (click to apply), instrument candidates, warnings for mediators, colliders and descendants of the outcome, every path with what blocks it, the assumptions and the evidence grading.
- **Causal question**: a domain-neutral structure (units, exposure, comparator, outcome, time horizon, estimand) with optional framings: PICO for clinical interventions, PECO for epidemiological exposures, policy counterfactuals for economics, interventions on structural causal models for ML, and longitudinal or mediation designs for psychology.
- **Data**: upload a CSV, map columns to variables, test implied independencies (partial correlation or the generalised covariance measure, with Holm and Benjamini–Hochberg correction), and estimate the effect with HC3-robust regression with a bootstrap, plus partially linear DML or AIPW with cross-fitted gradient boosting, overlap diagnostics and E-values.
- **Literature**: OpenAlex, Crossref, PubMed/Europe PMC, CORE, optional Semantic Scholar, SerpApi (Google Scholar) and Tavily. Legal full-text discovery via Unpaywall, Europe PMC open-access links, CORE and an optional institutional EZproxy prefix. Upload your own PDFs; they stay on your machine.
- **AI drafts and evaluation**: text-to-DAG and PDF-to-DAG drafts kept apart from your graph, expert reference graphs, and structural plus inferential evaluation between any two graphs.
- **Notebook, evidence matrix, lab briefing, worldviews** (fork and compare alternative graphs), and **exports** to DAGitty, R (dagitty/ggdag), Python (DoWhy), LaTeX TikZ and a Markdown dossier with the three tiers.
- **Citation styles**: APA 7, Vancouver, Chicago, Harvard, IEEE and BibTeX.

## Quickstart

```bash
git clone https://github.com/SlightlyHappy/DAG_Brain.git
cd DAG_Brain
```

| System | Command |
|:--|:--|
| Windows | `start.bat` or `.\start.ps1` |
| macOS / Linux | `chmod +x start.sh && ./start.sh` |
| Docker | `docker compose up --build -d` |
| Docker with a local Ollama model | `docker compose --profile local-ai up --build -d` |

The launcher creates a Python environment, initialises a SQLite database in `./data` with two worked examples, and opens `http://127.0.0.1:8000`.

### AI providers

Open **AI and search settings** (top right) or copy `.env.example` to `.env`. Supported: OpenRouter, Ollama (auto-detected on `localhost:11434`), Gemini, OpenAI, Anthropic, and any OpenAI-compatible server (LM Studio, vLLM). With no provider configured, a rule-based assistant still answers using the engine's results.

## Privacy and local-only mode

Using Ollama keeps *model calls* on your machine, but the app also makes other outbound requests by default: literature search, web search, Unpaywall and web fonts. To keep everything local, turn on **local-only mode** (`OFFLINE_MODE=true` or the Privacy setting). Then:

- every outbound HTTP request from the backend is refused before it leaves the process, except to `localhost` (Ollama, LM Studio) — enforced in one place, [`backend/net.py`](backend/net.py), and covered by tests;
- the interface does not load web fonts;
- uploaded datasets and PDFs stay in `./data/uploads`.

For a genuinely air-gapped deployment you must also control the host itself: no telemetry from the OS or the browser, no outbound network at the firewall, and dependencies installed from a local mirror. DAG Brain's local-only mode covers the application; it cannot certify the machine around it.

## Limitations

- Identification covers one exposure and one outcome at a time, by covariate adjustment, plus graphical instrument candidates. Front-door estimation and g-methods for time-varying treatments are not implemented.
- The direct effect uses the single-door criterion, which identifies a path coefficient in linear models; for non-linear models it is a controlled direct effect.
- Estimation supports the total effect. Implication tests can lack power, and statistically equivalent graphs cannot be told apart by data.
- AI summaries are labelled unverified until you check them against the paper.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md): layers, data model, algorithms and references.
- [docs/VALIDATION.md](docs/VALIDATION.md): the validation study.
- [docs/BENCHMARK.md](docs/BENCHMARK.md): AI-drafted DAGs against reference DAGs.
- [CONTRIBUTING.md](CONTRIBUTING.md): how to contribute.

## License

[Apache License 2.0](LICENSE).
