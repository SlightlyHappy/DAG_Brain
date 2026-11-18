import re
import asyncio
from urllib.parse import quote_plus
from typing import List, Dict, Any, Optional
import httpx
from .schemas import CitationSource, PaperAISummary, StudyDesign, CitationStyle
from .config import get_runtime_settings
from .llm_router import call_llm_json


def build_external_links(
    title: str, doi: str = "", url: str = "", pdf_url: str = ""
) -> Dict[str, str]:
    settings = get_runtime_settings()
    scihub_base = (
        settings.get("scihub_mirror_url") or "https://sci-hub.name"
    ).rstrip("/")
    clean_doi = (
        doi.replace("https://doi.org/", "")
        .replace("http://doi.org/", "")
        .strip()
    )
    query_target = clean_doi if clean_doi else title
    encoded_q = quote_plus(query_target or "causal inference")
    encoded_title = quote_plus(title or query_target)

    scholar_url = f"https://scholar.google.com/scholar?q={encoded_title}"
    core_url = f"https://core.ac.uk/search?q={encoded_title}"
    scihub_url = (
        f"{scihub_base}/{clean_doi}"
        if clean_doi
        else f"{scihub_base}/{encoded_title}"
    )
    primary_url = url or (f"https://doi.org/{clean_doi}" if clean_doi else scholar_url)

    return {
        "doi": clean_doi,
        "url": primary_url,
        "pdf_url": pdf_url,
        "scholar_url": scholar_url,
        "core_url": core_url,
        "scihub_url": scihub_url,
    }


def enrich_source_links(src: CitationSource) -> CitationSource:
    links = build_external_links(
        title=src.title, doi=src.doi, url=src.url, pdf_url=src.pdf_url
    )
    src.doi = links["doi"]
    src.url = links["url"]
    if not src.pdf_url:
        src.pdf_url = links["pdf_url"]
    src.scholar_url = links["scholar_url"]
    src.core_url = links["core_url"]
    src.scihub_url = links["scihub_url"]
    if not src.ai_summary:
        src.ai_summary = _heuristic_paper_summary(src)
    return src


def classify_study_design(title: str, abstract: str) -> StudyDesign:
    text = f"{title} {abstract}".lower()
    if any(
        k in text
        for k in [
            "meta-analysis",
            "systematic review",
            "umbrella review",
            "cochrane",
        ]
    ):
        return "Meta-Analysis / Systematic Review"
    if any(
        k in text
        for k in [
            "randomized controlled trial",
            "randomised controlled trial",
            "double-blind",
            "placebo-controlled",
            "rct",
            "clinical trial",
        ]
    ):
        return "Randomized Controlled Trial (RCT)"
    if any(
        k in text
        for k in [
            "mendelian randomization",
            "mendelian randomisation",
            "instrumental variable",
            "difference-in-differences",
            "regression discontinuity",
            "quasi-experimental",
            "natural experiment",
        ]
    ):
        return "Quasi-Experimental (IV / DiD / RDD / MR)"
    if any(
        k in text
        for k in [
            "cohort study",
            "prospective study",
            "longitudinal",
            "follow-up",
            "panel data",
            "framingham",
            "biobank",
        ]
    ):
        return "Longitudinal Cohort"
    if "case-control" in text or "case control" in text:
        return "Case-Control"
    if any(
        k in text
        for k in [
            "cross-sectional",
            "survey",
            "prevalence",
            "observational study",
        ]
    ):
        return "Cross-Sectional / Observational"
    if any(
        k in text
        for k in [
            "mechanism",
            "pathway",
            "structural causal model",
            "dag",
            "directed acyclic",
            "theoretical",
        ]
    ):
        return "Theoretical / Mechanism"
    return "Unclassified"


def extract_sample_size(abstract: str) -> str:
    if not abstract:
        return ""
    patterns = [
        r"\b[nN]\s*=\s*([0-9][0-9,\.]+)",
        r"\b([0-9][0-9,]{2,})\s+(?:participants|patients|adults|individuals|subjects|children|workers|students|firms)\b",
    ]
    for pat in patterns:
        m = re.search(pat, abstract)
        if m:
            return f"N = {m.group(1)}"
    return ""


def reconstruct_openalex_abstract(
    inverted_index: Optional[Dict[str, List[int]]]
) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                word_positions.append((pos, word))
    if not word_positions:
        return ""
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions)


# ============================================================================
# INDIVIDUAL SEARCH ADAPTERS (OpenAlex, CORE.ac.uk, EuropePMC, Scholar, Tavily)
# ============================================================================

async def _search_openalex(query: str, limit: int = 5) -> List[CitationSource]:
    settings = get_runtime_settings()
    email = settings.get("scholarly_email", "researcher@dagbrain.org")
    headers = {"User-Agent": f"DAG_Brain/0.2 (mailto:{email})"}
    results: List[CitationSource] = []
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            params = {
                "search": query,
                "per-page": str(limit),
                "select": "id,title,publication_year,authorships,primary_location,doi,abstract_inverted_index,cited_by_count,open_access",
            }
            resp = await client.get("https://api.openalex.org/works", params=params)
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue
                    year = item.get("publication_year")
                    authors = [
                        (a.get("author") or {}).get("display_name")
                        for a in item.get("authorships", [])[:6]
                        if (a.get("author") or {}).get("display_name")
                    ]
                    loc = item.get("primary_location") or {}
                    journal = (loc.get("source") or {}).get("display_name") or "OpenAlex"
                    raw_doi = item.get("doi") or ""
                    oa = item.get("open_access") or {}
                    pdf_url = oa.get("oa_url") or loc.get("pdf_url") or ""
                    abstract = reconstruct_openalex_abstract(
                        item.get("abstract_inverted_index")
                    )
                    design = classify_study_design(title, abstract)
                    sample_n = extract_sample_size(abstract)
                    cited_by = item.get("cited_by_count", 0)
                    links = build_external_links(
                        title, raw_doi, loc.get("landing_page_url") or "", pdf_url
                    )
                    src = CitationSource(
                        title=title,
                        authors=authors or ["Unknown Author"],
                        year=year,
                        journal=journal,
                        doi=links["doi"],
                        url=links["url"],
                        pdf_url=links["pdf_url"],
                        scholar_url=links["scholar_url"],
                        core_url=links["core_url"],
                        scihub_url=links["scihub_url"],
                        source_engine="OpenAlex",
                        abstract=abstract,
                        study_design=design,
                        sample_size=sample_n,
                        key_finding=(
                            abstract[:280] + "..."
                            if len(abstract) > 280
                            else (abstract or f"Peer-reviewed study ({cited_by} citations).")
                        ),
                    )
                    src.ai_summary = _heuristic_paper_summary(src)
                    results.append(src)
    except Exception:
        pass
    return results


async def _search_core_ac_uk(query: str, limit: int = 5) -> List[CitationSource]:
    """Queries CORE.ac.uk API v3 (world's largest open-access full-text aggregator)."""
    settings = get_runtime_settings()
    api_key = (settings.get("core_api_key") or "").strip()
    headers = {"Accept": "application/json", "User-Agent": "DAG_Brain/0.2"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: List[CitationSource] = []
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(
                "https://api.core.ac.uk/v3/search/works",
                params={"q": query, "limit": str(limit)},
            )
            if resp.status_code == 200:
                for item in resp.json().get("results", []):
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue
                    year = item.get("yearPublished")
                    authors = [
                        a.get("name")
                        for a in item.get("authors", [])[:6]
                        if isinstance(a, dict) and a.get("name")
                    ]
                    journal = (
                        item.get("publisher")
                        or (item.get("journals") or [{}])[0].get("title")
                        or "CORE Open Access"
                    )
                    doi = item.get("doi") or ""
                    download_url = item.get("downloadUrl") or ""
                    full_text = (item.get("fullText") or "")[:600]
                    abstract = (item.get("abstract") or full_text or "").strip()
                    design = classify_study_design(title, abstract)
                    links = build_external_links(
                        title, doi, download_url, download_url
                    )
                    src = CitationSource(
                        title=title,
                        authors=authors or ["CORE Author"],
                        year=int(year) if str(year).isdigit() else None,
                        journal=str(journal),
                        doi=links["doi"],
                        url=links["url"],
                        pdf_url=download_url,
                        scholar_url=links["scholar_url"],
                        core_url=links["core_url"],
                        scihub_url=links["scihub_url"],
                        source_engine="CORE.ac.uk",
                        abstract=abstract,
                        full_text_snippet=full_text,
                        study_design=design,
                        sample_size=extract_sample_size(abstract),
                        key_finding=(
                            abstract[:280] + "..."
                            if len(abstract) > 280
                            else (abstract or "Open-access full text indexed via CORE.ac.uk.")
                        ),
                    )
                    src.ai_summary = _heuristic_paper_summary(src)
                    results.append(src)
    except Exception:
        pass
    return results


async def _search_europepmc(query: str, limit: int = 5) -> List[CitationSource]:
    """Queries EuropePMC / PubMed REST API for biomedical & life-science literature."""
    results: List[CitationSource] = []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                params={
                    "query": query,
                    "format": "json",
                    "pageSize": str(limit),
                    "resultType": "core",
                },
            )
            if resp.status_code == 200:
                items = (
                    resp.json().get("resultList", {}).get("result", [])
                )
                for item in items:
                    title = (item.get("title") or "").strip().rstrip(".")
                    if not title:
                        continue
                    year_str = str(item.get("pubYear") or "")
                    year = int(year_str) if year_str.isdigit() else None
                    author_str = item.get("authorString") or ""
                    authors = [
                        a.strip() for a in author_str.split(",") if a.strip()
                    ][:6]
                    journal = item.get("journalTitle") or "PubMed / EuropePMC"
                    doi = item.get("doi") or ""
                    pmid = item.get("pmid") or ""
                    abstract = re.sub(
                        r"<[^>]+>", "", item.get("abstractText") or ""
                    ).strip()
                    design = classify_study_design(title, abstract)
                    links = build_external_links(
                        title,
                        doi,
                        f"https://europepmc.org/article/MED/{pmid}" if pmid else "",
                        "",
                    )
                    src = CitationSource(
                        title=title,
                        authors=authors or ["PubMed Author"],
                        year=year,
                        journal=journal,
                        doi=links["doi"],
                        pmid=pmid,
                        url=links["url"],
                        scholar_url=links["scholar_url"],
                        core_url=links["core_url"],
                        scihub_url=links["scihub_url"],
                        source_engine="PubMed / EuropePMC",
                        abstract=abstract,
                        study_design=design,
                        sample_size=extract_sample_size(abstract),
                        key_finding=(
                            abstract[:280] + "..."
                            if len(abstract) > 280
                            else (abstract or f"Indexed in PubMed (PMID: {pmid}).")
                        ),
                    )
                    src.ai_summary = _heuristic_paper_summary(src)
                    results.append(src)
    except Exception:
        pass
    return results


async def _search_google_scholar_and_s2(
    query: str, limit: int = 5
) -> List[CitationSource]:
    """
    Searches Google Scholar via SerpApi (if key configured) or Semantic Scholar Graph API,
    attaching direct scholar.google.com, core.ac.uk, and sci-hub.name resolution links.
    """
    settings = get_runtime_settings()
    serp_key = (settings.get("serpapi_api_key") or "").strip()
    s2_key = (settings.get("semantic_scholar_api_key") or "").strip()
    results: List[CitationSource] = []

    if serp_key:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://serpapi.com/search.json",
                    params={
                        "engine": "google_scholar",
                        "q": query,
                        "num": str(limit),
                        "api_key": serp_key,
                    },
                )
                if resp.status_code == 200:
                    for item in resp.json().get("organic_results", []):
                        title = (item.get("title") or "").strip()
                        if not title:
                            continue
                        snippet = (item.get("snippet") or "").strip()
                        pub_info = (
                            item.get("publication_info", {}).get("summary") or ""
                        )
                        year_match = re.search(r"\b(19\d\d|20\d\d)\b", pub_info)
                        year = int(year_match.group(1)) if year_match else None
                        link = item.get("link") or ""
                        resources = item.get("resources") or []
                        pdf_url = resources[0].get("link", "") if resources else ""
                        links = build_external_links(title, "", link, pdf_url)
                        src = CitationSource(
                            title=title,
                            authors=[pub_info.split("-")[0].strip() or "Scholar Author"],
                            year=year,
                            journal="Google Scholar",
                            url=links["url"],
                            pdf_url=links["pdf_url"],
                            scholar_url=links["scholar_url"],
                            core_url=links["core_url"],
                            scihub_url=links["scihub_url"],
                            source_engine="Google Scholar",
                            abstract=snippet,
                            study_design=classify_study_design(title, snippet),
                            sample_size=extract_sample_size(snippet),
                            key_finding=snippet or "Retrieved via Google Scholar.",
                        )
                        src.ai_summary = _heuristic_paper_summary(src)
                        results.append(src)
        except Exception:
            pass

    if results:
        return results

    # Fallback: Semantic Scholar Graph API (powers scholarly search with direct Google Scholar links)
    headers = {"User-Agent": "DAG_Brain/0.2"}
    if s2_key:
        headers["x-api-key"] = s2_key
    try:
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": str(limit),
                    "fields": "title,authors,year,venue,externalIds,url,abstract,openAccessPdf,citationCount",
                },
            )
            if resp.status_code == 200:
                for item in resp.json().get("data", []):
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue
                    authors = [
                        a.get("name")
                        for a in item.get("authors", [])[:6]
                        if a.get("name")
                    ]
                    year = item.get("year")
                    venue = item.get("venue") or "Google Scholar / S2"
                    ext = item.get("externalIds") or {}
                    doi = ext.get("DOI") or ""
                    pmid = str(ext.get("PubMed") or "")
                    oa_pdf = (item.get("openAccessPdf") or {}).get("url") or ""
                    abstract = (item.get("abstract") or "").strip()
                    links = build_external_links(
                        title, doi, item.get("url") or "", oa_pdf
                    )
                    src = CitationSource(
                        title=title,
                        authors=authors or ["Scholar Author"],
                        year=year,
                        journal=venue,
                        doi=links["doi"],
                        pmid=pmid,
                        url=links["url"],
                        pdf_url=links["pdf_url"],
                        scholar_url=links["scholar_url"],
                        core_url=links["core_url"],
                        scihub_url=links["scihub_url"],
                        source_engine="Google Scholar / S2",
                        abstract=abstract,
                        study_design=classify_study_design(title, abstract),
                        sample_size=extract_sample_size(abstract),
                        key_finding=(
                            abstract[:280] + "..."
                            if len(abstract) > 280
                            else (abstract or "Indexed in Google Scholar & Semantic Scholar.")
                        ),
                    )
                    src.ai_summary = _heuristic_paper_summary(src)
                    results.append(src)
    except Exception:
        pass
    return results


async def search_tavily_and_web(
    query: str, limit: int = 5
) -> Dict[str, Any]:
    """
    Searches the live internet using Tavily AI Search API (if TAVILY_API_KEY configured)
    or SearXNG (if SEARXNG_BASE_URL configured), returning web answers and CitationSource items.
    """
    settings = get_runtime_settings()
    tavily_key = (settings.get("tavily_api_key") or "").strip()
    searxng_url = (settings.get("searxng_base_url") or "").strip().rstrip("/")

    sources: List[CitationSource] = []
    web_answer = ""

    if tavily_key:
        try:
            async with httpx.AsyncClient(timeout=14.0) as client:
                payload = {
                    "api_key": tavily_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "max_results": limit,
                }
                resp = await client.post(
                    "https://api.tavily.com/search", json=payload
                )
                if resp.status_code == 200:
                    data = resp.json()
                    web_answer = data.get("answer") or ""
                    for item in data.get("results", []):
                        title = (item.get("title") or "").strip()
                        url = item.get("url") or ""
                        content = (item.get("content") or "").strip()
                        if not title:
                            continue
                        doi_match = re.search(
                            r"(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)",
                            f"{url} {content}",
                        )
                        doi = doi_match.group(1) if doi_match else ""
                        links = build_external_links(title, doi, url, "")
                        src = CitationSource(
                            title=title,
                            authors=["Tavily Web Research"],
                            year=2026,
                            journal=re.sub(r"^https?://(www\.)?", "", url).split("/")[0],
                            doi=links["doi"],
                            url=links["url"],
                            scholar_url=links["scholar_url"],
                            core_url=links["core_url"],
                            scihub_url=links["scihub_url"],
                            source_engine="Tavily Web",
                            abstract=content,
                            study_design=classify_study_design(title, content),
                            sample_size=extract_sample_size(content),
                            key_finding=content[:280] + ("..." if len(content) > 280 else ""),
                        )
                        src.ai_summary = _heuristic_paper_summary(src)
                        sources.append(src)
        except Exception:
            pass

    if not sources and searxng_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{searxng_url}/search",
                    params={"q": query, "format": "json"},
                )
                if resp.status_code == 200:
                    for item in resp.json().get("results", [])[:limit]:
                        title = (item.get("title") or "").strip()
                        url = item.get("url") or ""
                        content = (item.get("content") or "").strip()
                        links = build_external_links(title, "", url, "")
                        src = CitationSource(
                            title=title,
                            authors=["SearXNG Research"],
                            year=2026,
                            journal="Web / Scholarly Metasearch",
                            url=links["url"],
                            scholar_url=links["scholar_url"],
                            core_url=links["core_url"],
                            scihub_url=links["scihub_url"],
                            source_engine="SearXNG",
                            abstract=content,
                            study_design=classify_study_design(title, content),
                            key_finding=content[:280],
                        )
                        src.ai_summary = _heuristic_paper_summary(src)
                        sources.append(src)
        except Exception:
            pass

    return {"answer": web_answer, "sources": sources}


async def search_scholarly_literature(
    query: str, limit: int = 8, engine: str = "all"
) -> List[CitationSource]:
    """
    Federated search across OpenAlex, CORE.ac.uk, EuropePMC/PubMed, Google Scholar/Semantic Scholar,
    and Tavily Web Search. Deduplicates by normalized title/DOI and attaches direct links to
    Google Scholar, CORE.ac.uk, and Sci-Hub (sci-hub.name).
    """
    engine = (engine or "all").lower()
    tasks = []

    if engine == "openalex":
        tasks.append(_search_openalex(query, limit=limit))
    elif engine == "core":
        tasks.append(_search_core_ac_uk(query, limit=limit))
    elif engine == "pubmed":
        tasks.append(_search_europepmc(query, limit=limit))
    elif engine == "scholar":
        tasks.append(_search_google_scholar_and_s2(query, limit=limit))
    elif engine == "tavily":
        web_res = await search_tavily_and_web(query, limit=limit)
        if web_res["sources"]:
            return web_res["sources"]
        tasks.append(_search_openalex(query, limit=limit))
    else:
        # Federated "all" search in parallel!
        per_engine = max(3, limit // 2)
        tasks = [
            _search_openalex(query, limit=per_engine),
            _search_core_ac_uk(query, limit=per_engine),
            _search_europepmc(query, limit=per_engine),
            _search_google_scholar_and_s2(query, limit=per_engine),
        ]

    gathered = await asyncio.gather(*tasks, return_exceptions=True)
    combined: List[CitationSource] = []
    seen_keys = set()

    for batch in gathered:
        if not isinstance(batch, list):
            continue
        for src in batch:
            norm_key = (
                src.doi.lower().strip()
                if src.doi
                else re.sub(r"[^a-z0-9]", "", src.title.lower())[:60]
            )
            if not norm_key or norm_key in seen_keys:
                continue
            seen_keys.add(norm_key)
            combined.append(enrich_source_links(src))

    return combined[:limit]


# ============================================================================
# AI PAPER SUMMARIZER & CAUSAL CLAIM EXTRACTOR
# ============================================================================

def _heuristic_paper_summary(src: CitationSource) -> PaperAISummary:
    text = (src.abstract or src.key_finding or src.title or "").strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    tldr = sentences[0] if sentences else f"{src.study_design} examining {src.title}."
    claims = sentences[1:3] if len(sentences) > 1 else [tldr]

    design_lower = src.study_design.lower()
    if "rct" in design_lower or "meta" in design_lower:
        grade = "High Causal Internal Validity (Experimental / Pooled Synthesis)"
        lims = "Check external transportability to target population and treatment adherence."
        conf = ["Randomization balances baseline observed & unobserved confounders"]
    elif "quasi" in design_lower:
        grade = "Strong Quasi-Experimental Identification (IV / MR / DiD)"
        lims = "Verify exclusion restriction (no pleiotropy / parallel trends assumption)."
        conf = ["Instrumental / structural identification against unmeasured confounding"]
    elif "cohort" in design_lower:
        grade = "Moderate-to-High Observational Validity (Temporal Ordering Established)"
        lims = "Susceptible to residual unmeasured baseline confounding (U) and attrition bias."
        conf = ["Baseline demographics", "Longitudinal pre-exposure covariates"]
    else:
        grade = "Associational / Hypothesis-Generating Evidence"
        lims = "Cross-sectional or ungraded design cannot rule out reverse causality or unmeasured confounding."
        conf = ["Requires explicit backdoor adjustment in DAG"]

    return PaperAISummary(
        tldr=tldr,
        causal_claims=claims,
        confounders_controlled=conf,
        limitations_and_bias=lims,
        methodology_grade=grade,
    )


async def summarize_paper_with_ai(src: CitationSource) -> PaperAISummary:
    """
    Uses the active LLM provider (OpenRouter, Ollama, Gemini, OpenAI, Anthropic) to generate
    a deep causal reading digest of a paper, falling back to the heuristic analyzer if offline.
    """
    sys_prompt = """You are an expert Epidemiologist, Econometrician, and Research Methodologist.
Given a scientific paper's metadata and abstract/text, extract a concise, high-signal reading summary for a causal inference researcher.
Return ONLY valid JSON with this exact structure:
{
  "tldr": "1-2 sentence plain-English summary of what the paper found.",
  "causal_claims": [
    "Variable A -> Variable B (direction, magnitude, or mechanism)",
    "Secondary pathway or mediation finding"
  ],
  "confounders_controlled": [
    "List of likely baseline confounders, instruments, or study design controls used"
  ],
  "limitations_and_bias": "Key threats to causal validity (e.g., reverse causality, unmeasured confounding, selection/collider bias).",
  "methodology_grade": "Evidence hierarchy rating (e.g., High: RCT/MR, Moderate: Longitudinal Cohort, Low: Cross-Sectional Association)"
}"""

    user_prompt = f"""Title: {src.title}
Authors: {', '.join(src.authors)} ({src.year or 'n.d.'})
Venue: {src.journal}
Detected Study Design: {src.study_design}
Sample Size: {src.sample_size}
Abstract / Content:
{src.abstract or src.full_text_snippet or src.key_finding}
"""
    llm_res = await call_llm_json(sys_prompt, user_prompt)
    if llm_res and isinstance(llm_res, dict) and llm_res.get("tldr"):
        return PaperAISummary(
            tldr=str(llm_res.get("tldr", "")),
            causal_claims=[str(x) for x in (llm_res.get("causal_claims") or [])[:4]],
            confounders_controlled=[
                str(x) for x in (llm_res.get("confounders_controlled") or [])[:5]
            ],
            limitations_and_bias=str(llm_res.get("limitations_and_bias", "")),
            methodology_grade=str(llm_res.get("methodology_grade", src.study_design)),
        )
    return _heuristic_paper_summary(src)


# ============================================================================
# MULTI-STYLE ACADEMIC CITATION FORMATTER (APA 7, Vancouver/AMA, Chicago, etc.)
# ============================================================================

def _last_name(full_name: str) -> str:
    parts = full_name.strip().split()
    return parts[-1] if parts else "Author"


def _initials(full_name: str) -> str:
    parts = full_name.strip().split()
    if len(parts) <= 1:
        return ""
    return " ".join(f"{p[0].upper()}." for p in parts[:-1] if p)


def format_in_text_citation(
    src: CitationSource, style: CitationStyle = "apa", index: int = 1
) -> str:
    year = src.year or "n.d."
    first_author = _last_name(src.authors[0]) if src.authors else "Anonymous"

    if style in ("vancouver", "ieee"):
        return f"[{index}]"

    if len(src.authors) == 1:
        author_part = first_author
    elif len(src.authors) == 2:
        second_author = _last_name(src.authors[1])
        sep = " & " if style == "apa" else " and "
        author_part = f"{first_author}{sep}{second_author}"
    else:
        author_part = f"{first_author} et al."

    if style == "chicago":
        return f"({author_part} {year})"
    return f"({author_part}, {year})"


def format_reference_entry(
    src: CitationSource, style: CitationStyle = "apa", index: int = 1
) -> str:
    year = src.year or "n.d."
    journal = src.journal or "Working Paper / Preprint"
    doi_str = f"https://doi.org/{src.doi}" if src.doi else src.url

    if style == "bibtex":
        first_last = re.sub(
            r"[^a-zA-Z0-9]",
            "",
            _last_name(src.authors[0]).lower() if src.authors else "author",
        )
        first_word = re.sub(
            r"[^a-zA-Z0-9]",
            "",
            (src.title.split()[0].lower() if src.title else "work"),
        )
        cite_key = f"{first_last}{year}{first_word}"
        bib_authors = " and ".join(src.authors) if src.authors else "Unknown"
        lines = [
            f"@article{{{cite_key},",
            f"  author = {{{bib_authors}}},",
            f"  title = {{{src.title}}},",
            f"  journal = {{{journal}}},",
            f"  year = {{{year}}},",
        ]
        if src.doi:
            lines.append(f"  doi = {{{src.doi}}},")
        if doi_str:
            lines.append(f"  url = {{{doi_str}}}")
        lines.append("}")
        return "\n".join(lines)

    if style == "vancouver":
        van_authors = []
        for a in src.authors[:6]:
            inits = _initials(a).replace(".", "").replace(" ", "")
            van_authors.append(f"{_last_name(a)} {inits}".strip())
        if len(src.authors) > 6:
            van_authors.append("et al")
        auth_str = ", ".join(van_authors) or "Anonymous"
        ref = f"{index}. {auth_str}. {src.title}. {journal}. {year}."
        if src.doi:
            ref += f" doi:{src.doi}"
        return ref

    if style == "ieee":
        ieee_authors = []
        for a in src.authors[:6]:
            inits = _initials(a)
            ieee_authors.append(f"{inits} {_last_name(a)}".strip())
        auth_str = ", ".join(ieee_authors) or "Anonymous"
        ref = f'[{index}] {auth_str}, "{src.title}," *{journal}*, {year}.'
        if src.doi:
            ref += f" doi: {src.doi}."
        return ref

    if style == "chicago":
        if not src.authors:
            auth_str = "Anonymous"
        elif len(src.authors) == 1:
            a0 = src.authors[0]
            parts = a0.split()
            auth_str = (
                f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else a0
            )
        else:
            a0 = src.authors[0]
            parts = a0.split()
            first_fmt = (
                f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) > 1 else a0
            )
            others = ", ".join(src.authors[1:4])
            auth_str = f"{first_fmt}, and {others}"
        ref = f'{auth_str}. {year}. "{src.title}." *{journal}*.'
        if doi_str:
            ref += f" {doi_str}."
        return ref

    if style == "harvard":
        harv_authors = []
        for a in src.authors[:4]:
            harv_authors.append(f"{_last_name(a)}, {_initials(a)}".strip(", "))
        if len(src.authors) > 4:
            auth_str = f"{harv_authors[0]} et al."
        else:
            auth_str = ", ".join(harv_authors) or "Anonymous"
        ref = f"{auth_str} ({year}) '{src.title}', *{journal}*."
        if doi_str:
            ref += f" Available at: {doi_str}."
        return ref

    # Default: APA 7th Edition
    apa_authors = []
    for a in src.authors[:7]:
        inits = _initials(a)
        apa_authors.append(
            f"{_last_name(a)}, {inits}" if inits else _last_name(a)
        )
    if len(apa_authors) == 0:
        auth_str = "Anonymous."
    elif len(apa_authors) == 1:
        auth_str = apa_authors[0]
    else:
        auth_str = ", ".join(apa_authors[:-1]) + ", & " + apa_authors[-1]
    ref = f"{auth_str} ({year}). {src.title}. *{journal}*."
    if doi_str:
        ref += f" {doi_str}"
    return ref
