/**
 * DAG Brain: frontend controller.
 * Causal graph canvas, sources library with AI summaries, notebook,
 * evidence matrix, lab briefing, history and assistant.
 */

const state = {
  projects: [],
  currentProject: null,
  selectedNodeId: null,
  selectedEdgeId: null,
  connectMode: false,
  connectSourceId: null,
  drag: null, // { kind: "node"|"pan", ... }
  agentMode: "socratic",
  settings: null,
  liveLitResults: null,
  activeExportFormat: "dagitty",
  view: { x: 0, y: 0, w: 900, h: 500 },
  lastChatLen: -1,
  sending: false,
  focusNodeAfterRender: false,
  matrixSort: { key: "strength", dir: -1 },
  speaking: false,
  dataResults: { implications: null, estimate: null },
  offline: false,
};

const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_W = 172;
const NODE_H = 58;
const BASE_W = 900;
const LANES = [
  { t: "t₀", label: "Baseline" },
  { t: "t₁", label: "Exposure" },
  { t: "t₂", label: "Mediators" },
  { t: "t₃", label: "Outcome" },
];
const VIEW_MODES = ["canvas", "split", "notebook", "matrix", "data", "briefing", "audit"];
const FONT_HREF = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap";

const ROLE_LABELS = {
  exposure: "Exposure",
  outcome: "Outcome",
  confounder: "Confounder",
  mediator: "Mediator",
  collider: "Collider",
  instrument: "Instrument",
  effect_modifier: "Modifier",
  competing_exposure: "Competing exposure",
  latent: "Unobserved",
  descendant_of_outcome: "Descendant of the outcome",
  descendant_of_exposure: "Descendant of the exposure",
  other: "Variable",
};

const EVIDENCE_LABELS = {
  experimental: "Experimental evidence",
  quasi_experimental: "Quasi-experimental evidence",
  empirical_association: "Empirical association",
  mechanistic: "Mechanistic evidence",
  author_hypothesis: "Author hypothesis",
  researcher_hypothesis: "Researcher hypothesis",
  ai_hypothesis: "AI hypothesis",
};
const EVIDENCE_HELP = {
  experimental: "Investigated through an intervention, such as a randomised trial.",
  quasi_experimental: "Causal inference from a quasi-experimental design (IV, DiD, RDD, MR). Valid only under that design's assumptions.",
  empirical_association: "An observed statistical relationship. It does not establish direction or rule out confounding.",
  mechanistic: "Evidence for a plausible mechanism, not for the size of the effect.",
  author_hypothesis: "Proposed by the authors of a paper, not tested there.",
  researcher_hypothesis: "Your own assumption.",
  ai_hypothesis: "Proposed by the assistant. Reclassify it once you have assessed the evidence.",
};
const ASSESSMENTS = { not_assessed: "Not assessed", supports: "Supports", weakly_supports: "Weakly supports", contradicts: "Contradicts" };
const DIRECTIONALITY = { unclear: "Direction unclear", established: "Direction established", not_established: "Direction not established" };

/** Visual category for an arrow, from strongest to weakest evidence. */
function edgeCategory(e) {
  if (e.edge_type === "bidirected") return "bi";
  if (e.contested) return "contested";
  if (e.evidence_class === "experimental" || e.evidence_class === "quasi_experimental") return "causal";
  if (e.evidence_class === "empirical_association" || e.evidence_class === "mechanistic") return "assoc";
  if (e.evidence_class === "ai_hypothesis") return "ai";
  return "hyp";
}

const PROVIDER_LABELS = {
  openrouter: "OpenRouter",
  ollama: "Ollama",
  gemini: "Gemini",
  openai: "OpenAI",
  anthropic: "Anthropic",
  custom: "Custom endpoint",
};

const PERSONAS = {
  topic_refiner: {
    name: "Topic",
    hint: "Narrows a broad topic into a causal question you can answer, using PICO/PECO and the FINER criteria.",
    starters: [
      "Help me turn my topic into a specific causal question",
      "What population and comparison should I define?",
      "Is this question feasible with observational data?",
    ],
  },
  socratic: {
    name: "Interview",
    hint: "Asks one to three questions at a time to pin down variables, timing and mechanisms.",
    starters: [
      "Help me define the exposure precisely",
      "Which variables come before the exposure in time?",
      "What could cause both the exposure and the outcome?",
    ],
  },
  critic: {
    name: "Critique",
    hint: "Plays Reviewer 2: looks for missing confounders, collider traps, reverse causation and selection bias.",
    starters: [
      "Critique my current graph the way Reviewer 2 would",
      "Am I conditioning on a collider anywhere?",
      "Which confounders am I most likely missing?",
    ],
  },
  scout: {
    name: "Literature",
    hint: "Searches scholarly databases and the web for studies on a specific arrow, then grades their design.",
    starters: [
      "Find evidence for the main exposure to outcome arrow",
      "Find randomised trials on this question",
      "Which arrows have no citations yet?",
    ],
  },
  synthesis: {
    name: "Methods",
    hint: "Drafts the identification strategy and methods text from the current graph.",
    starters: [
      "Draft the identification strategy paragraph",
      "Summarise the adjustment set and why it is sufficient",
      "Write the limitations around unmeasured confounding",
    ],
  },
};

const PROPOSAL_KINDS = {
  add_node: "New variable",
  add_edge: "New arrow",
  update_node: "Change to a variable",
  remove_edge: "Remove an arrow",
};

const EXPORT_EXT = { dagitty: "dagitty.txt", r: "R", python: "py", latex: "tex", dossier: "md" };

// ============================================================================
// HELPERS
// ============================================================================

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
  const resp = await fetch(path, { ...options, headers });
  if (!resp.ok) {
    let msg = await resp.text();
    try {
      const j = JSON.parse(msg);
      msg = j.detail || j.error || msg;
    } catch (e) { /* plain text */ }
    throw new Error(msg || `Request failed (HTTP ${resp.status})`);
  }
  return resp.json();
}

function escapeHtml(str) {
  return String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function icon(name, cls = "") {
  return `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

const SUBSCRIPTS = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };

function inlineMd(s) {
  return s
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    // inline LaTeX like $t_0$ or $d$ becomes plain italics
    .replace(/\$([^$\n]{1,40})\$/g, (_, m) => `<em>${m.replace(/_\{?(\d)\}?/g, (__, d) => SUBSCRIPTS[d])}</em>`)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^\w])_(?!\s)([^_]+?)_(?!\w)/g, "$1<em>$2</em>")
    .replace(/\[(\d+(?:\s*[,–-]\s*\d+)*)\]/g, '<span class="cite">[$1]</span>');
}

/** Small, safe Markdown renderer: headings, paragraphs, lists, quotes, code. */
function simpleMarkdown(md) {
  if (!md) return "";
  const lines = escapeHtml(md).split(/\r?\n/);
  const out = [];
  let para = [];
  let list = null;
  let code = null;
  const flushP = () => { if (para.length) { out.push(`<p>${inlineMd(para.join(" "))}</p>`); para = []; } };
  const flushL = () => {
    if (list) { out.push(`<${list.t}>${list.items.map((i) => `<li>${inlineMd(i)}</li>`).join("")}</${list.t}>`); list = null; }
  };
  for (const raw of lines) {
    const line = raw.replace(/\s+$/, "");
    if (code !== null) {
      if (/^```/.test(line.trim())) { out.push(`<pre><code>${code.join("\n")}</code></pre>`); code = null; }
      else code.push(raw);
      continue;
    }
    let m;
    if (/^```/.test(line.trim())) { flushP(); flushL(); code = []; continue; }
    if (!line.trim()) { flushP(); flushL(); continue; }
    if ((m = line.match(/^(#{1,4})\s+(.*)$/))) { flushP(); flushL(); const l = m[1].length; out.push(`<h${l}>${inlineMd(m[2])}</h${l}>`); continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { flushP(); flushL(); out.push("<hr>"); continue; }
    if ((m = line.match(/^\s*[-*•]\s+(.*)$/))) { flushP(); if (!list || list.t !== "ul") { flushL(); list = { t: "ul", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) { flushP(); if (!list || list.t !== "ol") { flushL(); list = { t: "ol", items: [] }; } list.items.push(m[1]); continue; }
    if ((m = line.match(/^&gt;\s?(.*)$/))) { flushP(); flushL(); out.push(`<blockquote>${inlineMd(m[1])}</blockquote>`); continue; }
    if (list && /^\s{2,}\S/.test(raw)) { list.items[list.items.length - 1] += " " + line.trim(); continue; }
    flushL();
    para.push(line.trim());
  }
  if (code !== null) out.push(`<pre><code>${code.join("\n")}</code></pre>`);
  flushP();
  flushL();
  return out.join("");
}

function stripEmoji(s) {
  return String(s || "").replace(/[\p{Extended_Pictographic}\u{FE0F}\u{200D}]/gu, "").replace(/\s{2,}/g, " ").trim();
}

function fmtTime(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d)) return "";
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function fmtDay(iso) {
  const d = new Date(iso);
  if (isNaN(d)) return "";
  const today = new Date();
  const yest = new Date(); yest.setDate(today.getDate() - 1);
  if (d.toDateString() === today.toDateString()) return "Today";
  if (d.toDateString() === yest.toDateString()) return "Yesterday";
  return d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short", year: d.getFullYear() !== today.getFullYear() ? "numeric" : undefined });
}

function slugify(s) {
  return String(s || "dag").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 60) || "dag";
}

function nodeLabel(id) {
  const n = (state.currentProject?.nodes || []).find((x) => x.id === id);
  return n ? n.label : id;
}

function setLoading(btn, on) {
  if (!btn) return;
  btn.classList.toggle("is-loading", on);
  btn.disabled = on;
}

function isTyping(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || el.isContentEditable;
}

function store(key, val) {
  try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; }
  return null;
}

// ---------------------------------------------------------------- toasts
function toast(message, opts = {}) {
  const region = $("toastRegion");
  const el = document.createElement("div");
  el.className = `toast ${opts.error ? "is-error" : ""}`;
  el.setAttribute("role", opts.error ? "alert" : "status");
  el.innerHTML = `<span class="toast-msg">${escapeHtml(message)}</span>`;
  let timer;
  const dismiss = () => {
    clearTimeout(timer);
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), 160);
  };
  if (opts.action) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "btn btn-xs";
    b.textContent = opts.action;
    b.addEventListener("click", async () => { dismiss(); await opts.onAction(); });
    el.appendChild(b);
  }
  region.appendChild(el);
  while (region.children.length > 3) region.firstElementChild.remove();
  timer = setTimeout(dismiss, opts.duration || (opts.action ? 7000 : 3500));
  return dismiss;
}

function reportError(prefix, err) {
  console.error(prefix, err);
  toast(`${prefix}: ${err.message || err}`, { error: true, duration: 6000 });
}

// ============================================================================
// INITIALISATION & PROJECT STATE
// ============================================================================

document.addEventListener("DOMContentLoaded", async () => {
  initPanels();
  bindUIEvents();
  setPersona("socratic");
  loadHealth();
  loadSettings(); // runs alongside the study load; Ollama detection can be slow
  try {
    await loadProjectsList();
  } catch (err) {
    reportError("Could not load studies", err);
  }
});

async function loadHealth() {
  try {
    const h = await api("/api/health");
    applyOffline(Boolean(h.offline_mode));
  } catch (e) { applyOffline(false); }
}

/** Web fonts are fetched only outside local-only mode; otherwise system fonts are used. */
function applyOffline(on) {
  state.offline = on;
  $("offlineBadge").classList.toggle("hidden", !on);
  if (!on && !document.getElementById("webFonts")) {
    const l = document.createElement("link");
    l.id = "webFonts";
    l.rel = "stylesheet";
    l.href = FONT_HREF;
    document.head.appendChild(l);
  }
}

async function loadSettings() {
  try {
    state.settings = await api("/api/settings");
    renderProviderPill();
  } catch (err) {
    reportError("Could not load AI settings", err);
  }
}

function providerReady(s) {
  const prov = s.active_provider || "openrouter";
  if (prov === "ollama") return (s.detected_ollama_models || []).length > 0;
  if (prov === "custom") return true;
  return Boolean(s[`${prov}_api_key_configured`]);
}

function renderProviderPill() {
  const s = state.settings || {};
  const prov = s.active_provider || "openrouter";
  const ready = providerReady(s);
  $("activeProviderLabel").textContent = PROVIDER_LABELS[prov] || prov;
  const pill = $("btnSettingsModal");
  pill.dataset.ready = String(ready);
  pill.title = ready
    ? `Using ${PROVIDER_LABELS[prov] || prov}${s.active_model ? `, ${s.active_model}` : ""}`
    : `${PROVIDER_LABELS[prov] || prov} is not configured. The built-in rule-based assistant is used instead.`;
  $("assistantModel").textContent = ready ? (s.active_model || "") : "Offline mode";
  $("assistantModel").title = ready ? (s.active_model || "") : "No model key configured: the rule-based assistant answers";
}

async function loadProjectsList(selectId = null) {
  state.projects = await api("/api/projects");
  const sel = $("projectSelect");
  sel.innerHTML = "";
  for (const p of state.projects) {
    const opt = document.createElement("option");
    opt.value = p.id;
    opt.textContent = p.title;
    opt.title = `${p.title}: ${p.node_count} variables, ${p.edge_count} arrows`;
    sel.appendChild(opt);
  }
  const remembered = store("dagbrain.project");
  const targetId =
    selectId ||
    (remembered && state.projects.some((p) => p.id === remembered) ? remembered : null) ||
    (state.projects[0] && state.projects[0].id);
  if (targetId) {
    sel.value = targetId;
    await loadProject(targetId);
  } else {
    $("canvasEmpty").classList.remove("hidden");
  }
}

async function loadProject(projectId) {
  const proj = await api(`/api/projects/${projectId}`);
  store("dagbrain.project", projectId);
  $("projectSelect").value = projectId;
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  state.lastChatLen = -1;
  state.dataResults = { implications: null, estimate: null };
  stopSpeech();
  setProjectState(proj);
  requestAnimationFrame(fitView);
}

function setProjectState(proj) {
  state.currentProject = proj;
  if (state.selectedNodeId && !(proj.nodes || []).some((n) => n.id === state.selectedNodeId)) state.selectedNodeId = null;
  if (state.selectedEdgeId && !(proj.edges || []).some((e) => e.id === state.selectedEdgeId)) state.selectedEdgeId = null;
  $("citationStyleSelect").value = proj.citation_style || "apa";
  $("worldviewCountBadge").textContent = (proj.worldviews || []).length || "";
  document.querySelectorAll("[data-estimand]").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.estimand === (proj.estimand_type || "total_effect")));
  });

  renderVault();
  renderDiagnostics();
  renderCanvas();
  renderInspector();
  renderNotebook();
  renderEvidenceMatrix();
  renderDataView();
  renderQuestion();
  renderLabBriefing();
  renderAuditLog();
  renderProposals();
  renderChat();
  if (typeof proj.offline_mode === "boolean" && proj.offline_mode !== state.offline) applyOffline(proj.offline_mode);
}

function setSaveStatus(text, isError = false) {
  const el = $("saveStatus");
  el.textContent = text;
  el.classList.toggle("is-error", isError);
}

async function saveCurrentProject(auditAction = null, auditNote = null) {
  if (!state.currentProject) return;
  const p = state.currentProject;
  const payload = {
    title: p.title,
    discipline: p.discipline,
    research_question: p.research_question,
    estimand_type: p.estimand_type,
    citation_style: p.citation_style,
    notebook_markdown: p.notebook_markdown,
    lab_briefing_markdown: p.lab_briefing_markdown,
    nodes: p.nodes,
    edges: p.edges,
    sources: p.sources,
  };
  if (auditAction && auditNote) {
    payload.audit_action = auditAction;
    payload.audit_note = auditNote;
  }
  setSaveStatus("Saving…");
  try {
    const updated = await api(`/api/projects/${p.id}`, { method: "PUT", body: JSON.stringify(payload) });
    setProjectState(updated);
    setSaveStatus(`Saved ${fmtTime(new Date().toISOString())}`);
    return updated;
  } catch (err) {
    setSaveStatus("Not saved", true);
    reportError("Changes were not saved", err);
    throw err;
  }
}

function graphSnapshot() {
  const p = state.currentProject;
  return JSON.stringify({ nodes: p.nodes, edges: p.edges });
}

function offerUndo(message, snapshot, restoreLabel) {
  toast(message, {
    action: "Undo",
    onAction: async () => {
      const snap = JSON.parse(snapshot);
      state.currentProject.nodes = snap.nodes;
      state.currentProject.edges = snap.edges;
      await saveCurrentProject("Undo", restoreLabel);
    },
  });
}

// ============================================================================
// SOURCES PANEL
// ============================================================================

function evidenceOf(design) {
  const d = (design || "").toLowerCase();
  if (d.includes("meta")) return { level: 5, label: "Meta-analysis" };
  if (d.includes("randomi")) return { level: 5, label: "Randomised trial" };
  if (d.includes("quasi")) return { level: 4, label: "Quasi-experimental" };
  if (d.includes("cohort") || d.includes("longitudinal")) return { level: 3, label: "Cohort" };
  if (d.includes("case-control")) return { level: 2, label: "Case-control" };
  if (d.includes("cross")) return { level: 2, label: "Cross-sectional" };
  if (d.includes("theor") || d.includes("mechanism")) return { level: 1, label: "Theory or mechanism" };
  return { level: 0, label: "Design not classified" };
}

function evidenceHtml(src, withN = true) {
  const ev = evidenceOf(src.study_design);
  return `
    <div class="evidence" title="${escapeHtml(src.study_design || "Unclassified")}">
      <span class="meter" data-level="${ev.level}" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i></span>
      <span class="evidence-label">${escapeHtml(ev.label)}</span>
      ${withN && src.sample_size ? `<span class="evidence-n">${escapeHtml(src.sample_size)}</span>` : ""}
      <span class="sr-only">Evidence strength ${ev.level} of 5</span>
    </div>`;
}

function srcHref(src) {
  if (src.url) return src.url;
  if (src.doi) return `https://doi.org/${src.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//, "")}`;
  return "";
}

function srcTitleHtml(src) {
  const href = srcHref(src);
  const t = escapeHtml(src.title);
  return href ? `<a href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer">${t}</a>` : t;
}

function srcMetaHtml(src) {
  const authors = src.authors || [];
  const who = authors.length > 3 ? `${authors.slice(0, 3).join(", ")} et al.` : authors.join(", ");
  const parts = [];
  if (who) parts.push(escapeHtml(who));
  parts.push(`${src.year || "n.d."}`);
  let meta = parts.join(", ");
  if (src.journal) meta += `. <em>${escapeHtml(src.journal)}</em>`;
  if (src.source_engine) meta += ` <span class="src-engine">via ${escapeHtml(src.source_engine)}</span>`;
  return meta;
}

function srcLinksHtml(src) {
  const links = [];
  const add = (href, label, title) => {
    if (href) links.push(`<a class="ext-link" href="${escapeHtml(href)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(title)}">${label}${icon("external")}</a>`);
  };
  const pid = state.currentProject?.id;
  if (src.local_pdf && pid) add(`/api/projects/${pid}/sources/${src.id}/pdf`, "Your PDF", "Open the PDF you uploaded");
  add(src.oa_url, "Open access", "Legal open-access copy");
  add(src.pdf_url && src.pdf_url !== src.oa_url ? src.pdf_url : "", "PDF", "Open the PDF");
  add(src.doi ? `https://doi.org/${src.doi.replace(/^https?:\/\/(dx\.)?doi\.org\//, "")}` : "", "DOI", "Publisher page");
  add(src.europepmc_url, "Europe PMC", "Europe PMC record");
  add(src.proxy_url, "Library", "Through your institutional proxy");
  add(src.core_url, "CORE", "Search CORE for an open-access copy");
  add(src.scholar_url, "Scholar", "Find on Google Scholar");
  return links.length ? `<div class="src-links">${links.join("")}</div>` : "";
}

function flagsHtml(src) {
  const f = src.injection_flags || [];
  if (!f.length) return "";
  return `<div class="src-flags" role="note">${icon("alert")}<span><strong>Instruction-like text found</strong> (${escapeHtml(f.join("; "))}). It is treated as data only and excluded from automatic proposals.</span></div>`;
}

function digestHtml(src, withRefresh) {
  const s = src.ai_summary || {};
  const hasAi = Boolean(s.tldr || (s.causal_claims || []).length || s.limitations_and_bias);
  if (!hasAi && !withRefresh) return "";
  const claims = s.causal_claims || [];
  const controls = s.confounders_controlled || [];
  const byAi = s.extracted_by !== "heuristic";
  const head = byAi ? "AI summary" : "Keyword summary";
  const dirPill = s.directionality === "established" ? "pill-ok" : s.directionality === "not_established" ? "pill-bad" : "pill-warn";
  return `
    <details class="digest">
      <summary>${icon("sparkle")}<span>${head}</span><span class="digest-grade">${s.verified ? "verified by you" : "unverified"}</span>${icon("chevron-down", "chev")}</summary>
      <div class="digest-body">
        ${hasAi ? "" : `<p class="muted">No summary yet. Generate one from the abstract and metadata.</p>`}
        <div class="row gap-1" style="flex-wrap:wrap">
          <span class="pill ${dirPill}">${escapeHtml(DIRECTIONALITY[s.directionality] || "Direction unclear")}</span>
          ${s.methodology_grade ? `<span class="pill">${escapeHtml(s.methodology_grade)}</span>` : ""}
        </div>
        ${s.tldr ? `<div><span class="digest-k">In short</span>${escapeHtml(s.tldr)}</div>` : ""}
        ${s.population ? `<div><span class="digest-k">Population</span>${escapeHtml(s.population)}</div>` : ""}
        ${claims.length ? `<div><span class="digest-k">Reported relationships</span><ul>${claims.map((c) => `<li>${escapeHtml(c)}</li>`).join("")}</ul></div>` : ""}
        ${controls.length ? `<div><span class="digest-k">Adjusted for (as reported)</span>${controls.map(escapeHtml).join(", ")}</div>` : ""}
        ${s.limitations_and_bias ? `<div class="digest-limit"><span class="digest-k">${byAi ? "Limitations and bias" : "Design caveat"}</span>${escapeHtml(s.limitations_and_bias)}</div>` : ""}
        ${withRefresh ? `<div class="digest-verify">
            <label class="check-inline"><input type="checkbox" class="chk-verify" data-id="${escapeHtml(src.id)}" ${s.verified ? "checked" : ""} ${hasAi ? "" : "disabled"} />I checked this against the paper</label>
            <button class="btn btn-ghost btn-xs btn-ai-resummarize" data-id="${escapeHtml(src.id)}" type="button">${icon("refresh")}${hasAi ? "Regenerate" : "Generate"}</button>
          </div>` : ""}
      </div>
    </details>`;
}

function renderVault() {
  const proj = state.currentProject;
  if (!proj) return;
  const sources = proj.formatted_sources || proj.sources || [];
  $("vaultCount").textContent = sources.length;

  const listEl = $("vaultSourcesView");
  const bibEl = $("vaultBibliographyView");
  const hint = $("edgeLinkHint");
  const openDigests = new Set([...listEl.querySelectorAll(".digest[open]")].map((d) => d.dataset.id));

  const selectedEdge = state.selectedEdgeId ? (proj.edges || []).find((e) => e.id === state.selectedEdgeId) : null;
  if (selectedEdge) {
    hint.classList.remove("hidden");
    hint.innerHTML = `${icon("link")}<span>Linking evidence to <strong>${escapeHtml(nodeLabel(selectedEdge.source))} → ${escapeHtml(nodeLabel(selectedEdge.target))}</strong></span>`;
  } else {
    hint.classList.add("hidden");
  }

  if (sources.length === 0) {
    listEl.innerHTML = `
      <div class="empty">
        <p class="empty-title">No sources yet</p>
        <p class="empty-body">Search above, add a paper by hand, or use Research in the assistant to gather and summarise papers automatically. Sources can be cited in the notebook and linked to arrows as evidence.</p>
      </div>`;
    bibEl.innerHTML = `<div class="empty"><p class="empty-body">The bibliography appears once the library has sources.</p></div>`;
    return;
  }

  listEl.innerHTML = sources
    .map((src) => {
      const linked = selectedEdge && (selectedEdge.citation_ids || []).includes(src.id);
      return `
      <article class="src ${linked ? "is-linked" : ""}">
        <h3 class="src-title">${srcTitleHtml(src)}</h3>
        <p class="src-meta">${srcMetaHtml(src)}</p>
        ${evidenceHtml(src)}
        ${src.key_finding && !(src.ai_summary && src.ai_summary.tldr) ? `<p class="src-finding" title="Show more">${escapeHtml(src.key_finding)}</p>` : ""}
        ${flagsHtml(src)}
        ${digestHtml(src, true).replace('<details class="digest">', `<details class="digest" data-id="${escapeHtml(src.id)}"${openDigests.has(src.id) ? " open" : ""}>`)}
        ${srcLinksHtml(src)}
        <div class="src-foot">
          <span class="cite-key">${escapeHtml(src.in_text || "")}</span>
          <div class="src-actions">
            <button class="btn btn-ghost btn-xs btn-cite-nb" data-id="${escapeHtml(src.id)}" type="button" title="Insert this citation into the notebook">${icon("quote")}Cite</button>
            <button class="btn btn-ghost btn-xs btn-digest-nb" data-id="${escapeHtml(src.id)}" type="button" title="Add this paper's summary and citation to the notebook">${icon("notebook-plus")}Summary</button>
            ${src.doi && !src.oa_url && !state.offline ? `<button class="btn btn-ghost btn-xs btn-find-oa" data-id="${escapeHtml(src.id)}" type="button" title="Look for a legal open-access copy (Unpaywall)">${icon("search")}Open access</button>` : ""}
            ${selectedEdge ? `<button class="btn ${linked ? "btn-ok" : "btn-secondary"} btn-xs btn-cite-edge" data-id="${escapeHtml(src.id)}" type="button" aria-pressed="${linked}">${icon(linked ? "check" : "link")}${linked ? "Linked" : "Link"}</button>` : ""}
          </div>
        </div>
      </article>`;
    })
    .join("");

  listEl.querySelectorAll(".src-finding").forEach((p) => p.addEventListener("click", () => p.classList.toggle("is-open")));

  listEl.querySelectorAll(".chk-verify").forEach((chk) => {
    chk.addEventListener("change", async () => {
      const src = state.currentProject.sources.find((x) => x.id === chk.dataset.id);
      if (!src || !src.ai_summary) return;
      src.ai_summary.verified = chk.checked;
      await saveCurrentProject(chk.checked ? "Verified paper summary" : "Marked summary unverified", `'${src.title}'`).catch(() => {});
    });
  });

  listEl.querySelectorAll(".btn-find-oa").forEach((btn) => {
    btn.addEventListener("click", async () => {
      setLoading(btn, true);
      try {
        const updated = await api(`/api/projects/${proj.id}/sources/${btn.dataset.id}/open-access`, { method: "POST" });
        setProjectState(updated);
        const src = updated.sources.find((x) => x.id === btn.dataset.id);
        toast(src && src.oa_url ? "Open-access copy found" : "No open-access copy found. Add a contact email in settings if you have not.");
      } catch (err) {
        setLoading(btn, false);
        reportError("Open-access lookup failed", err);
      }
    });
  });

  listEl.querySelectorAll(".btn-ai-resummarize").forEach((btn) => {
    btn.addEventListener("click", async () => {
      setLoading(btn, true);
      try {
        const updated = await api(`/api/projects/${proj.id}/sources/${btn.dataset.id}/summarize`, { method: "POST" });
        setProjectState(updated);
        toast("Summary updated");
      } catch (err) {
        setLoading(btn, false);
        reportError("Could not summarise the paper", err);
      }
    });
  });

  const insertIntoNotebook = (text) => {
    const nb = $("notebookEditor");
    const pos = typeof nb.selectionStart === "number" && nb.value && nb.selectionStart > 0 ? nb.selectionStart : nb.value.length;
    const before = nb.value.slice(0, pos);
    nb.value = before + text + nb.value.slice(pos);
    state.currentProject.notebook_markdown = nb.value;
    renderNotebookPreview();
  };

  listEl.querySelectorAll(".btn-cite-nb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const src = sources.find((s) => s.id === btn.dataset.id);
      if (!src) return;
      const before = $("notebookEditor").value;
      insertIntoNotebook((before && !/\s$/.test(before) ? " " : "") + (src.in_text || ""));
      try {
        await saveCurrentProject("Cited source in notebook", `Inserted ${src.in_text}`);
        toast(`Cited ${src.in_text} in the notebook`);
      } catch (e) { /* reported */ }
    });
  });

  listEl.querySelectorAll(".btn-digest-nb").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const src = sources.find((s) => s.id === btn.dataset.id);
      if (!src) return;
      const s = src.ai_summary || {};
      const lines = [
        "",
        "",
        `### ${src.title} ${src.in_text || ""}`.trim(),
        `- **Design:** ${src.study_design}${src.sample_size ? ` (${src.sample_size})` : ""}`,
        `- **Finding:** ${s.tldr || src.key_finding || "Not summarised yet."}`,
      ];
      if ((s.confounders_controlled || []).length) lines.push(`- **Controls for:** ${s.confounders_controlled.join(", ")}`);
      if (s.limitations_and_bias) lines.push(`- **Limitations:** ${s.limitations_and_bias}`);
      lines.push("");
      const nb = $("notebookEditor");
      nb.value = nb.value.replace(/\s*$/, "") + lines.join("\n");
      state.currentProject.notebook_markdown = nb.value;
      renderNotebookPreview();
      try {
        await saveCurrentProject("Added literature digest to notebook", `Inserted summary of '${src.title}'`);
        toast("Summary added to the notebook", { action: "Open", onAction: async () => setViewMode("notebook") });
      } catch (e) { /* reported */ }
    });
  });

  listEl.querySelectorAll(".btn-cite-edge").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!state.selectedEdgeId) return;
      const edge = state.currentProject.edges.find((e) => e.id === state.selectedEdgeId);
      if (!edge) return;
      edge.evidence_links = edge.evidence_links || [];
      const sid = btn.dataset.id;
      const removing = edge.evidence_links.some((l) => l.source_id === sid);
      if (removing) {
        edge.evidence_links = edge.evidence_links.filter((l) => l.source_id !== sid);
      } else {
        const src = state.currentProject.sources.find((x) => x.id === sid) || {};
        const sum = src.ai_summary || {};
        edge.evidence_links.push({
          source_id: sid,
          design: src.study_design || "",
          population: sum.population || "",
          directionality: sum.directionality || "unclear",
          limitations: sum.limitations_and_bias || "",
          researcher_assessment: "not_assessed",
          extracted_by: sum.extracted_by === "ai" ? "ai" : "researcher",
        });
      }
      edge.citation_ids = edge.evidence_links.map((l) => l.source_id);
      await saveCurrentProject(
        removing ? "Unlinked edge evidence" : "Linked edge evidence",
        `${removing ? "Removed" : "Linked"} source ${sid} ${removing ? "from" : "to"} edge ${edge.source}->${edge.target}`
      ).then(() => { if (!removing) toast("Linked. Assess the study in the arrow's inspector."); }).catch(() => {});
    });
  });

  // Bibliography
  const style = proj.citation_style || "apa";
  bibEl.innerHTML =
    `<div class="bib-actions"><button id="btnCopyBib" class="btn btn-ghost btn-xs" type="button">${icon("copy")}Copy all</button></div>` +
    sources
      .map((src) =>
        style === "bibtex"
          ? `<div class="bib-entry"><pre>${escapeHtml(src.reference_entry)}</pre></div>`
          : `<div class="bib-entry"><div class="prose">${simpleMarkdown(src.reference_entry)}</div></div>`
      )
      .join("");
  $("btnCopyBib").addEventListener("click", () => {
    const text = sources.map((s) => (s.reference_entry || "").replace(/\*/g, "")).join(style === "bibtex" ? "\n\n" : "\n");
    copyText(text).then(() => toast("Bibliography copied"));
  });
}

function sourceAlreadyInLibrary(r) {
  const lib = state.currentProject?.sources || [];
  const t = (r.title || "").trim().toLowerCase();
  return lib.some((s) => (r.doi && s.doi && s.doi.toLowerCase() === r.doi.toLowerCase()) || (s.title || "").trim().toLowerCase() === t);
}

function renderLiveLitResults(query = null) {
  const container = $("litSearchResultsContainer");
  const list = $("litSearchResultsList");
  const results = state.liveLitResults;
  if (results === null) { container.classList.add("hidden"); return; }
  container.classList.remove("hidden");
  $("litResultCount").textContent = results.length || "";

  if (results.length === 0) {
    list.innerHTML = `<div class="empty"><p class="empty-body">No papers matched${query ? ` “${escapeHtml(query)}”` : ""}. Try fewer or broader terms, or search all sources.</p></div>`;
    return;
  }

  list.innerHTML = results
    .map((r, idx) => {
      const have = sourceAlreadyInLibrary(r);
      const tldr = (r.ai_summary && r.ai_summary.tldr) || r.key_finding || "";
      return `
      <article class="src">
        <h4 class="src-title">${srcTitleHtml(r)}</h4>
        <p class="src-meta">${srcMetaHtml(r)}</p>
        ${evidenceHtml(r)}
        ${tldr ? `<p class="src-finding">${escapeHtml(tldr)}</p>` : ""}
        ${flagsHtml(r)}
        ${srcLinksHtml(r)}
        <div class="src-foot">
          <span></span>
          <div class="src-actions">
            <button class="btn ${have ? "btn-ok" : "btn-secondary"} btn-xs btn-add-searched" data-idx="${idx}" type="button" ${have ? "disabled" : ""}>
              ${icon(have ? "check" : "plus")}${have ? "In library" : "Add to library"}
            </button>
          </div>
        </div>
      </article>`;
    })
    .join("");

  list.querySelectorAll(".src-finding").forEach((p) => p.addEventListener("click", () => p.classList.toggle("is-open")));
  list.querySelectorAll(".btn-add-searched").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const item = state.liveLitResults[Number(btn.dataset.idx)];
      if (!item || !state.currentProject) return;
      setLoading(btn, true);
      state.currentProject.sources.push(item);
      try {
        await saveCurrentProject("Added literature source", `Saved '${item.title}' (${item.source_engine || item.study_design})`);
        renderLiveLitResults();
        toast("Added to library");
      } catch (e) {
        state.currentProject.sources.pop();
        setLoading(btn, false);
      }
    });
  });
}

function currentSearchQuery() {
  const q = $("litSearchInput").value.trim();
  if (q) return q;
  return state.currentProject ? state.currentProject.title : "causal inference directed acyclic graph";
}

async function runLitSearch() {
  const q = $("litSearchInput").value.trim();
  if (!q) { $("litSearchInput").focus(); return; }
  const engine = $("litEngineSelect").value || "all";
  const btn = $("btnLitSearch");
  setLoading(btn, true);
  try {
    const res = await api(`/api/literature/search?q=${encodeURIComponent(q)}&engine=${encodeURIComponent(engine)}&limit=8`);
    state.liveLitResults = res.results || [];
    renderLiveLitResults(q);
  } catch (err) {
    reportError("Search failed", err);
  } finally {
    setLoading(btn, false);
  }
}

// ============================================================================
// IDENTIFICATION STRIP
// ============================================================================

function prettyPath(s) {
  return String(s || "").replace(/<->/g, "↔").replace(/->/g, "→").replace(/<-/g, "←");
}

function renderDiagnostics() {
  const proj = state.currentProject;
  if (!proj || !proj.diagnostics) return;
  const diag = proj.diagnostics;
  const badge = $("diagStatusBadge");
  const openPaths = diag.open_backdoor_paths || [];
  const openCount = openPaths.length;
  const hasX = (proj.nodes || []).some((n) => n.role === "exposure");
  const hasY = (proj.nodes || []).some((n) => n.role === "outcome");

  const countEl = $("openBackdoorCount");
  countEl.textContent = `${openCount} open`;
  countEl.classList.toggle("has-open", openCount > 0);

  let cls, ic, title, sub;
  if (!diag.is_dag) {
    cls = "is-bad"; ic = "octagon"; title = "Not a DAG"; sub = "the graph contains a cycle";
  } else if (!hasX || !hasY) {
    cls = "is-warn"; ic = "info"; title = "Set an exposure and an outcome"; sub = "";
  } else if (diag.current_adjustment_valid) {
    cls = "is-ok"; ic = "circle-check"; title = "Identifiable"; sub = "with the current adjustment";
  } else if (openCount > 0) {
    cls = "is-warn"; ic = "alert"; title = `${openCount} open backdoor ${openCount === 1 ? "path" : "paths"}`; sub = "";
  } else {
    cls = "is-warn"; ic = "alert"; title = "Adjustment introduces bias"; sub = "see the warnings below";
  }
  badge.className = `verdict ${cls}`;
  badge.innerHTML = `${icon(ic)}<span>${escapeHtml(title)}</span>${sub ? `<span class="verdict-sub">${escapeHtml(sub)}</span>` : ""}`;

  // Minimal sufficient adjustment sets
  const adjSummary = $("diagAdjustmentSummary");
  const minSets = diag.minimal_adjustment_sets || [];
  const current = (proj.nodes || []).filter((n) => n.adjusted).map((n) => n.id).sort().join(",");
  let html = "";
  if (diag.is_dag && hasX && hasY) {
    if (minSets.length === 0) {
      html += `<span class="adj-label">No valid adjustment set from observed variables</span>`;
    } else {
      html += `<span class="adj-label">${diag.current_adjustment_valid ? "Minimal sets" : "Adjust for"}</span>`;
      minSets.slice(0, 4).forEach((s) => {
        const ids = s.map((x) => x.id);
        const isCurrent = ids.slice().sort().join(",") === current;
        const label = s.length === 0 ? "nothing (empty set)" : s.map((x) => escapeHtml(x.label)).join(", ");
        html += `<button class="adj-chip ${isCurrent ? "is-current" : ""}" data-ids="${escapeHtml(ids.join(","))}" type="button"
          title="${isCurrent ? "This is your current adjustment set" : "Apply this adjustment set"}" ${isCurrent ? 'aria-current="true"' : ""}>
          ${isCurrent ? icon("check") : ""}<span class="brace">{</span>${label}<span class="brace">}</span></button>`;
      });
      if (minSets.length > 4) html += `<span class="adj-label">+${minSets.length - 4} more</span>`;
    }
    const ivs = diag.instrumental_variables || [];
    if (ivs.length > 0) {
      html += `<span class="adj-iv" title="${escapeHtml((diag.identification || {}).instrument_note || "")}">Instrument candidates: <strong>${ivs
        .map((i) => escapeHtml(i.label) + (i.conditional_on && i.conditional_on.length ? ` (given ${i.conditional_on.map((w) => escapeHtml(w.label)).join(", ")})` : ""))
        .join(", ")}</strong></span>`;
    }
  }
  adjSummary.innerHTML = html;
  adjSummary.querySelectorAll(".adj-chip:not(.is-current)").forEach((chip) => {
    chip.addEventListener("click", async () => {
      const nodeIds = chip.dataset.ids ? chip.dataset.ids.split(",").filter(Boolean) : [];
      const snap = graphSnapshot();
      try {
        const updated = await api(`/api/projects/${proj.id}/apply-adjustment-set`, {
          method: "POST",
          body: JSON.stringify({ node_ids: nodeIds }),
        });
        setProjectState(updated);
        offerUndo("Adjustment set applied", snap, "Restored previous adjustment set");
      } catch (err) {
        reportError("Could not apply the adjustment set", err);
      }
    });
  });

  // Warnings, most severe first
  const order = { critical: 0, warning: 1, info: 2 };
  const iconFor = { critical: "octagon", warning: "alert", info: "info" };
  const warns = (diag.warnings || []).slice().sort((a, b) => (order[a.severity] ?? 3) - (order[b.severity] ?? 3));
  $("diagWarningsBar").innerHTML = warns
    .map(
      (w) => `
      <details class="warn warn-${escapeHtml(w.severity || "info")}">
        <summary>${icon(iconFor[w.severity] || "info")}<span>${escapeHtml(prettyPath(w.type))}</span>${icon("chevron-down", "chev")}</summary>
        <div class="warn-msg">${escapeHtml(prettyPath(w.message))}</div>
      </details>`
    )
    .join("");

  // Paths drawer
  const pathItem = (p, stateHtml) => `
    <div class="path-item" tabindex="0" data-nodes="${escapeHtml((p.nodes || []).join(","))}">
      <span class="path-text">${escapeHtml(prettyPath(p.pretty))}
        ${p.blocking_nodes && p.blocking_nodes.length && !p.is_open ? `<span class="path-blockers">Blocked by ${p.blocking_nodes.map((id) => escapeHtml(nodeLabel(id))).join(", ")}</span>` : ""}
        ${p.opened_colliders && p.opened_colliders.length ? `<span class="path-blockers">Opened by conditioning on ${p.opened_colliders.map((id) => escapeHtml(nodeLabel(id))).join(", ")}</span>` : ""}
      </span>
      ${stateHtml}
    </div>`;

  $("openBackdoorList").innerHTML =
    openPaths.map((p) => pathItem(p, `<span class="path-state is-open">Open</span>`)).join("") ||
    `<div class="path-none">None. Every non-causal path is blocked.</div>`;
  $("causalPathsList").innerHTML =
    (diag.causal_paths || []).map((p) => pathItem(p, `<span class="path-state ${p.is_open ? "is-causal" : "is-blocked"}">${p.is_open ? "Open" : "Blocked"}</span>`)).join("") ||
    `<div class="path-none">No directed path from exposure to outcome.</div>`;
  $("allPathsList").innerHTML =
    (diag.paths || []).map((p) => pathItem(p, `<span class="path-state ${p.is_open ? "is-open" : "is-blocked"}">${p.is_open ? "Open" : "Blocked"}</span>`)).join("") ||
    `<div class="path-none">No paths between exposure and outcome.</div>`;

  renderTiers(diag);

  $("pathsDrawer").querySelectorAll(".path-item").forEach((el) => {
    const on = () => highlightPath(el.dataset.nodes ? el.dataset.nodes.split(",") : []);
    el.addEventListener("mouseenter", on);
    el.addEventListener("focus", on);
    el.addEventListener("mouseleave", clearHighlight);
    el.addEventListener("blur", clearHighlight);
  });
}

function renderTiers(diag) {
  const idn = diag.identification || {};
  const sup = diag.scientific_support || { edges: [] };
  const weak = (sup.edges || []).filter((r) => r.weak);
  const strong = (sup.edges || []).filter((r) => !r.weak);
  $("tiersPanel").innerHTML = `
    <div class="tier">
      <h4 class="subhead">What identification assumes</h4>
      <div class="tier-meta">${escapeHtml(idn.criterion || "Not computed")}${idn.search_complete === false ? ". Set search stopped early on a large graph." : ""}</div>
      <ul>${(idn.assumes || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("") || "<li>Mark an exposure and an outcome first.</li>"}</ul>
    </div>
    <div class="tier">
      <h4 class="subhead">How well the arrows are supported <span class="count">${weak.length}/${sup.total || 0} weak</span></h4>
      <div class="tier-meta">${escapeHtml(sup.note || "")}</div>
      ${weak.map((r) => `<div class="support-row"><span class="edge-name">${escapeHtml(prettyPath(r.edge))}</span><span class="pill ${r.evidence_class === "ai_hypothesis" ? "pill-warn" : ""}">${escapeHtml(EVIDENCE_LABELS[r.evidence_class] || r.evidence_class)}</span><span class="why">${escapeHtml(r.weak_reasons.join("; "))}</span></div>`).join("")}
      ${strong.map((r) => `<div class="support-row"><span class="edge-name">${escapeHtml(prettyPath(r.edge))}</span><span class="pill pill-ok">${escapeHtml(EVIDENCE_LABELS[r.evidence_class] || r.evidence_class)}</span><span class="why">${r.researcher_supported_links} supporting source(s)</span></div>`).join("")}
      ${!(sup.edges || []).length ? `<div class="muted">No arrows to grade yet.</div>` : ""}
    </div>`;
}

function highlightPath(nodeIds) {
  if (!nodeIds.length) return;
  const svg = $("dagSvg");
  svg.classList.add("has-hl");
  const set = new Set(nodeIds);
  svg.querySelectorAll(".dag-node").forEach((g) => g.classList.toggle("hl", set.has(g.dataset.id)));
  const pairs = new Set();
  for (let i = 0; i < nodeIds.length - 1; i++) {
    pairs.add(`${nodeIds[i]}|${nodeIds[i + 1]}`);
    pairs.add(`${nodeIds[i + 1]}|${nodeIds[i]}`);
  }
  svg.querySelectorAll(".dag-edge").forEach((g) => {
    const hl = pairs.has(`${g.dataset.src}|${g.dataset.tgt}`);
    g.classList.toggle("hl", hl);
    const line = g.querySelector(".edge-line");
    if (hl) line.setAttribute("marker-end", "url(#arrow-hl)");
    else line.setAttribute("marker-end", line.dataset.marker);
  });
}

function clearHighlight() {
  const svg = $("dagSvg");
  svg.classList.remove("has-hl");
  svg.querySelectorAll(".hl").forEach((el) => el.classList.remove("hl"));
  svg.querySelectorAll(".edge-line").forEach((l) => l.setAttribute("marker-end", l.dataset.marker));
}

// ============================================================================
// CANVAS: VIEWPORT, LANES, NODES, EDGES
// ============================================================================

function svgEl(tag, attrs = {}, cls = "") {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (cls) el.setAttribute("class", cls);
  return el;
}

function unitsPerPx() {
  const svg = $("dagSvg");
  return state.view.w / Math.max(1, svg.clientWidth);
}

function applyView() {
  const svg = $("dagSvg");
  const cw = svg.clientWidth;
  const ch = svg.clientHeight;
  if (!cw || !ch) return; // canvas hidden
  const v = state.view;
  const newH = v.w * (ch / cw);
  v.y += (v.h - newH) / 2;
  v.h = newH;
  svg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`);
  $("btnZoomLabel").textContent = `${Math.round((cw / v.w) * 100)}%`;
  drawLanes();
}

function zoomAt(factor, cx, cy) {
  const v = state.view;
  const svg = $("dagSvg");
  const minW = svg.clientWidth / 3.2;
  const maxW = svg.clientWidth * 4;
  const w = Math.min(maxW, Math.max(minW, v.w * factor));
  const k = w / v.w;
  if (cx === undefined) { cx = v.x + v.w / 2; cy = v.y + v.h / 2; }
  v.x = cx - (cx - v.x) * k;
  v.y = cy - (cy - v.y) * k;
  v.w = w;
  v.h = v.h * k;
  applyView();
}

function resetZoom() {
  const svg = $("dagSvg");
  const v = state.view;
  const cx = v.x + v.w / 2;
  const cy = v.y + v.h / 2;
  v.w = svg.clientWidth || 900;
  v.h = svg.clientHeight || 500;
  v.x = cx - v.w / 2;
  v.y = cy - v.h / 2;
  applyView();
}

function fitView() {
  const nodes = state.currentProject?.nodes || [];
  const svg = $("dagSvg");
  const cw = svg.clientWidth;
  const ch = svg.clientHeight;
  if (!cw || !ch) return;
  let minX = 0, maxX = BASE_W, minY = 40, maxY = 460;
  if (nodes.length) {
    minX = Math.min(...nodes.map((n) => n.x)) - NODE_W / 2 - 48;
    maxX = Math.max(...nodes.map((n) => n.x)) + NODE_W / 2 + 48;
    minY = Math.min(...nodes.map((n) => n.y)) - NODE_H / 2 - 64;
    maxY = Math.max(...nodes.map((n) => n.y)) + NODE_H / 2 + 56;
  }
  let w = Math.max(maxX - minX, cw / 1.6);
  let h = Math.max(maxY - minY, 1);
  if (h / w > ch / cw) w = h * (cw / ch);
  h = w * (ch / cw);
  state.view = { x: (minX + maxX) / 2 - w / 2, y: (minY + maxY) / 2 - h / 2, w, h };
  applyView();
}

function drawLanes() {
  const layer = $("dagGridLayer");
  layer.innerHTML = "";
  const v = state.view;
  const s = unitsPerPx();
  const laneW = BASE_W / 4;
  LANES.forEach((lane, i) => {
    if (i % 2 === 1) layer.appendChild(svgEl("rect", { x: i * laneW, y: v.y - 10, width: laneW, height: v.h + 20 }, "lane"));
    const text = svgEl("text", { x: i * laneW + laneW / 2, y: v.y + 22 * s, "text-anchor": "middle", "font-size": 11 * s }, "lane-label");
    const t1 = svgEl("tspan", {}, "lane-label-t");
    t1.textContent = lane.t;
    const t2 = svgEl("tspan", {});
    t2.textContent = `  ${lane.label}`;
    text.append(t1, t2);
    layer.appendChild(text);
  });
}

/** Point where the segment from the node centre towards (tx, ty) leaves the node box. */
function boxExit(cx, cy, tx, ty, pad) {
  const dx = tx - cx;
  const dy = ty - cy;
  const hw = NODE_W / 2 + pad;
  const hh = NODE_H / 2 + pad;
  if (dx === 0 && dy === 0) return { x: cx, y: cy };
  const t = Math.min(dx !== 0 ? hw / Math.abs(dx) : Infinity, dy !== 0 ? hh / Math.abs(dy) : Infinity);
  return { x: cx + dx * t, y: cy + dy * t };
}

function truncate(label, max) {
  return label.length > max ? label.slice(0, max - 1).trimEnd() + "…" : label;
}

/** Split a label into at most two lines of ~max chars, breaking on spaces. */
function wrapLabel(label, max) {
  if (label.length <= max) return [label];
  const words = label.split(/\s+/);
  let first = "";
  while (words.length && (first + " " + words[0]).trim().length <= max) first = (first + " " + words.shift()).trim();
  if (!first) return [truncate(label, max)];
  return [first, truncate(words.join(" "), max)];
}

function renderCanvas() {
  const proj = state.currentProject;
  if (!proj) return;
  const edgesLayer = $("dagEdgesLayer");
  const nodesLayer = $("dagNodesLayer");
  edgesLayer.innerHTML = "";
  nodesLayer.innerHTML = "";

  const nodes = proj.nodes || [];
  const edges = proj.edges || [];
  $("canvasEmpty").classList.toggle("hidden", nodes.length > 0);

  const nodeMap = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const pairKeys = new Set(edges.map((e) => `${e.source}|${e.target}`));

  // Edges
  edges.forEach((edge) => {
    const u = nodeMap[edge.source];
    const v = nodeMap[edge.target];
    if (!u || !v) return;
    const status = edgeCategory(edge);
    const bi = status === "bi";
    const reverse = pairKeys.has(`${edge.target}|${edge.source}`);
    const dx = v.x - u.x;
    const dy = v.y - u.y;
    const dist = Math.hypot(dx, dy) || 1;
    const bend = bi ? 0.22 : reverse ? 0.14 : 0;
    const cxp = (u.x + v.x) / 2 - (dy / dist) * dist * bend;
    const cyp = (u.y + v.y) / 2 + (dx / dist) * dist * bend;
    const p0 = boxExit(u.x, u.y, bend ? cxp : v.x, bend ? cyp : v.y, bi ? 4 : 2);
    const p2 = boxExit(v.x, v.y, bend ? cxp : u.x, bend ? cyp : u.y, 5);
    const d = bend ? `M ${p0.x} ${p0.y} Q ${cxp} ${cyp} ${p2.x} ${p2.y}` : `M ${p0.x} ${p0.y} L ${p2.x} ${p2.y}`;

    const g = svgEl("g", { "data-id": edge.id, "data-src": edge.source, "data-tgt": edge.target }, `dag-edge e-${status} ${state.selectedEdgeId === edge.id ? "selected" : ""}`);
    const title = svgEl("title");
    title.textContent = `${u.label} ${bi ? "↔" : "→"} ${v.label}: ${bi ? "shared latent cause" : EVIDENCE_LABELS[edge.evidence_class] || ""}${edge.contested ? " (contested)" : ""}${edge.mechanism ? `. ${edge.mechanism}` : ""}`;
    const marker = state.selectedEdgeId === edge.id ? "url(#arrow-hl)" : `url(#arrow-${status})`;
    const line = svgEl("path", { d, "marker-end": marker, "data-marker": marker }, "edge-line");
    if (bi) line.setAttribute("marker-start", marker);
    g.append(title, svgEl("path", { d }, "edge-hit"), line);

    const cites = (edge.citation_ids || []).length;
    const sign = { positive: "+", negative: "−", non_linear: "∪" }[edge.effect_direction] || "";
    if (cites || sign) {
      const mx = bend ? 0.25 * p0.x + 0.5 * cxp + 0.25 * p2.x : (p0.x + p2.x) / 2;
      const my = bend ? 0.25 * p0.y + 0.5 * cyp + 0.25 * p2.y : (p0.y + p2.y) / 2;
      const tag = svgEl("text", { x: mx, y: my - 6, "text-anchor": "middle" }, `edge-tag sign-${edge.effect_direction}`);
      tag.textContent = [sign, cites ? `${cites} ref${cites > 1 ? "s" : ""}` : ""].filter(Boolean).join("  ");
      g.appendChild(tag);
    }

    g.addEventListener("click", (ev) => {
      ev.stopPropagation();
      selectEdge(edge.id);
    });
    edgesLayer.appendChild(g);
  });

  // Nodes
  nodes.forEach((node) => {
    const latent = !node.observed || node.role === "latent";
    const role = ROLE_LABELS[node.role] ? node.role : "other";
    const sel = state.selectedNodeId === node.id;
    const isSource = state.connectSourceId === node.id;
    const g = svgEl(
      "g",
      {
        transform: `translate(${node.x - NODE_W / 2}, ${node.y - NODE_H / 2})`,
        "data-id": node.id,
        tabindex: "0",
        role: "button",
        "aria-pressed": String(sel),
        "aria-label": `${node.label}, ${ROLE_LABELS[role]}${node.adjusted ? ", adjusted for" : ""}, time ${node.time_index}`,
      },
      `dag-node role-${role} ${sel ? "selected" : ""} ${latent ? "is-latent" : ""} ${isSource ? "is-source" : ""}`
    );
    const title = svgEl("title");
    title.textContent = node.operationalization ? `${node.label}\n${node.operationalization}` : node.label;
    g.appendChild(title);
    g.appendChild(svgEl("rect", { x: -7, y: -7, width: NODE_W + 14, height: NODE_H + 14, rx: 13 }, "node-ring"));
    if (node.adjusted) g.appendChild(svgEl("rect", { x: -4, y: -4, width: NODE_W + 8, height: NODE_H + 8, rx: 10 }, "node-adjust"));
    g.appendChild(svgEl("rect", { width: NODE_W, height: NODE_H, rx: 7 }, "node-body"));
    g.appendChild(svgEl("circle", { cx: 13, cy: 16, r: 3 }, "node-dot"));
    const roleText = svgEl("text", { x: 21, y: 19.5 }, "node-role");
    roleText.textContent = ROLE_LABELS[role] + (node.adjusted ? ", adjusted" : "");
    const timeText = svgEl("text", { x: NODE_W - 10, y: 19.5, "text-anchor": "end" }, "node-time");
    timeText.textContent = `t${SUBSCRIPTS[node.time_index] ?? node.time_index}`;
    g.append(roleText, timeText);
    const lines = wrapLabel(node.label, 24);
    lines.forEach((ln, i) => {
      const t = svgEl("text", { x: 12, y: lines.length === 1 ? 41 : 37 + i * 14.5 }, "node-label");
      t.textContent = ln;
      g.appendChild(t);
    });

    g.addEventListener("pointerdown", (ev) => onNodePointerDown(ev, node));
    g.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        if (state.connectMode) handleConnectClick(node.id);
        else { state.focusNodeAfterRender = true; selectNode(node.id); }
      }
    });
    nodesLayer.appendChild(g);
  });

  if (state.focusNodeAfterRender && state.selectedNodeId) {
    const el = nodesLayer.querySelector(`[data-id="${CSS.escape(state.selectedNodeId)}"]`);
    if (el) el.focus({ preventScroll: true });
  }
  state.focusNodeAfterRender = false;
}

function selectNode(id) {
  state.selectedNodeId = id;
  state.selectedEdgeId = null;
  renderCanvas();
  renderInspector();
  renderVault();
}

function selectEdge(id) {
  state.selectedEdgeId = id;
  state.selectedNodeId = null;
  renderCanvas();
  renderInspector();
  renderVault();
}

function clearSelection() {
  if (!state.selectedNodeId && !state.selectedEdgeId) return;
  state.selectedNodeId = null;
  state.selectedEdgeId = null;
  renderCanvas();
  renderInspector();
  renderVault();
}

function getSvgPoint(ev) {
  const svg = $("dagSvg");
  const pt = svg.createSVGPoint();
  pt.x = ev.clientX;
  pt.y = ev.clientY;
  const ctm = svg.getScreenCTM();
  return ctm ? pt.matrixTransform(ctm.inverse()) : { x: 0, y: 0 };
}

function onNodePointerDown(ev, node) {
  if (ev.button !== 0) return;
  ev.stopPropagation();
  if (state.connectMode) {
    handleConnectClick(node.id);
    return;
  }
  const pt = getSvgPoint(ev);
  state.drag = { kind: "node", id: node.id, ox: pt.x - node.x, oy: pt.y - node.y, sx: ev.clientX, sy: ev.clientY, moved: false };
  if (state.selectedNodeId !== node.id) selectNode(node.id);
}

function onCanvasPointerDown(ev) {
  if (ev.button !== 0 && ev.button !== 1) return;
  if (ev.target.closest(".dag-node, .dag-edge")) return;
  state.drag = { kind: "pan", sx: ev.clientX, sy: ev.clientY, vx: state.view.x, vy: state.view.y, moved: false };
  $("dagSvg").classList.add("is-panning");
}

function onPointerMove(ev) {
  const d = state.drag;
  if (!d) return;
  if (!d.moved && Math.hypot(ev.clientX - d.sx, ev.clientY - d.sy) < 3) return;
  d.moved = true;
  if (d.kind === "pan") {
    const s = unitsPerPx();
    state.view.x = d.vx - (ev.clientX - d.sx) * s;
    state.view.y = d.vy - (ev.clientY - d.sy) * s;
    applyView();
  } else if (d.kind === "node") {
    const node = state.currentProject?.nodes.find((n) => n.id === d.id);
    if (!node) return;
    const pt = getSvgPoint(ev);
    node.x = Math.round(pt.x - d.ox);
    node.y = Math.round(pt.y - d.oy);
    const g = $("dagNodesLayer").querySelector(`[data-id="${CSS.escape(node.id)}"]`);
    if (g) g.setAttribute("transform", `translate(${node.x - NODE_W / 2}, ${node.y - NODE_H / 2})`);
    scheduleCanvasRender();
  }
}

function scheduleCanvasRender() {
  if (state._raf) return;
  state._raf = requestAnimationFrame(() => {
    state._raf = null;
    renderCanvas();
  });
}

async function onPointerUp() {
  const d = state.drag;
  if (!d) return;
  state.drag = null;
  $("dagSvg").classList.remove("is-panning");
  if (d.kind === "pan" && !d.moved) {
    if (state.connectMode) cancelConnect();
    else clearSelection();
  } else if (d.kind === "node" && d.moved) {
    await saveCurrentProject().catch(() => {});
  }
}

function setConnectMode(on) {
  state.connectMode = on;
  state.connectSourceId = null;
  $("btnConnectMode").setAttribute("aria-pressed", String(on));
  $("dagSvg").classList.toggle("is-connecting", on);
  updateConnectHint();
  renderCanvas();
}

function cancelConnect() { setConnectMode(false); }

function updateConnectHint() {
  const hint = $("connectHint");
  if (!state.connectMode) { hint.classList.add("hidden"); return; }
  hint.classList.remove("hidden");
  hint.innerHTML = state.connectSourceId
    ? `<span>Now choose the effect of <strong>${escapeHtml(nodeLabel(state.connectSourceId))}</strong></span><button class="btn btn-xs" type="button" id="btnCancelConnect">Cancel</button>`
    : `<span>Choose the cause</span><button class="btn btn-xs" type="button" id="btnCancelConnect">Cancel</button>`;
  $("btnCancelConnect").addEventListener("click", cancelConnect);
}

async function handleConnectClick(nodeId) {
  if (!state.connectSourceId) {
    state.connectSourceId = nodeId;
    updateConnectHint();
    renderCanvas();
    return;
  }
  const source = state.connectSourceId;
  const target = nodeId;
  setConnectMode(false);
  if (source === target) return;
  const exists = state.currentProject.edges.some((e) => e.source === source && e.target === target);
  if (exists) { toast("That arrow already exists"); return; }
  const id = "edge_" + Math.random().toString(36).slice(2, 9);
  state.currentProject.edges.push({
    id,
    source,
    target,
    edge_type: "directed",
    evidence_class: "researcher_hypothesis",
    contested: false,
    mechanism: "",
    effect_direction: "unknown",
    confidence: 0.75,
    evidence_links: [],
    citation_ids: [],
    origin: "researcher",
    rationale: "Drawn by researcher on canvas",
    approved_by_researcher: true,
  });
  state.selectedEdgeId = id;
  state.selectedNodeId = null;
  await saveCurrentProject("Added directed edge", `Created edge ${source} -> ${target}`).catch(() => {});
}

async function addNode() {
  if (!state.currentProject) return;
  const nodes = state.currentProject.nodes;
  const nCount = nodes.length + 1;
  const v = state.view;
  const cx = v.x + v.w / 2;
  const cy = v.y + v.h / 2;
  let x = Math.round(cx), y = Math.round(cy);
  for (let tries = 0; tries < 12 && nodes.some((n) => Math.abs(n.x - x) < NODE_W && Math.abs(n.y - y) < NODE_H + 16); tries++) {
    y += NODE_H + 24;
    if (tries === 5) { x += NODE_W + 30; y = Math.round(cy); }
  }
  const lane = Math.max(0, Math.min(3, Math.floor(x / (BASE_W / 4))));
  const newNode = {
    id: `var_${nCount}_${Math.random().toString(36).slice(2, 5)}`,
    label: `Variable ${nCount}`,
    role: "confounder",
    adjusted: false,
    observed: true,
    time_index: lane,
    operationalization: "",
    x,
    y,
  };
  nodes.push(newNode);
  state.selectedNodeId = newNode.id;
  state.selectedEdgeId = null;
  await saveCurrentProject("Added node", `Created node '${newNode.label}'`).catch(() => {});
  const lbl = $("inspNodeLabel");
  if (lbl) { lbl.focus(); lbl.select(); }
}

async function autoLayout() {
  if (!state.currentProject) return;
  const snap = graphSnapshot();
  const waves = { 0: [], 1: [], 2: [], 3: [] };
  state.currentProject.nodes.forEach((n) => waves[Math.min(3, Math.max(0, n.time_index || 0))].push(n));
  const laneW = BASE_W / 4;
  Object.keys(waves).forEach((w) => {
    const arr = waves[w].sort((a, b) => a.y - b.y);
    const gap = NODE_H + 40;
    const top = 260 - ((arr.length - 1) * gap) / 2;
    arr.forEach((node, idx) => {
      node.x = Math.round(Number(w) * laneW + laneW / 2);
      node.y = Math.round(top + idx * gap);
    });
  });
  await saveCurrentProject("Applied temporal layout", "Arranged nodes by temporal wave (t0 -> t3)").catch(() => {});
  fitView();
  offerUndo("Arranged by time wave", snap, "Restored previous layout");
}

async function deleteSelection() {
  const proj = state.currentProject;
  if (!proj) return;
  if (state.selectedNodeId) {
    const node = proj.nodes.find((n) => n.id === state.selectedNodeId);
    if (!node) return;
    const snap = graphSnapshot();
    proj.nodes = proj.nodes.filter((n) => n.id !== node.id);
    proj.edges = proj.edges.filter((e) => e.source !== node.id && e.target !== node.id);
    state.selectedNodeId = null;
    await saveCurrentProject("Deleted node", `Removed variable '${node.label}'`).catch(() => {});
    offerUndo(`Deleted ${node.label}`, snap, `Restored variable '${node.label}'`);
  } else if (state.selectedEdgeId) {
    const edge = proj.edges.find((e) => e.id === state.selectedEdgeId);
    if (!edge) return;
    const snap = graphSnapshot();
    proj.edges = proj.edges.filter((e) => e.id !== edge.id);
    state.selectedEdgeId = null;
    await saveCurrentProject("Deleted edge", `Removed edge ${edge.source} -> ${edge.target}`).catch(() => {});
    offerUndo("Arrow deleted", snap, `Restored edge ${edge.source} -> ${edge.target}`);
  }
}

// ============================================================================
// INSPECTOR
// ============================================================================

function option(value, label, current) {
  return `<option value="${value}" ${String(current) === String(value) ? "selected" : ""}>${label}</option>`;
}

function renderInspector() {
  const panel = $("inspectorPanel");
  const proj = state.currentProject;
  if (!proj || (!state.selectedNodeId && !state.selectedEdgeId)) {
    panel.classList.add("hidden");
    panel.innerHTML = "";
    return;
  }

  if (state.selectedNodeId) {
    const node = proj.nodes.find((n) => n.id === state.selectedNodeId);
    if (!node) { panel.classList.add("hidden"); return; }
    const inferred = proj.diagnostics?.structural_roles?.[node.id];
    const showInferred = inferred && inferred !== node.role && ROLE_LABELS[inferred];
    panel.dataset.for = node.id;
    panel.classList.remove("hidden");
    panel.innerHTML = `
      <div class="insp-head role-${escapeHtml(node.role)}">
        <div class="grow">
          <div class="insp-kicker">Variable</div>
          <div class="insp-title">${escapeHtml(node.label)}</div>
          ${showInferred ? `<div class="inferred role-${escapeHtml(inferred)}"><span class="swatch"></span>The graph structure makes this a ${escapeHtml(ROLE_LABELS[inferred].toLowerCase())}</div>` : ""}
        </div>
        <button class="icon-btn icon-btn-sm" id="btnCloseInspector" type="button" aria-label="Close inspector">${icon("x")}</button>
      </div>
      <div class="insp-body">
        <div class="field">
          <label for="inspNodeLabel">Name</label>
          <input type="text" id="inspNodeLabel" class="input" value="${escapeHtml(node.label)}" />
        </div>
        <div class="field-row">
          <div class="field grow">
            <label for="inspNodeRole">Role</label>
            <select id="inspNodeRole" class="select">
              ${option("exposure", "Exposure (X)", node.role)}
              ${option("outcome", "Outcome (Y)", node.role)}
              ${option("confounder", "Confounder", node.role)}
              ${option("mediator", "Mediator", node.role)}
              ${option("collider", "Collider", node.role)}
              ${option("instrument", "Instrument", node.role)}
              ${option("effect_modifier", "Effect modifier", node.role)}
              ${option("competing_exposure", "Competing exposure", node.role)}
              ${option("latent", "Unobserved (U)", node.role)}
              ${option("other", "Other", node.role)}
            </select>
          </div>
          <div class="field" style="width:112px;flex:none">
            <label for="inspNodeTime">Time</label>
            <select id="inspNodeTime" class="select">
              ${option(0, "t₀ baseline", node.time_index)}
              ${option(1, "t₁ exposure", node.time_index)}
              ${option(2, "t₂ mediator", node.time_index)}
              ${option(3, "t₃ outcome", node.time_index)}
            </select>
          </div>
        </div>
        <div class="field">
          <label for="inspNodeOp">How it is measured</label>
          <textarea id="inspNodeOp" class="input" rows="2" placeholder="Instrument, scale, coding, timing">${escapeHtml(node.operationalization || "")}</textarea>
        </div>
        <label class="check">
          <input type="checkbox" id="inspNodeAdjusted" ${node.adjusted ? "checked" : ""} ${node.role === "latent" ? "disabled" : ""} />
          <div><strong>Adjust for this variable</strong><span>${node.role === "latent" ? "Unobserved variables cannot be adjusted for." : "Condition on it in the analysis (part of Z). Applies immediately."}</span></div>
        </label>
      </div>
      <div class="insp-foot">
        <button id="btnDeleteNode" class="btn btn-danger btn-sm" type="button" title="Delete (Del)">${icon("trash")}Delete</button>
        <span class="spacer"></span>
        <button id="btnSaveNode" class="btn btn-primary btn-sm" type="button">Save</button>
      </div>`;

    $("btnCloseInspector").onclick = clearSelection;
    $("inspNodeAdjusted").onchange = async (ev) => {
      node.adjusted = ev.target.checked;
      await saveCurrentProject("Toggled conditioning", `${node.adjusted ? "Conditioned on" : "Unconditioned"} '${node.label}'`).catch(() => {});
    };
    const save = async () => {
      node.label = $("inspNodeLabel").value.trim() || node.label;
      node.role = $("inspNodeRole").value;
      node.observed = node.role !== "latent";
      if (!node.observed) node.adjusted = false;
      node.time_index = Number($("inspNodeTime").value);
      node.operationalization = $("inspNodeOp").value.trim();
      await saveCurrentProject("Updated node", `Updated variable '${node.label}'`).then(() => toast("Variable saved")).catch(() => {});
    };
    $("btnSaveNode").onclick = save;
    $("inspNodeLabel").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { ev.preventDefault(); save(); } });
    $("btnDeleteNode").onclick = deleteSelection;
    return;
  }

  const edge = proj.edges.find((e) => e.id === state.selectedEdgeId);
  if (!edge) { panel.classList.add("hidden"); return; }
  panel.dataset.for = edge.id;
  panel.classList.remove("hidden");
  const sources = proj.formatted_sources || proj.sources || [];
  const links = edge.evidence_links || [];
  const linkRows = links
    .map((l, i) => {
      const src = sources.find((x) => x.id === l.source_id) || { title: l.source_id };
      return `
      <div class="ev-link" data-i="${i}">
        <div class="ev-link-title">${escapeHtml(src.title)}</div>
        <div class="ev-link-meta">${escapeHtml(src.in_text || "")}${l.design ? `, ${escapeHtml(l.design)}` : ""}${l.population ? `, ${escapeHtml(l.population)}` : ""}${l.extracted_by === "ai" ? ", details pre-filled by AI" : ""}</div>
        <div class="field-row">
          <div class="field grow"><label for="evA${i}">Your assessment</label>
            <select id="evA${i}" class="select ev-assess">${Object.entries(ASSESSMENTS).map(([k, v]) => option(k, v, l.researcher_assessment)).join("")}</select></div>
          <div class="field grow"><label for="evD${i}">Direction</label>
            <select id="evD${i}" class="select ev-dir">${Object.entries(DIRECTIONALITY).map(([k, v]) => option(k, v, l.directionality)).join("")}</select></div>
        </div>
      </div>`;
    })
    .join("");
  panel.innerHTML = `
    <div class="insp-head">
      <div class="grow">
        <div class="insp-kicker">${edge.origin === "ai" ? "Arrow, proposed by the assistant" : "Arrow"}</div>
        <div class="insp-title">${escapeHtml(nodeLabel(edge.source))}<span class="arrow">${edge.edge_type === "bidirected" ? "↔" : "→"}</span>${escapeHtml(nodeLabel(edge.target))}</div>
      </div>
      <button class="icon-btn icon-btn-sm" id="btnCloseInspector" type="button" aria-label="Close inspector">${icon("x")}</button>
    </div>
    <div class="insp-body">
      <div class="field">
        <label for="inspEdgeType">Type</label>
        <select id="inspEdgeType" class="select">
          ${option("directed", "Causal effect (→)", edge.edge_type)}
          ${option("bidirected", "Shared unmeasured cause (↔)", edge.edge_type)}
        </select>
      </div>
      <div class="field">
        <label for="inspEdgeClass">Evidence class</label>
        <select id="inspEdgeClass" class="select">
          ${Object.entries(EVIDENCE_LABELS).map(([k, v]) => option(k, v, edge.evidence_class)).join("")}
        </select>
        <p class="help" id="inspEdgeClassHelp">${escapeHtml(EVIDENCE_HELP[edge.evidence_class] || "")}</p>
      </div>
      <div class="field-row">
        <div class="field grow">
          <label for="inspEdgeSign">Effect sign</label>
          <select id="inspEdgeSign" class="select">
            ${option("positive", "Positive (+)", edge.effect_direction)}
            ${option("negative", "Negative (−)", edge.effect_direction)}
            ${option("non_linear", "Non-linear", edge.effect_direction)}
            ${option("unknown", "Unknown", edge.effect_direction)}
          </select>
        </div>
      </div>
      <label class="check-inline"><input type="checkbox" id="inspEdgeContested" ${edge.contested ? "checked" : ""} />Contested: studies disagree about this arrow</label>
      <div class="field" style="margin-top:12px">
        <label for="inspEdgeMech">Mechanism</label>
        <textarea id="inspEdgeMech" class="input" rows="3" placeholder="The biological, economic or behavioural process behind this arrow">${escapeHtml(edge.mechanism || "")}</textarea>
      </div>
      <div class="ev-links">
        <div class="ev-links-head">Evidence (${links.length})</div>
        ${linkRows || `<div class="insp-note" style="margin-top:0">${icon("link")}<span>No linked sources. Use Link on a source in the Sources panel, then record whether it supports this arrow and establishes its direction.</span></div>`}
      </div>
    </div>
    <div class="insp-foot">
      <button id="btnDeleteEdge" class="btn btn-danger btn-sm" type="button" title="Delete (Del)">${icon("trash")}Delete</button>
      <button id="btnFlipEdge" class="btn btn-ghost btn-sm" type="button" title="Reverse the arrow" ${edge.edge_type === "bidirected" ? "disabled" : ""}>${icon("swap")}Reverse</button>
      <span class="spacer"></span>
      <button id="btnSaveEdge" class="btn btn-primary btn-sm" type="button">Save</button>
    </div>`;

  $("inspEdgeClass").addEventListener("change", (ev) => ($("inspEdgeClassHelp").textContent = EVIDENCE_HELP[ev.target.value] || ""));
  $("btnCloseInspector").onclick = clearSelection;
  $("btnSaveEdge").onclick = async () => {
    edge.edge_type = $("inspEdgeType").value;
    edge.evidence_class = $("inspEdgeClass").value;
    edge.contested = $("inspEdgeContested").checked;
    edge.effect_direction = $("inspEdgeSign").value;
    edge.mechanism = $("inspEdgeMech").value.trim();
    panel.querySelectorAll(".ev-link").forEach((row) => {
      const l = edge.evidence_links[Number(row.dataset.i)];
      if (!l) return;
      l.researcher_assessment = row.querySelector(".ev-assess").value;
      l.directionality = row.querySelector(".ev-dir").value;
    });
    if (edge.origin === "ai" && edge.evidence_class !== "ai_hypothesis") edge.origin = "researcher";
    await saveCurrentProject("Updated edge evidence", `Edge ${edge.source} -> ${edge.target}: ${edge.evidence_class}${edge.contested ? ", contested" : ""}`).then(() => toast("Arrow saved")).catch(() => {});
  };
  $("btnFlipEdge").onclick = async () => {
    const snap = graphSnapshot();
    [edge.source, edge.target] = [edge.target, edge.source];
    await saveCurrentProject("Flipped causal arrow", `Reversed direction to ${edge.source} -> ${edge.target}`).catch(() => {});
    offerUndo("Arrow reversed", snap, "Restored arrow direction");
  };
  $("btnDeleteEdge").onclick = deleteSelection;
}

// ============================================================================
// NOTEBOOK, EVIDENCE MATRIX, BRIEFING, HISTORY
// ============================================================================

let notebookTimer = null;

function renderNotebook() {
  const proj = state.currentProject;
  if (!proj) return;
  const editor = $("notebookEditor");
  const incoming = proj.notebook_markdown || "";
  // Never clobber what the researcher is typing.
  if (document.activeElement !== editor && editor.value !== incoming) editor.value = incoming;
  renderNotebookPreview();
}

function renderNotebookPreview() {
  const md = $("notebookEditor").value;
  $("notebookPreview").innerHTML = md.trim()
    ? simpleMarkdown(md)
    : `<p class="muted">The preview of your notebook appears here.</p>`;
}

async function saveNotebook(withAudit) {
  if (!state.currentProject) return;
  clearTimeout(notebookTimer);
  state.currentProject.notebook_markdown = $("notebookEditor").value;
  if (withAudit) await saveCurrentProject("Saved research notebook", "Updated protocol markdown notes").then(() => toast("Notebook saved")).catch(() => {});
  else await saveCurrentProject().catch(() => {});
}

function matrixRows() {
  const proj = state.currentProject;
  const sources = proj.formatted_sources || proj.sources || [];
  const labels = Object.fromEntries((proj.nodes || []).map((n) => [n.id, n.label]));
  const rows = sources.map((src) => ({
    src,
    strength: evidenceOf(src.study_design).level,
    year: Number(src.year) || 0,
    title: (src.title || "").toLowerCase(),
    edges: (proj.edges || [])
      .filter((e) => (e.citation_ids || []).includes(src.id))
      .map((e) => `${labels[e.source] || e.source} → ${labels[e.target] || e.target}`),
  }));
  const { key, dir } = state.matrixSort;
  rows.sort((a, b) => (a[key] < b[key] ? -1 : a[key] > b[key] ? 1 : 0) * dir || b.strength - a.strength);
  return rows;
}

function renderEvidenceMatrix() {
  const proj = state.currentProject;
  if (!proj) return;
  const container = $("evidenceMatrixContainer");
  const sources = proj.formatted_sources || proj.sources || [];
  if (!sources.length) {
    container.innerHTML = `
      <div class="empty" style="padding:32px 24px">
        <p class="empty-title">No evidence to compare yet</p>
        <p class="empty-body">Add papers from the Sources panel, or use Research in the assistant. Each paper becomes a row here with its design, findings and the arrows it supports.</p>
      </div>`;
    return;
  }
  const sortBtn = (key, label) => {
    const on = state.matrixSort.key === key;
    const dirLabel = on ? (state.matrixSort.dir === 1 ? "ascending" : "descending") : "none";
    return `<button type="button" data-sort="${key}" aria-label="Sort by ${label}">${label}${icon("sort")}</button><span class="sr-only">sorted ${dirLabel}</span>`;
  };
  const ariaSort = (key) => (state.matrixSort.key === key ? (state.matrixSort.dir === 1 ? "ascending" : "descending") : "none");
  container.innerHTML = `
    <table class="ev-table">
      <thead>
        <tr>
          <th scope="col" aria-sort="${ariaSort("title")}">${sortBtn("title", "Study")}</th>
          <th scope="col" aria-sort="${ariaSort("strength")}">${sortBtn("strength", "Design")}</th>
          <th scope="col">Finding</th>
          <th scope="col">Controls for</th>
          <th scope="col">Limitations</th>
          <th scope="col">Supports</th>
          <th scope="col" aria-sort="${ariaSort("year")}">${sortBtn("year", "Year")}</th>
        </tr>
      </thead>
      <tbody>
        ${matrixRows()
          .map(({ src, edges }) => {
            const s = src.ai_summary || {};
            const controls = s.confounders_controlled || [];
            return `
            <tr>
              <td class="col-title">
                <div class="ev-title">${srcTitleHtml(src)}</div>
                <div class="ev-sub">${escapeHtml(src.in_text || "")}${src.journal ? `, ${escapeHtml(src.journal)}` : ""}${src.source_engine ? `, via ${escapeHtml(src.source_engine)}` : ""}</div>
                ${srcLinksHtml(src)}
              </td>
              <td class="col-design">${evidenceHtml(src, false)}<div class="ev-sub">${escapeHtml(src.sample_size || "Sample not reported")}</div></td>
              <td class="col-sum">${escapeHtml(s.tldr || src.key_finding || "") || `<span class="ev-muted">Not summarised</span>`}</td>
              <td>${controls.length ? escapeHtml(controls.join(", ")) : `<span class="ev-muted">Not extracted</span>`}</td>
              <td>${s.limitations_and_bias ? `<span class="ev-limit">${escapeHtml(s.limitations_and_bias)}</span>` : `<span class="ev-muted">None noted</span>`}</td>
              <td>${edges.length ? edges.map((l) => `<span class="ev-edge">${escapeHtml(l)}</span>`).join("") : `<span class="ev-muted">Not linked</span>`}</td>
              <td style="font-variant-numeric:tabular-nums">${escapeHtml(src.year || "n.d.")}</td>
            </tr>`;
          })
          .join("")}
      </tbody>
    </table>`;

  container.querySelectorAll("[data-sort]").forEach((b) => {
    b.addEventListener("click", () => {
      const key = b.dataset.sort;
      state.matrixSort = { key, dir: state.matrixSort.key === key ? -state.matrixSort.dir : key === "title" ? 1 : -1 };
      renderEvidenceMatrix();
      container.querySelector(`[data-sort="${key}"]`)?.focus();
    });
  });
}

function matrixAsMarkdown() {
  const cell = (s) => String(s || "").replace(/\|/g, "\\|").replace(/\n/g, " ");
  const head = "| Study | Design | Sample | Finding | Controls for | Limitations | Supports |\n|---|---|---|---|---|---|---|";
  const body = matrixRows()
    .map(({ src, edges }) => {
      const s = src.ai_summary || {};
      return `| ${cell(src.title)} ${cell(src.in_text)} | ${cell(src.study_design)} | ${cell(src.sample_size)} | ${cell(s.tldr || src.key_finding)} | ${cell((s.confounders_controlled || []).join(", "))} | ${cell(s.limitations_and_bias)} | ${cell(edges.join("; "))} |`;
    })
    .join("\n");
  return `${head}\n${body}`;
}

/** Split the briefing into speaker turns: lines like "**Dr. Aria Vance (Lead PI):** text". */
function parseBriefing(md) {
  const turns = [];
  const speakers = [];
  for (const block of md.split(/\n{2,}/)) {
    const m = block.trim().match(/^\*\*([^*]+?):\*\*\s*([\s\S]*)$/);
    if (m) {
      const full = stripEmoji(m[1]);
      const nm = full.match(/^(.*?)\s*\((.*)\)\s*$/);
      const name = nm ? nm[1] : full;
      const role = nm ? nm[2] : "";
      if (!speakers.includes(name)) speakers.push(name);
      turns.push({ kind: "turn", name, role, host: speakers.indexOf(name) % 2, text: m[2].trim() });
    } else if (block.trim()) {
      turns.push({ kind: "md", text: block.trim() });
    }
  }
  return turns;
}

function initials(name) {
  const parts = name.replace(/^(Dr|Prof|Mr|Ms|Mrs)\.?\s+/i, "").split(/\s+/).filter(Boolean);
  return (parts[0]?.[0] || "") + (parts[1]?.[0] || "");
}

function renderLabBriefing() {
  const proj = state.currentProject;
  if (!proj) return;
  const el = $("briefingContent");
  const has = Boolean(proj.lab_briefing_markdown);
  $("btnGenerateBriefing").querySelector("span").textContent = has ? "Regenerate" : "Generate";
  $("btnPlayBriefingTTS").disabled = !has;
  if (!has) {
    el.innerHTML = `
      <div class="briefing-empty">
        <p class="empty-title">Hear the graph argued out loud</p>
        <p class="empty-body">Two hosts, a lead investigator and a methodologist playing Reviewer 2, talk through your current graph: the effect you want, the paths still open, and what the saved papers support. Read it here or listen to it.</p>
        <button id="btnGenerateBriefingInline" class="btn btn-primary" type="button">${icon("sparkle")}Generate briefing</button>
      </div>`;
    $("btnGenerateBriefingInline").onclick = generateBriefing;
    return;
  }
  const turns = parseBriefing(proj.lab_briefing_markdown);
  el.innerHTML = `<div class="transcript">${turns
    .map((t, i) =>
      t.kind === "turn"
        ? `<div class="turn host-${t.host === 0 ? "a" : "b"}" data-turn="${i}">
             <span class="turn-avatar" aria-hidden="true">${escapeHtml(initials(t.name))}</span>
             <div>
               <div class="turn-name">${escapeHtml(t.name)}${t.role ? ` <span class="turn-role">${escapeHtml(t.role)}</span>` : ""}</div>
               <div class="turn-text">${simpleMarkdown(t.text)}</div>
             </div>
           </div>`
        : `<div class="prose" data-turn="${i}">${simpleMarkdown(stripEmoji(t.text))}</div>`
    )
    .join("")}</div>`;
}

async function generateBriefing() {
  if (!state.currentProject) return;
  const btn = $("btnGenerateBriefing");
  const inline = $("btnGenerateBriefingInline");
  setLoading(btn, true);
  setLoading(inline, true);
  stopSpeech();
  try {
    const updated = await api(`/api/projects/${state.currentProject.id}/lab-briefing`, { method: "POST" });
    setProjectState(updated);
    toast("Briefing ready");
  } catch (err) {
    reportError("Could not generate the briefing", err);
  } finally {
    setLoading(btn, false);
    setLoading($("btnGenerateBriefingInline"), false);
  }
}

// Read the briefing aloud turn by turn, with a different voice per host.
function speakBriefing() {
  if (!("speechSynthesis" in window)) {
    toast("This browser cannot read text aloud", { error: true });
    return;
  }
  const md = state.currentProject?.lab_briefing_markdown;
  if (!md) return;
  stopSpeech();
  const turns = parseBriefing(md);
  const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang));
  const voiceFor = [voices[0], voices.find((v) => v !== voices[0] && v.lang === voices[0]?.lang) || voices[1] || voices[0]];
  state.speaking = true;
  $("btnPlayBriefingTTS").classList.add("hidden");
  $("btnStopBriefingTTS").classList.remove("hidden");
  const clean = (s) => s.replace(/[#*`_>$|]/g, "").replace(/\s+/g, " ").trim();
  turns.forEach((t, i) => {
    const u = new SpeechSynthesisUtterance(clean(t.text));
    u.rate = 1.03;
    if (t.kind === "turn" && voiceFor[t.host]) u.voice = voiceFor[t.host];
    u.onstart = () => {
      document.querySelectorAll(".turn.is-speaking").forEach((x) => x.classList.remove("is-speaking"));
      const el = document.querySelector(`#briefingContent [data-turn="${i}"]`);
      if (el && el.classList.contains("turn")) {
        el.classList.add("is-speaking");
        el.scrollIntoView({ block: "nearest", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      }
    };
    if (i === turns.length - 1) u.onend = stopSpeech;
    speechSynthesis.speak(u);
  });
}

function stopSpeech() {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  state.speaking = false;
  $("btnPlayBriefingTTS")?.classList.remove("hidden");
  $("btnStopBriefingTTS")?.classList.add("hidden");
  document.querySelectorAll(".turn.is-speaking").forEach((x) => x.classList.remove("is-speaking"));
}

const ACTOR_LABELS = { researcher: "You", ai_accepted: "Accepted from assistant", system: "System" };

function renderAuditLog() {
  const proj = state.currentProject;
  if (!proj) return;
  const list = $("auditLogList");
  const entries = (proj.audit_log || []).slice().sort((a, b) => String(b.timestamp).localeCompare(String(a.timestamp)));
  if (!entries.length) {
    list.innerHTML = `<li class="empty"><p class="empty-body">No changes recorded yet.</p></li>`;
    return;
  }
  let lastDay = "";
  list.innerHTML = entries
    .map((e) => {
      const day = fmtDay(e.timestamp);
      const head = day !== lastDay ? `<li class="tl-day">${escapeHtml(day)}</li>` : "";
      lastDay = day;
      return `${head}
        <li class="tl-item actor-${escapeHtml(e.actor || "researcher")}">
          <span class="tl-dot" aria-hidden="true"></span>
          <div>
            <div class="tl-action">${escapeHtml(stripEmoji(e.action))}</div>
            <div class="tl-detail">${escapeHtml(prettyPath(e.details))}</div>
            <div class="tl-actor">${escapeHtml(ACTOR_LABELS[e.actor] || e.actor || "")}</div>
          </div>
          <time class="tl-time" datetime="${escapeHtml(e.timestamp)}">${escapeHtml(fmtTime(e.timestamp))}</time>
        </li>`;
    })
    .join("");
}

// ============================================================================
// ASSISTANT: PROPOSALS, CHAT, RESEARCH
// ============================================================================

function diffHtml(v) {
  const d = v && v.diff;
  if (!d) return "";
  const sets = (x) => (x.sets && x.sets.length ? x.sets.map((ss) => `{${ss.join(", ") || "∅"}}`).join(" or ") : "none");
  const lines = [];
  lines.push(`<span class="${d.identifiability_changes ? "changed" : ""}">Identifiable by adjustment: ${d.before.identifiable ? "yes" : "no"} → ${d.after.identifiable ? "yes" : "no"}</span>`);
  if (d.adjustment_sets_change) lines.push(`<span class="changed">Adjustment sets: ${escapeHtml(sets(d.before))} → ${escapeHtml(sets(d.after))}</span>`);
  if (d.current_set_validity_changes) lines.push(`<span class="changed">Your current adjustment would become ${d.after.current_valid ? "valid" : "invalid"}</span>`);
  if (!d.identifiability_changes && !d.adjustment_sets_change && !d.current_set_validity_changes) lines.push("<span>No change to identification</span>");
  return `<div class="proposal-diff">${lines.join("")}</div>`;
}

function renderProposals() {
  const proj = state.currentProject;
  if (!proj) return;
  const open = (proj.proposals || []).filter((p) => p.status === "pending" || p.status === "invalid");
  const staged = proj.staged_sources || [];
  const container = $("proposalsContainer");
  const list = $("proposalsList");
  $("pendingProposalCount").textContent = open.length + staged.length;
  if (!open.length && !staged.length) {
    container.classList.add("hidden");
    list.innerHTML = "";
    return;
  }
  container.classList.remove("hidden");
  const sourceTitle = (sid) => (staged.find((x) => x.id === sid) || (proj.sources || []).find((x) => x.id === sid) || {}).title;
  list.innerHTML =
    open
      .map((p) => {
        const v = p.validation || {};
        const invalid = p.status === "invalid";
        const cites = p.edge && p.edge.citation_ids && p.edge.citation_ids.length
          ? `<div class="proposal-diff"><span>Evidence: ${p.edge.citation_ids.map((sid) => escapeHtml(sourceTitle(sid) || sid)).join("; ")}${p.edge.citation_ids.some((sid) => staged.some((x) => x.id === sid)) ? " (staged; enters the library if you accept)" : ""}</span></div>`
          : "";
        return `
        <article class="proposal ${invalid ? "is-invalid" : ""}">
          <div class="proposal-kind">${escapeHtml(PROPOSAL_KINDS[p.action] || "Proposed change")}${p.edge ? ", AI hypothesis" : ""}</div>
          <div class="proposal-title">${escapeHtml(stripEmoji(p.title))}</div>
          ${p.rationale ? `<p class="proposal-rationale">${escapeHtml(p.rationale)}</p>` : ""}
          ${p.critic_warning ? `<div class="proposal-warning">${icon("alert")}<span>${escapeHtml(stripEmoji(p.critic_warning))}</span></div>` : ""}
          ${invalid ? `<div class="proposal-errors"><strong>Cannot be applied:</strong> ${escapeHtml((v.errors || []).join(" "))}</div>` : diffHtml(v)}
          ${(v.warnings || []).length ? `<div class="proposal-diff"><span class="changed">${escapeHtml(v.warnings.join(" "))}</span></div>` : ""}
          ${cites}
          <div class="proposal-actions">
            <button class="btn btn-ghost btn-xs btn-prop-decide" data-id="${escapeHtml(p.id)}" data-dec="reject" type="button">${invalid ? "Dismiss" : "Reject"}</button>
            ${invalid ? "" : `<button class="btn btn-primary btn-xs btn-prop-decide" data-id="${escapeHtml(p.id)}" data-dec="accept" type="button">${icon("check")}Accept</button>`}
          </div>
        </article>`;
      })
      .join("") +
    staged
      .map(
        (src) => `
        <article class="proposal staged-src">
          <div class="proposal-kind">Source found by research, not yet in your library</div>
          <div class="proposal-title">${escapeHtml(src.title)}</div>
          <p class="proposal-rationale">${srcMetaHtml(src)}</p>
          ${evidenceHtml(src)}
          ${flagsHtml(src)}
          ${(src.ai_summary && src.ai_summary.tldr) ? `<p class="proposal-rationale">AI summary, unverified: ${escapeHtml(src.ai_summary.tldr)}</p>` : ""}
          <div class="proposal-actions">
            <button class="btn btn-ghost btn-xs btn-staged" data-id="${escapeHtml(src.id)}" data-dec="reject" type="button">Discard</button>
            <button class="btn btn-secondary btn-xs btn-staged" data-id="${escapeHtml(src.id)}" data-dec="accept" type="button">${icon("plus")}Add to library</button>
          </div>
        </article>`
      )
      .join("");

  list.querySelectorAll(".btn-prop-decide").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const prop = open.find((x) => x.id === btn.dataset.id);
      setLoading(btn, true);
      try {
        const updated = await api(`/api/projects/${proj.id}/proposals/${btn.dataset.id}/decide`, {
          method: "POST",
          body: JSON.stringify({ decision: btn.dataset.dec }),
        });
        setProjectState(updated);
        toast(`${btn.dataset.dec === "accept" ? "Accepted" : "Rejected"}: ${prop ? stripEmoji(prop.title) : "proposal"}`);
      } catch (err) {
        reportError("Not applied", err);
        loadProject(proj.id).catch(() => {});
      }
    });
  });
  list.querySelectorAll(".btn-staged").forEach((btn) => {
    btn.addEventListener("click", async () => {
      setLoading(btn, true);
      try {
        const updated = await api(`/api/projects/${proj.id}/staged/${btn.dataset.id}/decide`, {
          method: "POST",
          body: JSON.stringify({ decision: btn.dataset.dec }),
        });
        setProjectState(updated);
        toast(btn.dataset.dec === "accept" ? "Added to library" : "Source discarded");
      } catch (err) {
        setLoading(btn, false);
        reportError("Could not record the decision", err);
      }
    });
  });
}

function setPersona(mode) {
  if (!PERSONAS[mode]) mode = "socratic";
  state.agentMode = mode;
  document.querySelectorAll(".persona-btn").forEach((b) => {
    const on = b.dataset.mode === mode;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", String(on));
  });
  $("chatModeHint").textContent = PERSONAS[mode].hint;
  const history = state.currentProject?.chat_history || [];
  if (!history.length) renderChat(true);
}

function renderChat(force = false) {
  const proj = state.currentProject;
  const stream = $("chatStream");
  const chipsBar = $("socraticChipsBar");
  const history = proj ? proj.chat_history || [] : [];
  if (!force && history.length === state.lastChatLen && !state.sending) return;
  const grew = history.length !== state.lastChatLen;
  state.lastChatLen = history.length;

  if (!history.length) {
    const p = PERSONAS[state.agentMode];
    stream.innerHTML = `
      <div class="chat-empty">
        <p class="empty-title">Think the graph through with the assistant</p>
        <p class="empty-body">It reads the current graph and the identification checks. Any change it suggests waits in a review queue until you accept it.</p>
      </div>`;
    chipsBar.innerHTML = `<span class="suggestions-label">Try asking</span>` +
      p.starters.map((q) => `<button class="suggestion" type="button">${escapeHtml(q)}</button>`).join("");
  } else {
    stream.innerHTML = history
      .map((msg) => {
        if (msg.role === "user") {
          return `<div class="msg msg-user"><div class="bubble">${escapeHtml(msg.content)}</div></div>`;
        }
        if (msg.role === "system") {
          return `<div class="msg msg-system"><div class="bubble">${escapeHtml(stripEmoji(msg.content))}</div></div>`;
        }
        const props = (msg.proposal_ids || []).length;
        const persona = PERSONAS[msg.agent_mode]?.name || (msg.agent_mode === "deep_research" ? "Research" : "Assistant");
        return `
          <div class="msg msg-assistant">
            <div class="msg-meta"><span class="persona-tag">${escapeHtml(persona)}</span><span>${escapeHtml(fmtTime(msg.created_at))}</span></div>
            <div class="prose">${simpleMarkdown(stripEmoji(msg.content))}</div>
            ${props ? `<div class="msg-props">Proposed ${props} change${props > 1 ? "s" : ""} for review</div>` : ""}
          </div>`;
      })
      .join("");
    const last = history[history.length - 1];
    const qs = last && last.role === "assistant" ? last.socratic_questions || [] : [];
    chipsBar.innerHTML = qs.length
      ? `<span class="suggestions-label">Suggested replies</span>` + qs.map((q) => `<button class="suggestion" type="button">${escapeHtml(stripEmoji(q))}</button>`).join("")
      : "";
  }

  chipsBar.querySelectorAll(".suggestion").forEach((chip) => {
    chip.addEventListener("click", () => {
      const input = $("chatInput");
      input.value = chip.textContent;
      autoGrow(input);
      input.focus();
    });
  });
  if (grew || force) stream.scrollTop = stream.scrollHeight;
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(200, el.scrollHeight) + "px";
}

function showPending(userText, label, extraClass = "") {
  const stream = $("chatStream");
  if (!(state.currentProject.chat_history || []).length) stream.innerHTML = "";
  $("socraticChipsBar").innerHTML = "";
  if (userText) stream.insertAdjacentHTML("beforeend", `<div class="msg msg-user" id="pendingUserMsg"><div class="bubble">${escapeHtml(userText)}</div></div>`);
  stream.insertAdjacentHTML(
    "beforeend",
    `<div class="msg msg-assistant msg-pending ${extraClass}" id="pendingMsg"><div class="msg-meta"><span class="persona-tag">${escapeHtml(label)}</span><span>working</span></div><div class="dots" aria-label="${escapeHtml(label)} is working"><i></i><i></i><i></i></div></div>`
  );
  stream.scrollTop = stream.scrollHeight;
}

function clearPending() {
  $("pendingMsg")?.remove();
  $("pendingUserMsg")?.remove();
}

async function sendChatMessage() {
  const input = $("chatInput");
  const msg = input.value.trim();
  if (!msg || !state.currentProject || state.sending) return;
  state.sending = true;
  input.value = "";
  autoGrow(input);
  const btn = $("btnSendChat");
  btn.disabled = true;
  showPending(msg, PERSONAS[state.agentMode].name);

  try {
    const updated = await api(`/api/projects/${state.currentProject.id}/chat`, {
      method: "POST",
      body: JSON.stringify({ message: msg, agent_mode: state.agentMode }),
    });
    state.sending = false;
    if (updated.last_turn?.fetched_papers?.length) {
      state.liveLitResults = updated.last_turn.fetched_papers;
      renderLiveLitResults();
      if (document.body.dataset.left === "closed") setPanel("left", "open");
    }
    setProjectState(updated);
  } catch (err) {
    state.sending = false;
    clearPending();
    input.value = msg;
    autoGrow(input);
    reportError("The assistant did not respond", err);
  } finally {
    state.sending = false;
    btn.disabled = false;
  }
}

async function runDeepResearch() {
  if (!state.currentProject || state.sending) return;
  const input = $("chatInput");
  const focus = input.value.trim();
  const btn = $("btnDeepResearch");
  state.sending = true;
  setLoading(btn, true);
  $("btnSendChat").disabled = true;
  showPending(focus ? `Research: ${focus}` : "", "Research", "msg-research");
  try {
    const updated = await api(`/api/projects/${state.currentProject.id}/deep-research`, {
      method: "POST",
      body: JSON.stringify({ focus }),
    });
    state.sending = false;
    input.value = "";
    autoGrow(input);
    setProjectState(updated);
    const added = (updated.staged_sources || []).length;
    const pending = (updated.proposals || []).filter((p) => p.status === "pending").length;
    toast(`Research finished: ${added > 0 ? `${added} source${added > 1 ? "s" : ""} staged for review` : "no new sources"}${pending ? `, ${pending} proposal${pending > 1 ? "s" : ""}` : ""}. Nothing changes until you accept.`);
  } catch (err) {
    state.sending = false;
    clearPending();
    reportError("Research did not finish", err);
  } finally {
    state.sending = false;
    setLoading(btn, false);
    $("btnSendChat").disabled = false;
  }
}

// ============================================================================
// PANELS, THEME, VIEWS
// ============================================================================

function setPanel(side, value) {
  const key = side === "left" ? "left" : "right";
  document.body.dataset[key] = value;
  $(side === "left" ? "btnToggleLeft" : "btnToggleRight").setAttribute("aria-pressed", String(value === "open"));
  store(`dagbrain.${key}`, value);
  setTimeout(applyView, 240);
}

function initPanels() {
  const narrow = window.innerWidth < 1100;
  setPanel("left", store("dagbrain.left") || (narrow ? "closed" : "open"));
  setPanel("right", store("dagbrain.right") || "open");
}

function toggleTheme() {
  const root = document.documentElement;
  const current = root.dataset.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  root.dataset.theme = next;
  store("dagbrain.theme", next);
}

function setViewMode(mode) {
  if (!VIEW_MODES.includes(mode)) mode = "canvas";
  document.querySelectorAll(".mode-tab").forEach((t) => {
    const on = t.dataset.cmode === mode;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", String(on));
  });
  $("centerWorkspace").className = `center-workspace mode-${mode}`;
  $("dagCanvasPanel").classList.toggle("hidden", mode !== "canvas" && mode !== "split");
  $("notebookPanel").classList.toggle("hidden", mode !== "notebook" && mode !== "split");
  $("matrixPanel").classList.toggle("hidden", mode !== "matrix");
  $("dataPanel").classList.toggle("hidden", mode !== "data");
  $("briefingPanel").classList.toggle("hidden", mode !== "briefing");
  $("auditPanel").classList.toggle("hidden", mode !== "audit");
  document.body.classList.toggle("view-not-canvas", mode !== "canvas" && mode !== "split");
  store("dagbrain.viewmode", mode);
  if (mode === "canvas" || mode === "split") requestAnimationFrame(applyView);
}

function openDialog(id) {
  const d = $(id);
  if (!d.open) d.showModal();
}

function closeDialog(id) {
  const d = $(id);
  if (d.open) d.close();
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand("copy");
    ta.remove();
  }
}

// ============================================================================
// EVENT BINDINGS
// ============================================================================

function bindUIEvents() {
  // Study
  $("projectSelect").addEventListener("change", (ev) => loadProject(ev.target.value).catch((e) => reportError("Could not open the study", e)));

  // Estimand segmented control
  document.querySelectorAll("[data-estimand]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!state.currentProject || btn.getAttribute("aria-checked") === "true") return;
      state.currentProject.estimand_type = btn.dataset.estimand;
      await saveCurrentProject("Changed estimand", `Target estimand set to ${btn.dataset.estimand}`).catch(() => {});
    });
  });

  // Citation style
  $("citationStyleSelect").addEventListener("change", async (ev) => {
    if (!state.currentProject) return;
    state.currentProject.citation_style = ev.target.value;
    await saveCurrentProject("Changed citation style", `Reformatted citations to ${ev.target.value.toUpperCase()}`).catch(() => {});
  });

  // Paths drawer
  $("btnTogglePathsDrawer").addEventListener("click", () => {
    const drawer = $("pathsDrawer");
    drawer.classList.toggle("hidden");
    $("btnTogglePathsDrawer").setAttribute("aria-expanded", String(!drawer.classList.contains("hidden")));
    requestAnimationFrame(applyView);
  });

  // Sources tabs
  document.querySelectorAll(".vault-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".vault-tab").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", String(t === tab));
      });
      $("vaultSourcesView").classList.toggle("hidden", tab.dataset.vtab !== "sources");
      $("vaultBibliographyView").classList.toggle("hidden", tab.dataset.vtab !== "bibliography");
    });
  });

  // External search portals
  $("btnOpenScholarPortal").onclick = () => window.open(`https://scholar.google.com/scholar?q=${encodeURIComponent(currentSearchQuery())}`, "_blank", "noopener");
  $("btnOpenCorePortal").onclick = () => window.open(`https://core.ac.uk/search?q=${encodeURIComponent(currentSearchQuery())}`, "_blank", "noopener");
  $("btnOpenEpmcPortal").onclick = () => window.open(`https://europepmc.org/search?query=${encodeURIComponent(currentSearchQuery())}`, "_blank", "noopener");

  // PDF upload (stays on this machine)
  $("btnUploadPdf").onclick = () => $("pdfFileInput").click();
  $("pdfFileInput").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if (!file || !state.currentProject) return;
    const btn = $("btnUploadPdf");
    setLoading(btn, true);
    try {
      const updated = await upload(`/api/projects/${state.currentProject.id}/sources/upload`, file);
      setProjectState(updated);
      const added = updated.sources[updated.sources.length - 1];
      toast(added && added.injection_flags && added.injection_flags.length ? "PDF added. It contains instruction-like text, which is treated as data only." : "PDF added to the library");
    } catch (err) {
      reportError("PDF not added", err);
    } finally {
      setLoading(btn, false);
    }
  });

  // Dataset upload
  $("btnUploadCsv").onclick = () => $("csvFileInput").click();
  $("csvFileInput").addEventListener("change", async (ev) => {
    const file = ev.target.files && ev.target.files[0];
    ev.target.value = "";
    if (!file || !state.currentProject) return;
    const btn = $("btnUploadCsv");
    setLoading(btn, true);
    try {
      const updated = await upload(`/api/projects/${state.currentProject.id}/datasets`, file);
      state.dataResults = { implications: null, estimate: null };
      setProjectState(updated);
      const ds = updated.datasets[0];
      toast(`${ds.filename}: ${ds.n_rows} rows, ${Object.keys(ds.mapping).length} variables matched automatically`);
    } catch (err) {
      reportError("Dataset not loaded", err);
    } finally {
      setLoading(btn, false);
    }
  });

  // Causal question
  $("qTemplate").addEventListener("change", () => applyQuestionLabels($("qTemplate").value));
  $("btnSaveQuestion").onclick = saveQuestion;

  // Manual source form
  const drawer = $("manualSourceDrawer");
  $("btnAddManualSource").onclick = () => {
    drawer.classList.toggle("hidden");
    const open = !drawer.classList.contains("hidden");
    $("btnAddManualSource").setAttribute("aria-expanded", String(open));
    if (open) $("srcTitle").focus();
  };
  $("btnCancelManualSource").onclick = () => {
    drawer.classList.add("hidden");
    $("btnAddManualSource").setAttribute("aria-expanded", "false");
  };
  $("btnSaveManualSource").onclick = async () => {
    const titleEl = $("srcTitle");
    const title = titleEl.value.trim();
    if (!title) {
      titleEl.setAttribute("aria-invalid", "true");
      titleEl.focus();
      toast("Add a title to save the source", { error: true });
      return;
    }
    titleEl.removeAttribute("aria-invalid");
    if (!state.currentProject) return;
    const btn = $("btnSaveManualSource");
    const authors = $("srcAuthors").value.split(",").map((s) => s.trim()).filter(Boolean);
    const doi = $("srcDoi").value.trim();
    const finding = $("srcFinding").value.trim();
    const id = "src_" + Math.random().toString(36).slice(2, 9);
    state.currentProject.sources.push({
      id,
      title,
      authors: authors.length ? authors : ["Author"],
      year: parseInt($("srcYear").value, 10) || new Date().getFullYear(),
      journal: $("srcJournal").value.trim(),
      doi,
      url: doi ? `https://doi.org/${doi}` : "",
      source_engine: "Manual / DOI",
      abstract: finding,
      study_design: $("srcDesign").value,
      sample_size: $("srcSampleSize").value.trim(),
      key_finding: finding,
      stance: "supporting",
    });
    setLoading(btn, true);
    try {
      await saveCurrentProject("Added manual citation", `Added '${title}' to Citation Vault`);
      drawer.classList.add("hidden");
      drawer.querySelectorAll("input, textarea").forEach((el) => (el.value = ""));
      toast("Source added, summarising…");
      if (finding) {
        try {
          const updated = await api(`/api/projects/${state.currentProject.id}/sources/${id}/summarize`, { method: "POST" });
          setProjectState(updated);
        } catch (e) { /* the source is saved; the summary can be generated later */ }
      }
    } catch (e) {
      state.currentProject.sources.pop();
    } finally {
      setLoading(btn, false);
    }
  };

  // Literature search
  $("btnLitSearch").onclick = runLitSearch;
  $("litSearchInput").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") runLitSearch();
    if (ev.key === "Escape") ev.target.blur();
  });
  $("btnClearLitSearch").onclick = () => {
    state.liveLitResults = null;
    renderLiveLitResults();
  };

  // Workspace views
  document.querySelectorAll(".mode-tab").forEach((tab) => tab.addEventListener("click", () => setViewMode(tab.dataset.cmode)));

  // Canvas tools
  $("btnAddNodeQuick").onclick = addNode;
  $("btnEmptyAddNode").onclick = addNode;
  $("btnConnectMode").onclick = () => setConnectMode(!state.connectMode);
  $("btnAutoLayout").onclick = autoLayout;
  $("btnResetZoom").onclick = fitView;
  $("btnZoomIn").onclick = () => zoomAt(1 / 1.25);
  $("btnZoomOut").onclick = () => zoomAt(1.25);
  $("btnZoomLabel").onclick = resetZoom;

  const svg = $("dagSvg");
  svg.addEventListener("pointerdown", onCanvasPointerDown);
  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  svg.addEventListener(
    "wheel",
    (ev) => {
      ev.preventDefault();
      const pt = getSvgPoint(ev);
      const delta = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
      zoomAt(Math.exp(delta * 0.0015), pt.x, pt.y);
    },
    { passive: false }
  );
  new ResizeObserver(() => applyView()).observe(svg);

  // Notebook
  const editor = $("notebookEditor");
  editor.addEventListener("input", () => {
    renderNotebookPreview();
    setSaveStatus("Unsaved changes");
    clearTimeout(notebookTimer);
    notebookTimer = setTimeout(() => saveNotebook(false), 1500);
  });
  $("btnToggleNotebookPreview").onclick = () => {
    const body = $("notebookBody");
    body.classList.toggle("split-view");
    const on = body.classList.contains("split-view");
    $("notebookPreview").classList.toggle("hidden", !on);
    $("btnToggleNotebookPreview").setAttribute("aria-pressed", String(on));
  };
  $("btnSaveNotebook").onclick = () => saveNotebook(true);

  // Evidence matrix
  $("btnCopyMatrix").onclick = async () => {
    if (!state.currentProject) return;
    await copyText(matrixAsMarkdown());
    toast("Evidence matrix copied as a Markdown table");
  };

  // Briefing
  $("btnGenerateBriefing").onclick = generateBriefing;
  $("btnPlayBriefingTTS").onclick = speakBriefing;
  $("btnStopBriefingTTS").onclick = stopSpeech;
  if ("speechSynthesis" in window) speechSynthesis.getVoices(); // warm the voice list

  // Assistant
  document.querySelectorAll(".persona-btn").forEach((btn) => btn.addEventListener("click", () => setPersona(btn.dataset.mode)));
  $("btnSendChat").onclick = sendChatMessage;
  $("btnDeepResearch").onclick = runDeepResearch;
  const chatInput = $("chatInput");
  chatInput.addEventListener("input", () => autoGrow(chatInput));
  chatInput.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey && !ev.isComposing) {
      ev.preventDefault();
      sendChatMessage();
    }
  });

  // Panels, theme, mobile
  $("btnToggleLeft").onclick = () => setPanel("left", document.body.dataset.left === "open" ? "closed" : "open");
  $("btnToggleRight").onclick = () => setPanel("right", document.body.dataset.right === "open" ? "closed" : "open");
  $("btnTheme").onclick = toggleTheme;
  document.querySelectorAll(".mobile-tabs button").forEach((b) => {
    b.addEventListener("click", () => {
      document.body.dataset.mview = b.dataset.mview;
      document.querySelectorAll(".mobile-tabs button").forEach((x) => x.setAttribute("aria-current", String(x === b)));
      if (b.dataset.mview === "graph") requestAnimationFrame(applyView);
    });
  });

  // Dialogs: close buttons and backdrop clicks
  document.querySelectorAll("[data-close]").forEach((btn) => btn.addEventListener("click", () => closeDialog(btn.dataset.close)));
  document.querySelectorAll("dialog").forEach((d) => {
    d.addEventListener("click", (ev) => { if (ev.target === d) d.close(); });
  });

  // Settings
  const KEY_FIELDS = [
    ["settingOpenrouterKey", "openrouter_api_key", "badgeOpenrouterKey"],
    ["settingGeminiKey", "gemini_api_key", "badgeGeminiKey"],
    ["settingOpenaiKey", "openai_api_key", "badgeOpenaiKey"],
    ["settingAnthropicKey", "anthropic_api_key", "badgeAnthropicKey"],
    ["settingTavilyKey", "tavily_api_key", "badgeTavilyKey"],
    ["settingCoreKey", "core_api_key", "badgeCoreKey"],
    ["settingSerpapiKey", "serpapi_api_key", "badgeSerpapiKey"],
  ];
  const TEXT_FIELDS = [
    ["settingOllamaUrl", "ollama_base_url", "http://localhost:11434"],
    ["settingScholarlyEmail", "scholarly_email", ""],
    ["settingEzproxy", "ezproxy_prefix", ""],
    ["settingSearxngUrl", "searxng_base_url", ""],
  ];
  $("btnSettingsModal").onclick = async () => {
    await loadSettings();
    const s = state.settings || {};
    $("settingProvider").value = s.active_provider || "openrouter";
    $("settingModelCustom").value = s.active_model || "";
    KEY_FIELDS.forEach(([id, key, badge]) => {
      $(id).value = s[key] || "";
      $(badge).textContent = s[`${key}_configured`] ? "Saved" : "";
    });
    TEXT_FIELDS.forEach(([id, key, dflt]) => ($(id).value = s[key] || dflt));
    $("settingOffline").checked = Boolean(s.offline_mode);
    $("ollamaDetectedCount").textContent = (s.detected_ollama_models || []).length;
    populateModelPresets();
    openDialog("settingsModal");
  };
  $("settingProvider").addEventListener("change", () => {
    $("settingModelCustom").value = "";
    populateModelPresets();
  });
  $("settingModelPreset").addEventListener("change", (ev) => ($("settingModelCustom").value = ev.target.value));
  $("btnSaveSettings").onclick = async () => {
    const btn = $("btnSaveSettings");
    const updates = {
      active_provider: $("settingProvider").value,
      active_model: $("settingModelCustom").value.trim(),
    };
    KEY_FIELDS.forEach(([id, key]) => (updates[key] = $(id).value.trim()));
    TEXT_FIELDS.forEach(([id, key]) => (updates[key] = $(id).value.trim()));
    updates.offline_mode = $("settingOffline").checked;
    setLoading(btn, true);
    try {
      state.settings = await api("/api/settings", { method: "POST", body: JSON.stringify(updates) });
      renderProviderPill();
      applyOffline(Boolean(state.settings.offline_mode));
      closeDialog("settingsModal");
      toast(`Settings saved. Using ${PROVIDER_LABELS[state.settings.active_provider] || state.settings.active_provider}`);
    } catch (err) {
      reportError("Settings were not saved", err);
    } finally {
      setLoading(btn, false);
    }
  };

  // Export / import
  $("btnExportModal").onclick = async () => {
    openDialog("exportModal");
    await loadExportFormat(state.activeExportFormat);
  };
  document.querySelectorAll(".export-tab").forEach((tab) => {
    tab.addEventListener("click", async () => {
      document.querySelectorAll(".export-tab").forEach((t) => {
        t.classList.toggle("active", t === tab);
        t.setAttribute("aria-selected", String(t === tab));
      });
      state.activeExportFormat = tab.dataset.efmt;
      await loadExportFormat(state.activeExportFormat);
    });
  });
  $("btnCopyExport").onclick = async () => {
    await copyText($("exportCodeBox").value);
    const label = $("btnCopyExport").querySelector("span");
    label.textContent = "Copied";
    setTimeout(() => (label.textContent = "Copy"), 1500);
  };
  $("btnDownloadExport").onclick = () => {
    const blob = new Blob([$("exportCodeBox").value], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${slugify(state.currentProject?.title)}.${EXPORT_EXT[state.activeExportFormat] || "txt"}`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  $("btnImportDagitty").onclick = async () => {
    if (!state.currentProject) return;
    const code = $("exportCodeBox").value.trim();
    if (!code) { toast("Paste DAGitty code into the editor first", { error: true }); return; }
    const snap = graphSnapshot();
    const btn = $("btnImportDagitty");
    setLoading(btn, true);
    try {
      const updated = await api(`/api/projects/${state.currentProject.id}/import-dagitty`, {
        method: "POST",
        body: JSON.stringify({ dagitty_code: code }),
      });
      setProjectState(updated);
      closeDialog("exportModal");
      requestAnimationFrame(fitView);
      offerUndo("DAGitty graph imported", snap, "Restored graph from before the import");
    } catch (err) {
      reportError("Import failed", err);
    } finally {
      setLoading(btn, false);
    }
  };

  // Worldviews
  $("btnWorldviews").onclick = () => {
    renderWorldviewsModal();
    openDialog("worldviewsModal");
  };
  $("btnSaveWorldview").onclick = async () => {
    if (!state.currentProject) return;
    const name = $("newWorldviewName").value.trim() || `Snapshot ${(state.currentProject.worldviews || []).length + 1}`;
    const btn = $("btnSaveWorldview");
    setLoading(btn, true);
    try {
      const updated = await api(`/api/projects/${state.currentProject.id}/worldviews`, {
        method: "POST",
        body: JSON.stringify({ action: "create", name }),
      });
      $("newWorldviewName").value = "";
      setProjectState(updated);
      renderWorldviewsModal();
      toast(`Saved “${name}”`);
    } catch (err) {
      reportError("Snapshot not saved", err);
    } finally {
      setLoading(btn, false);
    }
  };
  $("newWorldviewName").addEventListener("keydown", (ev) => { if (ev.key === "Enter") $("btnSaveWorldview").click(); });

  // New study
  $("btnNewProject").onclick = () => {
    openDialog("newProjectModal");
    $("npTitle").focus();
  };
  $("btnCreateProjectConfirm").onclick = async () => {
    const btn = $("btnCreateProjectConfirm");
    setLoading(btn, true);
    try {
      const created = await api("/api/projects", {
        method: "POST",
        body: JSON.stringify({
          title: $("npTitle").value.trim() || "New causal study",
          discipline: $("npDiscipline").value,
          research_question: $("npQuestion").value.trim(),
          exposure_label: $("npExposure").value.trim() || "Exposure (X)",
          outcome_label: $("npOutcome").value.trim() || "Outcome (Y)",
        }),
      });
      closeDialog("newProjectModal");
      document.querySelectorAll("#newProjectModal input, #newProjectModal textarea").forEach((el) => (el.value = ""));
      await loadProjectsList(created.id);
      setPersona("topic_refiner");
      toast("Study created. Topic mode can help you narrow the question.");
    } catch (err) {
      reportError("Study not created", err);
    } finally {
      setLoading(btn, false);
    }
  };

  // Restore last view mode
  const vm = store("dagbrain.viewmode");
  if (vm && VIEW_MODES.includes(vm)) setViewMode(vm);

  // Keyboard shortcuts
  let nudgeTimer = null;
  document.addEventListener("keydown", async (ev) => {
    if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "s") {
      ev.preventDefault();
      if (document.activeElement === editor || !$("notebookPanel").classList.contains("hidden")) await saveNotebook(true);
      else await saveCurrentProject().catch(() => {});
      return;
    }
    if (document.querySelector("dialog[open]")) return;
    if (isTyping(ev.target)) {
      if (ev.key === "Escape") ev.target.blur();
      return;
    }
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const k = ev.key;
    const onCanvas = !$("dagCanvasPanel").classList.contains("hidden");

    if (k === "Escape") {
      if (state.speaking) stopSpeech();
      else if (state.connectMode) cancelConnect();
      else clearSelection();
      return;
    }
    if (k === "/") { ev.preventDefault(); if (document.body.dataset.left === "closed") setPanel("left", "open"); $("litSearchInput").focus(); return; }
    if (k === "[") { $("btnToggleLeft").click(); return; }
    if (k === "]") { $("btnToggleRight").click(); return; }
    if (/^[1-7]$/.test(k)) { setViewMode(VIEW_MODES[Number(k) - 1]); return; }
    if (!onCanvas) return;
    if (k === "n" || k === "N") { ev.preventDefault(); addNode(); }
    else if (k === "a" || k === "A") setConnectMode(!state.connectMode);
    else if (k === "l" || k === "L") autoLayout();
    else if (k === "f" || k === "F") fitView();
    else if (k === "+" || k === "=") zoomAt(1 / 1.25);
    else if (k === "-" || k === "_") zoomAt(1.25);
    else if (k === "0") resetZoom();
    else if (k === "Delete" || k === "Backspace") { ev.preventDefault(); deleteSelection(); }
    else if (k.startsWith("Arrow") && state.selectedNodeId) {
      // Keyboard alternative to dragging.
      ev.preventDefault();
      const node = state.currentProject.nodes.find((n) => n.id === state.selectedNodeId);
      if (!node) return;
      const step = ev.shiftKey ? 40 : 10;
      if (k === "ArrowLeft") node.x -= step;
      if (k === "ArrowRight") node.x += step;
      if (k === "ArrowUp") node.y -= step;
      if (k === "ArrowDown") node.y += step;
      state.focusNodeAfterRender = document.activeElement?.classList?.contains("dag-node");
      renderCanvas();
      clearTimeout(nudgeTimer);
      nudgeTimer = setTimeout(() => saveCurrentProject().catch(() => {}), 600);
    }
  });
}

function populateModelPresets() {
  const s = state.settings || {};
  const prov = $("settingProvider").value;
  const presets = s.provider_presets?.[prov]?.suggested_models || [];
  const ollamaModels = prov === "ollama" ? s.detected_ollama_models || [] : [];
  const combined = [...new Set([...ollamaModels, ...presets])];
  const sel = $("settingModelPreset");
  sel.innerHTML = combined.length ? "" : `<option value="">No suggestions for this provider</option>`;
  combined.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    sel.appendChild(opt);
  });
  const custom = $("settingModelCustom");
  if (combined.includes(custom.value)) sel.value = custom.value;
  else if (combined.length && !custom.value) custom.value = combined[0];
}

async function loadExportFormat(fmt) {
  if (!state.currentProject) return;
  const box = $("exportCodeBox");
  box.value = "Generating…";
  try {
    const res = await api(`/api/projects/${state.currentProject.id}/export?fmt=${fmt}`);
    box.value = res.content || "";
  } catch (err) {
    box.value = "";
    reportError("Export failed", err);
  }
}

function setsText(diag) {
  const sets = diag.minimal_adjustment_sets || [];
  if (!sets.length) return `<span class="muted">None available</span>`;
  return sets
    .map((s) => `<span class="set">{ ${s.length ? s.map((x) => escapeHtml(x.label)).join(", ") : "∅"} }</span>`)
    .join("");
}

function renderWorldviewsModal() {
  const proj = state.currentProject;
  if (!proj) return;
  const list = $("worldviewsList");
  const compBox = $("worldviewComparisonBox");
  compBox.classList.add("hidden");
  const wvs = proj.worldviews || [];
  if (!wvs.length) {
    list.innerHTML = `<div class="empty"><p class="empty-body">No snapshots yet. Save the current graph above, change an assumption, then compare the two.</p></div>`;
    return;
  }
  list.innerHTML = wvs
    .map(
      (wv) => `
      <div class="wv-row" data-id="${escapeHtml(wv.id)}">
        <div>
          <div class="wv-name">${escapeHtml(wv.name)}</div>
          <div class="wv-meta">${(wv.nodes || []).length} variables, ${(wv.edges || []).length} arrows${wv.created_at ? `, saved ${escapeHtml(fmtDay(wv.created_at))} ${escapeHtml(fmtTime(wv.created_at))}` : ""}</div>
        </div>
        <div class="wv-actions">
          <button class="btn btn-ghost btn-sm btn-wv-compare" data-id="${escapeHtml(wv.id)}" type="button">Compare</button>
          <button class="btn btn-secondary btn-sm btn-wv-restore" data-id="${escapeHtml(wv.id)}" type="button">Load</button>
        </div>
      </div>`
    )
    .join("");

  list.querySelectorAll(".btn-wv-restore").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const snap = graphSnapshot();
      setLoading(btn, true);
      try {
        const updated = await api(`/api/projects/${proj.id}/worldviews`, {
          method: "POST",
          body: JSON.stringify({ action: "restore", worldview_id: btn.dataset.id }),
        });
        setProjectState(updated);
        closeDialog("worldviewsModal");
        requestAnimationFrame(fitView);
        offerUndo("Worldview loaded", snap, "Restored graph from before loading the worldview");
      } catch (err) {
        setLoading(btn, false);
        reportError("Could not load the worldview", err);
      }
    });
  });

  list.querySelectorAll(".btn-wv-compare").forEach((btn) => {
    btn.addEventListener("click", async () => {
      list.querySelectorAll(".wv-row").forEach((r) => r.classList.toggle("is-selected", r.dataset.id === btn.dataset.id));
      setLoading(btn, true);
      try {
        const diff = await api(`/api/projects/${proj.id}/worldviews`, {
          method: "POST",
          body: JSON.stringify({ action: "compare", worldview_id: btn.dataset.id }),
        });
        const c = diff.current_diagnostics || {};
        const w = diff.worldview_diagnostics || {};
        const cOpen = (c.open_backdoor_paths || []).length;
        const wOpen = (w.open_backdoor_paths || []).length;
        const row = (label, a, b) => `<tr><th scope="row">${label}</th><td class="${a !== b ? "is-diff" : ""}">${a}</td><td class="${a !== b ? "is-diff" : ""}">${b}</td></tr>`;
        compBox.classList.remove("hidden");
        compBox.innerHTML = `
          <table>
            <thead><tr><th></th><th scope="col">Current graph</th><th scope="col">${escapeHtml(diff.worldview.name)}</th></tr></thead>
            <tbody>
              ${row("Variables", String((proj.nodes || []).length), String((diff.worldview.nodes || []).length))}
              ${row("Arrows", String((proj.edges || []).length), String((diff.worldview.edges || []).length))}
              ${row("Minimal adjustment sets", setsText(c), setsText(w))}
              ${row("Open backdoor paths with current adjustment", String(cOpen), String(wOpen))}
              ${row("Identifiable with current adjustment", c.current_adjustment_valid ? "Yes" : "No", w.current_adjustment_valid ? "Yes" : "No")}
            </tbody>
          </table>`;
      } catch (err) {
        reportError("Comparison failed", err);
      } finally {
        setLoading(btn, false);
      }
    });
  });
}


// ============================================================================
// UPLOADS, DATA VIEW, CAUSAL QUESTION
// ============================================================================

async function upload(path, file) {
  const fd = new FormData();
  fd.append("file", file);
  const resp = await fetch(path, { method: "POST", body: fd });
  if (!resp.ok) {
    let msg = await resp.text();
    try { msg = JSON.parse(msg).detail || msg; } catch (e) { /* text */ }
    throw new Error(msg || `Upload failed (HTTP ${resp.status})`);
  }
  return resp.json();
}

function fmtNum(x, d = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "–";
  const a = Math.abs(x);
  if (a !== 0 && (a < 0.001 || a >= 1e5)) return x.toExponential(2);
  return Number(x).toFixed(d);
}

function fmtP(p) {
  if (p === null || p === undefined) return "–";
  return p < 0.001 ? "< 0.001" : p.toFixed(3);
}

function renderDataView() {
  const proj = state.currentProject;
  if (!proj) return;
  const el = $("dataContent");
  const ds = (proj.datasets || [])[0];
  if (!ds) {
    el.innerHTML = `
      <div class="empty-data">
        <p class="empty-title">Bring data to the graph</p>
        <p class="empty-body">Upload a CSV with one column per measured variable. The app then (1) tests every conditional independence your graph implies, which can reveal a missing arrow or a wrong assumption, and (2) estimates the effect using only an adjustment set the engine has certified as valid, with regression and double machine learning, plus an E-value for sensitivity to unmeasured confounding.</p>
        <p class="empty-body">The file is stored in this project's data folder and never sent anywhere.</p>
      </div>`;
    return;
  }
  const cols = ds.columns.map((c) => c.name);
  const observed = proj.nodes.filter((n) => n.observed && n.role !== "latent");
  const kind = Object.fromEntries(ds.columns.map((c) => [c.name, c]));
  const mapRows = observed
    .map((n) => {
      const cur = ds.mapping[n.id] || "";
      const c = kind[cur];
      return `<tr>
        <td>${escapeHtml(n.label)} <span class="muted-xs">${escapeHtml(ROLE_LABELS[n.role] || n.role)}</span></td>
        <td><select class="select map-select" data-node="${escapeHtml(n.id)}" aria-label="Column for ${escapeHtml(n.label)}">
          <option value="">Not measured</option>
          ${cols.map((col) => `<option value="${escapeHtml(col)}" ${col === cur ? "selected" : ""}>${escapeHtml(col)}</option>`).join("")}
        </select></td>
        <td class="num">${c ? `${escapeHtml(c.kind)}${c.missing ? `, ${c.missing} missing` : ""}` : ""}</td>
      </tr>`;
    })
    .join("");

  const imp = state.dataResults.implications;
  const est = state.dataResults.estimate;
  el.innerHTML = `
    <section class="data-block">
      <h3>${escapeHtml(ds.filename)}</h3>
      <p class="muted">${ds.n_rows.toLocaleString()} rows, ${ds.columns.length} columns. Match each variable in the graph to a column. Unmapped variables are treated as unmeasured.</p>
      <table class="map-table">
        <thead><tr><th scope="col">Variable</th><th scope="col">Column</th><th scope="col">Type</th></tr></thead>
        <tbody>${mapRows}</tbody>
      </table>
      <div class="data-actions">
        <button id="btnSaveMapping" class="btn btn-secondary btn-sm" type="button">Save mapping</button>
        <button id="btnRemoveDataset" class="btn btn-danger btn-sm" type="button">${icon("trash")}Remove dataset</button>
      </div>
    </section>

    <section class="data-block">
      <h3>Does the data contradict the graph?</h3>
      <p class="muted">Each missing arrow in a DAG implies that two variables are independent given some others. These implications are tested one by one. A rejection means the data disagree with the graph; no rejection means the data are consistent with it, not that it is proven.</p>
      <div class="data-actions">
        <label class="sr-only" for="ciMethod">Test</label>
        <select id="ciMethod" class="select select-sm">
          <option value="auto">Automatic choice</option>
          <option value="partial_correlation">Partial correlation (linear, Fisher's z)</option>
          <option value="gcm">Generalised covariance measure (non-linear, boosted trees)</option>
        </select>
        <button id="btnRunImplications" class="btn btn-primary btn-sm" type="button">Test implications</button>
      </div>
      <div id="implicationsOut">${imp ? implicationsHtml(imp) : ""}</div>
    </section>

    <section class="data-block">
      <h3>Estimate the effect</h3>
      <p class="muted">Uses your current adjustment set if the engine certifies it, otherwise the first minimal valid set. If the effect is not identifiable by adjustment, no number is produced.</p>
      <div class="data-actions">
        <button id="btnRunEstimate" class="btn btn-primary btn-sm" type="button">Estimate ${escapeHtml(((proj.diagnostics || {}).exposure || {}).label || "exposure")} → ${escapeHtml(((proj.diagnostics || {}).outcome || {}).label || "outcome")}</button>
      </div>
      <div id="estimateOut">${est ? estimateHtml(est) : ""}</div>
    </section>`;

  $("btnSaveMapping").onclick = async () => {
    const mapping = {};
    el.querySelectorAll(".map-select").forEach((s) => { if (s.value) mapping[s.dataset.node] = s.value; });
    try {
      const updated = await api(`/api/projects/${proj.id}/datasets/${ds.id}`, { method: "PUT", body: JSON.stringify({ mapping }) });
      state.dataResults = { implications: null, estimate: null };
      setProjectState(updated);
      toast("Mapping saved");
    } catch (err) { reportError("Mapping not saved", err); }
  };
  $("btnRemoveDataset").onclick = async () => {
    try {
      const updated = await api(`/api/projects/${proj.id}/datasets/${ds.id}`, { method: "DELETE" });
      state.dataResults = { implications: null, estimate: null };
      setProjectState(updated);
      toast("Dataset removed from this machine");
    } catch (err) { reportError("Dataset not removed", err); }
  };
  $("btnRunImplications").onclick = async () => {
    const btn = $("btnRunImplications");
    setLoading(btn, true);
    try {
      state.dataResults.implications = await api(`/api/projects/${proj.id}/datasets/${ds.id}/implications`, {
        method: "POST", body: JSON.stringify({ method: $("ciMethod").value }),
      });
      $("implicationsOut").innerHTML = implicationsHtml(state.dataResults.implications);
    } catch (err) { reportError("Tests did not run", err); }
    finally { setLoading(btn, false); }
  };
  $("btnRunEstimate").onclick = async () => {
    const btn = $("btnRunEstimate");
    setLoading(btn, true);
    try {
      state.dataResults.estimate = await api(`/api/projects/${proj.id}/datasets/${ds.id}/estimate`, { method: "POST", body: "{}" });
      $("estimateOut").innerHTML = estimateHtml(state.dataResults.estimate);
    } catch (err) { reportError("No estimate", err); }
    finally { setLoading(btn, false); }
  };
}

function implicationsHtml(r) {
  const ok = r.tested > 0 && r.rejected === 0;
  return `
    <div class="verdict-line ${ok ? "is-ok" : r.rejected ? "is-bad" : ""}">${icon(ok ? "circle-check" : r.rejected ? "octagon" : "info")}<span>${escapeHtml(r.verdict.charAt(0).toUpperCase() + r.verdict.slice(1))}</span></div>
    <p class="muted">${r.tested} tested, ${r.rejected} rejected at α = ${r.alpha} after Holm correction. ${escapeHtml(r.interpretation)}</p>
    ${r.results.length ? `<table class="res-table" style="margin-top:10px">
      <thead><tr><th scope="col">Implied independence</th><th scope="col">n</th><th scope="col">Test</th><th scope="col">Partial r</th><th scope="col">p</th><th scope="col">p (Holm)</th><th scope="col">p (BH)</th></tr></thead>
      <tbody>${r.results.map((x) => `<tr class="${x.rejected ? "is-rejected" : ""}">
        <td>${escapeHtml(x.implication)}${x.rejected ? ` <span class="pill pill-bad">rejected</span>` : ""}</td>
        <td class="num">${x.n}</td><td>${x.method === "gcm" ? "GCM" : "Partial corr."}</td>
        <td class="num">${fmtNum(x.partial_r)}</td><td class="num">${fmtP(x.p_value)}</td><td class="num">${fmtP(x.p_holm)}</td><td class="num">${fmtP(x.p_bh)}</td></tr>`).join("")}</tbody>
    </table>` : ""}
    ${r.skipped.length ? `<p class="muted" style="margin-top:8px">Not tested: ${r.skipped.map((x) => `${escapeHtml(x.implication)} (${escapeHtml(x.reason)})`).join("; ")}</p>` : ""}`;
}

function forestHtml(items) {
  const lo = Math.min(0, ...items.map((i) => i.ci_low));
  const hi = Math.max(0, ...items.map((i) => i.ci_high));
  const pad = (hi - lo) * 0.08 || 1;
  const a = lo - pad, b = hi + pad;
  const W = 600, rowH = 30, left = 220, right = 20, H = items.length * rowH + 30;
  const X = (v) => left + ((v - a) / (b - a)) * (W - left - right);
  const rows = items.map((it, k) => {
    const y = 18 + k * rowH;
    return `<text x="0" y="${y + 4}">${escapeHtml(it.name)}</text>
      <line class="ci" x1="${X(it.ci_low)}" x2="${X(it.ci_high)}" y1="${y}" y2="${y}"/>
      <circle class="pt" cx="${X(it.estimate)}" cy="${y}" r="5"/>`;
  }).join("");
  return `<div class="forest" role="img" aria-label="Estimates with 95% confidence intervals">
    <svg viewBox="0 0 ${W} ${H}">
      <line class="zero" x1="${X(0)}" x2="${X(0)}" y1="4" y2="${H - 18}"/>
      <line class="axis" x1="${left}" x2="${W - right}" y1="${H - 18}" y2="${H - 18}"/>
      <text x="${left}" y="${H - 4}">${fmtNum(a, 2)}</text>
      <text x="${W - right}" y="${H - 4}" text-anchor="end">${fmtNum(b, 2)}</text>
      <text x="${X(0)}" y="${H - 4}" text-anchor="middle">0</text>
      ${rows}
    </svg></div>`;
}

function estimateHtml(r) {
  const est = r.estimators;
  const cards = Object.entries(est).map(([k, e]) => `
    <div class="estimate">
      <div class="estimate-label">${escapeHtml(e.label)}</div>
      <div class="estimate-value">${fmtNum(e.estimate)}</div>
      <div class="estimate-ci">95% CI ${fmtNum(e.ci_low)} to ${fmtNum(e.ci_high)}${e.bootstrap_ci ? `; bootstrap ${fmtNum(e.bootstrap_ci[0])} to ${fmtNum(e.bootstrap_ci[1])}` : ""}</div>
      ${e.risk_ratio ? `<div class="estimate-ci">Risk ratio ${fmtNum(e.risk_ratio.estimate)} (${fmtNum(e.risk_ratio.ci_low)} to ${fmtNum(e.risk_ratio.ci_high)})</div>` : ""}
      ${e.overlap ? `<div class="estimate-ci">Propensity range ${fmtNum(e.overlap.propensity_min)} to ${fmtNum(e.overlap.propensity_max)}; ${e.overlap.clipped_units} units clipped at ${e.overlap.clip_at}</div>` : ""}
      <div class="estimate-meaning">${escapeHtml(e.meaning || "")}</div>
    </div>`).join("");
  const ev = r.e_value || {};
  return `
    <p class="muted" style="margin-top:10px">${escapeHtml(r.exposure)} → ${escapeHtml(r.outcome)}, n = ${r.n.toLocaleString()}, adjusted for ${r.adjusted_for_labels.length ? escapeHtml(r.adjusted_for_labels.join(", ")) : "nothing (the empty set is valid)"}, using ${escapeHtml(r.adjustment_source)}. Graph version ${r.graph_version}.</p>
    <div class="estimate-grid">${cards}
      <div class="estimate">
        <div class="estimate-label">E-value (${escapeHtml(ev.scale || "")})</div>
        <div class="estimate-value">${fmtNum(ev.point, 2)}</div>
        <div class="estimate-ci">For the confidence limit: ${fmtNum(ev.ci_limit, 2)}</div>
        <div class="estimate-meaning">${escapeHtml(ev.meaning || "")}</div>
      </div>
    </div>
    ${forestHtml(Object.values(est).map((e) => ({ name: e.label.split(" (")[0].split(",")[0], estimate: e.estimate, ci_low: e.ci_low, ci_high: e.ci_high })))}
    <p class="muted" style="margin-top:12px">These numbers are causal only under these assumptions:</p>
    <ul class="assumption-list">${(r.assumptions || []).map((a) => `<li>${escapeHtml(a)}</li>`).join("")}</ul>`;
}

const QUESTION_LABELS = {
  general: { unit: "Population or units", exposure: "Exposure or treatment", comparator: "Comparator", outcome: "Outcome", time: "Time horizon" },
  pico: { unit: "Patients (P)", exposure: "Intervention (I)", comparator: "Comparator (C)", outcome: "Outcome (O)", time: "Follow-up" },
  peco: { unit: "Population (P)", exposure: "Exposure (E)", comparator: "Comparator exposure (C)", outcome: "Outcome (O)", time: "Exposure window and follow-up" },
  policy: { unit: "Units (people, firms, regions)", exposure: "Treatment or policy", comparator: "Counterfactual condition", outcome: "Outcome", time: "Post-treatment period" },
  scm: { unit: "System or population", exposure: "Intervention do(X = x)", comparator: "Reference intervention do(X = x′)", outcome: "Target variable", time: "Time step" },
  longitudinal: { unit: "Sample", exposure: "Exposure construct", comparator: "Comparison level", outcome: "Outcome construct", time: "Waves and lags" },
};

function applyQuestionLabels(tpl) {
  const labels = QUESTION_LABELS[tpl] || QUESTION_LABELS.general;
  document.querySelectorAll("[data-qlabel]").forEach((l) => (l.textContent = labels[l.dataset.qlabel]));
}

function renderQuestion() {
  const q = (state.currentProject && state.currentProject.question) || {};
  const set = (id, v) => { const el = $(id); if (el && document.activeElement !== el) el.value = v || ""; };
  set("qTemplate", q.template || "general");
  set("qEstimand", q.estimand || "");
  set("qUnit", q.unit); set("qSetting", q.setting); set("qExposure", q.exposure);
  set("qComparator", q.comparator); set("qOutcome", q.outcome); set("qTime", q.time_horizon);
  applyQuestionLabels(q.template || "general");
  const parts = [q.exposure && `effect of ${q.exposure}`, q.comparator && `versus ${q.comparator}`, q.outcome && `on ${q.outcome}`, q.unit && `in ${q.unit}`, q.time_horizon && `over ${q.time_horizon}`].filter(Boolean);
  $("questionSummary").textContent = parts.length ? parts.join(" ") : "Not defined yet";
}

async function saveQuestion() {
  if (!state.currentProject) return;
  const question = {
    template: $("qTemplate").value, estimand: $("qEstimand").value, unit: $("qUnit").value.trim(),
    setting: $("qSetting").value.trim(), exposure: $("qExposure").value.trim(), comparator: $("qComparator").value.trim(),
    outcome: $("qOutcome").value.trim(), time_horizon: $("qTime").value.trim(),
  };
  try {
    const updated = await api(`/api/projects/${state.currentProject.id}`, {
      method: "PUT",
      body: JSON.stringify({ question, audit_action: "Updated causal question", audit_note: `${question.template.toUpperCase()} framing, estimand ${question.estimand || "unset"}` }),
    });
    setProjectState(updated);
    toast("Question saved");
  } catch (err) { reportError("Question not saved", err); }
}
