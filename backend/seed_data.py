# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 SlightlyHappy. DAG Brain: https://github.com/SlightlyHappy/DAG_Brain
# Created and maintained by SlightlyHappy. See NOTICE for the attribution terms.
from typing import List
from .schemas import (
    ProjectState,
    CausalNode,
    CausalEdge,
    CitationSource,
    CompetingWorldview,
    ChatMessage,
    GraphPatchProposal,
    AuditLogEntry,
)


def get_seed_projects() -> List[ProjectState]:
    # =========================================================================
    # PROJECT 1: Epidemiology / Neuroscience / Behavioral Science
    # =========================================================================
    src_walker = CitationSource(
        id="src_walker2017",
        title="The sleep-deprived human brain",
        authors=["Matthew P. Walker", "use_et_al"],
        year=2017,
        journal="Nature Reviews Neuroscience",
        doi="10.1038/nrn.2017.55",
        url="https://doi.org/10.1038/nrn.2017.55",
        study_design="Meta-Analysis / Systematic Review",
        sample_size="Review of 140+ neuroimaging studies",
        key_finding="Sleep deprivation impairs prefrontal cortical control, hippocampal memory consolidation, and attentional network stability.",
        stance="supporting",
    )
    src_walker.authors = ["Matthew P. Walker", "Bryce A. Mander", "Simon N. Archer"]

    src_kalmbach = CitationSource(
        id="src_kalmbach2018",
        title="Sleep reactivity as a vulnerability to shift work disorder and stress-related insomnia",
        authors=["David A. Kalmbach", "Jason R. Anderson", "Christopher L. Drake"],
        year=2018,
        journal="Sleep",
        doi="10.1093/sleep/zsy025",
        url="https://doi.org/10.1093/sleep/zsy025",
        study_design="Longitudinal Cohort",
        sample_size="N = 1,741",
        key_finding="High baseline psychosocial stress and HPA-axis hyperarousal precede and prospectively predict chronic sleep disruption.",
        stance="supporting",
    )

    src_mendelian = CitationSource(
        id="src_dashti2019",
        title="Genome-wide association study identifies genetic loci for self-reported habitual sleep duration",
        authors=["Hassan S. Dashti", "Samuel E. Jones", "Andrew R. Wood", "Richa Saxena"],
        year=2019,
        journal="Nature Communications",
        doi="10.1038/s41467-019-08917-4",
        url="https://doi.org/10.1038/s41467-019-08917-4",
        study_design="Quasi-Experimental (IV / DiD / RDD / MR)",
        sample_size="N = 446,118 (UK Biobank)",
        key_finding="78 genetic loci associated with habitual sleep duration provide valid instrumental variables for Mendelian Randomization studies.",
        stance="supporting",
    )

    p1_nodes = [
        CausalNode(
            id="chronotype_prs",
            label="Circadian Polygenic Score (IV)",
            role="instrument",
            adjusted=False,
            observed=True,
            time_index=0,
            operationalization="78-SNP sleep duration polygenic score (Dashti et al., 2019)",
            x=90.0,
            y=240.0,
        ),
        CausalNode(
            id="baseline_stress",
            label="Baseline Psychosocial Stress",
            role="confounder",
            adjusted=True,
            observed=True,
            time_index=0,
            operationalization="Perceived Stress Scale (PSS-10) at baseline wave t0",
            x=260.0,
            y=70.0,
        ),
        CausalNode(
            id="shift_employment",
            label="Paid Night/Shift Work",
            role="confounder",
            adjusted=False,
            observed=True,
            time_index=0,
            operationalization="Weekly hours worked after 22:00",
            x=520.0,
            y=70.0,
        ),
        CausalNode(
            id="sleep_deprivation",
            label="Chronic Sleep Deprivation",
            role="exposure",
            adjusted=False,
            observed=True,
            time_index=1,
            operationalization="Actigraphy-verified <6h sleep/night over 14 days",
            x=260.0,
            y=240.0,
        ),
        CausalNode(
            id="executive_fatigue",
            label="Prefrontal Executive Fatigue",
            role="mediator",
            adjusted=False,
            observed=True,
            time_index=2,
            operationalization="Psychomotor Vigilance Task (PVT) lapse rate",
            x=470.0,
            y=240.0,
        ),
        CausalNode(
            id="academic_perf",
            label="Academic & Cognitive Performance",
            role="outcome",
            adjusted=False,
            observed=True,
            time_index=3,
            operationalization="Standardized end-of-term exam z-score",
            x=700.0,
            y=240.0,
        ),
        CausalNode(
            id="clinic_visit",
            label="Campus Health Clinic Visit (Collider)",
            role="collider",
            adjusted=False,
            observed=True,
            time_index=3,
            operationalization="Binary indicator of student health clinic presentation",
            x=470.0,
            y=390.0,
        ),
    ]

    p1_edges = [
        CausalEdge(
            id="e_iv_sleep",
            source="chronotype_prs",
            target="sleep_deprivation",
            epistemic_status="empirical",
            mechanism="Endogenous circadian phase & sleep homeostatic drive",
            effect_direction="positive",
            confidence=0.92,
            citation_ids=["src_dashti2019"],
            rationale="Validated genetic instrument from UK Biobank GWAS.",
        ),
        CausalEdge(
            id="e_stress_sleep",
            source="baseline_stress",
            target="sleep_deprivation",
            epistemic_status="empirical",
            mechanism="HPA-axis hyperarousal and pre-sleep cognitive rumination",
            effect_direction="positive",
            confidence=0.90,
            citation_ids=["src_kalmbach2018"],
            rationale="Prospective cohort evidence showing stress precedes insomnia.",
        ),
        CausalEdge(
            id="e_stress_perf",
            source="baseline_stress",
            target="academic_perf",
            epistemic_status="empirical",
            mechanism="Anxiety-induced working memory interference during assessment",
            effect_direction="negative",
            confidence=0.84,
            citation_ids=["src_kalmbach2018"],
            rationale="Direct confounding pathway independent of sleep duration.",
        ),
        CausalEdge(
            id="e_shift_sleep",
            source="shift_employment",
            target="sleep_deprivation",
            epistemic_status="empirical",
            mechanism="Circadian misalignment and truncated sleep opportunity window",
            effect_direction="positive",
            confidence=0.94,
            citation_ids=["src_kalmbach2018"],
            rationale="Strong occupational determinant of sleep restriction.",
        ),
        CausalEdge(
            id="e_shift_perf",
            source="shift_employment",
            target="academic_perf",
            epistemic_status="hypothesized",
            mechanism="Reduced study hours and time allocation conflict",
            effect_direction="negative",
            confidence=0.78,
            citation_ids=[],
            rationale="Time-budget substitution effect.",
        ),
        CausalEdge(
            id="e_sleep_exec",
            source="sleep_deprivation",
            target="executive_fatigue",
            epistemic_status="empirical",
            mechanism="Adenosine accumulation and reduced prefrontal-thalamic connectivity",
            effect_direction="positive",
            confidence=0.95,
            citation_ids=["src_walker2017"],
            rationale="Well-established neurobiological pathway.",
        ),
        CausalEdge(
            id="e_exec_perf",
            source="executive_fatigue",
            target="academic_perf",
            epistemic_status="empirical",
            mechanism="Impaired sustained attention and hippocampal encoding",
            effect_direction="negative",
            confidence=0.93,
            citation_ids=["src_walker2017"],
            rationale="Primary cognitive mediation pathway.",
        ),
        CausalEdge(
            id="e_sleep_perf_direct",
            source="sleep_deprivation",
            target="academic_perf",
            epistemic_status="hypothesized",
            mechanism="Direct mood/motivation pathway not captured by PVT vigilance",
            effect_direction="negative",
            confidence=0.72,
            citation_ids=["src_walker2017"],
            rationale="Direct effect pathway outside executive vigilance.",
        ),
        CausalEdge(
            id="e_sleep_clinic",
            source="sleep_deprivation",
            target="clinic_visit",
            epistemic_status="hypothesized",
            mechanism="Somatic complaints and daytime fatigue prompt clinic visits",
            effect_direction="positive",
            confidence=0.75,
            citation_ids=[],
            rationale="Selection into clinical care.",
        ),
        CausalEdge(
            id="e_perf_clinic",
            source="academic_perf",
            target="clinic_visit",
            epistemic_status="hypothesized",
            mechanism="Students failing exams seek medical documentation/accommodations",
            effect_direction="negative",
            confidence=0.70,
            citation_ids=[],
            rationale="Converging arrow creating a collider at Campus Health Clinic Visit.",
        ),
    ]

    p1_notebook = """# Research Protocol: Sleep Deprivation & Cognitive Performance

## 1. Causal Question & Target Estimand
We aim to estimate the **Total Causal Effect** of **Chronic Sleep Deprivation** ($X$, defined as $<6$ hours/night verified via wrist actigraphy at $t_1$) on **Academic & Cognitive Performance** ($Y$, standardized assessment score at $t_3$).

## 2. Identification & Backdoor Path Analysis
Based on our Structural Causal Model:
- **Confounders ($t_0$):**Both *Baseline Psychosocial Stress* (Kalmbach et al., 2018) and *Paid Night/Shift Work* open backdoor paths between $X$ and $Y$.
- **Mediators ($t_2$):** *Prefrontal Executive Fatigue* (Walker et al., 2017) lies on the causal pathway $X \\rightarrow M \\rightarrow Y$. Because our target estimand is the **Total Effect**, we **must not** condition on *Prefrontal Executive Fatigue* (doing so induces overadjustment bias).
- **Collider Trap ($t_3$):** *Campus Health Clinic Visit* is a common effect of both sleep deprivation and academic distress ($X \\rightarrow C \\leftarrow Y$). Restricting the sample to clinic patients or adjusting for `clinic_visit` opens a spurious non-causal path (Berkson's / Collider Bias).

## 3. Instrumental Variable Backup Strategy
To address potential unmeasured baseline confounding ($U$), *Circadian Polygenic Score* (Dashti et al., 2019) serves as a candidate genetic instrument ($I \\rightarrow X$) for Mendelian Randomization sensitivity analysis.
"""

    p1_proposal = GraphPatchProposal(
        id="prop_caffeine_demo",
        action="add_node",
        title="Evaluate Confounder vs. Mediator: Evening Caffeine Intake",
        rationale="Reviewer #2 often asks about caffeine consumption. However, its role depends on time-ordering: habitual baseline caffeine (t0) is a confounder, whereas compensatory caffeine intake in response to sleep loss (t2) is a mediator.",
        critic_warning="Do not lump baseline caffeine and reactive daytime caffeine into a single time-invariant variable.",
        node=CausalNode(
            id="baseline_caffeine",
            label="Baseline Evening Caffeine (t0)",
            role="confounder",
            time_index=0,
            operationalization="mg/day of caffeine consumed after 16:00 at baseline",
            x=110.0,
            y=70.0,
        ),
    )

    p1 = ProjectState(
        id="proj_sleep_cognition",
        title="Sleep Deprivation & Cognitive Performance",
        discipline="Epidemiology & Behavioral Neuroscience",
        research_question="What is the total causal effect of chronic sleep deprivation (<6h/night) on academic and cognitive performance in university cohorts?",
        target_population="Full-time undergraduate and graduate students (ages 18–26)",
        estimand_type="total_effect",
        citation_style="apa",
        notebook_markdown=p1_notebook,
        nodes=p1_nodes,
        edges=p1_edges,
        sources=[src_walker, src_kalmbach, src_mendelian],
        proposals=[p1_proposal],
        worldviews=[
            CompetingWorldview(
                id="wv_model_a",
                name="Model A: Stress as Pre-Exposure Confounder (t0)",
                description="Baseline stress precedes sleep deprivation and requires adjustment.",
                nodes=p1_nodes,
                edges=p1_edges,
            )
        ],
        chat_history=[
            ChatMessage(
                role="assistant",
                agent_mode="socratic",
                content=(
                    "Welcome to **DAG_Brain Causal Studio**! I've loaded your **Sleep Deprivation & Cognitive Performance** model.\n\n"
                    "Notice in the **Causal Diagnostics Banner** that **Paid Night/Shift Work** currently leaves an **unblocked backdoor path** (`Chronic Sleep Deprivation <- Paid Night/Shift Work -> Academic & Cognitive Performance`), because only *Baseline Psychosocial Stress* is currently marked as `Adjusted`.\n\n"
                    "You can:\n"
                    "1. Click **Paid Night/Shift Work** on the canvas and toggle **Condition / Adjust for Variable**, or click **Apply Minimal Adjustment Set** in the top banner.\n"
                    "2. Test what happens if you mistakenly condition on **Campus Health Clinic Visit (Collider)** — watch the engine immediately flag **Collider Bias**!\n"
                    "3. Review the pending AI Graph Proposal below for **Baseline Evening Caffeine (t0)**."
                ),
                socratic_questions=[
                    "Should 'Paid Night/Shift Work' be included in your primary regression adjustment set, or is there positivity violation in certain student subgroups?",
                    "How will you distinguish baseline anxiety/stress (t0 confounder) from sleep-loss-induced irritability (t2 mediator)?",
                ],
                proposal_ids=["prop_caffeine_demo"],
            )
        ],
        audit_log=[
            AuditLogEntry(
                actor="system",
                action="Initialized Project",
                details="Loaded Sleep Deprivation & Cognitive Performance DAG with 7 nodes, 10 edges, and 3 verified citations.",
            )
        ],
    )

    # =========================================================================
    # PROJECT 2: Economics & Organizational Science
    # =========================================================================
    src_bloom = CitationSource(
        id="src_bloom2015",
        title="Does Working from Home Work? Evidence from a Chinese Experiment",
        authors=["Nicholas Bloom", "James Liang", "John Roberts", "Zhichun Jenny Ying"],
        year=2015,
        journal="The Quarterly Journal of Economics",
        doi="10.1093/qje/qju032",
        url="https://doi.org/10.1093/qje/qju032",
        study_design="Randomized Controlled Trial (RCT)",
        sample_size="N = 996 employees",
        key_finding="Home working led to a 13% performance increase and 50% reduction in attrition, though promotion rates conditional on performance shifted.",
        stance="supporting",
    )

    p2_nodes = [
        CausalNode(
            id="fiber_rollout",
            label="Municipal Fiber Rollout (IV)",
            role="instrument",
            time_index=0,
            operationalization="Staggered municipal FTTH broadband availability",
            x=100.0,
            y=220.0,
        ),
        CausalNode(
            id="task_modularity",
            label="Task Digitalizability & Modularity",
            role="confounder",
            adjusted=True,
            time_index=0,
            operationalization="O*NET occupational teleworkability index",
            x=320.0,
            y=70.0,
        ),
        CausalNode(
            id="hybrid_work",
            label="Hybrid / Remote Work Policy",
            role="exposure",
            time_index=1,
            operationalization=">= 2 days/week remote work schedule",
            x=290.0,
            y=220.0,
        ),
        CausalNode(
            id="deep_work_time",
            label="Uninterrupted Deep Work Hours",
            role="mediator",
            time_index=2,
            operationalization="Hours/week in >=90min uninterrupted focus blocks",
            x=490.0,
            y=220.0,
        ),
        CausalNode(
            id="worker_output",
            label="Employee Output & Retention",
            role="outcome",
            time_index=3,
            operationalization="Quarterly output quality index & 12-month retention",
            x=690.0,
            y=220.0,
        ),
    ]

    p2_edges = [
        CausalEdge(
            source="fiber_rollout",
            target="hybrid_work",
            epistemic_status="empirical",
            mechanism="Home bandwidth threshold enables video & cloud workflows",
            effect_direction="positive",
            citation_ids=["src_bloom2015"],
        ),
        CausalEdge(
            source="task_modularity",
            target="hybrid_work",
            epistemic_status="empirical",
            mechanism="Firms grant remote flexibility preferentially to modular roles",
            effect_direction="positive",
            citation_ids=["src_bloom2015"],
        ),
        CausalEdge(
            source="task_modularity",
            target="worker_output",
            epistemic_status="hypothesized",
            mechanism="Modular tasks have clearer KPIs and lower coordination overhead",
            effect_direction="positive",
        ),
        CausalEdge(
            source="hybrid_work",
            target="deep_work_time",
            epistemic_status="empirical",
            mechanism="Elimination of commute and open-office interruptions",
            effect_direction="positive",
            citation_ids=["src_bloom2015"],
        ),
        CausalEdge(
            source="deep_work_time",
            target="worker_output",
            epistemic_status="empirical",
            mechanism="Higher cognitive throughput on complex deliverables",
            effect_direction="positive",
            citation_ids=["src_bloom2015"],
        ),
    ]

    p2 = ProjectState(
        id="proj_remote_work_econ",
        title="Remote Work Flexibility & Labor Productivity",
        discipline="Labor Economics & Operations Research",
        research_question="What is the causal effect of hybrid remote work policies on knowledge-worker output and retention?",
        target_population="Enterprise software, finance, and operations analysts",
        estimand_type="total_effect",
        citation_style="chicago",
        notebook_markdown="# Labor Economics DAG: Remote Work & Productivity\n\nEvaluating the causal effect of hybrid work using **Task Modularity** as an observed confounder and **Municipal Fiber Rollout** as an instrumental variable (Bloom et al. 2015).\n",
        nodes=p2_nodes,
        edges=p2_edges,
        sources=[src_bloom],
        chat_history=[
            ChatMessage(
                role="assistant",
                agent_mode="socratic",
                content="Loaded the **Remote Work Flexibility & Labor Productivity** causal model (Chicago author-date citation style).",
                socratic_questions=[
                    "Could firm-level management quality confound both the adoption of hybrid work policies and employee retention?",
                    "Does Municipal Fiber Rollout satisfy the exclusion restriction, or could local infrastructure investment directly improve regional economic optimism?",
                ],
            )
        ],
    )

    return [p1, p2]
