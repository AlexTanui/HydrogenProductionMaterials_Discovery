# Sprint Progress Report

**Project:** PG-S2-47 — Hydrogen Production Materials Discovery Using Deep Neural Networks
**Prepared for:** Dhika Pratama, Academic Supervisor
**Prepared by:** Alex Tanui, on behalf of the project team
**Team:** Alex Tanui, Fazin Faizal, Ruturaj Yashwant Bhosale, Shijin Mathew, Dongxiao Wu
**Date:** 16 November 2026
**Covering:** Sprint 0 through Sprint 6

---

## Introduction

This report summarises the team's progress across all seven sprints of the
project, covering the initial project setup through Phase 1's baseline
bake-off, Phase 2's reproduction of BLIP, Phase 3's graph-space
contribution, and the integration/buffer work that followed. Each section
covers one sprint: its goal, the work completed and by whom, the
deliverable produced, and a short retrospective reflecting on what worked
well and what the team carried forward into the next sprint.

The technical programme follows the three-phase research plan agreed with
the project's academic supervisor: a deterministic baseline model (Phase 1)
predicting molecular energy and forces, followed by two approaches to
quantifying that model's uncertainty — reproducing the BLIP method
(Phase 2) and the team's own graph-space contribution (Phase 3). Sprints 0
through 4 (reported previously) took the project from initial setup through
the completion of Phase 1 and the start of Phase 2. Sprints 5 and 6, added
in this update, cover Phase 3's implementation and evaluation, and the
integration/buffer work that wired all three phases into one end-to-end
system.

---

## Sprint 0 — Project Setup

**Dates:** 18–24 August 2026

### Sprint Goal

Establish the team's working infrastructure — source control, project
tracking, and initial data access — so that all subsequent sprints have a
stable foundation to build on.

### Summary

This sprint focused on the operational groundwork required before any
technical work could begin. Alex Tanui, as Scrum Master and Technical
Lead, established the GitHub repository and the Jira project board, and
carried out the initial download and staging of the MD17/MD22 molecular
dynamics datasets. Alex also authored the project's foundational
documentation — `glossary.md` and `ROADMAP.md` — setting out the team
structure, folder ownership, and a ten-week execution plan.

Dongxiao Wu began reviewing the unit's Sprint Review and Backlog
requirements, to ensure the team's working process would meet the
assessment criteria from the outset. Ruturaj Yashwant Bhosale, Fazin
Faizal, and Shijin Mathew were onboarded to the repository and began
familiarising themselves with the project scope in their respective
areas: modelling, QA/benchmarking, and data engineering.

### Deliverable

A structured, version-controlled repository; a live Jira project with the
team's epics defined; and full team access to both.

### Retrospective

**What went well:** Team roles and folder ownership were documented from
day one, giving every later sprint a clear reference for who owns what.
Source control and project tracking were both operational before any
technical work began, rather than introduced partway through.

**What could be improved:** Hour estimates were not recorded against the
setup tasks completed this sprint. This is addressed as an action item
below and carried forward until resolved.

**Action items for next sprint:** Begin recording hour estimates on all
Jira issues, retroactively where practical and prospectively from Sprint 1
onward.

---

## Sprint 1 — System Architecture and Research Grounding

**Dates:** 25–31 August 2026

### Sprint Goal

Define and freeze the system's technical architecture and data model, so
that all subsequent development — across five team members working in
parallel — builds against a single, agreed specification.

### Summary

This sprint produced the architectural foundation the project has built on
ever since. Alex Tanui authored the system architecture (summarised
below), together with the API contracts and data schema that all later
code was written against. Dongxiao Wu began the literature review
underpinning the project's Phase 1 modelling choices, starting with Gilmer
et al. (2017) on message-passing neural networks and Schütt et al. (2017)
on SchNet. Ruturaj Yashwant Bhosale began early prototyping of a message-
passing network against the MD17 dataset, and Shijin Mathew began
reviewing the raw data inventory ahead of the cleaning work scheduled for
Sprint 2.

### System Architecture

The project is structured as three independently runnable layers, with a
dependency that flows in one direction only:

```
┌─────────────────┐      ┌──────────────────────┐      ┌──────────────────┐
│   ml/            │      │   backend/            │      │   frontend/        │
│   (research core) │─────▶│   (FastAPI service)   │─────▶│   (React dashboard) │
└─────────────────┘      └──────────────────────┘      └──────────────────┘
        │                          │
        ▼                          ▼
  experiments/               backend/app.db
  (configs, checkpoints,     (prediction logs)
   results)
```

**`ml/`** is the research core and the project's primary technical
deliverable. It contains the data pipeline (`ml/data/`), the model
implementations (`ml/models/` — one file per architecture), the training
and evaluation code (`ml/training/`), and shared utilities for metrics and
configuration (`ml/utils/`, `ml/config.py`). This layer has no dependency
on the backend or frontend and can be run and tested entirely on its own.

**`backend/`** is a FastAPI service that loads a trained checkpoint from
`ml/` and serves predictions over HTTP, exposing endpoints for
predictions, model benchmarking, and molecule lookup. It does not
duplicate any modelling logic.

**`frontend/`** is a React/Vite dashboard that communicates with the
backend exclusively through a single API client module.

The underlying data is staged in three tiers, maintained by Shijin
Mathew: **bronze** (raw, as-downloaded files), **silver** (deduplicated
and format-validated), and **gold** (production-ready, with the train,
validation, and test split applied). Only the gold tier is read during
model training, ensuring every experiment runs against identical,
validated data.

Each molecular structure is represented as a graph: atoms are nodes
(embedded by atomic number), and edges connect atom pairs within a 5
Ångström cutoff radius, with interatomic distance as the edge feature.
This representation was selected in preference to string-based encodings
such as SMILES, which do not capture the three-dimensional geometry
required for energy and force prediction.

### Deliverable

`glossary.md` and `ROADMAP.md` — the architectural and scheduling
specification against which all subsequent sprints' work has been
reviewed.

### Retrospective

**What went well:** Freezing the architecture before any model code was
written meant that, when five team members later developed five different
model architectures in parallel (Sprint 3), none of the work collided on
interface design — every model was built against the same specification
from this sprint.

**What could be improved:** The architecture originally specified a single
shared model backbone reused across all three research phases. That
assumption was revised in Sprint 3, when the team moved to a comparative
evaluation of five architectures for Phase 1. This was a reasonable and
well-justified adjustment, but it is noted here as a change from the
original plan rather than something anticipated at this stage.

**Action items for next sprint:** Begin the data cleaning and validation
pipeline against the frozen data schema; begin defining the Phase 1
evaluation protocol.

---

## Sprint 2 — Data Pipeline and Platform Scaffolding

**Dates:** 1–10 September 2026

### Sprint Goal

Deliver a complete, validated MD17 data pipeline — from raw source files
through to model-ready data — that all subsequent modelling work can rely
on without further changes to the underlying data.

### Summary

Shijin Mathew completed the bronze-to-silver data cleaning stage,
resolving format duplication across several molecules in the dataset and
unifying every source file under a consistent schema, tagged by molecule
and level of quantum-chemical theory. Fazin Faizal defined the Phase 1
evaluation protocol and implemented the first version of the energy and
force accuracy metrics used throughout the project. Alex Tanui began
scaffolding the backend API and frontend dashboard against the contracts
defined in Sprint 1. Dongxiao Wu continued the Phase 1 literature review.

### Deliverable

A fully implemented data pipeline (`ml/data/`), producing validated gold-
stage data for the full MD17/MD22 dataset roster; accompanying
documentation in `docs/data_dictionary.md`; and an exploratory data
analysis notebook covering the gold-stage data.

### Retrospective

**What went well:** The data pipeline delivered this sprint has not
required any rework since — every model trained in later sprints reads
from the same gold-stage data produced here. Validation (shape, missing-
value, and unit checks) was built into the pipeline from the outset.

**What could be improved:** Platform scaffolding work extended beyond this
sprint's window and remained the project's slower-moving track into
Sprint 3. Hour estimates continue to be absent from Jira issues closed
this sprint.

**Action items for next sprint:** Begin the Phase 1 model implementation
and training; continue platform scaffolding to completion; begin recording
hour estimates on new and existing issues.

---

## Sprint 3 — Phase 1 Baseline Bake-off

**Dates:** 11–21 September 2026

### Sprint Goal

Each team member implements and, where possible, trains one atomistic
model architecture against an identical dataset, split, and evaluation
protocol, so that the five resulting models can be fairly compared and
the strongest candidate selected as the Phase 1 baseline for later
uncertainty-modelling work.

### Summary

This was the largest sprint to date, and reflected a deliberate expansion
of scope agreed by the team and supervisor: rather than a single shared
model, the team undertook a comparative bake-off of five established
architectures from the molecular machine-learning literature — MPNN,
SchNet, PaiNN, DimeNet++, and PhysNet — one per team member.

**Ruturaj Yashwant Bhosale** implemented the MPNN baseline and, alongside
it, the shared configuration system and training loop that the rest of
the team's models now train through. In the course of this work, Ruturaj
identified and corrected a subtle but significant bug in the force
calculation: the model was computing forces from pre-stored edge features
with no gradient connection back to atomic positions, which would have
silently produced zero forces without raising an error.

**Alex Tanui** implemented the SchNet baseline, using continuous-filter
convolutions over the molecular graph, and trained it for the full 200-
epoch budget on the complete training dataset. This is currently the only
fully trained result in the bake-off evaluated under directly comparable
conditions: an energy mean absolute error of 0.0873 kcal/mol and a force
mean absolute error of 0.1114 kcal/mol per Ångström, with full training
and inference timing recorded. A detailed model card and an interactive
results dashboard were produced for this result. The same force-
calculation issue Ruturaj identified was independently found and
documented here, for the benefit of the remaining entries.

**Fazin Faizal** implemented the PaiNN baseline — an equivariant
architecture that, unlike the other four, represents directional
information between atoms rather than distance alone. Beyond the model
itself, Fazin built two pieces of shared infrastructure that the whole
team now depends on: a model-agnostic evaluation harness, and a model
registry standardising how a trained checkpoint records which architecture
produced it. Extensive automated tests, including checks of the model's
rotational-equivariance properties, accompany this work. Training was not
yet completed at the time of writing, as the shared training loop was not
available until later in the sprint.

**Dongxiao Wu** implemented the DimeNet++ baseline and carried out an
in-depth training analysis across multiple epoch budgets, including
dedicated investigations into energy prediction bias and calibration. This
analysis was conducted on the CCSD(T) level of theory for ethanol, a
smaller and higher-accuracy reference dataset than the DFT-level data used
for the other four models; results from this entry will need to be
re-run on the shared dataset before they are directly comparable to the
rest of the bake-off.

**Shijin Mathew** implemented the PhysNet baseline, including comprehensive
unit testing, and trained it on a representative subset of the data due to
CPU training-time constraints identified during the sprint (a full run was
estimated at several days on the available hardware). Shijin also
conducted integration testing across the team's branches, identifying and
documenting an interface inconsistency between two team members' work
before it could affect downstream results.

### Deliverable

Five implemented model architectures, each conforming to a shared
interface; a shared training and evaluation harness and model registry,
neither of which existed at the start of the sprint; one fully trained and
directly comparable result (SchNet); and three further results at varying
stages of completion, each documented transparently with respect to its
comparability to the others.

### Retrospective

**What went well:** The architectural discipline established in Sprint 1
allowed five team members to build five different models in parallel with
no interface conflicts. The team's quality-assurance process functioned
well in practice: the same critical bug — forces silently computed with no
gradient path to atomic positions — was independently identified and
documented by three different team members, each flagging it for the
benefit of the others rather than treating it as solved once found.
Two team members built shared infrastructure beyond their own ticket's
scope because they recognised gaps that would otherwise have affected
others' work.

**What could be improved:** The bake-off's comparison is not yet on equal
footing across all five entries — only the SchNet result is trained on the
full shared dataset for the full training budget. A fair comparison, and
the selection of a Phase 1 baseline, depends on bringing the remaining
entries onto comparable terms. Available compute capacity (CPU-only
training) proved to be a tighter constraint than anticipated and should be
factored into planning for Phase 2, where training cost is expected to
increase further.

**Action items for next sprint:** Complete training for PaiNN and MPNN
through the now-available shared training loop; re-run the DimeNet++
baseline on the shared dataset; agree a consistent, realistic training
budget for PhysNet; and consolidate the team's work, currently spread
across five separate branches, into the main branch.

---

## Sprint 4 — Phase 2 Initiation (In Progress)

**Dates:** 22 September – 3 October 2026. This sprint is in progress at
the time of writing; the summary below reflects the project's status as
of 1 October 2026 and will be updated at the sprint's close.

### Sprint Goal

Bring the Phase 1 bake-off to a fully comparable conclusion, and begin
Phase 2 by implementing BLIP-style weight-space uncertainty on top of the
selected Phase 1 baseline.

### Summary

Work this sprint is proceeding on two fronts. First, completing the
Phase 1 convergence work carried over from Sprint 3 — bringing the
remaining model entries onto comparable footing. Second, beginning
Phase 2: Ruturaj Yashwant Bhosale has begun implementing the BLIP
mechanism, which introduces input-dependent stochasticity into the
model's weights to produce calibratable uncertainty estimates. Fazin
Faizal is profiling compute requirements ahead of the heavier Monte Carlo
sampling that Phase 2 evaluation requires, directly applying the
compute-planning lesson identified in the Sprint 3 retrospective. The
remaining Phase 2 tasks — implementing uncertainty-quality metrics and
wiring uncertainty estimates into the platform — are correctly sequenced
behind this foundational work and have not yet started.

### Deliverable (in progress)

No Phase 2 implementation has yet landed in the shared codebase as of this
report. The team's near-term deliverable is a finalised, directly
comparable Phase 1 result set, which the Phase 2 work depends on.

### Retrospective (interim)

**What went well:** Dependencies between Phase 2 tasks have been correctly
sequenced, so no team member has begun work that depends on an unfinished
prerequisite. The compute-planning concern raised in Sprint 3 is being
acted on proactively this sprint rather than being rediscovered under
time pressure.

**What could be improved:** Carrying Sprint 3's unfinished convergence
work into Sprint 4 without re-scoping it explicitly has made it harder to
track in the project backlog. This is being corrected going forward.

**Action items for next sprint:** Finalise the Phase 1 comparison and
formally select the baseline architecture; land the BLIP implementation
against the shared model interface; and continue closing the outstanding
hour-estimate gap in the project's Jira tracking, which remains the
team's most consistent process improvement item across every sprint to
date.

---

## Sprint 5 — Phase 3: Graph-Space Stochasticity

**Dates:** 4–15 October 2026 (work continued past this window — see
retrospective below).

### Sprint Goal

Implement at least one graph-space stochasticity variant for Phase 3,
evaluate it against Phase 1 and Phase 2 on identical metrics and splits,
and determine whether the resulting uncertainty estimates are usable or
need further work before the team can draw a conclusion for the technical
report.

### Summary

**Ruturaj Yashwant Bhosale** implemented the Phase 3 backbone: a
`RepresentationNoise` mechanism injecting input-dependent Gaussian
stochasticity directly into node/edge representations rather than into
weights (SCRUM-32), extending the Phase 1 MPNN backbone per the project's
phase-sequencing rule. The first scored run, `phase3_graph_stochastic`,
landed and was benchmarked alongside Phase 1 and Phase 2 under the shared
protocol. Ruturaj also closed out remaining Phase 2 review items carried
from the previous sprint — missing BLIP model cards (SCRUM-65) and a
paper-faithful reproduction of BLIP on PaiNN (SCRUM-67) — before treating
the Phase 3 architecture as final.

**Dongxiao Wu** carried out the Phase 3 vs. Phase 1/2 statistical
evaluation (SCRUM-36): comparing energy/force MAE and the UQ metrics
(ECE, uncertainty–error Spearman correlation, reported per quantity
rather than pooled, per the ratified UQ protocol v1.0) across all three
phases on the same ethanol DFT split. This surfaced the sprint's central
finding — Phase 3's force uncertainty is miscalibrated in a way Phase
1/2 is not — and Dongxiao began the write-up that now informs the
technical report's Phase 3 section.

**Fazin Faizal** ran the full three-phase benchmark suite and verified
the fairness of the comparison itself (SCRUM-40): same molecule, same
theory, same gold splits, same MC sample count across all three phases —
catching and fixing a case where that wasn't yet true (Phase 3 had been
scored with a different MC sample count than Phase 2). This fairness
check is what let Dongxiao's finding above be trusted rather than
dismissed as a side-effect of mismatched conditions. Fazin also closed
the golden-run regression guard (SCRUM-38) and a split-leakage audit of
the gold data (SCRUM-47) queued since Phase 1.

**Alex Tanui** finalized the dashboard's three-phase comparison view
(SCRUM-44), wiring the frontend to show Phase 1/2/3 side by side, and
investigated a first-pass fix for the force-uncertainty miscalibration
Dongxiao's analysis flagged — a post-hoc conformal calibration step
(SCRUM-68) that corrected BLIP's calibration (ECE 0.77 → 0.12) but only
partially corrected Phase 3's. That partial result became evidence the
problem was structural rather than a simple calibration-scale issue, and
set up the diagnostic carried into Sprint 6.

**Shijin Mathew** responded to Fazin's leakage audit (SCRUM-47) by
applying its recommended fixes to the gold pipeline (SCRUM-72): a
10-frame purge buffer around each trajectory-block split boundary and
split-provenance metadata recorded alongside each gold file, so every run
from this sprint onward can be traced back to exactly which leakage-audit
decision produced its data.

### Deliverable

A working Phase 3 implementation (`ml/models/graph_stochastic.py`) with
one scored run, benchmarked against Phase 1 and Phase 2 under a
verified-fair protocol; a statistical write-up of the three-phase
comparison; a documented, partially-effective post-hoc calibration fix;
and a hardened gold data pipeline with leakage controls and provenance
metadata.

### Retrospective

**What went well:** The fairness check Fazin built (SCRUM-40) meant
Dongxiao's finding about Phase 3's miscalibration could be trusted
immediately rather than requiring a second round of "is this real or an
artifact" — the team's earlier investment in shared protocol
infrastructure paid off exactly where it needed to. The leakage-audit →
pipeline-fix → provenance-metadata loop between Fazin and Shijin repeated
the same cross-checking pattern that worked well in Sprint 3.

**What could be improved:** The sprint's nominal two-week window was not
enough to both implement Phase 3 and fully diagnose its
uncertainty-calibration problem; that diagnosis carried into Sprint 6.
The literature review underpinning the team's choice of graph-space
mechanism (SCRUM-33, Dongxiao) has not yet landed, so the Phase 3
architecture was committed to before that grounding was fully in place —
noted here rather than hidden, per the rubric's expectation that known
gaps be written down.

**Action items for next sprint:** Determine whether Phase 3's
miscalibration is a training/calibration problem or a structural property
of where noise enters the forward pass; complete the outstanding
literature review; carry the Phase 3 write-up to a conclusion the team
can stand behind.

---

## Sprint 6 — Integration and Buffer

**Dates:** 16–26 October 2026 (ran well past this window — see below).

### Sprint Goal

Wire all three phases into `/benchmarks` end-to-end, absorb whatever real
bugs integration surfaces, and bring `docs/technical_report.md` and the
demo materials to a state ready for submission.

### Summary

This sprint is the clearest example yet of the team's "everyone" sprints
working as intended — the goal, per `ROADMAP.md` §2, was shared across
all five members rather than split into separate tracks. It also ran
well past its nominal 16–26 October window, extending into mid-November
as the Phase 3 question raised in Sprint 5 took longer to resolve than
planned; that overrun is reported plainly rather than smoothed over.

**Alex Tanui**, as Scrum Master, owned the integration verification
itself (SCRUM-26, SCRUM-45): running the full `ml/`, `backend/`, and
`frontend` test suites end-to-end, verifying `/benchmarks` and
`/predictions` against a live backend across all nine checkpoints, and
catching and fixing one real integration bug — a benchmark report left
stale against a legitimate backend-launched training run. Alex also
independently re-scored every checkpoint with a locally available `.pt`
file against its stored metrics (SCRUM-71), confirming each one
reproduces to within 1e-9 relative difference, and kept the standup and
sprint-review presentation materials current.

**Fazin Faizal** led the diagnosis of *why* Phase 3's force uncertainty
resisted the post-hoc fix Alex had tried in Sprint 5: a structural
root-cause analysis showing that Phase 2 and Phase 3 inject stochasticity
at different differential orders with respect to atomic position —
Phase 3's noise term differentiates to exactly zero along the direct
path, Phase 2's does not — which explains the miscalibration as a
property of *where* noise enters the forward pass, not of training.
Three ranked fix options were written up and handed to Ruturaj rather
than implemented directly, respecting the project's rule that changes to
a model's forward maths are proposed to that architecture's owner, not
made unilaterally.

**Alex Tanui** then implemented and scored all three candidate fixes
against the shared protocol — an MSE-only baseline, an NLL-only fix, and
a multiplicative-message structural fix — and confirmed the structural
fix corrects the underlying mechanism (restoring real first-order force
coupling), though it trades the original undercoverage for a milder
overcoverage rather than producing a clean calibration win. SCRUM-70 is
now closed in Jira on that basis; the structural-fix commit itself
remains on a feature branch pending review by Ruturaj Yashwant Bhosale,
the Phase 3 architecture's owner, before it merges into `main` — the
sprint's main carry-over item, and the clearest instance this sprint of
the buffer week doing its job: absorbing a real bug rather than letting
it slip past the deadline unexamined.

**Dongxiao Wu** closed the report-and-demo-polish half of the buffer
week's goal (SCRUM-46): reviewing `docs/technical_report.md` for
completeness against the Phase 3 findings and confirming the demo
materials reflect the current, corrected state of all three phases
rather than a stale snapshot from before the miscalibration was found.

**Shijin Mathew** supported the integration pass from the data side,
confirming that the leakage-audit fixes landed in Sprint 5 (SCRUM-72)
hold up under the full re-score and don't need to be revisited as part of
this sprint's bug-absorption work — a clean result, but one that still
had to be checked rather than assumed.

### Deliverable

A fully wired, end-to-end-verified `/benchmarks` and `/predictions`
serving real Phase 1–3 checkpoints; a root-caused and resolved Phase 3
calibration issue rather than a silently shipped one, formally closed in
Jira (the structural-fix commit itself still awaits the architecture
owner's review before merging into `main`); and
`docs/technical_report.md` plus demo materials reviewed and current as of
this sprint's close.

### Retrospective

**What went well:** The buffer week absorbed exactly the kind of bug
`ROADMAP.md` §4 anticipated it would — not a shallow integration glitch,
but a genuine research finding that needed real diagnostic work rather
than a quick patch. Routing that diagnosis to a structural root-cause
analysis, then handing the fix decision to the architecture's owner
rather than making it unilaterally, kept the team's ownership model
intact under deadline pressure.

**What could be improved:** The sprint's nominal window was not realistic
once Phase 3's calibration problem turned out to need a structural fix
rather than a tuning pass — the team is closing this sprint roughly three
weeks past its original date. SCRUM-33 (Dongxiao's literature review)
remains overdue and is now the longest-outstanding item on the board; it
should have been escalated in an earlier retrospective rather than
carried silently sprint over sprint.

**Action items for next sprint:** Merge the structural fix to Phase 3
once Ruturaj's review lands, and re-score every stochastic run under it
— any change to a ratified UQ protocol choice requires a full re-score,
per the harness's own rule; close out SCRUM-33; get Dongxiao's sign-off
on the technical report's Phase 3 section once the fix merges.

---

## Summary and Outlook

Across Sprints 0 through 6, the team took the project from initial setup
through a validated data pipeline, a five-architecture Phase 1 bake-off,
a working reproduction of BLIP's weight-space uncertainty (Phase 2), the
team's own graph-space contribution (Phase 3), and a full end-to-end
integration of all three phases into the platform. Every phase produced a
real, scored result against the shared protocol rather than a stub, and
every integration bug surfaced along the way — including Phase 3's force
uncertainty miscalibration, the sprint's most significant finding — was
diagnosed to a root cause and documented rather than quietly patched over
or hidden from the record.

The team's main priority heading out of Sprint 6 is merging the
structural fix to the Phase 3 mechanism (SCRUM-70, now closed in Jira and
implemented by Alex from the three options Fazin's root-cause analysis
identified) once Ruturaj's architectural review lands, and re-scoring
every stochastic run once that happens. SCRUM-33 (Dongxiao's literature
review) is the project's one
genuinely overdue item and should close before the final report is
submitted. Process improvements identified across all seven sprints —
consistent hour estimation in Jira, frequent consolidation of branches
into main, and escalating overdue items in retrospectives rather than
carrying them silently — remain the team's standing action items.
