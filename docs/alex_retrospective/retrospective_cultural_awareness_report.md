<!--
AUTHOR TODO — remove this block before submitting.
Every [bracketed placeholder] below marks a spot that needs YOUR genuine
detail: your own background, a real teammate's background (confirm with
them before naming specifics — see the note in §3), a real incident you
recall, your student ID, submission date. Everything outside brackets is
grounded in the actual repo (glossary.md, ROADMAP.md, docs/sprint_process.md,
git history) so it's already accurate to the project — but a reflective
report with no personal specifics reads as generic, which is exactly what
keeps a submission at Credit/Distinction instead of HD. Spend most of your
editing time on §2–4, not the intro/conclusion.
-->

# Retrospective and Cultural Awareness Report

**Student:** Alex Tanui
**Student ID:** [your student ID]
**Course:** PG-S2-47 — ICT Capstone, College of Engineering & IT, University of Adelaide
**Project:** Hydrogen Production Materials Discovery (technical scope: Adaptive Graph-Space Uncertainty Modelling for Machine Learning Interatomic Potentials, benchmarked on MD17)
**Date:** [submission date]

## Introduction

This report reflects on the cultural dimensions of working within a five-person, internationally composed capstone team over a 10-week, seven-sprint delivery cycle (`ROADMAP.md`, `docs/sprint_process.md`). The team is split across distinct functional roles — Shijin (data pipeline), Ruturaj (model architecture), Dongxiao (research and UQ-metric definitions), Fazin (QA and benchmarking), and myself (platform: backend/frontend) — coordinated through Jira, a shared `glossary.md` specification, and weekly Sprint Reviews with our Academic Supervisor, Dhika. Because the client for this project is effectively our own supervisor and the broader research community MD17 results are benchmarked against, cultural awareness here shows up less as external client diplomacy and more as how a genuinely cross-cultural team negotiates technical ambiguity, authority, and communication style under a hard deadline. This report covers a cultural advisory relevant to our stakeholder interactions, how cultural awareness shaped requirements gathering and design, our internal team dynamics, and the resulting impact on the final deliverable.

## 1. Cultural Advisory Summary

Our team draws on distinct academic and professional traditions [confirm and name: e.g. South/Southeast Asian, East Asian, and Australian educational backgrounds — replace with what's actually true of the team]. Three considerations shaped how I approached engagement across the team and with Dhika:

**Communication directness.** Team members from higher-context academic cultures tend to signal disagreement indirectly — through delayed responses, qualified language ("maybe we could also consider…"), or raising a concern privately rather than in a shared Jira comment or standup. Recognising this mattered concretely during Sprint 2's five-architecture bake-off (SCRUM-50), where five people were independently implementing baselines against the same `ml/training/train.py` contract — a misread of a hesitant "that should work" as agreement, rather than an unresolved concern, would have surfaced as an integration bug rather than a design conversation. [Add a real instance: a time you deliberately followed up 1:1 with someone whose written feedback was ambiguous, rather than assuming silence meant agreement.]

**Authority and hierarchy in supervisor interactions.** Sprint Reviews with Dhika function as a formal checkpoint (`docs/sprint_process.md` §3), and team members' comfort challenging or reframing scope in that setting varies by academic background — some cultures treat a supervisor's framing as close to final, others expect open debate. My advisory to the team was to separate the two moments deliberately: raise disagreements and alternative framings in the pre-review team discussion, then present a single, team-aligned position to Dhika, so no one is put in the position of either staying silent or contradicting a teammate live in front of the supervisor.

**Written-English fluency and async load.** With ceremonies and specification documents (`glossary.md`, `ROADMAP.md`, Jira tickets) as the primary coordination surface, non-native-English speakers carry a heavier cognitive load reading and writing precise technical English under deadline than native speakers do. My practical recommendation — applied to `glossary.md` and the sprint ticket descriptions I own — was to favour short, literal sentences and explicit schemas (e.g. the fixed `z`/`R`/`E`/`F` array-naming contract) over idiomatic or ambiguous phrasing, since a specification is read many more times than it is written.

## 2. Reflection on Requirements Gathering and Design Approach

Requirements for this project were unusual in that they originated from a supervisor-issued technical plan rather than an external client brief, which shifted "requirements gathering" toward *interpreting and operationalising* that plan as a team, in Sprint 0 (`docs/sprint_process.md` §1). That process produced `glossary.md` — the single specification for team structure, API contracts, and data schema — and `ROADMAP.md`'s 10-week phase sequencing.

Cultural awareness influenced this process in a specific way: early Sprint 0 discussion revealed that team members weighted "done" differently — for some, a requirement was settled once documented in `glossary.md`; for others, it remained open until explicitly re-confirmed verbally. [Give the real example — a specific requirement, e.g. the MD17 unit convention (kcal/mol, Å) or the bronze/silver/gold staging discipline, that surfaced this gap and how it was resolved.] Rather than assuming a shared definition of consensus, we adopted an explicit convention: any decision affecting more than one person's folder ownership (`glossary.md` §3) had to be written into the spec document itself, not just agreed upon in conversation — this reduced reliance on any one communication style being correctly read by everyone.

On reflection, the requirements-gathering process would have benefited from naming these differing expectations of "agreement" explicitly and earlier — in Sprint 0 rather than discovering them reactively in Sprint 1–2. A concrete improvement for a future project: open Sprint 0 with a short, explicit round where each person states how they prefer disagreement and uncertainty to be raised, rather than inferring it from behaviour over several sprints.

## 3. Team Interactions and Internal Culture

The team's internal culture was shaped by the practical realities of coordinating five people across [confirm: time zones/locations] against a fixed sprint cadence. Two adaptations stand out. First, we moved toward documenting decisions in Jira ticket descriptions and `glossary.md` rather than relying on verbal agreement in synchronous calls (`docs/sprint_process.md` §6) — this was partly a Jira-hygiene requirement for the unit's individual-contribution marking, but it also had a cultural benefit: it gave team members who are less comfortable asserting a position live in a call equal footing to shape a decision in writing, on their own time.

Second, functional ownership (Shijin/data, Ruturaj/models, Dongxiao/research, Fazin/QA, myself/platform) mapped roughly onto areas where each person's academic background gave them the strongest grounding — [confirm/replace with real detail, e.g. "Dongxiao's research background in uncertainty quantification made her the natural owner of ECE/calibration definitions"]. Respecting that meant that even as platform owner I proposed changes to metric definitions rather than implementing them unilaterally, deferring to Dongxiao's ownership of UQ-metric definitions as documented in `CLAUDE.md`'s "Working as Fazin" section and the equivalent for research ownership — a small but deliberate practice of not letting functional confidence override someone else's domain authority.

[Add a genuine reflection here: a specific moment where a teammate's background changed how a technical decision was made or discussed — even a minor one, like a different default assumption about weekend availability, meeting punctuality norms, or how directly to word a code review comment.]

## 4. Impact on Final Project Proposal and Outcomes

Cultural awareness shaped the final deliverable in concrete, traceable ways rather than remaining abstract. The decision to keep `glossary.md` as the single source of truth for array names, units, and API contracts — rather than letting conventions live in individual contributors' heads — reduced the number of silent, culturally-inflected misunderstandings (e.g., over-politeness leading someone to implement a guessed convention rather than ask a clarifying question) that could otherwise have surfaced as integration bugs at the Sprint 6 three-phase integration checkpoint (`ROADMAP.md` §3).

The explicit fallback order agreed in `ROADMAP.md` §5 (Phase 3 scope first, then platform polish, then Phase 2 depth, with Phase 1 as non-negotiable) is itself a product of a team-wide, low-ego conversation about what to cut under time pressure — a conversation that required team members from more hierarchy-deferential backgrounds to feel safe proposing that even a supervisor-set phase be simplified. [Add the real outcome: did this fallback actually get invoked, and if so, how did the team decide, and did cultural dynamics show up in that decision?]

## Conclusion

Cultural awareness on this project was less about a single client-facing moment and more a continuous practice: separating disagreement-surfacing from supervisor-facing presentation, defaulting to written specification over assumed consensus, and respecting functional ownership as a proxy for domain authority. [Close with one honest, forward-looking sentence — what you would do differently on the next cross-cultural team you join.]
