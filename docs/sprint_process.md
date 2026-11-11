# Sprint process — Sprint Reviews & Backlog (continuous assessment)

This document is the team's operating guide for the **Sprint Reviews and
Backlog** component of the ICT Capstone unit. It's separate from
`glossary.md`/`ROADMAP.md` (which say what gets built and in what order) —
this doc says how that work must be tracked in Jira and demonstrated to the
Academic Supervisor so it actually counts.

**Non-negotiable, per the unit's rubric:** a minimum of 5 sprints, 5 sprint
reviews, 5 retrospectives, 5 burnup/burndown charts, a maintained sprint
backlog per sprint, and one evolving product backlog — each with a
recording (or detailed minutes if confidentiality blocks recording)
uploaded to Canvas. Jira is the system of record: if the board doesn't
show it, it didn't happen, no matter what actually got built.

**This team's board runs 7 sprints, numbered Sprint 0 – Sprint 6** (Jira's
actual numbering, not a 5-sprint plan). Sprint 0 is a short setup sprint —
if Dhika's marking caps graded Sprint Reviews at 5, treat Sprint 0 as
ungraded retroactive setup and confirm with her which of Sprints 1–6 are
the formally reviewed ones; either way it doesn't change what belongs in
each sprint's backlog below.

---

## 1. Sprint calendar

Dates are derived from the repo's actual commit history (Sprints 0–2
already happened — see the `git log` dates for the underlying commits),
projected forward to fit `ROADMAP.md`'s 10-week hard deadline (started
Monday 18 Aug 2026, ends ~26 Oct 2026). Confirm the end date against
whatever's on record with Dhika and adjust Sprints 4–6 if it's off — the
content/order is the part that matters more than the exact day.

| Sprint | Dates | Status | Roadmap focus |
|---|---|---|---|
| 0 | 18–24 Aug 2026 | **Done** | Setup: team structure, architecture, `glossary.md`/`ROADMAP.md`, backend/frontend scaffolding, Jira project created |
| 1 | 25–31 Aug 2026 | **Done** | Data pipeline: bronze→silver→gold (dedupe, format-unify, trajectory-block + literature splits), Git LFS staging, gold-stage EDA |
| 2 | 1–10 Sep 2026 | **Done** | **Phase 1 baseline bake-off** — MPNN/SchNet/PaiNN/DimeNet++/PhysNet trained + scored on MD17 ethanol (SCRUM-50 epic), shared train/eval harness |
| 3 | 11–21 Sep 2026 | **Current** | Phase 1 wrap-up: pick the strongest/most reliable baseline, assemble the comparison table + model cards, wire `POST /predictions` to the real checkpoint, start the BLIP literature deep-dive |
| 4 | 22 Sep – 3 Oct 2026 | Upcoming | Phase 2: implement BLIP-style stochastic weights, multi-sample MC inference, UQ metrics (ECE, uncertainty–error correlation), calibration analysis |
| 5 | 4–15 Oct 2026 | Upcoming | Phase 3: settle the graph-space stochasticity approach from the lit review, implement, evaluate vs Phases 1–2 on identical metrics/splits |
| 6 | 16–26 Oct 2026 | Upcoming | Integration + buffer: wire all three phases into `/benchmarks` end-to-end, absorb integration bugs, finalize technical report + demo |

We're currently in **Sprint 3** — Sprint 2's deliverable (the Phase 1
bake-off epic, [SCRUM-50](https://hydrogecapstone.atlassian.net/browse/SCRUM-50),
due 10 Sep) is done; Sprint 3 is about closing it out and picking a winner,
not more Phase 1 training. See §2 below for exactly what to put in each
sprint's backlog.

---

## 2. Jira setup checklist

Project: **SCRUM** — *Hydrogen Production Materials Discovery Using Deep
Neural Networks* on `hydrogecapstone.atlassian.net`.

- [ ] **Sprints enabled.** Team-managed software projects need the
      Backlog feature turned on before Sprints exist as a concept —
      check *Project settings → Features → Backlog*. Without this,
      nothing below is possible.
- [ ] **Product backlog = the Epics.** Each Epic (e.g. SCRUM-50) is one
      requirement area from the project's technical plan. Keep it
      evolving — new Epics get added as `ROADMAP.md` phases start
      (Phase 2, Phase 3, platform work).
- [ ] **Sprint backlog = Tasks pulled into the active sprint.** Every
      Task must be **estimated in hours**, not story points — the
      Academic Supervisor's evaluation is built entirely around the
      hours shown in the burnup chart.
- [ ] **Every issue has:** a clear, meaningful description (contributes
      to the sprint goal — no filler tasks), an assignee, an hour
      estimate, and a priority.
- [ ] **Burnup/burndown chart** configured and check-able at any time
      (*Reports → Burnup/Burndown* on the board) — this is what gets
      screen-shared live in the review, never exported/faked.
- [ ] **Sprint goal** stated up front for each sprint — one sentence,
      measurable, achievable in the sprint's window. For Sprint 3
      (current) that's: *"The strongest Phase 1 baseline is picked from
      the 5-architecture bake-off, backed by a comparison table, and
      `/predictions` serves real predictions from that checkpoint
      instead of the stub."*

**Hours target:** the rubric expects **10+ hours/week per person** — since
these 7 sprints aren't all the same length (Sprints 0–1 are 1 week,
Sprints 2–6 run ~1.5–2 weeks), scale the target to the sprint's actual
span rather than assuming a flat 20 hours:

| Sprint | Span | Target hours/person | Target team burnup (5 people) |
|---|---|---|---|
| 0 | 1 week | 10+ | 50+ |
| 1 | 1 week | 10+ | 50+ |
| 2 | 1.5 weeks | 14+ | 70+ |
| 3 | 1.5 weeks | 16+ | 80+ |
| 4 | 1.5 weeks | 17+ | 85+ |
| 5 | 1.5 weeks | 17+ | 85+ |
| 6 | 1.5 weeks | 16+ | 80+ |

If your chart is well under target for the sprint's span, either work is
missing from Jira (fix the board) or the sprint is genuinely
under-resourced (raise it in standup, don't paper over it).

---

## 3. Ceremony cadence, per sprint

1. **Sprint planning** (start of sprint) — pull tasks from the product
   backlog into the sprint backlog, estimate hours, assign, set the
   sprint goal. Whole team, not just the Scrum Master.
2. **Mid-sprint check-in** (informal, whenever) — Jira should already
   reflect reality; don't let updates pile up for the last day.
3. **Sprint Review** (per sprint, with Dhika, 30+ min) —
   - Demo the working MVP/deliverable for this sprint (not slides —
     the actual thing running).
   - Show the burnup/burndown chart live from Jira.
   - Show the **completed** sprint backlog and the **upcoming** one.
   - Walk through whether the sprint goal was met; if not, say so
     plainly and explain the fallback (see `ROADMAP.md` §5).
   - **Record the session** and upload to Canvas. If confidentiality
     blocks recording, write detailed minutes instead and upload those.
4. **Retrospective** (immediately after the review, team only) — use
   the template in [docs/retrospectives/TEMPLATE.md](retrospectives/TEMPLATE.md).
   Copy it to `docs/retrospectives/sprint_<n>.md` and fill it in every
   sprint; keep all 7 (Sprints 0–6), even though the rubric's minimum is 5 —
   backfill Sprints 0 and 1 now if they weren't written up at the time.

---

## 4. What "meaningful" looks like in a task

The rubric explicitly penalizes insignificant/unnecessary tasks. Before
adding something to the sprint backlog, it should trace to either:

- a folder/owner line in `glossary.md` §3, or
- a week/phase item in `ROADMAP.md` §2, or
- a finding from a completed sprint's retrospective ("Action items for
  next sprint").

If a task doesn't trace to one of those, ask whether it actually adds
value before creating it — a full board of busywork scores worse than a
short, sharply-scoped one.

---

## 5. Group assessment criteria — what gets checked, per sprint

These three map directly to the unit's Group Assessment rubric items 4–6.
Unlike §4 (which is about what belongs in the backlog), this is about what
Dhika is actually scoring at the review.

**Expected deliverable received** — is it complete and functional for
*this* sprint's goal, and are known issues acknowledged rather than hidden?
Per `ROADMAP.md` §4 (definition of done, per phase), what "functional"
means shifts sprint to sprint — don't over- or under-promise:

| Sprint | Expected deliverable |
|---|---|
| 0 | Repo/team scaffolding (`glossary.md`, `ROADMAP.md`, backend/frontend skeleton, Jira project) — no model expected yet |
| 1 | Gold-stage MD17 data + EDA — need not be a running model yet |
| 2 | Phase 1 checkpoints trained + scored on the held-out test split (SCRUM-50) — this one should be functional |
| 3 | A picked Phase 1 winner, comparison table, and `/predictions` serving that real checkpoint |
| 4 | Phase 2 (BLIP) UQ implemented, calibrated, and validated (ECE + uncertainty–error correlation) |
| 5 | Phase 3 (graph-space stochasticity) implemented and compared against Phases 1–2 on the same metrics/splits |
| 6 | Full three-phase integration end-to-end; report/demo polish |

Major or minor issues are fine *if* they're written down — as a Jira Bug,
or in that sprint's retrospective ("what could be improved") — and there's
a stated plan to address them. An undocumented gap discovered live in the
review is what actually costs marks, not the gap itself.

**Sprint goal** — was it met, did the sprint backlog's task priorities
actually support it, and does it add value to the project/client (not
just busywork — same bar as §4 above)? Set the goal at planning (§3.1),
keep every task in that sprint traceable to it (§4), and at the review
state plainly whether it was met — partial completion with a clear
reason beats a goal quietly redefined after the fact to match whatever
got done.

**Burndown/burnup chart** — must be generated from the board's actual
Jira data (*Reports → Burnup/Burndown*), not exported/recreated by hand,
and must be in **hours**, not story points. Team-managed Jira boards
default their Estimation statistic to *Story points* — check
*Board settings → Estimation* and switch it to **Time tracking / original
estimate** now, before Sprint 3's review, or every chart from here on is
showing the wrong unit and the rubric's "cannot be faked" hours check
fails regardless of how much work actually happened. Re-check this
setting hasn't drifted back at the start of each remaining sprint.

---

## 6. Individual contribution — what to keep true in Jira

Each person's mark is checked against what Jira shows, not what actually
happened off-platform. Practically, that means, per sprint, per person:

- Your assigned Tasks are the ones you did the work for — not
  reassigned after the fact to match what got done.
- Hour estimates on your Tasks roughly match your actual time (10+
  hrs/week). If a task ran long or short, update the estimate — don't
  leave stale numbers.
- Task descriptions are specific enough that Dhika can tell what you
  did without you narrating it live (this is also just good practice —
  see the per-model tickets under SCRUM-50 for the level of detail
  expected).
- You attend the Sprint Review. Contribution shown only in Jira, with
  no one there to speak to it, is a weaker showing than being present.

---

## 7. Cross-references

- `ROADMAP.md` — the technical plan this calendar is layered on top of.
- `glossary.md` §1, §3 — team structure and folder ownership; Jira
  assignees should match this unless a sprint's retrospective records a
  deliberate reassignment.
- `docs/retrospectives/` — one file per sprint, using the template.
