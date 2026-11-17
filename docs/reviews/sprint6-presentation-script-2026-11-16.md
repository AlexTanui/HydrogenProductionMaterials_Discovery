# Sprint 6 presentation — narration script & storyboard

**Project:** PG-S2-47 — Adaptive Graph-Space Uncertainty Modelling for Machine
Learning Interatomic Potentials (titled *Hydrogen Production Materials
Discovery* at the university level)
**For:** Sprint 6 demo / recording, submitted to Dhika Pratama, Academic
Supervisor
**Written:** 16 November 2026
**Revised:** 18 November 2026 — updated to reflect SCRUM-70's resolution (the
Phase 3 force-uncertainty miscalibration was root-caused, a structural fix
was implemented and scored, and the ticket is now closed in Jira — see
Scene 6 for the current, accurate status), and restructured to walk through
the project in build order: data engineering, then system architecture,
then Phases 1 through 3, naming each team member's role as their
contribution comes up, per the team's request.
**Voice:** Australian English narration (spelling and phrasing below is
written for that accent/register — natural, professional, not a caricature)
**Format:** scene-by-scene script with on-screen cues against the actual
running app (`frontend/`, backend on `localhost:8000`). Read each
**NARRATION** block aloud while performing the matching **ON SCREEN** action.
Estimated runtime if read at a natural pace: ~17–19 minutes. Length doesn't
matter for this deliverable — cut or extend any scene as needed.

> **A note on how this was produced:** this script was written by Claude
> Code from the project's own committed history — git log, Jira, the model
> cards, `docs/papers/`, and the live app — not generated from a template.
> No video, audio, or screen-recording tool is available in this session, so
> the actual recording (voice + screen capture) still needs a person to do
> it. Everything below is written to be read straight into any screen
> recorder (OBS, Loom, QuickTime, Zoom) in one take, or fed to a TTS tool
> with an Australian voice if a live presenter isn't available.

---

## Scene 0 — Cold open

**ON SCREEN:** Dashboard page, hero section visible (`Adaptive graph-space
uncertainty modelling for interatomic potentials`).

**NARRATION:**

> G'day. This is PG-S2-47 — Hydrogen Production Materials Discovery, out of
> Adelaide University's College of Engineering and IT. Over the next
> few minutes I'll walk you through how this was built, start to finish —
> the data, the architecture, and the three research phases on top of it —
> who on the team owns what, and — just as importantly — what we found out
> the hard way, including a real calibration problem we root-caused and
> fixed, and we're reporting that process honestly rather than hiding it.
>
> Five of us built this: **Alex Tanui** as Scrum Master and Technical Lead,
> **Fazin Faizal** as QA and Benchmarking Engineer, **Ruturaj Yashwant
> Bhosale**, **Dongxiao Wu**, and **Shijin Mathew** each owning one of the
> five Phase 1 model architectures alongside their other responsibilities.
> I'll name whoever owns the piece on screen as we get to it.

---

## Scene 1 — The research question

**ON SCREEN:** Scroll to the feature-card grid and the "Research plan"
stepper on the Dashboard.

**NARRATION:**

> The brief here is deliberately narrow, on purpose. Rather than pointing a
> model straight at real catalyst screening, the supervisor-issued technical
> plan scopes this around MD17 — a standard, well-understood molecular
> dynamics benchmark, from Chmiela et al.'s 2017 *Science Advances* paper on
> energy-conserving molecular force fields. The idea is to prove out a
> *methodology* first: a message-passing graph neural network that predicts
> a molecule's energy and atomic forces, and — the actual research
> question — reports when it's uncertain about that prediction, rigorously,
> not just a number that looks like a confidence interval.
>
> That question splits into three sequential phases, each one extending the
> last so any difference in results is down to *where the randomness lives*,
> not three unrelated models:
>
> - Phase 1 — a deterministic baseline.
> - Phase 2 — reproducing an existing method, BLIP, that puts stochasticity
>   in the model's weights.
> - Phase 3 — our own contribution: moving that stochasticity into the
>   graph's node and edge representations instead.
>
> But before any of that modelling work could start, the data had to be
> right, and the system had to have a shape everyone could build against in
> parallel. That's where I'll actually start.

---

## Scene 2 — Data engineering: getting MD17 ready to train on

**ON SCREEN:** Open `docs/data_dictionary.md` briefly, or the EDA page/
notebook if the dashboard exposes it, showing the bronze/silver/gold
folders and a sample molecule's shape.

**NARRATION:**

> **Shijin Mathew**, our data engineer on this project, owns everything
> under `ml/data/` — and this is the layer every later result ultimately
> depends on.
>
> Real MD17 data ships with the repo over Git LFS — about 814 megabytes
> across nine source files — and it moves through three tiers. Bronze is
> the raw, as-downloaded files. Silver is deduplicated and schema-validated:
> every file normalised to the same four arrays — atomic numbers,
> coordinates, energy, and forces — with units checked, not just assumed,
> and molecule identity tagged by *both* the molecule and the level of
> quantum-chemical theory, because the same molecule computed at two
> different theory levels is two different datasets, never pooled. Gold is
> the production-ready tier, with the train, validation, and test split
> already applied — and that's the only tier training ever reads from.
>
> One detail that matters more than it sounds like it should: those splits
> are contiguous blocks of simulation trajectory, never a random shuffle.
> Adjacent frames in a molecular dynamics trajectory are near-duplicates of
> each other, so a random split would leak near-identical structures
> between train and test and quietly inflate every accuracy number built on
> top of it.
>
> And we didn't just assume that was airtight — Fazin ran a dedicated
> leakage audit of those trajectory-block splits before the team locked the
> evaluation protocol, found the boundary wasn't as clean as assumed, and
> Shijin applied the fix: a ten-frame purge buffer around every split
> boundary, plus split-provenance metadata recorded alongside each gold
> file, so any result from that point on can be traced back to exactly
> which data decision produced it. That audit-and-harden loop is a pattern
> you'll see again later in this video.

---

## Scene 3 — System architecture

**ON SCREEN:** Open `glossary.md` or `docs/architecture.md`, showing the
three-layer diagram (`ml/` → `backend/` → `frontend/`).

**NARRATION:**

> With the data settled, the next thing that had to exist before five
> people could build five models in parallel was one agreed system shape.
>
> **Alex Tanui**, as Technical Lead, authored that architecture up front:
> three independently runnable layers, with a dependency that only ever
> flows one way. `ml/` is the research core — the data pipeline you just
> saw, the model implementations, training and evaluation — and it has no
> dependency on anything downstream, so it can be run and tested entirely
> on its own. `backend/` is a FastAPI service that loads a trained
> checkpoint and serves predictions and benchmark results over HTTP, with
> no modelling logic duplicated in it. `frontend/` is the React dashboard
> you're looking at right now, and it talks to the backend through exactly
> one API client module — nothing calls `fetch` directly from a page.
>
> **Fazin Faizal**, as QA and Benchmarking Engineer, owns the shared
> training loop, the experiment config schema, and the model registry and
> checkpoint contract that standardises how a trained checkpoint records
> which architecture produced it — infrastructure that came together
> slightly later, in the middle of Phase 1, but that every model now trains
> and is scored through.
>
> Freezing that shape before any model code existed is what let five team
> members build five different architectures at the same time with zero
> interface collisions — which is exactly what happens next.

---

## Scene 4 — Phase 1: the five-architecture bake-off

**ON SCREEN:** Navigate to Benchmarks. Scroll to the "Energy MAE by phase"
and "Force MAE by phase" charts, point at the blue Phase 1 bars.

**NARRATION:**

> Phase 1 wasn't just one model — the team expanded scope here and built
> five established architectures from the molecular ML literature side by
> side, one per team member, all trained under one identical protocol: same
> data, same loss, same training budget. That's what lets you compare them
> fairly in this chart.
>
> - **MPNN** — **Ruturaj Yashwant Bhosale**'s build, and the project's
>   model owner for it, a Gilmer-style message-passing network. That's
>   Gilmer, Schoenholz, Riley, Vinyals and Dahl, *Neural Message Passing
>   for Quantum Chemistry*, ICML 2017 — this is the backbone every later
>   phase extends.
> - **SchNet** — **Alex Tanui**'s build and model, using continuous-filter
>   convolutions over the molecular graph. Schütt et al., 2017.
> - **PaiNN** — **Fazin Faizal**'s build and model, an equivariant
>   architecture that represents *directional* information between atoms,
>   not just distance. Schütt, Unke and Gastegger, *Equivariant message
>   passing for the prediction of tensorial properties and molecular
>   spectra*, ICML 2021.
> - **DimeNet++** — **Dongxiao Wu**'s build and model, which goes further
>   again and uses angular information between triplets of atoms, not just
>   pairwise distance. Gasteiger et al.
> - **PhysNet** — **Shijin Mathew**'s build and model, on top of her data
>   engineering work, using atom-wise additive energy contributions. Unke
>   and Meuwly, *Journal of Chemical Theory and Computation*, 2019.
>
> One thing worth calling out because it's a genuinely good QA story: three
> different team members independently caught the same subtle bug during
> this bake-off — forces were being computed from pre-stored edge features
> with no gradient path back to atomic positions, which would've silently
> produced zero forces without ever raising an error. Three people finding
> the same bug and flagging it for each other, rather than assuming it was
> already fixed, is exactly the kind of cross-checking we want on a team
> this size.

---

## Scene 5 — Phase 2: reproducing BLIP

**ON SCREEN:** Benchmarks page, "Uncertainty quantification" table — point
at the Phase 2 row's ECE and correlation columns. Then switch to Predict,
pick `ethanol`, select `phase2_blip` from the phase selector if present,
run a prediction, and point at the uncertainty value returned.

**NARRATION:**

> Phase 1 is deterministic — one forward pass, one number, no sense of how
> much to trust it. Phase 2 fixes that by reproducing BLIP: input-dependent
> Gaussian stochasticity placed directly in the message and update layers'
> *weights*, owned on our side by **Alex Tanui** as the architecture's
> model owner. Run the same molecule through the model several times and
> you get a *distribution* of predictions instead of one point estimate —
> the spread of that distribution is the model's uncertainty. **Ruturaj
> Yashwant Bhosale** then built a paper-faithful version of this on top of
> PaiNN specifically, following Coscia et al., ICML 2026, to check the
> reproduction against the original method as closely as possible.
>
> Getting there wasn't a straight line, and I want to be upfront about
> that rather than skip to the clean result. BLIP's first implementation had
> a batch-dependence bug — weight noise was being gated by a batch mean
> instead of per-row — which we found and fixed. Training it with a proper
> negative-log-likelihood loss instead of just a fixed-variance assumption
> then exposed a variance-collapse problem, which needed a real fix, not
> just a variance floor papering over it. Once that landed, we had our
> first clean, trustworthy Phase 2 result.
>
> On uncertainty quality specifically: we report ECE — expected calibration
> error, how far a model's stated confidence is from its actual
> correctness — and Spearman correlation between predicted uncertainty and
> actual error, for energy and force *separately*, never pooled, because on
> the same checkpoint they can differ by two orders of magnitude. That's UQ
> protocol v1.0, defined and ratified by **Fazin Faizal**, and it's binding
> across every stochastic run from here on.

---

## Scene 6 — Phase 3: our own contribution, and the finding we're reporting honestly

**ON SCREEN:** Benchmarks page, orange Phase 3 bars on the MAE charts, then
scroll to the full metrics table, the "Post-hoc calibration" card, and the
Phase 3 force-calibration note directly below it. Read each callout on
screen while narrating.

**NARRATION:**

> Phase 3 is the team's own research contribution, not a reproduction. The
> question: what if stochasticity lives in the graph's node and edge
> *representations* instead of the weights? **Ruturaj Yashwant Bhosale**,
> as Phase 3's architecture owner, built a `RepresentationNoise` mechanism
> that injects input-dependent Gaussian noise straight into those
> representations, extending the same Phase 1 backbone so the comparison
> stays fair. **Dongxiao Wu** then ran the statistical comparison of Phase 3
> against Phase 1 and 2 on identical metrics and splits, and **Fazin
> Faizal** verified that comparison was actually fair — same molecule, same
> theory, same gold splits, same Monte Carlo sample count across all three
> phases — catching and fixing one case where that wasn't yet true.
>
> That comparison surfaced the finding that matters most in this video, and
> it's the part a lot of presentations would be tempted to gloss over.
> Phase 3's uncertainty wasn't well calibrated out of the box. **Alex
> Tanui** tried a post-hoc fix first — conformal calibration, following
> Kuleshov, Fenner and Ermon's 2018 ICML paper on recalibrating Bayesian
> models, and Bao, Colombo, Manokhin, Cao and Luo's 2025 survey of
> conformal prediction methods. That's a single correction factor, fit
> offline on held-out data, multiplied onto the predicted uncertainty at
> test time — no retraining. You can see it right here: it sharply fixes
> Phase 3's *energy* calibration.
>
> But it only partially fixed *force* calibration, and it was never going
> to fully fix it, because the problem wasn't a miscalibrated number — it
> was structural. **Fazin Faizal** led a root-cause analysis comparing how
> Phase 2 and Phase 3 actually inject their noise, line by line. BLIP's
> noise is *multiplicative*, directly on the activation — so when you
> differentiate the model's output with respect to atomic position to get
> the force, that noise term is still there, same order as the real
> signal. Phase 3's noise was *additive*, a single random draw per forward
> pass — so that same derivative came out to exactly zero along the direct
> path. The uncertainty wasn't responding to atomic position the way it
> needed to, and no scalar correction applied after the fact was ever going
> to fix something missing from the forward pass itself.
>
> Rather than changing someone else's model's maths unilaterally, Fazin
> wrote up three ranked options and handed them to Ruturaj, the
> architecture's owner. **Alex Tanui** then implemented and scored all
> three — an MSE-only baseline, an NLL-only fix, and a multiplicative-
> message structural fix applied before aggregation — and confirmed the
> structural fix restores real first-order force coupling, correcting the
> underlying mechanism, though it trades the original undercoverage for a
> milder overcoverage rather than producing a clean calibration win on
> every number. On that basis, this issue — tracked as SCRUM-70 on our
> board — is now formally closed. What you're looking at on screen right
> now is still the previously deployed checkpoint, though, because the
> structural-fix commit is sitting on its own branch pending Ruturaj's
> review as the architecture owner before it replaces the live checkpoint —
> we don't swap in an unreviewed change to a teammate's model ourselves,
> even once it's scored.

---

## Scene 7 — Sprint 6: integration and buffer

**ON SCREEN:** Benchmarks page, full view scrolling top to bottom once to
show all three phases together. Then briefly show a terminal or the Jira
board if available, for the "everyone" sprint framing.

**NARRATION:**

> Sprint 6 was integration and buffer week — wiring all three phases into
> one working `/benchmarks` endpoint and `/predictions` endpoint,
> end-to-end, and absorbing whatever real bug integration surfaced, which
> this time was exactly the calibration finding you just saw.
>
> Concretely, this sprint: **Alex Tanui** ran the full ML, backend, and
> frontend test suites end-to-end and fixed one real integration bug — a
> benchmark report left stale against a legitimate backend-launched
> training run — and independently re-scored every checkpoint we hold a
> local file for against its stored metrics, with every single one
> reproducing to within one part in a billion. **Dongxiao Wu** closed the
> report-and-demo-polish half of the buffer week, reviewing the written
> technical report for completeness against the Phase 3 findings. And it
> routed the Phase 3 finding to the structural diagnosis and fix you just
> heard about, rather than a quick patch, with the fix decision handed to
> the right person rather than made unilaterally.

---

## Scene 8 — Ethics and responsible use

**ON SCREEN:** Navigate to the Ethics & Safety page. Scroll through the case
cards, pausing on the first two.

**NARRATION:**

> This project has a dedicated Ethics and Security page, and it isn't a
> generic checklist — every case on it is this project's own finding, read
> through a responsible-use lens.
>
> The headline case: if a materials-screening pipeline trusted BLIP's
> original uncertainty as-is, a "95 percent confidence interval" that's
> actually right only about 12 percent of the time would make a weak
> prediction look trustworthy — and downstream, that's real lab time,
> compute, or hazardous synthesis work committed to a candidate the model
> was never entitled to be confident about.
>
> And directly linked to what you just saw in Phase 3: a calibration fix
> that sharply improves *one* quantity can look like the *whole* model is
> now calibrated, when the other quantity's problem is a different,
> structural issue underneath. That's exactly why we report energy and
> force calibration separately, why we routed that one to a root-cause
> analysis instead of letting the good-looking number stand in for the
> whole phase, and why a scored fix still waits on its architecture owner's
> review before it ships.

---

## Scene 9 — Wrap-up

**ON SCREEN:** Back to Dashboard hero, or hold on the Benchmarks page.

**NARRATION:**

> So, where things actually stand: a validated data pipeline, a frozen
> three-layer architecture, five Phase 1 architectures, a working
> reproduction of BLIP, and our own graph-space contribution — all live,
> all scored under one fair, shared protocol, and all wired into one real
> end-to-end platform, not stubs. Five people, five named pieces of
> ownership, and a calibration problem that got root-caused and fixed in
> the open rather than smoothed over.
>
> What's left: merging the structural fix to Phase 3 once the architecture
> owner's review lands, one outstanding literature review that's overdue,
> and a final sign-off on the written report once that fix merges. We'd
> rather show you that honestly than call it finished before it is.
>
> Thanks for watching.

---

## Reference list (for an end-card or spoken citation)

- Gilmer, Schoenholz, Riley, Vinyals & Dahl (2017). *Neural Message Passing
  for Quantum Chemistry*. ICML.
- Schütt et al. (2017). SchNet — a continuous-filter convolutional neural
  network for modeling quantum interactions.
- Schütt, Unke & Gastegger (2021). *Equivariant message passing for the
  prediction of tensorial properties and molecular spectra* (PaiNN). ICML.
- Gasteiger et al. DimeNet++ — directional message passing with angular
  (not just radial) information.
- Unke & Meuwly (2019). PhysNet. *Journal of Chemical Theory and
  Computation*.
- Chmiela et al. (2017). *Machine learning of accurate energy-conserving
  molecular force fields*. Science Advances. — origin of the MD17 dataset.
- Coscia et al. (2026). BLIP — weight-space Bayesian uncertainty for
  interatomic potentials. ICML.
- Kuleshov, Fenner & Ermon (2018). *Accurate Uncertainties for Deep
  Learning Using Calibrated Regression*. ICML.
- Bao, Colombo, Manokhin, Cao & Luo (2025). Survey of conformal prediction
  methods. PMLR 266.

## Team and roles, for quick reference

- **Alex Tanui** — Scrum Master, Technical Lead; system architecture,
  backend and frontend; model owner for SchNet and BLIP; implemented and
  scored the Phase 3 structural fix.
- **Fazin Faizal** — QA and Benchmarking Engineer; shared training loop,
  config schema, model registry, metrics, and UQ protocol v1.0; model
  owner for PaiNN; led the Phase 3 root-cause analysis.
- **Ruturaj Yashwant Bhosale** — model owner for MPNN (the shared backbone)
  and for the Phase 3 `RepresentationNoise` architecture; built the
  paper-faithful BLIP reproduction on PaiNN.
- **Dongxiao Wu** — model owner for DimeNet++; literature review; Phase 3
  vs. Phase 1/2 statistical evaluation; report and demo review.
- **Shijin Mathew** — data engineer, owns `ml/data/` end to end (bronze/
  silver/gold pipeline, leakage-audit fixes); model owner for PhysNet.

## Jira tickets referenced in this script

SCRUM-32 (Phase 3 implementation, Ruturaj), SCRUM-47/72 (leakage audit and
fix, Fazin/Shijin), SCRUM-56/61 (BLIP fixes), SCRUM-39/62 (UQ protocol
v1.0, Fazin), SCRUM-40 (fairness check, Fazin), SCRUM-36 (Phase 3 vs.
Phase 1/2 evaluation, Dongxiao), SCRUM-67 (paper-faithful BLIP on PaiNN,
Ruturaj), SCRUM-68 (post-hoc calibration, Alex), SCRUM-70 (structural root
cause and fix — diagnosed by Fazin, implemented and closed by Alex,
pending Ruturaj's architecture-owner review before merging to `main`),
SCRUM-26/45/46 (Sprint 6 integration & buffer, Alex/Dongxiao), SCRUM-71
(re-scoring, Alex), SCRUM-33 (outstanding literature review, Dongxiao).
