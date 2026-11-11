# CQR: a distribution-free alternative to scalar conformal calibration (4 Nov 2026)

**Update, same day (two rounds):** the "Phase 3 energy ensemble is heavier-tailed" reading in the
first version of this doc was an inference from a width gap, not a direct measurement. Asked to
check shape directly, `shape_diagnostics` was added and run on all four checkpoints - it does not
support that reading. See "Checking shape directly" below for that corrected finding. That section's
own follow-up proposal - that force's inflated residual kurtosis is a cross-atom-type pooling
artifact - was then tested directly with `shape_diagnostics_by_atom_type` and **also did not hold
up**: splitting by element does not bring any element back down to the Gaussian baseline. See
"Stratifying by atom type" below for the real result.

**Follow-up to:** SCRUM-68 (`docs/reviews/scrum-68-calibration-review-2026-11-01.md`) and SCRUM-70
(`phase3_graph_stochastic_nll` retrain). Prompted by re-reading `docs/papers/bao25_conformal_regression_review.md`
and `docs/papers/kuleshov18_calibrated_regression.md` against our own `ml/training/calibrate.py`,
asking specifically: what do these two papers recommend that we are *not* doing yet.

## The gap

`calibrate.py`'s existing method (`conformal_scale`, kept as `--method scalar`, the default) fits one
scalar per quantity that rescales the model's predicted std, then scores it by treating that std as
a genuine Normal standard deviation. Two assumptions ride along with that, neither examined before
now:

1. **The residual really is Normal(0, std²).** `conformal_scale` can only ever widen or shrink a
   *symmetric* interval. If the true error distribution relative to the predicted std is skewed or
   heavy-tailed, no scalar fixes that — this is Bao et al. 2025 §2.1's "Fixed Width" critique of
   plain Split Conformal, applied to our own pipeline.
2. **The ensemble is only used for its first two moments.** Phase 2/3 already run `num_mc_samples=20`
   independent stochastic forward passes per prediction (`predict_distribution`'s internal loop) and
   immediately collapse them to `mean`/`var`. The actual empirical shape of that ensemble — whatever
   it is — was being thrown away before this change.

Both gaps point at the same fix: conformalize the ensemble's own empirical quantiles instead of a
fitted Gaussian. That is exactly Romano et al. 2019's **Conformalized Quantile Regression (CQR)**,
as surveyed in Bao et al. 2025 §3 (Table 1). We picked CQR over the paper's other locally-adaptive
methods (CQR-m/CQR-r, CQRFM, CHR, LCP, CTI) because it is the smallest change that reuses an existing
asset: Phase 2/3's MC ensemble is already a free "quantile regressor" (Bao's own term for the thing
CQR conformalizes) — no extra model, no extra training, no new resource cost (Table 1's "Extra
resource cost" column: CQR is "No" when a quantile regressor already exists, which ours does).

## What was implemented

**`ml/models/blip.py`, `ml/models/graph_stochastic.py`:** `predict_distribution`'s existing MC loop
is now factored out into a new `predict_samples` method that returns the raw, uncollapsed stack
(`E_samples [n, B]`, `F_samples [n, N, 3]`) instead of only `mean`/`var`. `predict_distribution` is
now a two-line wrapper calling it. This changes no forward maths and no numeric output of
`predict_distribution` (verified: `tests/ml/test_graph_stochastic.py` and the full `tests/ml` suite
pass unchanged) — it is a shared-interface refactor, the kind CLAUDE.md's 15 Oct agreement allows
Fazin to make directly, done as one commit per model file for Ruturaj to review.

**`ml/models/scaling.py`:** `ScaledEnergyModel.predict_samples` — the same rescale
(`E = scale*E_model + n_atoms*shift`, `F = scale*F_model`) `predict_distribution` already applies,
now also available on the raw stack.

**`ml/training/calibrate.py`:**
- `cqr_bounds(samples, alpha)` — the `alpha/2`/`1-alpha/2` empirical quantiles of the MC ensemble,
  per point. No distributional assumption.
- `cqr_nonconformity(lo, hi, y)` — Romano's eq. (2): `max(lo - y, y - hi)`.
- `cqr_calibrate(samples, y, alpha)` — the split-conformal additive correction `q_hat`: the
  `(1-alpha)(1+1/n)`-th empirical quantile of the nonconformity scores (same finite-sample
  guarantee `conformal_scale` already relied on — factored both through a new shared
  `_split_conformal_level` helper).
- `cqr_block(lo, hi, y, q_hat)` — reports coverage and mean width at one nominal level, not a full
  ECE curve, because a single conformal pass is calibrated at exactly the one alpha it was fit for
  (this matches Bao et al.'s own evaluation protocol — Table 3 reports Coverage/Size/TCR at a target
  level, not a reliability-diagram ECE).
- `calibrate_cqr_and_report` / `--method cqr` on the CLI — same val/test split and subset
  conventions as the existing `--method scalar` path, fits `q_hat_energy`/`q_hat_force` on
  validation, reports before/after on test.
- `shape_diagnostics(samples, y)` — direct skewness/excess-kurtosis measurement of the ensemble
  itself (standardized per point, pooled) and of the studentized residual, attached to every CQR
  report under `"shape"`. Added after the discussion below; see "Checking shape directly."
- `shape_diagnostics_by_atom_type(samples, y, z)` — the same measurement, split by element before
  pooling, testing whether force's inflated residual kurtosis is a cross-atom-type scale-mixture
  artifact rather than genuine per-point non-Gaussian shape. Attached to every CQR report under
  `"shape_by_atom_type"` (force only — energy has no per-atom structure to split on). See
  "Stratifying by atom type."
- 18 unit tests in `tests/ml/test_calibrate.py` (pure-numpy primitives + three real-checkpoint
  integration tests against `phase3_graph_stochastic.pt`), all passing alongside
  `tests/ml/test_graph_stochastic.py`'s pre-existing 5 (23 total).

**Not changed:** `ml/utils/uq_metrics.py` (CQR does not use it — deliberately, see above), and
`backend/app/services/calibration.py` (Alex's file; not touched — see the caveat below on where the
new reports were written instead).

## Results — all four scored checkpoints, full 55,509-frame validation split

`q`/`std_scale` columns are the existing scalar method; `q_hat` and width columns are new. "Gaussian-equiv.
width" is `2 × 1.6449 × sharp_after` — the width a Normal interval with the scalar method's own
calibrated std would have, included only to compare against CQR's width on a common footing.

| Checkpoint | Qty | cov@90 before (cqr) | cov@90 after (cqr) | width before | width after (cqr) | Gaussian-equiv. width (scalar) | Δ |
|---|---|---|---|---|---|---|---|
| `phase3_graph_stochastic_nll` | energy | 0.9994 | 0.9030 | 1.4379 | **0.7411** | 0.5567 | **+33%** |
| `phase3_graph_stochastic_nll` | force | 0.8132 | 0.9166 | 0.6104 | 0.8390 | 0.8524 | −2% |
| `phase3_graph_stochastic` | energy | 0.9830 | 0.9058 | 0.6222 | **0.4433** | 0.3938 | **+13%** |
| `phase3_graph_stochastic` | force | 0.5176 | 0.9187 | 0.3293 | 0.9641 | 1.0107 | −5% |
| `phase2_blip_nll` | energy | 0.1784 | 0.9154 | 0.0365 | 0.3170 | 0.3333 | −5% |
| `phase2_blip_nll` | force | 0.0517 | 0.9174 | 0.0252 | 0.8373 | 0.8503 | −2% |
| `phase2_blip` (refused, dirty tree — informational only) | energy | 0.2188 | 0.9084 | 0.0666 | 0.3627 | 0.3774 | −4% |
| `phase2_blip` (refused, dirty tree — informational only) | force | 0.1201 | 0.9181 | 0.0595 | 0.8559 | 0.8733 | −2% |

Full before/after reports (coverage, width, `q_hat`): `reports/calibration/cqr/calibrate_cqr_*.json`.

## The finding this was worth doing for

**Both BLIP variants, and Phase 3's force, land within ±5% of what a Gaussian interval calibrated to
the same coverage would need.** That's the scalar method's assumption basically holding: those four
residual distributions, relative to their own predicted/ensemble spread, are close enough to Normal
that a symmetric std-rescale wasn't misrepresenting their shape.

**Phase 3's energy is the one outlier, in both the MSE and NLL checkpoints, and it gets worse after
the NLL fix, not better.** CQR needs a 13% wider interval than Gaussian for `phase3_graph_stochastic`,
and a **33% wider** one for `phase3_graph_stochastic_nll`, to hit the same 90% target.

**This paragraph originally read that gap as a shape miscalibration — a heavier-than-Gaussian tail in
the energy ensemble. Asked to check that directly (see below), it isn't: the ensemble's own shape
measures as Gaussian within the method's noise floor.** So the 13–33% gap is real but its cause is
still open. The two most likely mechanical candidates, neither confirmed here: (a) `conformal_scale`
applies one *multiplicative* correction (scales every point's own std by the same factor) while CQR's
`q_hat` is *additive* (shifts every point's bounds by the same absolute amount) — under
heteroscedastic per-point std, these two operations redistribute width across the test set
differently and can land at different averages while both still hitting ~90% marginal coverage, with
no shape difference required; or (b) something about `conformal_scale`'s calibration-set fit
specifically undershoots on energy in a way CQR's fit doesn't, which would show up in a residual
(not ensemble) diagnostic. Neither was isolated before moving on to the finding below, which turned
out to be the bigger and better-supported one.

## Checking shape directly

Everything above infers shape indirectly, from how wide an interval each method needs. `shape_diagnostics`
(`ml/training/calibrate.py`) measures it directly instead: skewness and excess kurtosis of (a) each
point's own MC ensemble, standardized by that point's own mean/std, pooled across every point
("ensemble" below — this is the predictive distribution's own shape), and (b) the studentized
residual `(y - ensemble_mean) / ensemble_std`, pooled the same way ("residual" below — this is
whether the real error matches the claimed uncertainty's shape, the thing a Gaussian calibration
assumes). Attached automatically to every `--method cqr` report, test split, full val-split fit.

**Important finite-sample correction, found while writing the tests for this:** at
`num_mc_samples=20`, neither excess-kurtosis figure is 0 even for a perfectly-calibrated Gaussian
model — `ensemble_excess_kurtosis`'s null is ≈ **−0.28** (self-studentization: dividing by a std
computed from the same 20 draws shrinks the pooled tails) and `residual_excess_kurtosis`'s null is
≈ **+0.4** (a Student-t(19) effect: `error / std` with `std` from only 20 draws has heavier tails
than Normal purely from the denominator's own sampling noise — matches the closed form
`6/(df-4)` at `df = num_mc_samples - 1 = 19`). Both verified by simulation, both documented in the
function's docstring. Every number below should be read against its own baseline, not against 0.

| Checkpoint | Qty | ensemble skew | ensemble exkurt (null ≈ −0.28) | residual skew | residual exkurt (null ≈ +0.4) |
|---|---|---|---|---|---|
| `phase3_graph_stochastic_nll` | energy | −0.013 | −0.308 | −0.176 | 0.50 |
| `phase3_graph_stochastic_nll` | force | 0.001 | −0.289 | 0.037 | **8.65** |
| `phase3_graph_stochastic` | energy | −0.011 | −0.309 | −0.054 | 1.16 |
| `phase3_graph_stochastic` | force | −0.002 | −0.285 | 0.123 | **11.20** |
| `phase2_blip_nll` | energy | −0.019 | −0.407 | 0.127 | 1.75 |
| `phase2_blip_nll` | force | 0.002 | −0.290 | −0.063 | **11.78** |
| `phase2_blip` (refused, informational only) | energy | −0.018 | −0.342 | −0.363 | 1.70 |
| `phase2_blip` (refused, informational only) | force | 0.001 | −0.282 | 0.019 | **9.81** |

**Two conclusions, and they point in opposite directions from where the width-gap inference above
was headed:**

1. **Every checkpoint's own predictive-ensemble shape is indistinguishable from Gaussian**, for both
   energy and force — `ensemble_excess_kurtosis` sits on top of the −0.28 null in all eight rows,
   and `ensemble_skew` is ~0 everywhere. This directly rules out "the model's stochasticity mechanism
   produces a non-Gaussian predictive distribution" as the explanation for anything observed in this
   project so far, for either BLIP's weight noise or Phase 3's representation noise. Whatever is
   wrong with the uncertainty, it isn't the shape of any single prediction's own ensemble.

2. **The real, large, and extremely consistent signal is `residual_excess_kurtosis` for force: 8.6–11.8
   against a null of 0.4, in every one of the four checkpoints, regardless of phase, mechanism, or
   training objective.** Combined with (1) — Gaussian ensembles, but wildly non-Gaussian pooled
   residuals — this is the textbook signature of a **Gaussian scale-mixture**: each individual
   prediction's uncertainty is roughly Normal, but the predicted std does not track the *true* error
   scale consistently across different points, so pooling residuals from points whose std is right
   with residuals from points whose std is badly wrong produces a heavy-tailed pool even though no
   single point's distribution is heavy-tailed. This is quantitative support — not just a
   plausibility argument — for the atom-type-heterogeneity hypothesis raised in the conversation that
   led to this change: ethanol's 2 C / 1 O / 6 H have very different force-magnitude distributions
   (stiff O–H/C–O bonds vs. floppy methyl hydrogens), and a per-quantity-only (never per-atom-type)
   calibration, scalar or CQR alike, cannot separate them. Energy shows the same pattern much more
   mildly (1.2–1.75 against the 0.4 null — real, but far smaller than force's), consistent with
   energy being a single scalar per configuration with much less structural heterogeneity to pool
   over than per-atom-per-component force is.

## Stratifying by atom type

The natural next test of the Gaussian-scale-mixture reading above: if pooling residuals across atom
types with different force scales (stiff O–H/C–O bonds vs. floppy methyl H) is the cause, splitting
`residual_excess_kurtosis` by element (`z`, via the new `shape_diagnostics_by_atom_type`) before
pooling should bring each element's own figure back down near the 0.4 null — a Gaussian of the
"wrong" width is still Gaussian; only the *pooling* of differently-scaled Gaussians inflates kurtosis
(verified synthetically in `test_shape_diagnostics_by_atom_type_recovers_each_slices_own_baseline`).
Ran on all four checkpoints, full val-split fit, same test split as above:

| Checkpoint | pooled resid. exkurt | H (n=90000) | C (n=30000) | O (n=15000) |
|---|---|---|---|---|
| `phase3_graph_stochastic_nll` | 8.65 | **9.24** | 7.00 | 6.13 |
| `phase3_graph_stochastic` | 11.20 | **11.92** | 8.85 | 5.22 |
| `phase2_blip_nll` | 11.78 | **14.42** | 5.42 | 4.13 |
| `phase2_blip` (refused, informational only) | 9.81 | **11.53** | 6.38 | 4.37 |

**This does not confirm the cross-atom-type pooling hypothesis.** If it were just a scale-mixture
across elements, every column would sit near 0.4. Instead, every element in every checkpoint is
still far above 0.4, H is consistently the *most* non-Gaussian of the three (above the pooled figure,
not below it, in all four checkpoints), and the ordering H > C > O is exact and consistent across
every checkpoint regardless of phase, mechanism, or training objective. Splitting by atom type
`ensemble_excess_kurtosis` (not shown — all land at the same −0.28 to −0.29 null in every element,
confirming each element's own predictive ensemble is still individually Gaussian-shaped, same as the
pooled figure) rules out a shape problem in the ensemble itself at any granularity.

So: the heavy tail is real and lives *within* each individual atom type's own residual distribution,
most strongly within hydrogen, not an artifact of mixing elements together. That redirects rather
than closes the earlier open question — it is not "different atom types have different scale,
pooled naively" (that hypothesis is now ruled out), it is "the model's predicted force uncertainty
for a given atom, especially H, doesn't track its true per-configuration error well enough, and this
is true per element, not just in aggregate." Two candidates this still leaves open, neither isolated
here: (a) within-H heterogeneity that atom type alone doesn't capture — not every H in ethanol is
equivalent (the hydroxyl H bonded to O vs. the five methyl H's bonded to C have different force
distributions, and `z` alone cannot tell them apart from bonding context); or (b) MC-sample-count
noise interacting with H specifically, e.g. if H's true force magnitude is small enough that 20
stochastic passes are a noisier relative estimate of its scale than for the heavier atoms. Both
remain open.

## Caveats

- **`num_mc_samples=20` makes each point's own `alpha/2`/`1-alpha/2` quantile a noisy estimate** (the
  1st/19th order statistic of 20 draws). This does not break the coverage guarantee — Romano et al.
  2019's Thm 1 only needs calibration/test exchangeability, not per-point quantile accuracy — but it
  does mean this is a global, not locally-adaptive, correction: one `q_hat` per quantity, the same
  "Fixed Width" limitation Bao et al. 2025 §2.1 raises about plain Split, just applied to a better
  base interval. Their own recommendation for "highly complex/skewed" data is CHR, or CQR combined
  with LCP (§4, "Choice-of-method recommendations") — genuine local adaptivity (e.g. per-atom-type
  energy intervals) remains open as a follow-up, not attempted here.
- **New reports live in `reports/calibration/cqr/`, not `reports/calibration/` directly.**
  `backend/app/services/calibration.py` globs `reports/calibration/*.json` (non-recursive) and keys
  purely by checkpoint filename, with last-loaded-wins on a collision. A CQR report and a scalar
  report for the same checkpoint share that key but have incompatible schemas (`coverage_90`/
  `mean_width` vs. `ece`/`coverage`/`sharpness_mean_std`) — writing both into the same flat directory
  would silently feed the wrong schema to `/benchmarks` or `/predictions` depending on alphabetical
  sort order. Keeping CQR output one directory level down avoids that without needing a change to
  Alex's file. If CQR is ever adopted as the production calibration, `calibration.py` needs an
  explicit per-checkpoint method choice, not implicit overwrite.
- **Energy and force are still scored independently, never pooled** — same convention as the scalar
  method, same UQ protocol v1.0 principle (CLAUDE.md), just carried over to a method that doesn't
  use `uq_metrics.quantity_block` at all.

## Reproduce

```bash
python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase3_graph_stochastic_nll.pt --method cqr
python -m ml.training.calibrate --checkpoint experiments/checkpoints/phase3_graph_stochastic_nll.pt --method scalar  # unchanged default
```
