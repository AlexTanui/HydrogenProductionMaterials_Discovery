# Papers — post-hoc calibration for Phase 2/3's uncertainty

Added 2026-10-23, read against SCRUM-61 (BLIP's uncalibrated uncertainty) and the honest
Phase 2 vs. Phase 3 comparison (`docs/model_cards/graph_stochastic.md`,
`reports/phase2_vs_phase3.md`). Both papers describe **post-processing** a model's uncertainty
*after* training — no retraining, no loss-function surgery — which is the one approach this
project hasn't tried yet. Everything tried so far (BLIP's NLL loss switch, the variance-floor fix)
changed *training*; these papers fix the *output*.

- [`kuleshov18_calibrated_regression.md`](kuleshov18_calibrated_regression.md) — Kuleshov,
  Fenner & Ermon, ICML 2018. Post-hoc recalibration of a Bayesian model's predicted mean/variance
  via isotonic regression on a held-out calibration set. Directly describes our exact setup
  (Section 3.3, "Recalibrating Bayesian Models"): a model that outputs N(mean, var) via repeated
  sampling, miscalibrated because the true error distribution isn't Gaussian.

- [`bao25_conformal_regression_review.md`](bao25_conformal_regression_review.md) — Bao, Colombo,
  Manokhin, Cao & Luo, PMLR 266, 2025. Survey of conformal prediction methods, which give a
  **finite-sample coverage guarantee** from calibration-set residuals alone — no probabilistic
  model output required at all, so it also applies to Phase 1 (MPNN), which currently has none.

See the chat discussion this sprint (2026-10-23) for how these map onto BLIP's measured
overconfidence and Phase 3's measured over-caution, and the proposed next step: a studentized-
residual conformal correction on BLIP/GraphStochastic's existing mean/variance output, using the
held-out validation split as the calibration set.
