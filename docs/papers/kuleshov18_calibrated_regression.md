# Accurate Uncertainties for Deep Learning Using Calibrated Regression

> **Note on this file:** this is a text transcription of the paper's content, extracted from a PDF
> pasted into a Claude Code conversation on 2026-10-23 — not the original PDF. Drop the actual
> `.pdf` into this directory if a byte-identical copy is needed for citation. Figures are described,
> not reproduced.

Volodymyr Kuleshov, Nathan Fenner, Stefano Ermon. *Proceedings of the 35th International
Conference on Machine Learning*, Stockholm, Sweden, PMLR 80, 2018.

**Relevance to this project:** read alongside SCRUM-61 (BLIP's uncalibrated uncertainty) and the
Phase 2/3 comparison. See `docs/papers/README.md` for how this connects to our own measured
numbers.

## Abstract

Methods for reasoning under uncertainty are a key building block of accurate and reliable machine
learning systems. Bayesian methods provide a general framework to quantify uncertainty. However,
because of model misspecification and the use of approximate inference, Bayesian uncertainty
estimates are often inaccurate — for example, a 90% credible interval may not contain the true
outcome 90% of the time. Here, we propose a simple procedure for calibrating any regression
algorithm; when applied to Bayesian and probabilistic models, it is guaranteed to produce calibrated
uncertainty estimates given enough data. Our procedure is inspired by Platt scaling and extends
previous work on classification. We evaluate this approach on Bayesian linear regression,
feedforward, and recurrent neural networks, and find that it consistently outputs well-calibrated
credible intervals while improving performance on time series forecasting and model-based
reinforcement learning tasks.

## 1. Introduction

Bayesian approaches provide a general framework for dealing with uncertainty (Gal, 2016).
Bayesian methods define a probability distribution over model parameters and derive uncertainty
estimates by integrating over all possible model weights. Recent advances in variational inference
have greatly increased the scalability and usefulness of these approaches (Blundell et al., 2015).

In practice, however, Bayesian uncertainty estimates often fail to capture the true data distribution
(Lakshminarayanan et al., 2017) — e.g., a 90% posterior credible interval generally does not contain
the true outcome 90% of the time. In such cases, we say that the model is **miscalibrated**. This
problem arises because of model bias: a predictor may not be sufficiently expressive to assign the
right probability to every credible interval, just as it may not be able to always assign the right
label to a datapoint.

Recently, Gal et al. (2017) and Lakshminarayanan et al. (2017) proposed uncertainty estimation
techniques for deep neural networks (ensemble methods, heteroscedastic regression, concrete
dropout). These methods require modifying the model and may not always produce perfectly
calibrated forecasts.

**Figure 1** (described): a Bayesian neural network's raw 90% credible interval around a sales
forecast misses most of the true data points (top); after the paper's recalibration method, the same
90% credible interval correctly contains 9/10 points (bottom).

**Contributions.** A simple technique for recalibrating the output of any regression algorithm,
extending recalibration methods such as Platt scaling (previously only for classification) to
regression and to the miscalibration of Bayesian credible intervals specifically. Demonstrated
useful in time series forecasting and model-based reinforcement learning.

## 2. Calibrated Classification

Background section reinterpreting Platt scaling for the regression extension in Section 3.

**Calibration (binary classification).** A forecaster H is calibrated if, as T → ∞:

```
sum_t y_t * I{H(x_t) = p} / sum_t I{H(x_t) = p}  ->  p   for all p in [0, 1]
```

Sufficient condition: P(Y = 1 | H(X) = p) = p for all p.

**Calibration vs. sharpness.** Calibration alone is not enough — a forecaster that always predicts
E[Y] is calibrated but not useful. Good predictions also need to be **sharp**: probabilities close to
0 or 1 (for regression: tight intervals around a single value). An ideal forecaster is both calibrated
and maximally sharp.

**Training calibrated classifiers.** Recalibration trains an auxiliary model R: [0,1] → [0,1] on top
of a pre-trained forecaster H such that R∘H is calibrated. Setting R(p) = P(Y=1 | H(X)=p) is
sufficient. Platt scaling approximates this with a sigmoid; other methods use isotonic regression or
kernel density estimation.

**Diagnostic tools.** Calibration curves: bin predictions into intervals, plot predicted average vs.
observed average per bin — a straight line is perfect calibration. Sharpness: look at how close
predictions cluster to 0/1 (classification) or how tight the predicted variance is (regression).

## 3. Calibrated Regression

Extends recalibration to continuous Y, and applies it to Bayesian deep learning models. The
forecaster H outputs at each step t a CDF F_t targeting y_t. F_t^{-1} is the quantile function.

### 3.1. Calibration

A forecaster H is calibrated if, as T → ∞:

```
(1/T) * sum_t I{y_t <= F_t^{-1}(p)}  ->  p    for all p in [0, 1]
```

i.e. the empirical and predicted CDFs match in the limit. This also extends to general confidence
intervals: the fraction of y_t falling in [F_t^{-1}(p1), F_t^{-1}(p2)] should → p2 − p1.

**Calibration and sharpness (regression).** As in classification, calibration alone is insufficient —
F(y) = P(Y ≤ y), ignoring the features X entirely, is calibrated but useless. Forecasts must also be
sharp: the variance var(F_t) should be small. **Figure 4** (described): a calibrated-but-unsharp
forecaster has uniformly wide intervals everywhere; a calibrated-and-sharp forecaster has wider
intervals only where genuinely uncertain, with its 100% interval closer to the mean on average —
the sharper, more useful forecast.

### 3.2. Training Calibrated Regression Models — Algorithm 1

Given a pre-trained forecaster H, train an auxiliary model R: [0,1] → [0,1] such that R∘F_t is
calibrated.

**Algorithm 1 (Recalibration of Regression Models)**

> **Input:** Uncalibrated model H : X → (Y → [0,1]) and calibration set S = {(x_t, y_t)}_{t=1}^T.
> **Output:** Auxiliary recalibration model R : [0,1] → [0,1].
>
> 1. Construct a recalibration dataset:
>    `D = { ([H(x_t)](y_t), P_hat([H(x_t)](y_t))) }_{t=1}^T`
>    where `P_hat(p) = |{ y_t : [H(x_t)](y_t) <= p }| / T`.
> 2. Train a model R (e.g., isotonic regression) on D.

Intuition: for any confidence level p, estimate from data the true probability that Y falls below the
p-th quantile of F_X. If p = 95% but only 80/100 observed y_t actually fall below the 95% quantile
of F_t, adjust the 95% quantile to the 80% quantile instead (Figure 3, described: forecast plot with
uncalibrated vs. recalibrated 90% bands; a CDF-calibration scatter; and an expected-vs-observed
confidence calibration plot).

**Recommended regressor for R:** isotonic regression (Niculescu-Mizil & Caruana, 2005) — it
respects that P(Y ≤ F_X^{-1}(p)) is monotonically increasing in p, and is non-parametric (learns the
true distribution given enough i.i.d. data). Fit R on a held-out calibration set distinct from training
(or use K-fold cross-validation: each fold's held-out portion serves as its calibration set, and
predictions are averaged across the K models) to avoid overfitting the recalibrator itself.

### 3.3. Recalibrating Bayesian Models

**This is the section most directly relevant to BLIP/GraphStochastic.** Probabilistic forecasts F_t
are often obtained from Bayesian neural networks or Gaussian processes. In practice, one often uses
the mean and variance μ, σ² of dropout (or weight-noise) samples evaluated at x_t to obtain a
predictive distribution over y_t (Gal & Ghahramani, 2016a) — the result is a Gaussian forecast
F_t(x_t) = N(μ(x_t), σ²(x_t)). **If the true data distribution P(Y|X) is not Gaussian, uncertainty
estimates derived this way will not be calibrated.** Algorithm 1 recalibrates uncertainty estimates
from *any* black-box Bayesian model, making them accurate and useful, without retraining the
original model.

### 3.4. Features for Recalibration

Algorithm 1 can also recalibrate **non-probabilistic** forecasters (just as Platt scaling recalibrates
SVM margins) by generalizing F(y) to any increasing function Φ(y) that correlates with confidence,
then fitting R: Φ → [0,1] to P(Y ≤ F_X^{-1}(φ)).

- **Simplest feature:** distance from the mean prediction, F_x(y) = y − μ(x). Counting the fraction
  of points at any given distance from μ(x) produces calibrated probabilistic forecasts even for an
  **arbitrary, non-probabilistic regressor** — but confidence intervals come out the same width
  everywhere regardless of x, which is less useful for identifying which inputs the model is actually
  uncertain about.
- **A better feature accounts for uncertainty as a function of x:** fit heteroscedastic regression to
  get μ(x), σ(x), and use F_x(y) = (y − μ(x)) / σ(x) — the **studentized residual**. Combining
  features can further improve sharpness.

### 3.5. Diagnostic Tools

**Calibration plot:** choose confidence levels p_1 < ... < p_m; compute empirical frequency
`p_hat_j = |{ y_t : F_t(y_t) <= p_j }| / T`; plot (p_j, p_hat_j) — calibrated forecasts fall on a straight
line. Use a diagnostic set distinct from the calibration and training sets.

**Calibration error (numerical score):**
```
cal(F_1,y_1,...,F_T,y_T) = sum_j w_j * (p_j - p_hat_j)^2
```
(w_j ≡ 1 in the paper's experiments, or w_j ∝ count in that bin to downweight sparse bins).

**Sharpness score:** mean predicted variance,
```
sha(F_1,...,F_T) = (1/T) * sum_t var(F_t)
```
Low-variance predictions are tightly centered; this is the "refinement" term from the classical Brier
score decomposition for the binary case.

## 4. Experiments

### 4.1 Setup
8 UCI datasets (194–8192 examples, 6–159 features), 25%/75% test/train split, averaged over 5
random splits; also depth estimation on Make3D, and time-series forecasting + model-based RL on
a Kaggle grocery-sales dataset (daily sales, item #1503844, stores #1–4, 2014-01-01–2016-12-31).

**Models:** Bayesian Ridge Regression (closed-form, conjugate prior); feedforward NN (2×128
hidden, dropout 0.5) and GRU-based recurrent NN, using MC-dropout (Gal & Ghahramani, 2016a)
for uncalibrated forecasts; DenseNet for the depth task. Recalibration: fit isotonic regression on the
training set itself (no significant overfitting observed, so no separate calibration set was used in
these experiments). Baselines: concrete dropout (Gal et al., 2017), deep ensembles
(Lakshminarayanan et al., 2017, 5-model ensemble with heteroscedastic σ(x) output).

### 4.2 UCI Experiments (Table 1, summarized)
Across 8 datasets (mpg, auto, crime, kinematics, stocks, cpu, bank, wine), recalibrating Bayesian
linear regression and a dense NN substantially lowers calibration error vs. the uncalibrated model
in nearly every case, and the recalibrated models' calibration is better than both baselines'
(concrete dropout, deep ensembles) in most cases, at similar accuracy (MAPE) to the baselines
despite the baselines having more parameters.

### 4.3 Depth Estimation
Per-pixel dropout uncertainty on Make3D, recalibrated on all training-set pixels: calibration error
improved from 5.50E-02 to 2.41E-02, with accuracy preserved (Figure 5, described: near-identical
rms before/after, but the recalibrated calibration curve hugs the diagonal much more closely).

### 4.4 Time Series Forecasting
Recurrent network forecasting daily sales, MAPE 17.3–21.8% across 4 stores. An uncalibrated 90%
interval misses most true points; the recalibrated 90% interval correctly contains ~90%. The
paper's method achieves near-perfect calibration where both baselines (concrete dropout, deep
ensembles) only partially improve on the uncalibrated model (Figure 6, described).

### 4.5 Model-Based Reinforcement Learning
Inventory-management task (a classic RL benchmark): an agent plans 14 days ahead using the
learned (recalibrated) state-transition model. A calibrated transition model lets the agent plan
better and earn higher reward: $60,082 (concrete dropout) vs. $60,894 (deep ensemble) vs. **$61,690
(this method)**, averaged over 4 stores (Table 2).

## 5. Discussion

The approach mixes frequentist and Bayesian ideas — closer to model-criticism / posterior-
predictive-checking than to a "pure" Bayesian stance: fit a model, compare its predictions to real
data, adjust if necessary, rather than trying to get a single model family exactly right up front.

A fully probabilistic model is *not* necessary to get calibrated confidence estimates — simple
features (e.g. distance from a point prediction) suffice (Section 3.4). The advantage of a richer
underlying model is only that it provides a better raw signal to recalibrate *from*, which can
produce *sharper* calibrated forecasts (Figure 4).

**Important caveat, directly relevant to us:** this recalibration preserves the accuracy of the
original point estimates — it only reshapes the stated uncertainty. If the original forecaster was
*underestimating* its uncertainty (intervals too tight), the recalibrated forecast will necessarily
become **less sharp** (wider) to achieve calibration — you cannot get calibration for free if the
starting point is badly overconfident; the price is honesty about how uncertain the model really is.

## 6. Previous Work (condensed)

Platt scaling (Platt, 1999) and isotonic regression (Niculescu-Mizil & Caruana, 2005) are the
standard classification recalibration techniques, extended since to multi-class, structured
prediction, and modern neural nets (Guo et al. 2017; Gal et al. 2017; Lakshminarayanan et al. 2017).
Calibration itself originates in the statistics literature (Murphy 1973; Dawid 1984) via proper
scoring rules, and has been extensively studied in meteorological probabilistic forecasting (Gneiting
& Raftery, 2005). Gneiting et al. (2007) proposed definitions of calibration for continuous variables
but did not explore *how* to produce calibrated forecasts — which is this paper's contribution.

## 7. Conclusion

Formalizes calibration for continuous variables, connecting it to calibrated classification; proposes
a simple recalibration technique producing calibrated probabilistic forecasts given enough i.i.d.
data; introduces diagnostics for calibration and sharpness; demonstrates practical value on
Bayesian neural networks across time-series forecasting, reinforcement learning, and more broadly
for building reliable, interpretable, interactive ML systems.

## References (selected, as cited above)

Gal, Y. *Uncertainty in Deep Learning*. PhD thesis, Cambridge, 2016. · Blundell et al. Weight
uncertainty in neural networks, 2015. · Lakshminarayanan, Pritzel & Blundell. Simple and scalable
predictive uncertainty estimation using deep ensembles, 2017. · Gal, Hron & Kendall. Concrete
dropout, NeurIPS 2017. · Platt, J. Probabilistic outputs for SVMs, 1999. · Gneiting, Balabdaoui &
Raftery. Probabilistic forecasts, calibration and sharpness, JRSS-B 2007. · Niculescu-Mizil &
Caruana. Predicting good probabilities with supervised learning, ICML 2005. · Gal & Ghahramani.
Dropout as a Bayesian approximation, ICML 2016a. · Murphy, A.H. A new vector partition of the
probability score, 1973. · Saxena, Sun & Ng. Make3D, PAMI 2009. · Kendall & Gal. What
uncertainties do we need in Bayesian deep learning for computer vision?, NeurIPS 2017. ·
MacKay, D. Bayesian interpolation, 1992. · Gal & Ghahramani. Theoretically grounded dropout in
RNNs, NeurIPS 2016b. · (full reference list of 24 entries in the original PDF, page 9 —
abbreviated here; see the original for the complete bibliography).
