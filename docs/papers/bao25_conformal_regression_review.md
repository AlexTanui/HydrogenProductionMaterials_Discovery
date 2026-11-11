# A Review and Comparative Analysis of Univariate Conformal Regression Methods

> **Note on this file:** text transcription of the paper's content, extracted from a PDF pasted into
> a Claude Code conversation on 2026-10-23 — not the original PDF. Drop the actual `.pdf` into
> this directory if a byte-identical copy is needed for citation. Figures/tables of experimental
> results are condensed; equations and method definitions are kept in full.

Jie Bao, Nicolo Colombo, Valery Manokhin, Suqun Cao, Rui Luo. *Proceedings of Machine Learning
Research* 266:1–23, 2025 (Conformal and Probabilistic Prediction with Applications).

**Relevance to this project:** read alongside `kuleshov18_calibrated_regression.md`. Where
Kuleshov et al. recalibrate an *existing* probabilistic forecast (our BLIP/GraphStochastic mean +
variance), conformal prediction instead builds an interval with a **finite-sample coverage
guarantee** directly from calibration-set residuals — it needs no probabilistic output from the
base model at all, so it also applies to Phase 1 (MPNN), which currently has none.

## Abstract

As machine learning models continue to evolve and improve, quantifying their uncertainty has
become increasingly crucial in high-stakes applications. Conformal prediction has emerged as a
powerful tool and has been widely applied in univariate regression tasks. While numerous
conformal regression methods and models have been developed, few studies have provided a
unified summary and comparison of these approaches. This paper discusses, summarizes, and
overviews the majority of existing univariate conformal regression methods, and conducts a
detailed examination and experimentation of eight major, popular and advanced conformal
regression methods.

## 1. Introduction

Conformal Prediction (CP) presents a statistically rigorous framework that provides probabilistic
guarantees, transforming point predictions from traditional ML models into prediction intervals
with well-defined coverage probabilities — **grounded only in the assumption of data
exchangeability**, independent of model architecture or the correctness of any assumed data
distribution. Used in medical diagnosis, autonomous driving, financial risk control, and (per the
paper's own citation list) graph-based applications.

**Contributions:** categorization of existing conformal regression techniques; detailed exposition
of 8 leading methods; simulation visualizations of the interval differences each method produces;
experiments on 12 real-world datasets with in-depth comparison.

## 2. Background

Univariate regression: predict scalar y ∈ R from x ∈ R^p, given i.i.d. data D = {(x_i, y_i)}_{i=1}^n
from a true joint distribution F_XY. CP transforms any point predictor h_hat into a calibrated,
distribution-free prediction interval C_hat(x) ⊆ Y with the finite-sample guarantee:

```
P(Y ∈ C_hat(X)) >= 1 - alpha
```

### 2.1 Split Conformal Prediction — limitations of the classic approach
Randomly partitions data into training and calibration sets; builds intervals from quantiles of
nonconformity scores on the calibration data. Two common limitations:
- **Fixed width:** relies on the global score distribution, giving near-uniform interval width
  across all inputs — doesn't adapt to local heterogeneity.
- **Simple residual calibration:** absolute residuals as the nonconformity score don't use local
  information, so intervals can be too conservative in some regions and too tight in others.

### 2.2–2.4 Advances in base models, prediction-interval construction, and other extensions
(Condensed survey of the literature — base-model choice from linear/tree/kNN/random-forest/
quantile-regression/neural approaches through to deep learning; conditional-density-estimation
approaches (SPICE, CP2); residual-distribution methods (jackknife+, split conformal prediction
sets); localized/input-dependent adaptation (RLCP's randomized local weighting, PLCP's
covariate-space partitioning, input-dependent conformity measures); training-aware methods that
embed conformal objectives directly into training; and real-world adaptations — federated
learning, missing covariates, discrete responses, spatial/covariate-shift settings. See the original
PDF pages 2–5 for the full literature map with citations; the eight methods it selects for
head-to-head comparison are detailed fully below.)

## 3. Conformal Prediction for Regression — the 8 methods compared

**Split Conformal Regression** (Papadopoulos et al. 2002; Vovk et al. 2005). Partition data into
I_train, I_cal, I_test. Train a point predictor y_hat = f_hat(x) on I_train. Compute nonconformity
scores on I_cal:
```
S_i^split = | y_i - f_hat(x_i) |
```
Interval for a new point x_{n+1}:
```
C_{n,alpha}(x_{n+1}) = { y : s(x_{n+1}, y) <= t_{1-alpha}^Split }
```
where t_{1-alpha}^Split is the (1-alpha)(1+1/|I_cal|)-th empirical quantile of {S_i^Split} ∪ {infinity}.
Guarantees P(Y_{n+1} ∈ C(x_{n+1})) ≥ 1-alpha under exchangeability.

**Conformal Quantile Regression — CQR** (Romano et al. 2019). Uses a quantile-regression base
model predicting q_hat_{alpha/2}(x), q_hat_{1-alpha/2}(x):
```
C^CQR(x) = [ q_hat_{a/2}(x) - t^CQR, q_hat_{1-a/2}(x) + t^CQR ]
S_i^CQR = max( q_hat_{a/2}(x_i) - y_i,  y_i - q_hat_{1-a/2}(x_i) )
```

**CQR-m / CQR-r** (Kivaranovic et al. 2020; Sesia & Candès 2020). Normalize the CQR margin by
the estimated local spread between the median and each quantile, rather than adding a flat margin
— CQR-m additionally estimates the median quantile q_hat_{1/2}; CQR-r avoids needing that extra
estimate:
```
S_i^{CQR-m} = max( (q_a/2(x_i)-y_i)/(q_1/2(x_i)-q_a/2(x_i)),  (y_i-q_{1-a/2}(x_i))/(q_{1-a/2}(x_i)-q_1/2(x_i)) )
S_i^{CQR-r} = max( (q_a/2(x_i)-y_i)/(q_{1-a/2}(x_i)-q_a/2(x_i)),  (y_i-q_{1-a/2}(x_i))/(q_{1-a/2}(x_i)-q_a/2(x_i)) )
```

**CQRFM — Conformal Quantile Regression with Full Model** (Kivaranovic et al. 2020). One
neural network N(x) = (l(x), m(x), u(x)) jointly predicts lower/median/upper quantiles (constrained
l ≤ m ≤ u), trained on a combined pinball loss:
```
L_tau(N(x),y) = h_{tau/2}(y-l(x)) + h_{1/2}(y-m(x)) + h_{1-tau/2}(y-u(x))
```
with the conformal margin computed analogously to CQR-m.

**Conformal Histogram Regression — CHR** (Sesia & Romano 2021). Estimates the *full*
conditional density f_{Y|X} via histograms, then finds the shortest interval (a,b) whose predicted
probability mass meets the coverage target:
```
C^CHR(x) = argmin_{a<b} (b-a)   s.t.   integral_a^b f_hat_{Y|X}(y|x) dy >= 1-alpha
```

**Localized Conformal Prediction — LCP** (Guan 2023). Generalizes CP with a localizer function
H(x,x') ∈ [0,1] measuring similarity between inputs (H(x,x)=1). Calibration points near the test
point get more weight in the local empirical score distribution used to set the threshold — can be
paired with any underlying conformity score, e.g. the plain residual score.

**Conformal Thresholded Intervals — CTI** (Luo & Zhou 2025b). Minimizes prediction-set size by
selectively including only narrower probability ranges of the predicted distribution, built from
quantile bins I_k(x) = (q_hat_{k-1}(x), q_hat_k(x)]. **Not necessarily a single continuous interval** —
its big advantage (smallest average size, best robustness to conditional shift in the paper's own
ranking) trades off continuity, which may or may not matter depending on the use case.

### Table 1 — qualitative comparison (condensed from the original)

| Method | Base model | Extra resource cost | Robust to conditional shift | Small avg. size | Interval continuous |
|---|---|---|---|---|---|
| Split | point predictor | No | ★★☆☆☆ | ★☆☆☆☆ | Yes |
| CQR | quantile regressor | Yes | ★☆☆☆☆ | ★★☆☆☆ | Yes |
| CQR-m | quantile regressor | No | ★★☆☆☆ | ★★★☆☆ | Yes |
| CQR-r | quantile regressor | No | ★☆☆☆☆ | ★★☆☆☆ | Yes |
| CQRFM | one joint network | Yes | ★★★☆☆ | ★★★☆☆ | Yes |
| CHR | quantile regressor | Yes | ★★★★☆ | ★★★★☆ | Yes |
| LCP | any model | Yes | ★★★☆☆ | ★★★☆☆ | Yes |
| CTI | quantile regressor | Yes | ★★★★★ | ★★★★★ | **No** |

(Asymptotic complexity, omitted here: Split/CQR/CQR-m/CQR-r/CQRFM/CTI are O(N log N) time;
CHR and LCP are O(N²) or worse — CTI, CHR and LCP were the slowest in practice.)

## 4. Experiments (condensed)

**Simulation study:** synthetic 2D data, Y ~ Triangular(0, x, x). Visualized (Figures 1–2 in the
original) how each method's intervals pack together across x — CTI and CHR produce visibly
narrower, more discontinuous coverage with better tail behavior (Y<0.1, Y>0.9); the CQR family
is unnecessarily wide for small x, with CQR-r worst and CQR-m/CQRFM improved; LCP gives the
most efficient (compact, still well-covering) intervals among the non-CTI/CHR methods.

**12 real-world UCI/Kaggle datasets** (bike, bio, blog, community, concrete, facebook×2, homes,
meps×3, star — sizes 1030–81311, features 8–280). Each method evaluated on **Coverage**,
interval **Size**, and **Tail Coverage Rate (TCR)**, plus a new composite metric the authors
introduce:

**Balanced Interval Score (BIS)** — combines conditional-coverage robustness (Worst-Slab
Coverage, WSC) and interval size relative to the plain Split baseline into one number:
```
BIS = WSC_method / WSC_split  -  Size_method / Size_split
```
Higher BIS = better net benefit over the simple Split baseline (gains in conditional-coverage
robustness, net of the width cost paid for them).

**Headline findings (Table 3 in the original, summarized):**
- **CTI** consistently produced the smallest prediction intervals across most datasets.
- **TCR** was generally best for CTI/CHR overall.
- **CHR** (2nd-ranked) showed the strongest adaptability on skewed datasets (Facebook, MEPS).
- **CQR** had stable WSC across datasets but intervals tended to be larger on skewed data.
- **CQR-m** produced significantly narrower intervals than plain CQR on every dataset tested;
  **CQR-r** sat between CQR and CQR-m; **CQRFM** matched CQR-m's interval size while
  improving both TCR and BIS.
- **LCP** showed consistent improvement over plain Split on most datasets, particularly on BIS.

### Choice-of-method recommendations (the paper's own, stated directly)
- Many x-features (>100): use Random Forest as the base predictive model.
- No need for a single continuous interval: use **CTI** exclusively.
- Highly complex/skewed data distribution: use **CHR**, or combine **CQR with LCP**.
- Very small sample size, or a simple distribution: use plain **Split**.

## 5. Conclusion

Pronounced discrepancies across methods on interval width, coverage probability and tail-coverage
performance. Scope intentionally centered on split conformal prediction (favored for computational
efficiency and adoption) — flagged directions for future work: extending the comparison to full
conformal, cross-conformal and Jackknife+ procedures; deeper integration of conformal prediction
into model training pipelines; efficient inference for resource-constrained settings; and improved
conditional-coverage guarantees via new methodological formulations.

## References (selected — full list of ~90 entries in the original PDF, pages 18–23)

Vovk, Gammerman & Shafer. *Algorithmic Learning in a Random World*, Springer 2005. · Shafer &
Vovk. A tutorial on conformal prediction, JMLR 2008. · Romano, Patterson & Candès. Conformalized
quantile regression, NeurIPS 2019. · Kivaranovic, Johnson & Leeb. Adaptive, distribution-free
prediction intervals for deep networks, AISTATS 2020. · Sesia & Candès. A comparison of some
conformal quantile regression methods, Stat 2020. · Sesia & Romano. Conformal prediction using
conditional histograms, NeurIPS 2021. · Guan, L. Localized conformal prediction: a generalized
inference framework for conformal prediction, Biometrika 2023. · Luo & Zhou. Conformal
thresholded intervals for efficient regression, AAAI 2025b. · Barber, Candès, Ramdas & Tibshirani.
Predictive inference with the jackknife+, Annals of Statistics 2021. · Vovk. Cross-conformal
predictors, Annals of Mathematics and AI 2015. · Angelopoulos, Bates et al. Conformal prediction:
a gentle introduction, Foundations and Trends in ML 2023. · (full bibliography in the original PDF;
see its reference section for the complete list, including the many 2024–2025 extensions on
training-aware CP, federated/missing-data/discrete-response variants, and graph-based
applications cited in Sections 2.3–2.4.)
