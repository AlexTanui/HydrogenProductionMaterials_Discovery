"""Scalar-energy split conformal prediction; requires exchangeable held-out scores."""
from decimal import Decimal, ROUND_CEILING
import numpy as np


def vector(x, name):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected nonempty finite one-dimensional array")
    return x


def absolute_residuals(reference, prediction):
    reference, prediction = vector(reference, "reference"), vector(prediction, "prediction")
    if reference.shape != prediction.shape:
        raise ValueError("Shapes must match; broadcasting is forbidden")
    return vector(np.abs(reference - prediction), "residuals")


def conformal_quantile(scores, confidence=0.9):
    """Exact ceil((n+1)*confidence)-th order statistic; infinity when rank > n."""
    scores = vector(scores, "scores")
    if np.any(scores < 0) or not 0 < confidence < 1:
        raise ValueError("Nonnegative scores and confidence in (0,1) required")
    rank = int((Decimal(str(confidence)) * (len(scores)+1)).to_integral_value(rounding=ROUND_CEILING))
    return float(np.partition(scores, rank-1)[rank-1]) if rank <= len(scores) else float("inf")


def prediction_intervals(prediction, quantile):
    prediction = vector(prediction, "prediction")
    if np.isnan(quantile) or quantile < 0:
        raise ValueError("Quantile must be nonnegative")
    return prediction - quantile, prediction + quantile


def evaluate(cal_reference, cal_prediction, test_reference, test_prediction, levels=(0.8,0.9,0.95)):
    scores = absolute_residuals(cal_reference, cal_prediction)
    errors = absolute_residuals(test_reference, test_prediction)
    if not len(levels):
        raise ValueError("At least one level required")
    rows = []
    for level in levels:
        q = conformal_quantile(scores, level)
        lo, hi = prediction_intervals(test_prediction, q)
        rows.append(dict(target_coverage=float(level),
            empirical_coverage=float(np.mean((np.asarray(test_reference)>=lo)&(np.asarray(test_reference)<=hi))),
            average_interval_width=2*q, quantile=q, energy_mae=float(errors.mean()),
            energy_rmse=float(np.sqrt(np.mean(errors**2))), n_calibration=len(scores), n_test=len(errors)))
    return rows
