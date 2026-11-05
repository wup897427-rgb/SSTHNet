"""Unpenalised calibration diagnostics; never used to update test predictions."""
import math

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import expit

EPS = 1e-6


def logits(probability):
    p = np.clip(np.asarray(probability, dtype=float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def arrays(labels, x, weights=None):
    y, x = np.asarray(labels, dtype=float), np.asarray(x, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if y.ndim != 1 or y.shape != x.shape or w.shape != y.shape:
        raise ValueError('Calibration arrays must be aligned one-dimensional vectors')
    if not (np.isfinite(y).all() and np.isfinite(x).all() and np.isfinite(w).all()):
        raise ValueError('Nonfinite calibration input')
    if not np.isin(y, (0, 1)).all() or (w < 0).any():
        raise ValueError('Invalid binary outcomes or frequency weights')
    keep = w > 0
    return y[keep], x[keep], w[keep]


def offset_intercept(labels, x, weights=None):
    y, x, w = arrays(labels, x, weights)
    if not len(y) or np.unique(y).size < 2:
        return float('nan')
    def score(a):
        return float(np.dot(w, y - expit(x + a)))
    return float(brentq(score, -100, 100, xtol=1e-11))


def joint_slope(labels, x, weights=None, verify=False, start=None):
    """Damped Newton fit, with an independent BFGS check for point estimates."""
    y, x, w = arrays(labels, x, weights)
    result = dict(intercept=float('nan'), slope=float('nan'), status='no_both_classes',
                  iterations=0, scaled_score=float('nan'), information_condition=float('nan'),
                  independent_solver_agreement=False, independent_slope=float('nan'))
    if not len(y) or np.unique(y).size < 2:
        return result
    center = float(np.average(x, weights=w))
    scale = float(np.sqrt(np.average((x - center) ** 2, weights=w)))
    if scale <= 1e-12:
        result['status'] = 'constant_logits'
        return result
    negative, positive = x[y == 0], x[y == 1]
    if negative.max() <= positive.min() or positive.max() <= negative.min():
        result['status'] = 'complete_or_quasi_separation'
        return result
    z = (x - center) / scale
    total = float(w.sum())
    prevalence = float(np.dot(w, y) / total)
    beta = np.array([math.log(prevalence / (1 - prevalence)), 0.])
    if start is not None and np.isfinite(start).all():
        intercept, slope = start
        beta = np.array([intercept + slope * center, slope * scale], dtype=float)

    def loss(b):
        eta = b[0] + b[1] * z
        return float(np.dot(w, np.logaddexp(0, eta) - y * eta) / total)

    def gradient(b):
        residual = w * (expit(b[0] + b[1] * z) - y)
        return np.array([residual.sum(), np.dot(residual, z)]) / total

    for iteration in range(1, 201):
        fitted = expit(beta[0] + beta[1] * z)
        variance = w * fitted * (1 - fitted)
        information = np.array([[variance.sum(), np.dot(variance, z)],
                                [np.dot(variance, z), np.dot(variance, z * z)]]) / total
        g = gradient(beta)
        condition = float(np.linalg.cond(information))
        result.update(iterations=iteration, scaled_score=float(np.max(np.abs(g))),
                      information_condition=condition)
        if not np.isfinite(condition) or condition > 1e12:
            result['status'] = 'ill_conditioned_information'
            return result
        if np.max(np.abs(g)) <= 1e-10:
            result['status'] = 'converged'
            break
        step = np.linalg.solve(information, g)
        old = loss(beta)
        accepted = False
        for power in range(40):
            candidate = beta - step * (0.5 ** power)
            if np.isfinite(candidate).all() and loss(candidate) <= old + 1e-14:
                beta = candidate
                accepted = True
                break
        if not accepted:
            result['status'] = 'line_search_failed'
            return result
    else:
        result['status'] = 'iteration_limit'
        return result
    slope = float(beta[1] / scale)
    intercept = float(beta[0] - slope * center)
    if verify:
        reference = minimize(loss, np.array([math.log(prevalence / (1 - prevalence)), 0.]),
                             jac=gradient, method='BFGS', options={'gtol': 1e-10, 'maxiter': 1000})
        reference_slope = float(reference.x[1] / scale)
        result['independent_slope'] = reference_slope
        agrees = (np.max(np.abs(gradient(reference.x))) <= 1e-7
                  and np.allclose(beta, reference.x, rtol=1e-4, atol=1e-5))
        result['independent_solver_agreement'] = bool(agrees)
        if not agrees:
            result['status'] = 'independent_solver_disagreement'
            return result
    result.update(intercept=intercept, slope=slope)
    return result


def ece(labels, probability, weights=None):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probability, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if not w.sum():
        return float('nan')
    bins = np.minimum((p * 10).astype(int), 9)
    return float(sum(abs(np.dot(w[bins == b], p[bins == b] - y[bins == b]))
                     for b in range(10)) / w.sum())


def metrics(labels, probability, weights=None, verify=False, start=None):
    y = np.asarray(labels, dtype=float)
    p = np.asarray(probability, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid eligible probability')
    x = logits(p)
    y, _, w = arrays(y, x, weights)
    keep = np.ones(len(p), dtype=bool) if weights is None else np.asarray(weights) > 0
    p, x = p[keep], x[keep]
    if not len(y):
        raise ValueError('No positive frequency weights')
    slope = joint_slope(y, x, w, verify=verify, start=start)
    clipped = np.clip(p, EPS, 1 - EPS)
    return dict(brier_score=float(np.average((p - y) ** 2, weights=w)),
                log_loss=float(np.average(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped), weights=w)),
                ece=ece(y, p, w), calibration_in_the_large=offset_intercept(y, x, w),
                calibration_slope=slope['slope'], **{f'slope_{k}': v for k, v in slope.items() if k != 'slope'})


def macro(rows, fields):
    """Do not silently change a 21-cell estimand to a finite-subset mean."""
    output = {}
    for field in fields:
        values = np.array([r[field] for r in rows], dtype=float)
        count = int(np.isfinite(values).sum())
        output[field + '_estimable_cells'] = count
        output[field] = float(values.mean()) if len(values) == 21 and count == 21 else float('nan')
    return output
