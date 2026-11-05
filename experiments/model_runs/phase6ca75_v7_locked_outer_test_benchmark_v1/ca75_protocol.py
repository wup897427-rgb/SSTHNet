"""Metric protocol only; dataset construction and splitting omitted."""
from __future__ import annotations
from typing import Iterable
import numpy as np
TASKS = ('hypotension', 'low_depth_of_anesthesia', 'heart_rate_out_of_range',
         'hypoxemia', 'monitor_recorded_low_temperature', 'hypocapnia',
         'sustained_elevated_peak_airway_pressure')
HORIZONS = (5, 10, 15)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def binary_counts(labels: np.ndarray, probability: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    predicted = probability >= threshold
    positive = labels == 1
    return (
        int(np.count_nonzero(predicted & positive)),
        int(np.count_nonzero(predicted & ~positive)),
        int(np.count_nonzero(~predicted & ~positive)),
        int(np.count_nonzero(~predicted & positive)),
    )


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return float(precision), float(recall), float(f1)


def best_f1_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    probability = np.asarray(probability, dtype=np.float64)
    require(len(labels) == len(probability) and len(np.unique(labels)) == 2, "CA75 threshold cell is not estimable")
    order = np.argsort(-probability, kind="stable")
    y = labels[order].astype(np.int64)
    score = probability[order]
    total = int(y.sum())
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    boundary = np.r_[score[1:] != score[:-1], True]
    denominator = tp + fp + total
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=np.float64), where=denominator > 0)
    f1[~boundary] = -1.0
    return float(score[int(np.argmax(f1))])


def validate_threshold_rows(rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    observed = {(str(row["task_id"]), int(row["horizon_minutes"])) for row in materialized}
    require(len(materialized) == 21 and observed == set(expected_cell_keys()), "CA75 threshold geometry changed")
    require(all(0.0 <= float(row["threshold"]) <= 1.0 for row in materialized), "CA75 threshold out of range")


def expected_cell_keys() -> tuple[tuple[str, int], ...]:
    return tuple((task, horizon) for task in TASKS for horizon in HORIZONS)
