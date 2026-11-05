"""Training-only helpers extracted without changing function bodies."""
from __future__ import annotations
import numpy as np
TASKS = ('hypotension', 'low_depth_of_anesthesia', 'heart_rate_out_of_range',
         'hypoxemia', 'monitor_recorded_low_temperature', 'hypocapnia',
         'sustained_elevated_peak_airway_pressure')
HORIZONS = (5, 10, 15)
def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fit_scaler(features: np.ndarray, representative: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = 0
    total = np.zeros((3, 22, 10), dtype=np.float64)
    total_square = np.zeros_like(total)
    for start in range(0, len(representative), 10000):
        values = np.asarray(features[representative[start : start + 10000]], dtype=np.float32).reshape(-1, 3, 22, 10)
        total += values.sum(axis=0, dtype=np.float64)
        total_square += np.square(values, dtype=np.float64).sum(axis=0, dtype=np.float64)
        count += len(values)
    mean = total / count
    variance = np.maximum(total_square / count - np.square(mean), 1.0e-6)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def positive_weights(labels: np.ndarray, masks: np.ndarray, groups: np.ndarray) -> np.ndarray:
    result = np.ones((len(TASKS), len(HORIZONS)), dtype=np.float32)
    for task in range(len(TASKS)):
        for horizon in range(len(HORIZONS)):
            valid = masks[groups, task, horizon] > 0
            positives = int(labels[groups, task, horizon][valid].sum())
            negatives = int(valid.sum()) - positives
            require(positives > 0 and negatives > 0, f"CA84 train cell not estimable: {TASKS[task]}/H{HORIZONS[horizon]}")
            result[task, horizon] = float(np.clip(np.sqrt(negatives / positives), 1.0, 30.0))
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
