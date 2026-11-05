"""Metric summaries only; dataset and unrelated benchmark code omitted."""
from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from ca75_protocol import (TASKS, HORIZONS, require, binary_counts,
    precision_recall_f1, best_f1_threshold, validate_threshold_rows)


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for slot in range(bins):
        selected = (probability >= edges[slot]) & (probability < edges[slot + 1] if slot + 1 < bins else probability <= 1.0)
        if selected.any():
            result += float(selected.mean()) * abs(float(labels[selected].mean()) - float(probability[selected].mean()))
    return float(result)


def threshold_rows(prediction: dict[str, np.ndarray], arm: str) -> list[dict[str, Any]]:
    rows = []
    for task_slot, task in enumerate(TASKS):
        for horizon in HORIZONS:
            valid = (prediction["horizons"] == horizon) & (prediction["masks"][:, task_slot] > 0)
            labels = prediction["labels"][valid, task_slot].astype(np.int8)
            probability = prediction["probability"][valid, task_slot]
            rows.append({
                "arm": arm, "task_id": task, "horizon_minutes": horizon,
                "eligible_rows": len(labels), "positive_rows": int(labels.sum()),
                "threshold": best_f1_threshold(labels, probability),
                "threshold_source": "locked historical 10 percent validation role",
                "architecture_selection_allowed": False, "test_labels_read": False,
            })
    validate_threshold_rows(rows)
    return rows


def metric_rows(
    prediction: dict[str, np.ndarray], arm: str, role: str, thresholds: dict[tuple[str, int], float]
) -> list[dict[str, Any]]:
    rows = []
    for task_slot, task in enumerate(TASKS):
        for horizon in HORIZONS:
            valid = (prediction["horizons"] == horizon) & (prediction["masks"][:, task_slot] > 0)
            labels = prediction["labels"][valid, task_slot].astype(np.int8)
            probability = prediction["probability"][valid, task_slot].astype(np.float64)
            require(len(np.unique(labels)) == 2, f"CA75 metric cell not estimable: {arm}/{role}/{task}/{horizon}")
            threshold = float(thresholds[(task, horizon)])
            tp, fp, tn, fn = binary_counts(labels, probability, threshold)
            precision, recall, f1 = precision_recall_f1(tp, fp, fn)
            ftp, ffp, ftn, ffn = binary_counts(labels, probability, 0.5)
            fixed_precision, fixed_recall, fixed_f1 = precision_recall_f1(ftp, ffp, ffn)
            rows.append({
                "arm": arm, "role": role, "task_id": task, "horizon_minutes": horizon,
                "eligible_rows": len(labels), "positive_rows": int(labels.sum()), "prevalence": float(labels.mean()),
                "auprc": float(average_precision_score(labels, probability)),
                "auroc": float(roc_auc_score(labels, probability)),
                "brier": float(brier_score_loss(labels, probability)),
                "ece_10bin": expected_calibration_error(labels, probability),
                "validation_locked_threshold": threshold,
                "precision": precision, "recall": recall, "f1": f1,
                "fixed_0_5_precision": fixed_precision, "fixed_0_5_recall": fixed_recall, "fixed_0_5_f1": fixed_f1,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "test_labels_read": role == "test",
            })
    require(len(rows) == 21, "CA75 metric geometry changed")
    return rows


def summary_row(rows: list[dict[str, Any]], arm: str, role: str, section: str) -> dict[str, Any]:
    micro_tp = int(sum(int(row["tp"]) for row in rows))
    micro_fp = int(sum(int(row["fp"]) for row in rows))
    micro_fn = int(sum(int(row["fn"]) for row in rows))
    micro_precision, micro_recall, micro_f1 = precision_recall_f1(micro_tp, micro_fp, micro_fn)
    return {
        "arm": arm, "table_section": section, "role": role,
        "equal_cell_macro_auprc": float(np.mean([row["auprc"] for row in rows])),
        "equal_cell_macro_auroc": float(np.mean([row["auroc"] for row in rows])),
        "equal_cell_macro_f1": float(np.mean([row["f1"] for row in rows])),
        "equal_cell_macro_precision": float(np.mean([row["precision"] for row in rows])),
        "equal_cell_macro_recall": float(np.mean([row["recall"] for row in rows])),
        "micro_f1": micro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "fixed_0_5_equal_cell_macro_f1": float(np.mean([row["fixed_0_5_f1"] for row in rows])),
        "equal_cell_macro_brier": float(np.mean([row["brier"] for row in rows])),
        "equal_cell_macro_ece_10bin": float(np.mean([row["ece_10bin"] for row in rows])),
        "test_labels_read": role == "test",
    }
