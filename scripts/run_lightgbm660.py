"""Shared cached-array I/O only; no tree-model training is included.

The historical filename is retained for research-script import compatibility.
This module reads existing feature/label files; it does not build a dataset.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
TASKS = ('hypotension', 'low_depth_of_anesthesia', 'heart_rate_out_of_range',
         'hypoxemia', 'monitor_recorded_low_temperature', 'hypocapnia',
         'sustained_elevated_peak_airway_pressure')
HORIZONS = (5, 10, 15)
FOLDS = (0, 1, 2, 3)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("wb") as destination:
        np.savez_compressed(destination, **arrays)
    temporary.replace(path)


def save_csv(path, rows):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sources(args):
    root = args.project_root.resolve()
    ca84 = root / "reports/phase6ca84_v7_scale_source_task_horizon_sparse_moe_v1/cache"
    reference = root / "reports/phase6ca84ro_v7_matched_seed_frozen_outer_v1/M12R-SingleExpert-Seed02"
    defaults = {
        "development_features": root / "reports/phase6ca82_v7_protected_lowdoa_multiscale_residual_v1/cache/CA82_DEVELOPMENT_MULTISCALE_FEATURES.npy",
        "development_groups": ca84 / "CA84_DEVELOPMENT_ANCHOR_GROUPS.npz",
        "validation_features": ca84 / "validation/CA84_MULTISCALE_FEATURES.npy",
        "validation_reference": reference / "CA84O_VALIDATION_PREDICTIONS.npz",
        "test_features": ca84 / "test/CA84_MULTISCALE_FEATURES.npy",
        "test_reference": reference / "CA84O_TEST_PREDICTIONS.npz",
    }
    return {key: (getattr(args, key) or value).resolve() for key, value in defaults.items()}


def validate_labels(labels, masks, context):
    require(np.isin(masks, (0, 1)).all(), f"Non-binary {context} masks")
    valid = masks > 0
    invalid_eligible = valid & ~np.isin(labels, (0, 1))
    if invalid_eligible.any():
        values, counts = np.unique(labels[invalid_eligible], return_counts=True)
        details = [(repr(value), int(count)) for value, count in zip(values[:10], counts[:10])]
        raise RuntimeError(f"Non-binary eligible {context} labels: {details}; inspect the source cache")
    ignored = int(np.count_nonzero(~valid & ~np.isin(labels, (0, 1))))
    print(f"Label audit ({context}): eligible={int(valid.sum())} masked={int((~valid).sum())} "
          f"masked_nonbinary_ignored={ignored}", flush=True)


def development(paths):
    for key in ("development_features", "development_groups"):
        require(paths[key].is_file(), f"Missing {key}: {paths[key]}")
    x = np.load(paths["development_features"], mmap_mode="r", allow_pickle=False)
    with np.load(paths["development_groups"], allow_pickle=False) as archive:
        groups = {key: np.asarray(archive[key]) for key in (
            "representative_index", "group_fold", "labels", "masks"
        )}
    count = len(groups["representative_index"])
    require(x.shape[1:] == (3, 220), f"Expected [N,3,220] development features, got {x.shape}")
    require(groups["labels"].shape == groups["masks"].shape == (count, 7, 3), "Expected grouped labels/masks [G,7,3]")
    require(groups["group_fold"].shape == (count,), "Invalid grouped folds")
    require(set(np.unique(groups["group_fold"]).tolist()) == set(FOLDS), "Expected original four development folds 0,1,2,3")
    indices = groups["representative_index"]
    require(np.issubdtype(indices.dtype, np.integer), "Representative indices must be integers")
    require(len(np.unique(indices)) == count and indices.min() >= 0 and indices.max() < len(x), "Invalid representative indices")
    validate_labels(groups["labels"], groups["masks"], "development")
    return x, groups


def outer(paths, role):
    for key in (role + "_features", role + "_reference"):
        require(paths[key].is_file(), f"Missing {key}: {paths[key]}")
    x = np.load(paths[role + "_features"], mmap_mode="r", allow_pickle=False)
    with np.load(paths[role + "_reference"], allow_pickle=False) as archive:
        reference = {key: np.asarray(archive[key]) for key in (
            "global_position", "case_id", "anchor_seconds", "labels", "masks"
        )}
        reference["horizons"] = np.asarray(archive["horizons" if "horizons" in archive else "horizon_minutes"])
        if "subject_id" in archive:
            reference["subject_id"] = np.asarray(archive["subject_id"])
    count = len(reference["global_position"])
    require(x.shape == (count, 3, 220), f"{role}: feature/reference geometry mismatch: {x.shape}")
    require(reference["labels"].shape == reference["masks"].shape == (count, 7), f"{role}: label geometry mismatch")
    require(all(len(value) == count for value in reference.values()), f"{role}: metadata row mismatch")
    require(len(np.unique(reference["global_position"])) == count, f"{role}: duplicate global positions")
    require(np.isin(reference["horizons"], HORIZONS).all(), f"{role}: invalid horizons")
    validate_labels(reference["labels"], reference["masks"], role)
    return x, reference


def matrix(x, indices):
    result = np.asarray(x[indices], dtype=np.float32).reshape(len(indices), 660)
    require(not np.isinf(result).any(), "Infinite feature value; inspect the frozen cache")
    return result


def cell_metrics(reference, probability, role):
    from sklearn.metrics import average_precision_score, roc_auc_score
    rows = []
    for task, name in enumerate(TASKS):
        for horizon in HORIZONS:
            selected = (reference["horizons"] == horizon) & (reference["masks"][:, task] > 0)
            y, p = reference["labels"][selected, task], probability[selected, task]
            require(len(y) > 0 and set(np.unique(y).tolist()) == {0, 1}, f"One-class {role} cell: {name}/H{horizon}")
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid predicted probability")
            rows.append(dict(role=role, task_id=name, horizon_minutes=horizon,
                             eligible_rows=len(y), positive_rows=int(y.sum()),
                             average_precision=float(average_precision_score(y, p)),
                             auroc=float(roc_auc_score(y, p)),
                             brier_score=float(np.mean((p - y) ** 2))))
    return rows
