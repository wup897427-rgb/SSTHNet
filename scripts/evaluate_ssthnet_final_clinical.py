#!/usr/bin/env python3
"""Evaluate frozen final-model predictions against Clinical LR; never train.

Uses the original observable-state cache, not imputed clinical rule scores.
All confidence intervals condition on the five fixed fitted SSTHNet seeds.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import sys
import time

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import bootstrap_ssthnet_five_seed_ablations as boot
    from . import ssthnet_calibration_qc as qc
except ImportError:
    import bootstrap_ssthnet_five_seed_ablations as boot
    import ssthnet_calibration_qc as qc

VERSION = "ssthnet_final_clinical_evaluation_v1"
HISTORY = "reports/ssthnet_history_window_test_evaluation_v1"
CLINICAL = "reports/phase6bja2t_v7_simple_clinical_comparators_one_time_test_v1"
STATUS_ARCHIVE = (
    "deliverables/BJA_SERVER1_FIGURE_TABLE_EVIDENCE_20260909_174328/reports/"
    "phase6bja5b_v7_current_normal_uncertainty_v1"
)
SEEDS = (1, 2, 3, 4, 5)
HORIZONS = (5, 10, 15)
POPULATIONS = ("overall", "current_normal")
MODELS = ("Clinical_LR", "SSTHNet")
RANK_FIELDS = ("Macro_AP", "Macro_AUROC")
CAL_FIELDS = ("brier_score", "log_loss", "ece", "calibration_in_the_large", "calibration_slope")
METHODS = ("raw", "validation_intercept_only")
CLINICAL_KEY = "target_only_clinical_logistic"
EXPECTED_HELPERS = {
    "bootstrap_ssthnet_five_seed_ablations.py": "0d96acfc608b1e13892004dcb5aa7dff0fb0da26714fbf3dbd81d2b55882643f",
    "ssthnet_calibration_qc.py": "4536219e4ba0e9936da539a6bd4f0dab21cf8f7cc4e771d3d84ef8362aa5805a",
}
require = boot.require


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {key: value[key] for key in value.files}


def record(path):
    path = Path(path).resolve()
    require(path.is_file(), "Missing existing evidence: " + str(path))
    return dict(path=str(path), sha256=boot.sha256(path))


def verify(entry):
    path = Path(entry["path"])
    require(path.is_file() and boot.sha256(path) == entry["sha256"], "Frozen file changed: " + str(path))
    return path


def clean(value):
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    boot.save_json(Path(path), clean(value))


def save_csv(path, rows):
    require(bool(rows), "Refusing an empty result table")
    boot.save_csv(Path(path), [clean(row) for row in rows])


def save_npz(path, **values):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def subset_indices(superset, subset):
    superset, subset = np.asarray(superset), np.asarray(subset)
    require(len(np.unique(superset)) == len(superset), "Duplicate source row IDs")
    require(len(np.unique(subset)) == len(subset), "Duplicate requested row IDs")
    order = np.argsort(superset)
    slots = np.searchsorted(superset[order], subset)
    require(bool((slots < len(order)).all()), "Requested row absent from source")
    indices = order[slots]
    require(np.array_equal(superset[indices], subset), "Requested row absent from source")
    return indices


def eligible_selection(reference, task, horizon, status=None):
    selected = (reference["horizons"] == horizon) & (reference["masks"][:, task] > 0)
    if status is not None:
        require(status.shape == reference["masks"].shape, "Status geometry changed")
        selected &= status[:, task] == 0
    return selected


def finite_probability(value, reference, name):
    p = np.asarray(value["probability"])
    require(p.shape == reference["labels"].shape, name + ": wrong probability shape")
    eligible = reference["masks"] > 0
    require(np.isfinite(p[eligible]).all() and ((p[eligible] >= 0) & (p[eligible] <= 1)).all(),
            name + ": invalid eligible probabilities")


def patient_names(reference, mapping_rows):
    lookup = {int(row["case_id"]): str(row["subject_id"]) for row in mapping_rows}
    require(len(lookup) == len(mapping_rows), "Duplicate case-to-patient mappings")
    require(all(int(case) in lookup for case in reference["case_id"]), "Unmapped operation")
    names = np.asarray([lookup[int(case)] for case in reference["case_id"]])
    require(not any(name.lower().strip() in ("", "nan", "none", "null", "-1") for name in names),
            "Missing patient identifiers")
    return names


def prepare(args):
    root = args.project_root.resolve()
    history = root / HISTORY
    validation_lock = read_json(history / "VALIDATION_LOCK.json")
    test_lock = read_json(history / "TEST_PREDICTION_LOCK.json")
    parent = read_json(history / "protocol.json")
    parent_hash = boot.canonical_hash(parent)
    require(parent["epoch"] == 15 and parent["candidate"] == "A_constant_weighted", "Wrong final model")
    require(tuple(parent["seeds"]) == SEEDS and tuple(parent["folds"]) == (0, 1, 2, 3), "Wrong fixed seed/fold set")
    require(validation_lock["contract_hash"] == test_lock["contract_hash"] == parent_hash, "Parent lock mismatch")
    require(boot.sha256(history / "VALIDATION_LOCK.json") == test_lock["validation_lock_sha256"],
            "Parent validation lock changed")
    require(read_json(history / "COMPLETE.json")["contract_hash"] == parent_hash, "Parent evaluation incomplete")

    paths = dict(history_protocol=history / "protocol.json", history_completion=history / "COMPLETE.json",
        validation_lock=history / "VALIDATION_LOCK.json", test_lock=history / "TEST_PREDICTION_LOCK.json",
        prior_seed_results=history / "TEST_SEED_RESULTS.csv",
        clinical_predictions=root / CLINICAL / "BJA2T_TEST_PREDICTIONS.npz",
        clinical_completion=root / CLINICAL / "PHASE6BJA2T_TEST_COMPLETE.json",
        clinical_protocol=root / "PHASE6BJA2_SIMPLE_CLINICAL_COMPARATOR_PROTOCOL.json",
        current_status=root / STATUS_ARCHIVE / "BJA5B_CURRENT_NORMAL_STATUS.npz",
        status_completion=root / STATUS_ARCHIVE / "PHASE6BJA5B_CURRENT_NORMAL_UNCERTAINTY_COMPLETE.json",
        current_support=root / "reports/phase6bja5_v7_current_normal_future_event_v1/BJA5_CURRENT_NORMAL_DENOMINATORS.csv",
        patient_mapping=root / "reports/phase6ca51_v7_iaenet_single_split_bridge_v1/CA51_CASE_SPLIT.csv",
        runner=Path(__file__), bootstrap_engine=Path(boot.__file__), calibration_engine=Path(qc.__file__))
    inputs = {key: record(path) for key, path in paths.items()}
    for module in (boot, qc):
        path = Path(module.__file__)
        require(boot.sha256(path) == EXPECTED_HELPERS[path.name], "Frozen analysis helper changed: " + path.name)
    require(inputs["clinical_predictions"]["sha256"] == test_lock["patient_mapping"]["sha256"],
            "Clinical archive differs from final-model patient mapping")
    require(read_json(paths["clinical_completion"])["predictions_sha256"] == inputs["clinical_predictions"]["sha256"],
            "Clinical prediction provenance changed")
    old = read_json(paths["status_completion"])["source_hashes"]
    require(old["current_normal_status"] == inputs["current_status"]["sha256"], "Unverified observable-state cache")
    require(old["bja2t_predictions"] == inputs["clinical_predictions"]["sha256"], "Status and clinical source differ")
    require(old["bja5b_bja5_denominators"] == inputs["current_support"]["sha256"], "Original support table changed")
    clinical = read_json(paths["clinical_protocol"])
    require(tuple(task["task_id"] for task in clinical["tasks"]) == boot.TASKS, "Clinical target order changed")
    logistic = clinical["comparators"][CLINICAL_KEY]
    require(logistic["penalty"] == "l2" and logistic["C"] == 1.0 and len(logistic["features"]) == 6,
            "Wrong Clinical LR specification")
    models = {}
    for seed in SEEDS:
        key = f"120_300_600s/seed{seed:02d}"
        entries = dict(test=test_lock["predictions"][key],
                       validation=validation_lock["models"][key]["validation_prediction"],
                       validation_thresholds=validation_lock["models"][key]["thresholds"])
        models[str(seed)] = {name: record(verify(entry)) for name, entry in entries.items()}
    protocol = dict(version=VERSION, inputs=inputs, models=models, final_model="non-EMA epoch15; all three windows",
        parent_contract_hash=parent_hash, seeds=list(SEEDS), folds=[0, 1, 2, 3], targets=list(boot.TASKS),
        horizons_minutes=list(HORIZONS), populations=list(POPULATIONS), models_compared=list(MODELS),
        clinical_lr_policy="reuse one frozen development-fitted Clinical LR prediction set; no refit or seed selection",
        current_normal="original frozen observable target state == 0; never imputed current_value/current_state scores",
        aggregation="four-fold probabilities already averaged within seed; 21-cell metrics then arithmetic mean of all five seeds",
        discrimination=list(RANK_FIELDS), calibration=list(CAL_FIELDS), calibration_clip=qc.EPS,
        calibration_fit="eligible validation labels only, separate intercept per seed/target/horizon; slope fixed at 1",
        diagnostic_fit="test CITL and free-intercept calibration slope are diagnostics only; never prediction updates",
        uncertainty="paired patient-cluster percentile 95% CI conditional on the fixed five fitted seeds and validation intercepts",
        resampling="one patient count vector shared by both models, all seeds, horizons, targets and populations",
        missing_policy="no redraw or finite-cell averaging; preserve the 21-cell/five-seed estimand",
        interval_gate="at least 100 and 95% estimable draws per reported metric",
        replicates=args.replicates, base_seed=args.base_seed, chunk_size=args.chunk_size,
        ranking_units="percent; model differences in percentage points", workers=args.workers,
        test_role="posthoc additional evaluation on an already viewed internal-test partition",
        training=False, model_selection=False, threshold_fitting=False, probability_ensemble_across_seeds=False,
        versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__))
    output = (args.out or root / "reports" / VERSION).resolve()
    output.mkdir(parents=True, exist_ok=True)
    target = output / "protocol.json"
    if target.exists():
        require(read_json(target) == protocol, "Resume protocol changed; preserve this output and choose a new version")
    else:
        require(not list(output.iterdir()), "New output directory must be empty")
        save_json(target, protocol)
    print("PREFLIGHT_VERIFIED", boot.canonical_hash(protocol), "fixed_seeds=1,2,3,4,5 no_training=True", flush=True)
    return protocol, output


def fit_validation_intercepts(protocol, output):
    intercepts, validation_subjects, reference = {}, set(), None
    mapping = boot.read_csv(verify(protocol["inputs"]["patient_mapping"]))
    for seed in SEEDS:
        data = boot.read_npz(verify(protocol["models"][str(seed)]["validation"]))
        if reference is None:
            reference = data
        boot.check_identity(reference, data, f"validation seed {seed}")
        finite_probability(data, reference, "validation")
        validation_subjects.update(patient_names(data, mapping).tolist())
        for task, name in enumerate(boot.TASKS):
            for horizon in HORIZONS:
                selected = eligible_selection(data, task, horizon)
                alpha = qc.offset_intercept(data["labels"][selected, task], qc.logits(data["probability"][selected, task]))
                require(np.isfinite(alpha), f"Unestimable validation intercept: {seed}/{name}/{horizon}")
                intercepts[f"{seed}/{name}/{horizon}"] = alpha
    lock = dict(contract_hash=boot.canonical_hash(protocol), intercepts=intercepts,
        fitting_population="validation only", test_labels_used_for_fitting=False,
        input_sha256={str(seed): protocol["models"][str(seed)]["validation"]["sha256"] for seed in SEEDS})
    path = output / "CALIBRATION_VALIDATION_LOCK.json"
    if path.exists():
        require(read_json(path) == lock, "Validation-fitted intercepts changed")
    else:
        save_json(path, lock)
    return intercepts, validation_subjects


def build_context(protocol, output):
    intercepts, validation_subjects = fit_validation_intercepts(protocol, output)
    clinical = read_npz(verify(protocol["inputs"]["clinical_predictions"]))
    prediction = [boot.read_npz(verify(protocol["models"][str(seed)]["test"])) for seed in SEEDS]
    reference = prediction[0]
    for seed, data in zip(SEEDS, prediction):
        boot.check_identity(reference, data, f"test seed {seed}")
        finite_probability(data, reference, "test")
    indices = subset_indices(clinical["global_position"], reference["global_position"])
    aligned = {key: clinical[key][indices] for key in (*boot.IDENTITY, "subject_id")}
    boot.check_identity(reference, aligned, "Clinical LR")
    clinical_probability = clinical[CLINICAL_KEY][indices]
    finite_probability(dict(probability=clinical_probability), reference, "Clinical LR")
    inverse, patients = boot.map_patients(reference, clinical)
    mapped_names = patient_names(reference, boot.read_csv(verify(protocol["inputs"]["patient_mapping"])))
    require(np.array_equal(mapped_names, aligned["subject_id"].astype(str)), "Independent patient mapping disagrees")
    require(not set(mapped_names) & validation_subjects, "Validation/test patient overlap")
    state = read_npz(verify(protocol["inputs"]["current_status"]))
    idx = subset_indices(state["global_position"], reference["global_position"])
    status = state["status"][idx]
    require(status.shape == reference["labels"].shape and np.isin(status, (-1, 0, 1)).all(), "Invalid original target status")

    support_rows, selections, support, cache = [], {}, {}, {}
    old_support = {(r["task_id"], int(r["horizon_minutes"])): r
                   for r in boot.read_csv(verify(protocol["inputs"]["current_support"]))}
    for population in POPULATIONS:
        selections[population], support[population] = [], []
        for task, name in enumerate(boot.TASKS):
            for horizon in HORIZONS:
                selected = eligible_selection(reference, task, horizon, status if population == "current_normal" else None)
                labels, ids = reference["labels"][selected, task], inverse[selected]
                require(set(np.unique(labels)) == {0, 1}, f"Original cell lacks both classes: {population}/{name}/{horizon}")
                selections[population].append(selected)
                support[population].append((np.bincount(ids[labels == 1], minlength=patients),
                                            np.bincount(ids[labels == 0], minlength=patients)))
                row = dict(population=population, task_id=name, horizon_minutes=horizon,
                    records=int(selected.sum()), positive_records=int(labels.sum()),
                    patients=len(np.unique(ids)), positive_patients=len(np.unique(ids[labels == 1])))
                support_rows.append(row)
                expected = old_support[(name, horizon)]
                if population == "current_normal":
                    for field, old_field in (("records", "current_normal_rows"), ("patients", "current_normal_subjects"),
                        ("positive_records", "current_normal_future_positive_rows"),
                        ("positive_patients", "current_normal_future_positive_subjects")):
                        require(row[field] == int(expected[old_field]), "Original current-normal support changed: " + old_field)
                else:
                    require(row["records"] == int(expected["eligible_rows"]), "Overall support changed")
        cache[population] = []
        for probabilities in ([clinical_probability], [p["probability"] for p in prediction]):
            cache[population].append([[boot.prepare_cell(reference["labels"][sel, cell // 3], p[sel, cell // 3], inverse[sel], .5)
                for cell, sel in enumerate(selections[population])] for p in probabilities])

    counts = np.ones(patients)
    point, seed_rows, cell_rows = [], [], []
    expected_seed = {int(r["seed"]): r for r in boot.read_csv(verify(protocol["inputs"]["prior_seed_results"]))
                     if r["configuration"] == "120_300_600s"}
    for population in POPULATIONS:
        models = []
        for model, seeds in zip(MODELS, cache[population]):
            values = []
            for slot, cells in enumerate(seeds):
                result = np.asarray([boot.weighted_metrics(cell, counts)[:2] for cell in cells])
                value = result.mean(axis=0)
                values.append(value)
                seed = SEEDS[slot] if model == "SSTHNet" else "fixed"
                seed_rows.append(dict(population=population, model=model, seed=seed,
                    Macro_AP=100 * value[0], Macro_AUROC=100 * value[1]))
                for cell, metrics in enumerate(result):
                    cell_rows.append(dict(population=population, model=model, seed=seed,
                        task_id=boot.TASKS[cell // 3], horizon_minutes=HORIZONS[cell % 3],
                        AP=100 * metrics[0], AUROC=100 * metrics[1]))
                if population == "overall" and model == "SSTHNet":
                    expected = expected_seed[seed]
                    require(np.allclose(value, [float(expected[k]) for k in ("equal_cell_macro_auprc", "equal_cell_macro_auroc")],
                                        atol=1e-12, rtol=0), "Final-model point estimates do not reproduce prior results")
            models.append(np.mean(values, axis=0))
        point.append(models)
    point = np.asarray(point)
    summary = [dict(population=population, model=model, seeds=5 if model == "SSTHNet" else 1,
        Macro_AP=100 * point[pi, mi, 0], Macro_AUROC=100 * point[pi, mi, 1])
        for pi, population in enumerate(POPULATIONS) for mi, model in enumerate(MODELS)]
    save_csv(output / "DISCRIMINATION_SEED_RESULTS.csv", seed_rows)
    save_csv(output / "DISCRIMINATION_21CELL_RESULTS.csv", cell_rows)
    save_csv(output / "DISCRIMINATION_FIVE_SEED_SUMMARY.csv", summary)
    save_csv(output / "SUPPORT_21CELL.csv", support_rows)
    save_json(output / "COHORT_SUPPORT.json", dict(patients=patients, operations=len(np.unique(reference["case_id"])),
        records=len(inverse), eligible_labels=int((reference["masks"] > 0).sum()),
        current_normal_labels=int(((reference["masks"] > 0) & (status == 0)).sum()),
        unavailable_current_labels=int(((reference["masks"] > 0) & (status == -1)).sum())))

    calibration_cells, diagnostics, curves = [], [], []
    calibration_point = np.full((5, 21, 2, len(CAL_FIELDS)), np.nan)
    for slot, (seed, data) in enumerate(zip(SEEDS, prediction)):
        for cell, selected in enumerate(selections["overall"]):
            task, horizon = boot.TASKS[cell // 3], HORIZONS[cell % 3]
            y, raw, ids = data["labels"][selected, cell // 3], data["probability"][selected, cell // 3], inverse[selected]
            alpha = intercepts[f"{seed}/{task}/{horizon}"]
            updated = qc.expit(qc.logits(raw) + alpha)
            starts, ranking = [], []
            for mi, (method, p) in enumerate(zip(METHODS, (raw, updated))):
                result = qc.metrics(y, p, verify=True)
                calibration_point[slot, cell, mi] = [result[field] for field in CAL_FIELDS]
                starts.append((result["slope_intercept"], result["calibration_slope"]))
                ranking.append((average_precision_score(y, p), roc_auc_score(y, p)))
                diagnostics.append(dict(seed=seed, task_id=task, horizon_minutes=horizon, method=method,
                    validation_intercept=alpha, **result))
                for bin_number, selected_bin in enumerate(np.array_split(np.argsort(p, kind="stable"), 10), 1):
                    curves.append(dict(seed=seed, task_id=task, horizon_minutes=horizon, method=method, bin=bin_number,
                        records=len(selected_bin), predicted=float(p[selected_bin].mean()), observed=float(y[selected_bin].mean())))
            diagnostics[-1].update(AP_change=ranking[1][0] - ranking[0][0], AUROC_change=ranking[1][1] - ranking[0][1])
            diagnostics[-2].update(AP_change=0., AUROC_change=0.)
            calibration_cells.append((y, raw, updated, ids, *starts))
    save_csv(output / "CALIBRATION_21CELL_DIAGNOSTICS.csv", diagnostics)
    save_csv(output / "CALIBRATION_PER_SEED_CURVES.csv", curves)
    cal_seed = calibration_point.mean(axis=1)
    save_csv(output / "CALIBRATION_PER_SEED_MACRO.csv", [dict(seed=seed, method=method,
        **{field: cal_seed[slot, mi, fi] for fi, field in enumerate(CAL_FIELDS)})
        for slot, seed in enumerate(SEEDS) for mi, method in enumerate(METHODS)])
    cal_point = calibration_point.mean(axis=(0, 1))
    save_csv(output / "CALIBRATION_FIVE_SEED_MACRO.csv", [dict(method=method, metric=field, estimate=cal_point[mi, fi])
        for mi, method in enumerate(METHODS) for fi, field in enumerate(CAL_FIELDS)])
    print("POINTS_VERIFIED", json.dumps(clean(summary)), "patients", patients, flush=True)
    print("CALIBRATION_POINTS", json.dumps(clean(cal_point)), flush=True)
    return dict(cache=cache, support=support, patients=patients, point=point,
                calibration_cells=calibration_cells, calibration_point=cal_point)


def ranking_replicate(context, base_seed, index):
    counts, _ = boot.draw(base_seed, index, context["patients"])
    values, valid = np.full((2, 2, 2), np.nan), np.zeros(2, dtype=bool)
    for pi, population in enumerate(POPULATIONS):
        valid[pi] = all(np.dot(pos, counts) > 0 and np.dot(neg, counts) > 0 for pos, neg in context["support"][population])
        if valid[pi]:
            for mi, models in enumerate(context["cache"][population]):
                values[pi, mi] = np.mean([np.mean([boot.weighted_metrics(cell, counts)[:2] for cell in cells], axis=0)
                                        for cells in models], axis=0)
    require(np.isfinite(values[valid]).all(), "Supported ranking draw returned nonfinite metrics")
    return dict(values=values, valid=valid)


def calibration_replicate(context, base_seed, index):
    counts, _ = boot.draw(base_seed, index, context["patients"])
    values = np.full((5, 21, 2, len(CAL_FIELDS)), np.nan)
    supported = True
    for ci, (y, raw, updated, ids, raw_start, updated_start) in enumerate(context["calibration_cells"]):
        weights = counts[ids]
        if not weights[y == 1].sum() or not weights[y == 0].sum():
            supported = False
            continue
        for mi, (p, start) in enumerate(((raw, raw_start), (updated, updated_start))):
            result = qc.metrics(y, p, weights=weights, start=start)
            values[ci // 21, ci % 21, mi] = [result[field] for field in CAL_FIELDS]
    return dict(values=values.mean(axis=(0, 1)), valid=np.asarray(supported),
                slope_fits=np.isfinite(values[..., -1]).sum(axis=(0, 1)))


def confidence_row(point, values, attempted):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    enough = np.isfinite(point) and int(finite.sum()) >= max(100, .95 * attempted)
    lower, upper = np.quantile(values[finite], [.025, .975]) if enough else (np.nan, np.nan)
    return dict(estimate=float(point), ci95_lower=float(lower), ci95_upper=float(upper),
        attempted_replicates=attempted, estimable_replicates=int(finite.sum()),
        estimable_fraction=float(finite.mean()), interval_status="reported" if enough and np.isfinite(point) else "not_estimable")


def bootstrap_stage(context, protocol, output, phase):
    folder = output / (phase + "_bootstrap")
    folder.mkdir(exist_ok=True)
    fingerprint = boot.canonical_hash(dict(parent=boot.canonical_hash(protocol), phase=phase))
    ledger_path = folder / "CHUNK_MANIFEST.json"
    if ledger_path.exists():
        ledger = read_json(ledger_path)
        require(ledger["fingerprint"] == fingerprint, "Chunk ledger identity changed")
    else:
        ledger = dict(fingerprint=fingerprint, chunks={})
        save_json(ledger_path, ledger)
    function = ranking_replicate if phase == "discrimination" else calibration_replicate
    blocks, started, computed = [], time.monotonic(), 0
    with ThreadPoolExecutor(max_workers=protocol["workers"]) as pool:
        for start in range(1, protocol["replicates"] + 1, protocol["chunk_size"]):
            end = min(start + protocol["chunk_size"] - 1, protocol["replicates"])
            path = folder / f"CHUNK_{start:06d}_{end:06d}.npz"
            if path.exists():
                require(path.name in ledger["chunks"], "Uncommitted chunk; verify/recompute it before resuming: " + str(path))
                require(boot.sha256(path) == ledger["chunks"][path.name], "Saved chunk hash changed: " + str(path))
                block = read_npz(path)
                require(str(block.pop("fingerprint")) == fingerprint, "Saved chunk fingerprint changed")
                require(np.array_equal(block.pop("replicates"), np.arange(start, end + 1)), "Saved replicate IDs changed")
            else:
                require(path.name not in ledger["chunks"], "Previously committed chunk is missing: " + str(path))
                results = list(pool.map(lambda index: function(context, protocol["base_seed"], index), range(start, end + 1)))
                block = {key: np.stack([row[key] for row in results]) for key in results[0]}
                save_npz(path, **block, fingerprint=np.asarray(fingerprint), replicates=np.arange(start, end + 1))
                ledger["chunks"][path.name] = boot.sha256(path)
                save_json(ledger_path, ledger)
                computed += end - start + 1
            count = end - start + 1
            shape = (count, 2, 2, 2) if phase == "discrimination" else (count, 2, len(CAL_FIELDS))
            valid_shape = (count, 2) if phase == "discrimination" else (count,)
            require(block["values"].shape == shape and block["valid"].shape == valid_shape, "Invalid chunk geometry")
            if phase == "discrimination":
                require(np.isfinite(block["values"][block["valid"]]).all(), "Invalid supported ranking chunk")
                require(np.isnan(block["values"][~block["valid"]]).all(), "Unsupported ranking chunk is not NaN")
            blocks.append(block)
            elapsed = time.monotonic() - started
            progress = dict(stage=phase, completed=end, attempted=protocol["replicates"],
                pid=os.getpid(), workers=protocol["workers"], updated_unix=time.time(),
                estimated_remaining_seconds=(elapsed / computed * (protocol["replicates"] - end)) if computed else None)
            save_json(output / "STATUS.json", progress)
            save_json(folder / "PROGRESS.json", progress)
            print(phase.upper() + "_BOOTSTRAP", end, "/", protocol["replicates"], flush=True)
    combined = {key: np.concatenate([block[key] for block in blocks]) for key in blocks[0]}
    save_npz(folder / "DISTRIBUTIONS.npz", **combined, fingerprint=np.asarray(fingerprint))
    rows = []
    if phase == "discrimination":
        for pi, population in enumerate(POPULATIONS):
            for fi, field in enumerate(RANK_FIELDS):
                for mi, model in enumerate(MODELS):
                    rows.append(dict(population=population, model=model, metric=field, units="percent",
                        **confidence_row(100 * context["point"][pi, mi, fi], 100 * combined["values"][:, pi, mi, fi], protocol["replicates"])))
                rows.append(dict(population=population, model="SSTHNet_minus_Clinical_LR", metric=field, units="percentage points",
                    **confidence_row(100 * (context["point"][pi, 1, fi] - context["point"][pi, 0, fi]),
                        100 * (combined["values"][:, pi, 1, fi] - combined["values"][:, pi, 0, fi]), protocol["replicates"])))
    else:
        for fi, field in enumerate(CAL_FIELDS):
            units = "log odds" if field == "calibration_in_the_large" else "unscaled"
            for mi, method in enumerate(METHODS):
                rows.append(dict(method=method, metric=field, units=units,
                    **confidence_row(context["calibration_point"][mi, fi], combined["values"][:, mi, fi], protocol["replicates"])))
            rows.append(dict(method="recalibrated_minus_raw", metric=field, units=units,
                **confidence_row(context["calibration_point"][1, fi] - context["calibration_point"][0, fi],
                    combined["values"][:, 1, fi] - combined["values"][:, 0, fi], protocol["replicates"])))
    save_csv(output / (phase.upper() + "_CONFIDENCE_INTERVALS.csv"), rows)
    save_json(folder / "COMPLETE.json", dict(status="COMPLETE", contract_hash=fingerprint,
        attempted=protocol["replicates"], reported_intervals=sum(row["interval_status"] == "reported" for row in rows),
        requested_intervals=len(rows), distributions=record(folder / "DISTRIBUTIONS.npz")))


@contextmanager
def run_lock(output):
    path = output / "RUN.lock"
    stream = path.open("a+b")
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt
            if not path.stat().st_size:
                stream.write(b" ")
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise FileExistsError("An evaluation process already holds RUN.lock") from error
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError("An evaluation process already holds RUN.lock") from error
        acquired = True
        stream.seek(0)
        stream.truncate()
        stream.write(json.dumps(dict(pid=os.getpid(), started_unix=time.time())).encode())
        stream.flush()
        os.fsync(stream.fileno())
        yield
    finally:
        if acquired:
            if os.name == "nt":
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def verify_completion(completion, protocol):
    require(completion["contract_hash"] == boot.canonical_hash(protocol), "Completion identity changed")
    for entry in completion["outputs"].values():
        verify(entry)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "points", "smoke", "run", "status"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--base-seed", type=int, default=20261022)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    require(args.replicates >= 100 and args.chunk_size > 0 and 1 <= args.workers <= 4, "Invalid bootstrap settings")
    if args.command == "status":
        output = (args.out or args.project_root / "reports" / VERSION).resolve()
        for name in ("STATUS.json", "COMPLETE.json"):
            path = output / name
            if path.exists():
                print(name, path.read_text())
        return
    protocol, output = prepare(args)
    if args.command == "preflight":
        return
    with run_lock(output):
        if (output / "COMPLETE.json").exists():
            completion = read_json(output / "COMPLETE.json")
            verify_completion(completion, protocol)
            print("ALREADY_COMPLETE_VERIFIED", flush=True)
            return
        save_json(output / "STATUS.json", dict(stage="points", pid=os.getpid(), updated_unix=time.time()))
        context = build_context(protocol, output)
        if args.command == "points":
            save_json(output / "STATUS.json", dict(stage="POINTS_COMPLETE_BOOTSTRAP_NOT_RUN", updated_unix=time.time()))
            return
        if args.command == "smoke":
            started = time.monotonic()
            rank = ranking_replicate(context, args.base_seed, 0)
            cal = calibration_replicate(context, args.base_seed, 0)
            require(rank["valid"].all() and bool(cal["valid"]), "Smoke draw lost class support")
            require(np.isfinite(rank["values"]).all() and np.isfinite(cal["values"]).all(), "Smoke produced nonfinite metrics")
            save_json(output / "SMOKE.json", dict(status="PASS", seconds=time.monotonic() - started,
                ranking_shape=rank["values"].shape, calibration_shape=cal["values"].shape,
                calibration_slope_fits=cal["slope_fits"], contract_hash=boot.canonical_hash(protocol)))
            save_json(output / "STATUS.json", dict(stage="SMOKE_PASSED_BOOTSTRAP_NOT_RUN", updated_unix=time.time()))
            print("SMOKE_PASSED", flush=True)
            return
        for phase in ("discrimination", "calibration"):
            bootstrap_stage(context, protocol, output, phase)
        excluded = {output / name for name in ("STATUS.json", "COMPLETE.json", "SMOKE.json", "LAUNCH.json")}
        outputs = {str(path.relative_to(output)): record(path) for path in output.rglob("*")
                   if path.is_file() and path.suffix in (".csv", ".json", ".npz") and path not in excluded}
        save_json(output / "COMPLETE.json", dict(status="COMPLETE_FINAL_MODEL_CLINICAL_EVALUATION_PENDING_REVIEW",
            contract_hash=boot.canonical_hash(protocol), completed_unix=time.time(), outputs=outputs,
            training_performed=False, fixed_seeds=list(SEEDS), bootstrap_replicates=protocol["replicates"]))
        save_json(output / "STATUS.json", dict(stage="complete", updated_unix=time.time()))
        print("COMPLETE_FINAL_MODEL_CLINICAL_EVALUATION_PENDING_REVIEW", flush=True)


if __name__ == "__main__":
    main()
