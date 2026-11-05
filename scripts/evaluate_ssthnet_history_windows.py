#!/usr/bin/env python3
"""Frozen posthoc internal-test evaluation of four matched history inputs.

No training or test-guided selection. All validation thresholds are locked
before test labels are loaded. Existing training outputs remain read-only.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssthnet_history_window_ablations as windows

opt = windows.opt
shared, original = opt.shared, opt.original
boot, require = shared.boot, opt.require
VERSION = "ssthnet_history_window_test_evaluation_v1"
CONFIGS = ("120_300_600s", "120s_only", "300s_only", "600s_only")
SECONDS = (None, 120, 300, 600)
FIELDS = shared.SUMMARY_FIELDS
POINT_FIELDS = FIELDS[:3]
SEEDS = windows.SEEDS
REPLICATES, BASE_SEED, CHUNK_SIZE = 5000, 20261020, 25


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bind_file(helper, path):
    return dict(path=str(path.resolve()), sha256=helper.sha256(path))


def check_file(helper, entry):
    path = Path(entry["path"])
    require(path.is_file() and helper.sha256(path) == entry["sha256"],
            "Frozen file changed or missing: " + str(path))
    return path


def contract_of(protocol):
    return boot.canonical_hash(protocol)


def persist_identity(helper, path, value):
    if path.exists():
        require(read_json(path) == value, "Frozen identity changed: " + str(path))
    else:
        helper.save_json(path, value)


def prepare(root, output):
    loaded, sources = original.modules(root)
    helper = loaded["io"]
    training = root / "reports" / windows.VERSION
    parent = root / "reports" / opt.VERSION
    wp, pp = read_json(training / "protocol.json"), read_json(parent / "protocol.json")
    wc, pc = contract_of(wp), contract_of(pp)
    require(read_json(training / "COMPLETE.json")["status"] ==
            "COMPLETE_MATCHED_WINDOW_VALIDATION_TEST_NOT_RUN", "Window training incomplete")
    require(read_json(training / "COMPLETE.json")["contract_hash"] == wc,
            "Window training completion contract changed")
    require(wp["parent_contract_hash"] == pc, "Parent training contract changed")
    require(helper.sha256(parent / "protocol.json") == wp["plan"]["parent_optimization_protocol_sha256"],
            "Parent protocol file changed")
    require(helper.sha256(parent / "COMPLETE.json") == wp["parent_completion_sha256"] and
            helper.sha256(parent / "SELECTION.json") == wp["selection_sha256"], "Frozen selection changed")
    reference = wp["reference"]
    require(reference["kind"] == "optimized" and reference["candidate"] == "A_constant_weighted"
            and reference["evaluation_epoch"] == reference["training_epochs"] == 15,
            "This version only evaluates the already selected A/epoch15 protocol")
    require(read_json(parent / "COMPLETE.json")["contract_hash"] == pc, "Parent incomplete")
    for entry in pp["sources"].values():
        check_file(helper, entry)
    for entry in wp["plan"]["sources"].values():
        check_file(helper, entry)
    for entry in wp["full_reference_files"].values():
        check_file(helper, entry)

    old = root / "reports" / shared.VERSION / "protocol.json"
    frozen = read_json(old)
    for key, path in sources.items():
        require(helper.sha256(path) == frozen["sources"][key], "Metric/model source changed: " + key)
    paths = helper.sources(SimpleNamespace(project_root=root, **{key: None for key in (
        *opt.INPUT_KEYS, "test_features", "test_reference")}))
    inputs = {}
    # Hashing verifies bytes; target arrays are not loaded until VALIDATION_LOCK.
    for key in ("validation_features", "validation_reference", "test_features", "test_reference"):
        entry = bind_file(helper, paths[key])
        require(entry["sha256"] == frozen["inputs"][key], "Frozen input changed: " + key)
        inputs[key] = entry
    mapping = bind_file(helper, root / shared.MAPPING)
    require(mapping["sha256"] == frozen["patient_mapping_sha256"], "Patient mapping changed")

    checkpoints = {}
    for config, seconds in zip(CONFIGS, SECONDS):
        local_hash = pc if seconds is None else contract_of(dict(parent=wc, window_seconds=seconds))
        location = parent if seconds is None else training / f"window{seconds}"
        for seed in SEEDS:
            for fold in original.FOLDS:
                directory = opt.job_dir(location, reference["candidate"], seed, fold)
                marker = read_json(directory / "COMPLETE.json")
                job = dict(candidate=reference["candidate"], seed=seed, fold=fold,
                           epochs=20 if seconds is None and seed in (1, 2) else 15)
                require(marker["contract_hash"] == local_hash and marker["job"] == job,
                        "Training identity mismatch: " + str(directory))
                entries = {}
                for name in ("epoch15.pt", "epoch15_validation.npz"):
                    entry = bind_file(helper, directory / name)
                    require(entry["sha256"] == marker["files"][name], "Training output changed: " + name)
                    entries[name] = entry
                checkpoints[f"{config}/seed{seed:02d}/fold{fold}"] = dict(
                    **entries, training_contract_hash=local_hash, job=job,
                    complete=bind_file(helper, directory / "COMPLETE.json"))

    sources.update(evaluation_runner=Path(__file__), window_runner=Path(windows.__file__),
                   optimization_runner=Path(opt.__file__), shared_runner=Path(shared.__file__),
                   original_runner=Path(original.__file__), bootstrap_engine=Path(boot.__file__))
    protocol = dict(version=VERSION, inputs=inputs, patient_mapping=mapping,
        sources={key: bind_file(helper, path) for key, path in sources.items()},
        training_protocol=bind_file(helper, training / "protocol.json"),
        parent_protocol=bind_file(helper, parent / "protocol.json"), checkpoints=checkpoints,
        configurations=list(CONFIGS), seeds=list(SEEDS), folds=list(original.FOLDS), epoch=15,
        candidate="A_constant_weighted", validation_threshold_rule="original per-cell maximum validation F1; original tie rule",
        probability_policy="raw; four-fold mean per seed, metrics then five-seed arithmetic mean; no cross-seed probability ensemble",
        primary_uncertainty="patient-cluster paired bootstrap conditional on the same five fitted seeds",
        sensitivity_uncertainty="crossed patient and five-seed resampling, paired across configurations",
        bootstrap_replicates=REPLICATES, bootstrap_seed=BASE_SEED, chunk_size=CHUNK_SIZE,
        intervals="percentile 2.5/97.5; AP also Bonferroni 98.333333% intervals for the three full-minus-single contrasts",
        invalid_replicates="no redraw; all 21 cells required for all configurations; report missing classes per cell",
        interval_support_rule="at least 95% of attempted replicates and at least 100 estimable draws",
        test_role="posthoc evaluation on a previously viewed internal-test partition, not an untouched confirmation set",
        selection="none; all four configurations reported; no new epochs, training or test-based threshold fitting",
        torch_version=torch.__version__, numpy_version=np.__version__)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "protocol.json").exists():
        require(not list(output.iterdir()), "New evaluation directory must be empty")
    persist_identity(helper, output / "protocol.json", protocol)
    print("EVALUATION_PREFLIGHT_VERIFIED", contract_of(protocol), "checkpoints=80", flush=True)
    return loaded, protocol


def model_dir(output, config, seed):
    return output / config / f"seed{seed:02d}"


def threshold_map(rows):
    result = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in rows}
    require(len(rows) == len(result) == 21, "Missing or duplicate validation thresholds")
    require(set(result) == {(task, h) for task in shared.TASKS for h in shared.HORIZONS},
            "Wrong target-horizon threshold set")
    require(all(np.isfinite(v) and 0 <= v <= 1 for v in result.values()), "Invalid frozen threshold")
    return result


def prediction_arrays(reference, probability):
    p = np.asarray(probability)
    require(p.shape == reference["labels"].shape and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(), "Prediction shape or range invalid")
    return {**{key: reference[key] for key in boot.IDENTITY}, "probability": p}


def average_probabilities(probabilities):
    require(len(probabilities) == 4, "Exactly four fold predictions are required")
    shape = probabilities[0].shape
    require(all(p.shape == shape and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
                for p in probabilities), "Fold prediction identity/range changed")
    return (np.sum(probabilities, axis=0, dtype=np.float64) / 4).astype(np.float32)


def validation_lock(helper, output, protocol):
    path = output / "VALIDATION_LOCK.json"
    require(path.exists(), "Test access refused: validation thresholds are not frozen")
    lock = read_json(path)
    require(lock["contract_hash"] == contract_of(protocol), "Validation lock protocol changed")
    require(lock["test_labels_loaded_during_lock"] is False, "Invalid threshold freeze chronology")
    require(len(lock["models"]) == 20, "Incomplete threshold lock")
    for config in CONFIGS:
        for seed in SEEDS:
            entry = lock["models"][f"{config}/seed{seed:02d}"]
            for key in ("thresholds", "validation_prediction"):
                check_file(helper, entry[key])
    return lock


def freeze_validation(loaded, output, protocol):
    helper, metrics = loaded["io"], loaded["metrics"]
    if (output / "VALIDATION_LOCK.json").exists():
        return validation_lock(helper, output, protocol)
    require(not list(output.glob("*/seed*/TEST_PREDICTIONS.npz")), "Test predictions precede validation lock")
    reference = boot.read_npz(Path(protocol["inputs"]["validation_reference"]["path"]))
    entries = {}
    for config in CONFIGS:
        for seed in SEEDS:
            folds = []
            for fold in original.FOLDS:
                frozen = protocol["checkpoints"][f"{config}/seed{seed:02d}/fold{fold}"]
                with np.load(check_file(helper, frozen["epoch15_validation.npz"]), allow_pickle=False) as z:
                    require(str(z["contract_hash"]) == frozen["training_contract_hash"] and
                            str(z["validation_reference_sha256"]) == protocol["inputs"]["validation_reference"]["sha256"],
                            "Validation fold contract changed")
                    folds.append(z["probability"])
            prediction = prediction_arrays(reference, average_probabilities(folds))
            rows = metrics.threshold_rows(prediction, config)
            threshold_map(rows)
            directory = model_dir(output, config, seed)
            directory.mkdir(parents=True, exist_ok=True)
            vp, tp = directory / "VALIDATION_PREDICTIONS.npz", directory / "VALIDATION_THRESHOLDS.csv"
            helper.save_npz(vp, **prediction)
            helper.save_csv(tp, rows)
            entries[f"{config}/seed{seed:02d}"] = dict(
                validation_prediction=bind_file(helper, vp), thresholds=bind_file(helper, tp))
    lock = dict(contract_hash=contract_of(protocol), models=entries,
                test_labels_loaded_during_lock=False,
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    helper.save_json(output / "VALIDATION_LOCK.json", lock)
    print("VALIDATION_THRESHOLDS_FROZEN models=20 cells_per_model=21", flush=True)
    return validation_lock(helper, output, protocol)


def worker(root, output, device_name, lane, lanes, allow_shared):
    loaded, _ = original.modules(root)
    helper = loaded["io"]
    protocol = read_json(output / "protocol.json")
    for entry in protocol["sources"].values():
        check_file(helper, entry)
    lock = validation_lock(helper, output, protocol)
    lock_hash = helper.sha256(output / "VALIDATION_LOCK.json")
    device = shared.choose_device(device_name, allow_shared=allow_shared)
    paths = {key: check_file(helper, entry) for key, entry in protocol["inputs"].items()}
    x, reference = helper.outer(paths, "test")
    fingerprint = contract_of(protocol)
    jobs = [(config, seconds, seed) for config, seconds in zip(CONFIGS, SECONDS) for seed in SEEDS]
    for config, seconds, seed in jobs[lane::lanes]:
        directory = model_dir(output, config, seed)
        marker_path = directory / "PREDICTION_COMPLETE.json"
        if marker_path.exists():
            marker = read_json(marker_path)
            require(marker["contract_hash"] == fingerprint and marker["validation_lock_sha256"] == lock_hash,
                    "Saved prediction contract changed")
            check_file(helper, marker["test_prediction"])
            print("TEST_PREDICTION_REUSED", config, seed, flush=True)
            continue
        with opt.exclusive_lock(directory / "PREDICTING.lock"):
            probabilities, fold_files = [], {}
            for fold in original.FOLDS:
                entry = protocol["checkpoints"][f"{config}/seed{seed:02d}/fold{fold}"]
                cp = check_file(helper, entry["epoch15.pt"])
                prediction_path = directory / f"fold{fold}_TEST.npz"
                if prediction_path.exists():
                    with np.load(prediction_path, allow_pickle=False) as z:
                        require(str(z["contract_hash"]) == fingerprint and str(z["checkpoint_sha256"]) == entry["epoch15.pt"]["sha256"]
                                and str(z["test_reference_sha256"]) == protocol["inputs"]["test_reference"]["sha256"],
                                "Resumed fold prediction identity changed")
                        probability = z["probability"]
                else:
                    checkpoint = original.load_checkpoint(cp)
                    require(checkpoint["contract_hash"] == entry["training_contract_hash"] and
                            checkpoint["job"] == entry["job"] and checkpoint["epoch"] == 15,
                            "Checkpoint internal identity changed")
                    base = loaded["models"] if seconds is None else windows.window_model_base(loaded["models"], seconds)
                    model = original.make_model(base, "ssthnet").to(device)
                    require(sum(p.numel() for p in model.parameters()) == 155271, "Model parameter identity changed")
                    model.load_state_dict(checkpoint["model_state"], strict=True)
                    probability = opt.predict(model, x, reference, device)
                    helper.save_npz(prediction_path, probability=probability, contract_hash=np.array(fingerprint),
                        checkpoint_sha256=np.array(entry["epoch15.pt"]["sha256"]),
                        test_reference_sha256=np.array(protocol["inputs"]["test_reference"]["sha256"]))
                    del model, checkpoint
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    print("TEST_FOLD_COMPLETE", config, seed, fold, str(device), flush=True)
                prediction_arrays(reference, probability)
                probabilities.append(probability)
                fold_files[str(fold)] = bind_file(helper, prediction_path)
            prediction = prediction_arrays(reference, average_probabilities(probabilities))
            path = directory / "TEST_PREDICTIONS.npz"
            helper.save_npz(path, **prediction)
            helper.save_json(marker_path, dict(contract_hash=fingerprint, validation_lock_sha256=lock_hash,
                test_prediction=bind_file(helper, path), folds=fold_files,
                threshold_sha256=lock["models"][f"{config}/seed{seed:02d}"]["thresholds"]["sha256"]))
            print("TEST_SEED_COMPLETE", config, seed, flush=True)


def launch_predictions(args, output, helper, protocol):
    children = []
    try:
        for lane, device in enumerate(args.devices):
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "predict-worker",
                "--project-root", str(args.project_root), "--output", str(output), "--devices", device,
                "--lane", str(lane), "--lanes", str(len(args.devices)), "--cpu-threads", str(args.cpu_threads)]
            if args.allow_shared_gpu:
                command.append("--allow-shared-gpu")
            with (output / f"prediction_worker{lane}.log").open("a", buffering=1) as stream:
                children.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        helper.save_json(output / "PREDICTION_WORKERS.json", dict(
            contract_hash=contract_of(protocol), pids=[p.pid for p in children], devices=args.devices))
        while any(p.poll() is None for p in children):
            require(not any(p.poll() not in (None, 0) for p in children), "Inference worker failed; inspect worker logs")
            time.sleep(3)
        require(all(p.returncode == 0 for p in children), "Inference failed; inspect prediction_worker logs")
    finally:
        for p in children:
            if p.poll() is None:
                p.terminate()
        for p in children:
            p.wait()


def load_bootstrap_inputs(loaded, output, protocol):
    helper, metrics = loaded["io"], loaded["metrics"]
    lock = validation_lock(helper, output, protocol)
    lock_hash = helper.sha256(output / "VALIDATION_LOCK.json")
    reference = boot.read_npz(check_file(helper, protocol["inputs"]["test_reference"]))
    patient_index, patients = boot.map_patients(reference, boot.read_npz(check_file(helper, protocol["patient_mapping"])))
    require(np.isin(reference["masks"], (0, 1)).all() and
            np.isin(reference["labels"][reference["masks"] > 0], (0, 1)).all(), "Invalid test labels/masks")
    selected, support, support_rows = [], [], []
    for task_index, task in enumerate(shared.TASKS):
        for horizon in shared.HORIZONS:
            sel = (reference["masks"][:, task_index] > 0) & (reference["horizons"] == horizon)
            y, subjects = reference["labels"][sel, task_index], patient_index[sel]
            pos = np.bincount(subjects[y == 1], minlength=patients)
            neg = np.bincount(subjects[y == 0], minlength=patients)
            require(pos.sum() > 0 and neg.sum() > 0, "Test cell is not estimable")
            selected.append((task_index, task, horizon, sel))
            support.append((pos, neg))
            support_rows.append(dict(task_id=task, horizon_minutes=horizon, eligible_records=int(sel.sum()),
                positive_records=int(pos.sum()), positive_patients=int(np.count_nonzero(pos)),
                negative_patients=int(np.count_nonzero(neg))))
    cache = [[None for _ in CONFIGS] for _ in SEEDS]
    point = np.zeros((5, 4, 3))
    cell_rows, seed_rows, files = [], [], {}
    for s, seed in enumerate(SEEDS):
        for m, config in enumerate(CONFIGS):
            directory = model_dir(output, config, seed)
            marker = read_json(directory / "PREDICTION_COMPLETE.json")
            require(marker["contract_hash"] == contract_of(protocol) and marker["validation_lock_sha256"] == lock_hash,
                    "Test prediction completion identity changed")
            for entry in marker["folds"].values():
                check_file(helper, entry)
            path = check_file(helper, marker["test_prediction"])
            files[f"{config}/seed{seed:02d}"] = marker["test_prediction"]
            prediction = boot.read_npz(path)
            shared.check_reference(reference, prediction, config)
            prediction_arrays(reference, prediction["probability"])
            entry = lock["models"][f"{config}/seed{seed:02d}"]["thresholds"]
            require(marker["threshold_sha256"] == entry["sha256"], "Frozen classification threshold changed")
            thresholds = threshold_map(boot.read_csv(check_file(helper, entry)))
            rows = metrics.metric_rows(prediction, config, "test", thresholds)
            summary = metrics.summary_row(rows, config, "test", VERSION)
            cell_rows.extend(dict(configuration=config, seed=seed, epoch=15, **r) for r in rows)
            seed_rows.append(dict(configuration=config, seed=seed, epoch=15, **summary))
            cells = [boot.prepare_cell(reference["labels"][sel, t], prediction["probability"][sel, t],
                        patient_index[sel], thresholds[(task, h)]) for t, task, h, sel in selected]
            cache[s][m] = cells
            point[s, m] = np.mean([boot.weighted_metrics(c, np.ones(patients)) for c in cells], axis=0)
            require(np.allclose(point[s, m], [summary[k] for k in POINT_FIELDS], atol=1e-10, rtol=0),
                    "Bootstrap metric engine disagrees with original evaluation")
            print("BOOTSTRAP_INPUT_VERIFIED", config, seed, flush=True)
    summaries = []
    for config in CONFIGS:
        rows = [r for r in seed_rows if r["configuration"] == config]
        values = {key: float(np.mean([r[key] for r in rows])) for key in FIELDS}
        values.update({key + "_seed_sd": float(np.std([r[key] for r in rows], ddof=1)) for key in FIELDS})
        summaries.append(dict(configuration=config, epoch=15, seeds=5, **values))
    differences = [dict(seed=seed, contrast="120_300_600s_minus_" + CONFIGS[m],
        **{key + "_difference_pp": float(100 * (point[s, 0, k] - point[s, m, k]))
           for k, key in enumerate(boot.METRICS)}) for s, seed in enumerate(SEEDS) for m in (1, 2, 3)]
    for name, rows in (("TEST_SUMMARY.csv", summaries), ("TEST_SEED_RESULTS.csv", seed_rows),
                       ("TEST_21CELL_RESULTS.csv", cell_rows), ("TEST_PAIRED_SEED_DIFFERENCES.csv", differences),
                       ("PATIENT_SUPPORT_21CELL.csv", support_rows)):
        helper.save_csv(output / name, rows)
    identity = dict(contract_hash=contract_of(protocol), validation_lock_sha256=lock_hash,
                    predictions=files, patient_mapping=protocol["patient_mapping"], patients=patients,
                    test_records=len(patient_index), primary="fixed_five_seed_patient_cluster_paired")
    persist_identity(helper, output / "TEST_PREDICTION_LOCK.json", identity)
    return cache, support, support_rows, patients, point, contract_of(identity)


def bootstrap_chunk(cache, support, patients, start, end, workers):
    def one(rep):
        m, slots, valid = boot.evaluate_replicate(cache, support, patients, BASE_SEED, rep)
        counts, _ = boot.draw(BASE_SEED, rep, patients)
        missing = np.array([np.sum(pos * counts) <= 0 or np.sum(neg * counts) <= 0 for pos, neg in support])
        require(valid == (not missing.any()), "Bootstrap support check disagrees")
        return m.mean(axis=0), m[slots].mean(axis=0), valid, missing
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, range(start, end)))
    return dict(replicates=np.arange(start, end), fixed=np.array([r[0] for r in rows]),
                crossed=np.array([r[1] for r in rows]), valid=np.array([r[2] for r in rows]),
                missing_cells=np.array([r[3] for r in rows]))


def validate_chunk(data, start, end, fingerprint):
    n = end - start
    require(str(data["contract_hash"]) == fingerprint and
            np.array_equal(data["replicates"], np.arange(start, end)), "Bootstrap resume identity changed")
    require(data["valid"].shape == (n,) and data["valid"].dtype == np.dtype(bool) and
            data["missing_cells"].shape == (n, 21), "Invalid bootstrap support geometry")
    valid = data["valid"]
    require(np.array_equal(valid, ~data["missing_cells"].any(axis=1)), "Bootstrap support flags disagree")
    for key in ("fixed", "crossed"):
        require(data[key].shape == (n, 4, 3) and np.isfinite(data[key][valid]).all()
                and np.isnan(data[key][~valid]).all(), "Invalid saved bootstrap metrics")


def interval_rows(point, fixed, crossed, valid):
    estimable = int(valid.sum())
    supported = estimable >= 100 and estimable >= .95 * len(valid)
    rows = []
    for design, values in (("fixed_five_seed_patient_cluster_primary", fixed),
                           ("crossed_seed_patient_sensitivity", crossed)):
        for m in (1, 2, 3):
            for k, metric in enumerate(boot.METRICS):
                delta = values[valid, 0, k] - values[valid, m, k]
                lo, hi = np.percentile(delta, [2.5, 97.5]) * 100 if supported else (float("nan"), float("nan"))
                a, b = (np.percentile(delta, [100 * .05 / 6, 100 * (1 - .05 / 6)]) * 100
                        if supported and k == 0 else (float("nan"), float("nan")))
                rows.append(dict(analysis=design, contrast="120_300_600s_minus_" + CONFIGS[m], metric=metric,
                    difference_pp=float(100 * (point[:, 0, k].mean() - point[:, m, k].mean())),
                    ci95_lower_pp=float(lo), ci95_upper_pp=float(hi), ap_bonferroni_lower_pp=float(a),
                    ap_bonferroni_upper_pp=float(b), attempted=len(valid), estimable=estimable,
                    interval_support_sufficient=supported))
    return rows


def run_bootstrap(loaded, output, protocol, workers):
    helper = loaded["io"]
    boot.verify_metric_engine()
    cache, support, support_rows, patients, point, fingerprint = load_bootstrap_inputs(loaded, output, protocol)
    folder = output / "bootstrap_chunks"
    folder.mkdir(exist_ok=True)
    chunks, records = [], {}
    attempted, estimable = 0, 0
    for start in range(0, REPLICATES, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, REPLICATES)
        path = folder / f"CHUNK_{start + 1:06d}_{end:06d}.npz"
        digest_path = path.with_suffix(".json")
        if path.exists():
            # A crash between atomic NPZ and its hash record leaves a readable,
            # identity-checked chunk; record the digest before further work.
            if digest_path.exists():
                check_file(helper, read_json(digest_path))
            with np.load(path, allow_pickle=False) as z:
                chunk = {k: z[k] for k in z.files}
            validate_chunk(chunk, start, end, fingerprint)
        else:
            chunk = dict(bootstrap_chunk(cache, support, patients, start, end, workers),
                         contract_hash=np.array(fingerprint))
            validate_chunk(chunk, start, end, fingerprint)
            helper.save_npz(path, **chunk)
        persist_identity(helper, digest_path, bind_file(helper, path))
        records[path.name] = helper.sha256(path)
        chunks.append(chunk)
        attempted += end - start
        estimable += int(chunk["valid"].sum())
        helper.save_json(output / "PROGRESS.json", dict(status="BOOTSTRAPPING", contract_hash=contract_of(protocol),
            bootstrap_contract_hash=fingerprint, completed=attempted, planned=REPLICATES,
            estimable=estimable, updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        print(f"BOOTSTRAP_PROGRESS {attempted}/{REPLICATES} estimable={estimable}", flush=True)
    fixed = np.concatenate([c["fixed"] for c in chunks])
    crossed = np.concatenate([c["crossed"] for c in chunks])
    valid = np.concatenate([c["valid"] for c in chunks])
    missing = np.concatenate([c["missing_cells"] for c in chunks])
    rows = interval_rows(point, fixed, crossed, valid)
    helper.save_csv(output / "PAIRED_BOOTSTRAP_INTERVALS.csv", rows)
    helper.save_csv(output / "BOOTSTRAP_SUPPORT_AUDIT.csv", [dict(**r, draws_losing_either_class=int(missing[:, i].sum()))
        for i, r in enumerate(support_rows)])
    helper.save_npz(output / "BOOTSTRAP_DISTRIBUTIONS.npz", fixed=fixed, crossed=crossed, valid=valid,
                    missing_cells=missing, contract_hash=np.array(fingerprint))
    files = {p.name: helper.sha256(p) for p in output.iterdir() if p.suffix in (".csv", ".npz")}
    helper.save_json(output / "COMPLETE.json", dict(status="COMPLETE_FROZEN_HISTORY_WINDOW_INTERNAL_TEST_EVALUATION",
        contract_hash=contract_of(protocol), bootstrap_contract_hash=fingerprint, attempted_replicates=REPLICATES,
        estimable_replicates=int(valid.sum()), interval_support_sufficient=rows[0]["interval_support_sufficient"],
        test_role=protocol["test_role"], output_sha256=files, chunks_sha256=records,
        completed_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print("WINDOW_TEST_EVALUATION_COMPLETE", "attempted=5000", "estimable=" + str(valid.sum()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "predict-worker"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--devices", nargs="+", default=["auto"])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--bootstrap-workers", type=int, default=2)
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    output = (args.output or args.project_root / "reports" / VERSION).resolve()
    require(args.cpu_threads >= 1 and args.bootstrap_workers >= 1, "Worker counts must be positive")
    require(0 <= args.lane < args.lanes, "Invalid inference shard")
    require(len(set(args.devices)) == len(args.devices), "GPU device list must not contain duplicates")
    require(len(args.devices) == 1 or all(d.startswith("cuda:") for d in args.devices),
            "Multiple workers require distinct explicit GPUs")
    torch.set_num_threads(args.cpu_threads)
    if args.command == "predict-worker":
        require(len(args.devices) == 1, "Worker requires one device")
        worker(args.project_root, output, args.devices[0], args.lane, args.lanes, args.allow_shared_gpu)
        return
    loaded, protocol = prepare(args.project_root, output)
    helper = loaded["io"]
    with opt.exclusive_lock(output / "EVALUATING.lock"):
        freeze_validation(loaded, output, protocol)
        if args.command == "preflight":
            print("PREFLIGHT_COMPLETE thresholds_locked=20 test_target_arrays_loaded=0", flush=True)
            return
        if (output / "COMPLETE.json").exists():
            marker = read_json(output / "COMPLETE.json")
            require(marker["contract_hash"] == contract_of(protocol), "Completed protocol changed")
            for name, expected in marker["output_sha256"].items():
                require(helper.sha256(output / name) == expected, "Completed evaluation output changed")
            validation_lock(helper, output, protocol)
            print("EVALUATION_ALREADY_COMPLETE", flush=True)
            return
        helper.save_json(output / "STATUS.json", dict(status="GENERATING_FROZEN_TEST_PREDICTIONS",
                         contract_hash=contract_of(protocol), pid=os.getpid()))
        launch_predictions(args, output, helper, protocol)
        helper.save_json(output / "STATUS.json", dict(status="PAIRED_PATIENT_BOOTSTRAP",
                         contract_hash=contract_of(protocol), pid=os.getpid()))
        run_bootstrap(loaded, output, protocol, args.bootstrap_workers)
        helper.save_json(output / "STATUS.json", dict(status="COMPLETE_PENDING_AUTHOR_REVIEW",
                         contract_hash=contract_of(protocol)))


if __name__ == "__main__":
    main()
