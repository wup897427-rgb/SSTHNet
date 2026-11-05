#!/usr/bin/env python3
"""Five-seed, four-fold cross-target sharing experiment for final SSTHNet.

Each independent model trains one target and all three horizons from all 660
features. Existing models and reports are read-only. No LightGBM dependency.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssthnet_ablations as original
import bootstrap_ssthnet_five_seed_ablations as boot

VERSION = "ssthnet_joint_vs_independent_targets_v1"
SEEDS = (1, 2, 3, 4, 5)
FOLDS = original.FOLDS
TASKS = boot.TASKS
HORIZONS = (5, 10, 15)
SYSTEMS = ("joint", "independent_targets")
SUMMARY_FIELDS = (
    "equal_cell_macro_auprc", "equal_cell_macro_auroc", "equal_cell_macro_f1",
    "equal_cell_macro_precision", "equal_cell_macro_recall", "micro_f1",
    "equal_cell_macro_brier", "equal_cell_macro_ece_10bin",
)
MAPPING = "reports/phase6bja2t_v7_simple_clinical_comparators_one_time_test_v1/BJA2T_TEST_PREDICTIONS.npz"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def choose_device(requested, allow_shared=False, min_free_mib=6000):
    require(min_free_mib >= 6000, "At least 6000 MiB free must be required")
    if allow_shared:
        require(requested.startswith("cuda:") and requested[5:].isdigit(),
                "Shared GPU use requires an explicit logical device, such as --device cuda:4")
        require(torch.cuda.is_available(), "CUDA unavailable; activate the original training environment")
        inventory = original.gpu_inventory()
        selected = next((row for row in inventory if row["device"] == requested), None)
        require(selected is not None, f"Requested GPU is not visible: {requested}; inventory: {inventory}")
        require(selected["free_mib"] >= min_free_mib,
                f"Shared GPU needs >= {min_free_mib} MiB free; selected: {selected}")
        print("GPU_SHARED_SELECTED", json.dumps(dict(selected, sharing_explicitly_allowed=True,
                                                     minimum_free_mib=min_free_mib,
                                                     inference_benchmark="deferred")), flush=True)
        return torch.device(selected["device"])
    device = original.choose_device(requested)
    if device.type == "cuda" and min_free_mib > 6000:
        selected = next(row for row in original.gpu_inventory() if row["device"] == str(device))
        require(selected["free_mib"] >= min_free_mib,
                f"Selected GPU needs >= {min_free_mib} MiB free; selected: {selected}")
    return device


def make_model(base, task=None):
    if task is None:
        return original.make_model(base, "ssthnet")
    require(isinstance(task, int) and 0 <= task < 7, "Target index must be 0..6")

    class IndependentTargetSSTHNet(base.SSTHMoEAblation):
        def __init__(self):
            # Initialise the full model first to match the original backbone,
            # query/head initial values and post-initialisation RNG stream.
            super().__init__("single_expert")
            self.target_index = task
            self.task_query = nn.Parameter(self.task_query.detach()[task:task + 1].clone())
            self.interaction_query = nn.Parameter(self.interaction_query.detach()[task:task + 1].clone())
            self.task_heads = nn.ModuleList([self.task_heads[task]])

        def _queries(self):
            return (self.task_query[:, None] + self.horizon_query[None] +
                    self.interaction_query).reshape(3, base.D_MODEL)

        def forward(self, values):
            physiology = self._physiology_tokens(values)
            queries = self._queries()[None].expand(len(values), -1, -1)
            attended, _ = self.query_attention(queries, physiology, physiology, need_weights=True)
            representation = self.query_norm(queries + attended)
            private, auxiliary = self._route(representation)
            routed = self.output_norm(representation + self.shared_expert(representation) + private)
            return self.task_heads[0](routed).squeeze(-1).unsqueeze(1), auxiliary

    return IndependentTargetSSTHNet()


def target_labels(groups, selected, task):
    labels, masks = groups["labels"][selected], groups["masks"][selected]
    if task is not None:
        labels, masks = labels[:, task:task + 1], masks[:, task:task + 1]
    require(np.isin(labels[masks > 0], (0, 1)).all(), "Non-binary eligible labels")
    return np.where(masks > 0, labels, 0).astype(np.float32), masks.astype(np.float32)


def masked_loss(logits, labels, masks, positive_weights):
    require(logits.shape == labels.shape == masks.shape, "Loss geometry mismatch")
    denominator = masks.sum(dim=0)
    estimable = denominator > 0
    if not bool(estimable.any()):
        return None
    weights = torch.where(labels > .5, positive_weights.unsqueeze(0), torch.ones_like(labels))
    raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights * masks
    loss = (raw.sum(dim=0)[estimable] / denominator[estimable]).mean()
    require(bool(torch.isfinite(loss)), "Non-finite training loss")
    return loss


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def job_name(task, seed):
    return f"joint_seed{seed:02d}" if task is None else f"independent_target{task:02d}_seed{seed:02d}"


def validate_checkpoint(checkpoint, seed, fold, task):
    require((checkpoint.get("seed_index"), checkpoint.get("fold"), checkpoint.get("epoch")) ==
            (seed, fold, 5), "Checkpoint seed/fold/epoch mismatch")
    if task is not None:
        require(checkpoint.get("target_index") == task, "Checkpoint belongs to another target")


def train_fold(loaded, x, groups, stats, output, seed, fold, task, device, contract_hash):
    helper = loaded["io"]
    name = job_name(task, seed)
    directory = output / name / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    marker_path, checkpoint_path = directory / "COMPLETE.json", directory / "epoch5.pt"
    if marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        require(marker["contract_hash"] == contract_hash, "Completed fold protocol changed")
        require(marker["checkpoint_sha256"] == helper.sha256(checkpoint_path), "Completed checkpoint changed")
        print("FOLD_REUSED", name, fold, flush=True)
        return
    seed_value = 20261884 + (seed - 1) * 1000 + fold
    original.seed_everything(seed_value)
    model = make_model(loaded["models"], task).to(device)
    mean, scale, positive = stats[fold]
    model.set_scaler(torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device))
    positive = positive if task is None else positive[task:task + 1]
    positive = torch.as_tensor(positive, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    train = np.flatnonzero(groups["group_fold"] != fold)
    rng = np.random.default_rng(seed_value)
    history, start_epoch = [], 1
    latest = directory / "latest.pt"
    if latest.exists():
        saved = original.load_checkpoint(latest)
        require(saved["contract_hash"] == contract_hash, "Resume protocol changed")
        require(saved["device_type"] == device.type, "Resume must use the same CPU/CUDA device type")
        require((saved["seed_index"], saved["fold"], saved["target_index"]) ==
                (seed, fold, task), "Resume model identity changed")
        model.load_state_dict(saved["model_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer_state"])
        rng.bit_generator.state = saved["numpy_generator_state"]
        original.random.setstate(saved["python_rng_state"])
        np.random.set_state(saved["numpy_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
        history, start_epoch = saved["history"], saved["epoch"] + 1
        # Repair an interruption between writing latest.pt and epoch5.pt.
        if saved["epoch"] == 5 and not checkpoint_path.exists():
            original.atomic_checkpoint(checkpoint_path, {key: saved[key] for key in
                                       ("seed_index", "fold", "epoch", "target_index", "contract_hash", "model_state")})
    for epoch in range(start_epoch, 7):
        model.train()
        shuffled = rng.permutation(train)
        loss_sum, used, skipped, processed = 0., 0, 0, 0
        sync(device)
        started = time.perf_counter()
        for start in range(0, len(shuffled), 128):
            selected = shuffled[start:start + 128]
            labels, masks = target_labels(groups, selected, task)
            processed += len(selected)
            if not masks.any():
                skipped += 1
                continue
            values = np.asarray(x[groups["representative_index"][selected]], dtype=np.float32).reshape(-1, 3, 22, 10)
            require(np.isfinite(values).all(), "Non-finite frozen input features")
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(torch.as_tensor(values, device=device))
            loss = masked_loss(logits, torch.as_tensor(labels, device=device),
                               torch.as_tensor(masks, device=device), positive)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
            used += len(selected)
            if processed // 25000 != (processed - len(selected)) // 25000:
                print(f"TRAIN {name} fold={fold} epoch={epoch} groups={processed}/{len(train)}", flush=True)
        sync(device)
        require(used > 0, f"No eligible training labels: {name}/fold{fold}")
        history.append(dict(epoch=epoch, mean_training_loss=loss_sum / used,
                            elapsed_seconds=time.perf_counter() - started,
                            groups_with_labels=used, skipped_all_masked_batches=skipped))
        saved = dict(seed_index=seed, fold=fold, epoch=epoch, target_index=task,
                     contract_hash=contract_hash, device_type=device.type, model_state=model.state_dict(),
                     optimizer_state=optimizer.state_dict(), history=history,
                     numpy_generator_state=rng.bit_generator.state, python_rng_state=original.random.getstate(),
                     numpy_rng_state=np.random.get_state(), torch_rng_state=torch.get_rng_state(),
                     cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        # Write the evaluation checkpoint before latest so resume cannot lose it.
        if epoch == 5:
            original.atomic_checkpoint(checkpoint_path, {key: saved[key] for key in
                                       ("seed_index", "fold", "epoch", "target_index", "contract_hash", "model_state")})
        original.atomic_checkpoint(latest, saved)
        helper.save_csv(directory / "TRAINING_HISTORY.csv", history)
        print(f"EPOCH_COMPLETE {name} fold={fold} epoch={epoch} loss={history[-1]['mean_training_loss']:.6f}", flush=True)
    require(checkpoint_path.is_file(), f"Missing epoch-5 checkpoint: {directory}")
    helper.save_json(marker_path, dict(contract_hash=contract_hash, seed_index=seed, fold=fold,
                                     target_index=task, training_epochs=6, evaluation_epoch=5,
                                     training_seconds=sum(row["elapsed_seconds"] for row in history),
                                     skipped_all_masked_batches=sum(row["skipped_all_masked_batches"] for row in history),
                                     checkpoint_sha256=helper.sha256(checkpoint_path)))
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


def check_reference(reference, prediction, name):
    boot.check_identity(reference, prediction, name)
    p = prediction["probability"]
    require(p.shape == reference["labels"].shape, f"{name}: probability shape mismatch")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), f"{name}: invalid probabilities")


def audit_joint_sources(args, loaded, input_hashes, source_hashes, outer_data):
    if args.joint_mode == "train":
        return {}
    root = args.project_root.resolve()
    folders = args.joint_roots or [root / "reports/ssthnet_single_expert_matched_ablations_v1",
                                   root / "reports/ssthnet_single_expert_extra_seeds_v1"]
    candidates = {}
    for folder in folders:
        folder = folder.resolve()
        protocol_path = folder / "protocol.json"
        require(protocol_path.is_file(), f"Missing joint protocol: {protocol_path}; upload old results or use --joint-mode train")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        require(protocol["script_sha256"] == loaded["io"].sha256(root / "scripts/run_ssthnet_ablations.py"),
                "Old joint runner differs; cannot silently reuse it")
        require(protocol["source_sha256"] == {k: source_hashes[k] for k in protocol["source_sha256"]},
                "Old joint source files differ")
        require(protocol["input_sha256"] == input_hashes, "Old joint input fingerprints differ")
        expected = dict(training_epochs=6, evaluation_epoch=5, batch_size=128,
                        learning_rate=2e-4, weight_decay=1e-4, folds=list(FOLDS))
        require(all(protocol.get(k) == v for k, v in expected.items()), "Old joint training settings differ")
        require(protocol["torch_version"] == torch.__version__ and protocol["numpy_version"] == np.__version__,
                "Old joint runtime differs; use the original environment or --joint-mode train")
        require(set(protocol["source_sha256"]) == {"io", "models", "training", "metrics", "metric_protocol"},
                "Old joint source manifest incomplete")
        lock = json.loads((folder / "VALIDATION_LOCK.json").read_text(encoding="utf-8"))
        for seed in protocol["seed_indices"]:
            if seed not in SEEDS:
                continue
            require(seed not in candidates, f"Duplicate joint source for seed {seed}")
            directory = folder / f"ssthnet_seed{seed:02d}"
            files = {role: directory / f"{role.upper()}_PREDICTIONS.npz" for role in ("validation", "test")}
            files["thresholds"] = directory / "VALIDATION_THRESHOLDS.csv"
            require(lock["thresholds_sha256"][directory.name] == loaded["io"].sha256(files["thresholds"]),
                    "Old validation thresholds changed")
            for fold in FOLDS:
                checkpoint_path = directory / f"fold{fold}/epoch5.pt"
                complete = json.loads((directory / f"fold{fold}/COMPLETE.json").read_text(encoding="utf-8"))
                require(complete["checkpoint_sha256"] == loaded["io"].sha256(checkpoint_path), "Old joint checkpoint changed")
                saved = original.load_checkpoint(checkpoint_path)
                validate_checkpoint(saved, seed, fold, None)
                require(saved.get("variant") == "ssthnet", "Not the final joint SSTHNet control")
                make_model(loaded["models"]).load_state_dict(saved["model_state"], strict=True)
            for role in ("validation", "test"):
                check_reference(outer_data[role][1], boot.read_npz(files[role]), f"old joint seed{seed}/{role}")
            validation_rows = loaded["metrics"].threshold_rows(boot.read_npz(files["validation"]), directory.name)
            reproduced = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in validation_rows}
            recorded = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in boot.read_csv(files["thresholds"])}
            require(reproduced == recorded, "Old joint validation thresholds cannot be reproduced; stop before training")
            candidates[seed] = dict(directory=str(directory),
                                    files={k: str(v) for k, v in files.items()},
                                    sha256={k: loaded["io"].sha256(v) for k, v in files.items()},
                                    protocol_sha256=loaded["io"].sha256(protocol_path))
            print("JOINT_REUSE_VERIFIED", seed, directory, flush=True)
    require(set(candidates) == set(SEEDS), f"Need all five joint seeds, found {sorted(candidates)}")
    return candidates


@contextmanager
def run_lock(output):
    path = output / "RUNNING.lock"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        require(old["host"] == socket.gethostname(), "Run lock belongs to another host; verify that run first")
        require(os.name != "nt", "Existing Windows run lock; verify the old process before removing the lock")
        try:
            os.kill(int(old["pid"]), 0)
        except ProcessLookupError:
            path.unlink()
            print("RECOVERED_STALE_LOCK", old, flush=True)
        else:
            raise RuntimeError(f"Process may still be running: {old}")
    with path.open("x", encoding="utf-8") as f:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), f)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def checkpoint_model(loaded, output, seed, fold, task, device, joint_sources):
    if task is None and joint_sources:
        directory = Path(joint_sources[seed]["directory"])
    else:
        directory = output / job_name(task, seed)
    saved = original.load_checkpoint(directory / f"fold{fold}/epoch5.pt")
    validate_checkpoint(saved, seed, fold, task)
    model = make_model(loaded["models"], task)
    model.load_state_dict(saved["model_state"], strict=True)
    return model.to(device).eval()


def predict(loaded, output, seed, task, device, features, reference, joint_sources):
    count = len(reference["horizons"])
    slots = np.searchsorted(HORIZONS, reference["horizons"])
    probability = np.zeros((count, 7 if task is None else 1), dtype=np.float64)
    for fold in FOLDS:
        model = checkpoint_model(loaded, output, seed, fold, task, device, joint_sources)
        with torch.inference_mode():
            for start in range(0, count, 256):
                end = min(count, start + 256)
                values = np.array(features[start:end], dtype=np.float32, copy=True).reshape(-1, 3, 22, 10)
                require(np.isfinite(values).all(), "Non-finite outer features")
                logits, _ = model(torch.as_tensor(values, device=device))
                scores = torch.sigmoid(logits).cpu().numpy()
                probability[start:end] += scores[np.arange(end - start), :, slots[start:end]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return (probability / len(FOLDS)).astype(np.float32)


def cached_prediction(loaded, output, seed, task, device, features, reference, joint_sources, role, contract_hash):
    helper = loaded["io"]
    directory = output / job_name(task, seed)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{role.upper()}_FORWARD_CACHE.npz"
    marker_path = path.with_suffix(".json")
    if path.exists() and marker_path.exists():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        require(marker["contract_hash"] == contract_hash and marker["target_index"] == task,
                "Forward cache identity changed")
        require(marker["sha256"] == helper.sha256(path), "Forward cache changed")
        with np.load(path, allow_pickle=False) as z:
            require(np.array_equal(z["global_position"], reference["global_position"]) and
                    np.array_equal(z["horizons"], reference["horizons"]), "Forward cache rows changed")
            probability = z["probability"]
        require(probability.shape == (len(reference["horizons"]), 7 if task is None else 1), "Forward cache shape mismatch")
        return probability
    probability = predict(loaded, output, seed, task, device, features, reference, joint_sources)
    helper.save_npz(path, global_position=reference["global_position"], horizons=reference["horizons"], probability=probability)
    helper.save_json(marker_path, dict(contract_hash=contract_hash, target_index=task, sha256=helper.sha256(path)))
    return probability


def evaluate(loaded, output, outer_data, device, joint_sources, contract_hash):
    helper, metrics = loaded["io"], loaded["metrics"]
    run_rows, cell_rows = [], []
    thresholds = {}
    # Complete validation and freeze all thresholds before test evaluation.
    for role in ("validation", "test"):
        features, reference = outer_data[role]
        for seed in SEEDS:
            for system in SYSTEMS:
                name = f"{system}_seed{seed:02d}"
                directory = output / name
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{role.upper()}_PREDICTIONS.npz"
                marker_path = path.with_suffix(".json")
                assembled_cached = path.exists() and marker_path.exists()
                if assembled_cached:
                    marker = json.loads(marker_path.read_text(encoding="utf-8"))
                    require(marker["contract_hash"] == contract_hash and marker["sha256"] == helper.sha256(path),
                            "Assembled prediction cache changed")
                    prediction = boot.read_npz(path)
                    check_reference(reference, prediction, name)
                    probability = prediction["probability"]
                elif system == "joint" and joint_sources:
                    source = joint_sources[seed]
                    source_path = Path(source["files"][role])
                    require(helper.sha256(source_path) == source["sha256"][role], "Reused predictions changed during run")
                    probability = boot.read_npz(source_path)["probability"]
                elif system == "joint":
                    probability = cached_prediction(loaded, output, seed, None, device, features, reference,
                                                    joint_sources, role, contract_hash)
                else:
                    probability = np.concatenate([cached_prediction(loaded, output, seed, task, device, features,
                                                                    reference, joint_sources, role, contract_hash)
                                                  for task in range(7)], axis=1)
                prediction = dict(**reference, probability=probability)
                check_reference(reference, prediction, name)
                if not assembled_cached:
                    helper.save_npz(path, **prediction)
                    helper.save_json(marker_path, dict(contract_hash=contract_hash, sha256=helper.sha256(path)))
                if role == "validation":
                    rows = metrics.threshold_rows(prediction, name)
                    thresholds[name] = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in rows}
                    if system == "joint" and joint_sources:
                        old = boot.read_csv(Path(joint_sources[seed]["files"]["thresholds"]))
                        old_values = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in old}
                        require(thresholds[name] == old_values, "Reused joint thresholds cannot be reproduced")
                    helper.save_csv(directory / "VALIDATION_THRESHOLDS.csv", rows)
                cells = metrics.metric_rows(prediction, name, role, thresholds[name])
                row = metrics.summary_row(cells, name, role, VERSION)
                run_rows.append(dict(system=system, seed_index=seed, **row))
                cell_rows.extend(dict(system=system, seed_index=seed, **r) for r in cells)
                helper.save_csv(output / "RUN_RESULTS.csv", run_rows)
                helper.save_csv(output / "RESULTS_21CELL.csv", cell_rows)
                print(f"RESULT {role} {name} AP={100 * row['equal_cell_macro_auprc']:.4f} "
                      f"AUROC={100 * row['equal_cell_macro_auroc']:.4f}", flush=True)
        if role == "validation":
            helper.save_json(output / "VALIDATION_LOCK.json", dict(
                architecture_or_epoch_selection_performed=False,
                thresholds_sha256={name: helper.sha256(output / name / "VALIDATION_THRESHOLDS.csv") for name in thresholds}))
    summaries, differences, cells_summary = [], [], []
    for role in ("validation", "test"):
        for system in SYSTEMS:
            rows = [r for r in run_rows if r["role"] == role and r["system"] == system]
            summary = dict(role=role, system=system, seeds=len(rows))
            for field in SUMMARY_FIELDS:
                values = np.array([r[field] for r in rows])
                summary[field + "_mean"] = float(values.mean())
                summary[field + "_seed_sd"] = float(values.std(ddof=1))
            summaries.append(summary)
        for seed in SEEDS:
            a = next(r for r in run_rows if r["role"] == role and r["system"] == "joint" and r["seed_index"] == seed)
            b = next(r for r in run_rows if r["role"] == role and r["system"] == "independent_targets" and r["seed_index"] == seed)
            differences.append(dict(role=role, seed_index=seed,
                                    **{field + "_joint_minus_independent": a[field] - b[field] for field in SUMMARY_FIELDS}))
        for task in TASKS:
            for horizon in HORIZONS:
                selected = [r for r in cell_rows if r["role"] == role and r["task_id"] == task and r["horizon_minutes"] == horizon]
                result = dict(role=role, task_id=task, horizon_minutes=horizon)
                for field in ("auprc", "auroc", "f1", "brier", "ece_10bin"):
                    for system in SYSTEMS:
                        result[system + "_" + field] = float(np.mean([r[field] for r in selected if r["system"] == system]))
                    result["joint_minus_independent_" + field] = result["joint_" + field] - result["independent_targets_" + field]
                cells_summary.append(result)
    helper.save_csv(output / "FIVE_SEED_SUMMARY.csv", summaries)
    helper.save_csv(output / "SEED_PAIRED_DIFFERENCES.csv", differences)
    helper.save_csv(output / "PAIRED_21CELL_RESULTS.csv", cells_summary)
    return run_rows


def bootstrap_intervals(args, loaded, output, inverse, patients, contract_hash):
    helper = loaded["io"]
    prediction_hashes = {f"{system}_seed{seed:02d}/{filename}":
                         helper.sha256(output / f"{system}_seed{seed:02d}" / filename)
                         for seed in SEEDS for system in SYSTEMS
                         for filename in ("TEST_PREDICTIONS.npz", "VALIDATION_THRESHOLDS.csv")}
    bootstrap_hash = boot.canonical_hash(dict(contract_hash=contract_hash, predictions=prediction_hashes))
    reference = boot.read_npz(output / "joint_seed01/TEST_PREDICTIONS.npz")
    selections = [(reference["masks"][:, t] > 0) & (reference["horizons"] == h) for t in range(7) for h in HORIZONS]
    support, cache = [], []
    for selected, (task, _) in zip(selections, ((t, h) for t in range(7) for h in HORIZONS)):
        y = reference["labels"][selected, task]
        support.append((np.bincount(inverse[selected][y == 1], minlength=patients),
                        np.bincount(inverse[selected][y == 0], minlength=patients)))
    point = np.empty((5, 2, 3))
    for s, seed in enumerate(SEEDS):
        seed_cells = []
        for m, system in enumerate(SYSTEMS):
            name = f"{system}_seed{seed:02d}"
            prediction = boot.read_npz(output / name / "TEST_PREDICTIONS.npz")
            check_reference(reference, prediction, name)
            rows = boot.read_csv(output / name / "VALIDATION_THRESHOLDS.csv")
            thresholds = {(r["task_id"], int(r["horizon_minutes"])): float(r["threshold"]) for r in rows}
            cells = []
            for c, selected in enumerate(selections):
                task, hslot = divmod(c, 3)
                cells.append(boot.prepare_cell(reference["labels"][selected, task], prediction["probability"][selected, task],
                                              inverse[selected], thresholds[(TASKS[task], HORIZONS[hslot])]))
            seed_cells.append(cells)
            point[s, m] = np.mean([boot.weighted_metrics(c, np.ones(patients)) for c in cells], axis=0)
        cache.append(seed_cells)

    def replicate(number):
        counts, slots = boot.draw(args.bootstrap_seed, number, patients)
        valid = all(np.dot(pos, counts) > 0 and np.dot(neg, counts) > 0 for pos, neg in support)
        values = np.full((5, 2, 3), np.nan)
        if valid:
            for s in range(5):
                for m in range(2):
                    values[s, m] = np.mean([boot.weighted_metrics(c, counts) for c in cache[s][m]], axis=0)
            require(np.isfinite(values).all(), "Supported bootstrap produced non-finite metrics")
        return values, slots, valid

    directory = output / "paired_bootstrap"
    directory.mkdir(exist_ok=True)
    chunks, slot_chunks, valid_chunks = [], [], []
    with ThreadPoolExecutor(max_workers=args.bootstrap_workers) as executor:
        for start in range(0, args.replicates, 25):
            end = min(start + 25, args.replicates)
            path = directory / f"CHUNK_{start:06d}_{end:06d}.npz"
            if path.exists():
                with np.load(path, allow_pickle=False) as z:
                    require(str(z["contract_hash"]) == bootstrap_hash, "Bootstrap inputs or chunk protocol changed")
                    values, slots, valid = z["metrics"], z["seed_slots"], z["valid"]
                    require(values.shape == (end - start, 5, 2, 3) and slots.shape == (end - start, 5)
                            and valid.shape == (end - start,), "Bootstrap chunk geometry changed")
            else:
                draws = list(executor.map(replicate, range(start, end)))
                values = np.stack([r[0] for r in draws])
                slots = np.stack([r[1] for r in draws])
                valid = np.array([r[2] for r in draws], dtype=bool)
                helper.save_npz(path, metrics=values, seed_slots=slots, valid=valid, contract_hash=np.array(bootstrap_hash))
            chunks.append(values)
            slot_chunks.append(slots)
            valid_chunks.append(valid)
            helper.save_json(directory / "PROGRESS.json", dict(completed=end, attempted=args.replicates,
                                                               estimable=int(sum(int(v.sum()) for v in valid_chunks))))
            print(f"BOOTSTRAP {end}/{args.replicates}", flush=True)
    values, slots, valid = np.concatenate(chunks), np.concatenate(slot_chunks), np.concatenate(valid_chunks)
    require(int(valid.sum()) >= max(2, .95 * args.replicates), "Fewer than 95% of bootstrap draws estimable; review support")
    differences = values[:, :, 0] - values[:, :, 1]
    fixed = differences.mean(axis=1)
    crossed = np.take_along_axis(differences, slots[:, :, None], axis=1).mean(axis=1)
    result = []
    for design, estimates in (("patient_clustered_fixed_five_seeds", fixed), ("crossed_seed_patient_sensitivity", crossed)):
        for m, metric in enumerate(("Macro_AP", "Macro_AUROC", "Macro_F1")):
            lo, hi = np.percentile(estimates[valid, m], [2.5, 97.5])
            result.append(dict(design=design, metric=metric, difference="joint_minus_independent_targets",
                               joint=float(point[:, 0, m].mean()), independent_targets=float(point[:, 1, m].mean()),
                               delta=float((point[:, 0, m] - point[:, 1, m]).mean()), ci_low=float(lo), ci_high=float(hi),
                               attempted_replicates=args.replicates, estimable_replicates=int(valid.sum()),
                               unit="proportion; multiply by 100 for percentage points"))
    helper.save_csv(output / "PAIRED_BOOTSTRAP_INTERVALS.csv", result)
    helper.save_json(directory / "COMPLETE.json", dict(contract_hash=contract_hash, attempted=args.replicates,
                                                     estimable=int(valid.sum()), fixed_seeds_principal=True))


def check_benchmark_isolation(device):
    if device.type != "cuda":
        return
    selected = next(row for row in original.gpu_inventory() if row["device"] == str(device))
    uuid = original.subprocess.run([
        "nvidia-smi", "--id=" + selected["physical"], "--query-gpu=uuid",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.strip()
    require(uuid and "\n" not in uuid, "Cannot identify the benchmark GPU")
    processes = original.subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    other = [row[1].strip() for row in csv.reader(processes.stdout.splitlines())
             if len(row) >= 2 and row[0].strip() == uuid and row[1].strip() != str(os.getpid())]
    require(not other, f"Benchmark GPU has other compute processes: {other}; retry on an idle GPU")


def benchmark(args, loaded, output, x, device, joint_sources):
    # This measures resident, single-fold model forward passes, not a deployed
    # four-fold ensemble, feature construction, transfers, or clinical latency.
    require(not getattr(args, "allow_shared_gpu", False), "Do not benchmark on an explicitly shared GPU")
    check_benchmark_isolation(device)
    joint = checkpoint_model(loaded, output, 1, 0, None, device, joint_sources)
    independent = [checkpoint_model(loaded, output, 1, 0, task, device, joint_sources) for task in range(7)]
    rows = []
    for batch in (1, min(256, len(x))):
        values = torch.as_tensor(np.array(x[:batch], dtype=np.float32, copy=True).reshape(-1, 3, 22, 10), device=device)
        with torch.inference_mode():
            for system, models in (("joint", [joint]), ("independent_targets", independent)):
                for _ in range(10):
                    for model in models:
                        model(values)
                sync(device)
                elapsed = []
                for _ in range(30):
                    sync(device)
                    start = time.perf_counter()
                    for model in models:
                        model(values)
                    sync(device)
                    elapsed.append(1000 * (time.perf_counter() - start))
                check_benchmark_isolation(device)
                rows.append(dict(system=system, batch_size=batch, seed_index=1, fold=0,
                                 repeats=30, median_ms=float(np.median(elapsed)), p95_ms=float(np.percentile(elapsed, 95)),
                                 device=str(device), device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                                 scope="resident single-fold forward passes producing all 21 logits; excludes preprocessing/transfers/ensemble"))
    loaded["io"].save_csv(output / "FORWARD_BENCHMARK.csv", rows)
    del joint, independent
    if device.type == "cuda":
        torch.cuda.empty_cache()


def benchmark_or_defer(args, loaded, output, x, device, joint_sources, contract_hash):
    helper = loaded["io"]
    path = output / "FORWARD_BENCHMARK_STATUS.json"
    csv_path = output / "FORWARD_BENCHMARK.csv"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        require(previous["contract_hash"] == contract_hash, "Benchmark protocol changed")
        if previous["status"] == "COMPLETE":
            require(previous["csv_sha256"] == helper.sha256(csv_path), "Benchmark CSV changed")
            print("BENCHMARK_REUSED", csv_path, flush=True)
            return previous
    if getattr(args, "allow_shared_gpu", False):
        require(not csv_path.exists(), "Unexpected unverified benchmark CSV; preserve it and review before proceeding")
        status = dict(status="DEFERRED_SHARED_GPU", contract_hash=contract_hash,
                      reason="Other GPU users may affect timings; measure separately without --allow-shared-gpu")
        helper.save_json(path, status)
        print("BENCHMARK_DEFERRED_SHARED_GPU; prediction evaluation and bootstrap will continue", flush=True)
        return status
    for task in (None, *range(7)):
        if task is None and joint_sources:
            continue
        directory = output / job_name(task, 1) / "fold0"
        marker = json.loads((directory / "COMPLETE.json").read_text(encoding="utf-8"))
        require(marker["contract_hash"] == contract_hash, "Benchmark checkpoint protocol changed")
        require(marker["checkpoint_sha256"] == helper.sha256(directory / "epoch5.pt"),
                "Benchmark checkpoint changed")
    benchmark(args, loaded, output, x, device, joint_sources)
    status = dict(status="COMPLETE", contract_hash=contract_hash, csv_sha256=helper.sha256(csv_path),
                  device=str(device), sharing_explicitly_allowed=False,
                  scope="resident single-fold forward only; isolation checked before and after timing blocks")
    helper.save_json(path, status)
    print("BENCHMARK_COMPLETE", csv_path, flush=True)
    return status


def validate_args(args):
    require(args.cpu_threads > 0 and args.bootstrap_workers > 0, "Thread counts must be positive")
    require(args.replicates >= 100, "At least 100 bootstrap draws required; use 5000 for the planned analysis")
    require(args.min_free_mib >= 6000, "--min-free-mib must be at least 6000")
    if args.allow_shared_gpu:
        require(args.device.startswith("cuda:") and args.device[5:].isdigit(),
                "--allow-shared-gpu requires --device cuda:N; auto and cpu are not allowed")
        require(args.command != "benchmark", "The benchmark command must use an idle GPU without --allow-shared-gpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "benchmark"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-shared-gpu", action="store_true",
                        help="Allow sharing an explicitly named cuda:N device; defer inference timing")
    parser.add_argument("--min-free-mib", type=int, default=6000,
                        help="Minimum free GPU memory at selection (at least 6000 MiB; not a reservation)")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--joint-mode", choices=("reuse", "train"), default="reuse")
    parser.add_argument("--joint-roots", type=Path, nargs="+")
    parser.add_argument("--patient-mapping", type=Path)
    parser.add_argument("--replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-workers", type=int, default=2)
    parser.add_argument("--bootstrap-seed", type=int, default=20261018)
    for name in ("development_features", "development_groups", "validation_features", "validation_reference", "test_features", "test_reference"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    validate_args(args)
    root = args.project_root.resolve()
    loaded, source_files = original.modules(root)
    require(Path(original.__file__).resolve() == root / "scripts/run_ssthnet_ablations.py",
            "Run the experiment script inside its own project root")
    helper = loaded["io"]
    source_files.update(joint_runner=Path(__file__).resolve(), ablation_runner=root / "scripts/run_ssthnet_ablations.py",
                        bootstrap_engine=root / "scripts/bootstrap_ssthnet_five_seed_ablations.py")
    paths = helper.sources(args)
    x, groups = helper.development(paths)
    outer_data = {role: helper.outer(paths, role) for role in ("validation", "test")}
    mapping_path = (args.patient_mapping or root / MAPPING).resolve()
    test_reference = outer_data["test"][1]
    mapping = test_reference if "subject_id" in test_reference else boot.read_npz(mapping_path)
    inverse, patients = boot.map_patients(test_reference, mapping)
    if "subject_id" in outer_data["validation"][1]:
        require(not len(np.intersect1d(outer_data["validation"][1]["subject_id"], mapping["subject_id"])),
                "Validation/test patient overlap")
    print("Hashing frozen inputs and implementation...", flush=True)
    input_hashes = {k: helper.sha256(v) for k, v in paths.items()}
    source_hashes = {k: helper.sha256(v) for k, v in source_files.items()}
    joint_sources = audit_joint_sources(args, loaded, input_hashes, source_hashes, outer_data)
    contract = dict(version=VERSION, inputs=input_hashes, sources=source_hashes, seeds=list(SEEDS), folds=list(FOLDS),
                    targets=list(TASKS), horizons=list(HORIZONS), features=660, training_epochs=6, evaluation_epoch=5,
                    batch_size=128, evaluation_batch_size=256, learning_rate=2e-4, weight_decay=1e-4,
                    gradient_clip=5., loss="eligible-cell-mean class-weighted BCE; target slice only for independent models",
                    joint_mode=args.joint_mode, joint_sources=joint_sources, torch_version=torch.__version__, numpy_version=np.__version__,
                    patient_mapping_sha256=helper.sha256(mapping_path) if "subject_id" not in test_reference else input_hashes["test_reference"],
                    bootstrap_replicates=args.replicates, bootstrap_seed=args.bootstrap_seed,
                    principal_uncertainty="patient_clustered_fixed_five_seeds", secondary_uncertainty="crossed_seed_patient_sensitivity",
                    primary_metric="overall equal-cell macro AP; joint minus independent",
                    interpretation="posthoc same-backbone fixed-protocol cross-target sharing comparison; not equal total compute",
                    current_normal="not calculated by this runner; requires a separately verified target-specific anchor-normal mask")
    contract_hash = boot.canonical_hash(contract)
    print(f"PREFLIGHT_COMPLETE seeds=5 folds=4 independent_training_jobs=140 joint_mode={args.joint_mode} patients={patients}", flush=True)
    if args.command == "preflight":
        return
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = choose_device(args.device, args.allow_shared_gpu, args.min_free_mib)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    output = (args.out or root / "reports" / VERSION).resolve()
    protected = [*paths.values(), mapping_path, *[Path(v["directory"]) for v in joint_sources.values()]]
    require(all(output != p and output not in p.parents and p not in output.parents for p in protected), "Output overlaps an input or old result")
    if args.command == "benchmark":
        require((output / "protocol.json").is_file() and (output / "COMPLETE.json").is_file(),
                "Finish the prediction-analysis run before requesting benchmark-only execution")
        require(json.loads((output / "protocol.json").read_text(encoding="utf-8")) == contract,
                "Inputs/code/protocol changed; do not benchmark mismatched results")
        with run_lock(output):
            completion = json.loads((output / "COMPLETE.json").read_text(encoding="utf-8"))
            require(completion["contract_hash"] == contract_hash, "Completed analysis protocol changed")
            status = benchmark_or_defer(args, loaded, output, outer_data["test"][0], device, joint_sources, contract_hash)
            completion.update(status="COMPLETE_POSTHOC_JOINT_VS_INDEPENDENT_TARGETS",
                              benchmark_status=status["status"], all_planned_outputs_complete=True)
            helper.save_json(output / "COMPLETE.json", completion)
        return
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        manifest = output / "protocol.json"
        if manifest.exists():
            require(json.loads(manifest.read_text(encoding="utf-8")) == contract, "Inputs/code/protocol changed; use a new --out directory")
        else:
            helper.save_json(manifest, contract)
        helper.save_json(output / "RUN_ENVIRONMENT.json", dict(python=sys.version, torch=torch.__version__,
                         numpy=np.__version__, device=str(device), cpu_threads=args.cpu_threads,
                         bootstrap_workers=args.bootstrap_workers, cuda_version=torch.version.cuda,
                         allow_shared_gpu=args.allow_shared_gpu, minimum_free_gpu_mib=args.min_free_mib,
                         cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
        specs = []
        for task in (None, *range(7)):
            model = make_model(loaded["models"], task)
            specs.append(dict(configuration="joint" if task is None else TASKS[task], targets=7 if task is None else 1,
                              horizons=3, inputs=660, parameters=sum(p.numel() for p in model.parameters())))
        helper.save_csv(output / "MODEL_SPECIFICATIONS.csv", specs)
        stats = {}
        for fold in FOLDS:
            train = np.flatnonzero(groups["group_fold"] != fold)
            mean, scale = loaded["training"].fit_scaler(x, groups["representative_index"][train])
            require(np.isfinite(mean).all() and np.isfinite(scale).all(), "Non-finite training-fold scaler")
            stats[fold] = mean, scale, loaded["training"].positive_weights(groups["labels"], groups["masks"], train)
        for seed in SEEDS:
            for task in ((None, *range(7)) if args.joint_mode == "train" else range(7)):
                for fold in FOLDS:
                    train_fold(loaded, x, groups, stats, output, seed, fold, task, device, contract_hash)
        helper.save_json(output / "TRAINING_COMPLETE.json", dict(contract_hash=contract_hash, independent_folds=140,
                         joint_folds_reused=20 if joint_sources else 0, joint_folds_trained=0 if joint_sources else 20))
        evaluate(loaded, output, outer_data, device, joint_sources, contract_hash)
        benchmark_status = benchmark_or_defer(args, loaded, output, outer_data["test"][0], device, joint_sources, contract_hash)
        bootstrap_intervals(args, loaded, output, inverse, patients, contract_hash)
        all_complete = benchmark_status["status"] == "COMPLETE"
        helper.save_json(output / "COMPLETE.json", dict(status="COMPLETE_POSTHOC_JOINT_VS_INDEPENDENT_TARGETS" if all_complete
                                                            else "COMPLETE_PREDICTION_ANALYSIS_BENCHMARK_DEFERRED",
                         contract_hash=contract_hash, seeds=list(SEEDS), independent_training_folds=140,
                         bootstrap_replicates=args.replicates, current_normal_evaluated=False,
                         benchmark_status=benchmark_status["status"], all_planned_outputs_complete=all_complete))
        print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
