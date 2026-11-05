#!/usr/bin/env python3
"""Bounded, validation-only SSTHNet training diagnosis; no test-set access."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssthnet_joint_vs_independent as shared

original = shared.original
require = shared.require
VERSION = "ssthnet_training_optimization_v1"
CANDIDATES = ("A_constant_weighted", "B_cosine_weighted", "C_cosine_unweighted")
SCREEN_SEEDS = (1, 2)
CONFIRM_SEEDS = (3, 4, 5)
SELECT_EPOCHS = (5, 10, 15, 20)
METRICS = ("average_precision", "auroc", "brier_score", "log_loss", "ece_10bin")
INPUT_KEYS = ("development_features", "development_groups", "validation_features", "validation_reference")


def learning_rate(candidate, epoch):
    require(candidate in CANDIDATES and 1 <= epoch <= 20, "Invalid training setting")
    if candidate == CANDIDATES[0]:
        return 2e-4
    # Epoch 1 starts at 2e-4; epoch 20 starts at 2e-5, also on resumed jobs.
    return 2e-5 + (2e-4 - 2e-5) * (1 + math.cos(math.pi * (epoch - 1) / 19)) / 2


def class_weights(candidate, original_weights):
    require(candidate in CANDIDATES, "Invalid candidate")
    return np.ones_like(original_weights) if candidate == CANDIDATES[2] else original_weights.copy()


def input_paths(helper, root):
    args = SimpleNamespace(project_root=root, **{key: None for key in (
        *INPUT_KEYS, "test_features", "test_reference")})
    defaults = helper.sources(args)
    return {key: defaults[key] for key in INPUT_KEYS}


def job_dir(output, candidate, seed, fold):
    return output / f"{candidate}_seed{seed:02d}" / f"fold{fold}"


def screening_jobs():
    return [dict(candidate=candidate, seed=seed, fold=fold, epochs=20)
            for seed in SCREEN_SEEDS for fold in original.FOLDS for candidate in CANDIDATES]


def confirmation_jobs(selection):
    require(selection["candidate"] in CANDIDATES and selection["epoch"] in SELECT_EPOCHS,
            "Invalid frozen selection")
    return [dict(candidate=selection["candidate"], seed=seed, fold=fold, epochs=selection["epoch"])
            for seed in CONFIRM_SEEDS for fold in original.FOLDS]


@contextmanager
def exclusive_lock(path):
    with path.open("x") as stream:
        json.dump(dict(pid=os.getpid(), host=socket.gethostname()), stream)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def prepare(root, output, create=False):
    loaded, sources = original.modules(root)
    helper = loaded["io"]
    paths = input_paths(helper, root)
    old = root / "reports" / shared.VERSION
    frozen = json.loads((old / "protocol.json").read_text())
    for key, path in sources.items():
        require(helper.sha256(path) == frozen["sources"][key], "Original implementation changed: " + key)
    require(helper.sha256(Path(original.__file__)) == frozen["sources"]["ablation_runner"],
            "Original architecture/training helpers changed")
    hashes = {key: helper.sha256(path) for key, path in paths.items()}
    require(all(hashes[key] == frozen["inputs"][key] for key in INPUT_KEYS),
            "Development/validation inputs differ from the original experiment")
    baseline_paths = {str(seed): old / f"joint_seed{seed:02d}/VALIDATION_PREDICTIONS.npz"
                      for seed in (*SCREEN_SEEDS, *CONFIRM_SEEDS)}
    baseline_hashes = {seed: helper.sha256(path) for seed, path in baseline_paths.items()}
    require(all(baseline_hashes[seed] == frozen["joint_sources"][seed]["sha256"]["validation"]
                for seed in baseline_paths), "Original validation predictions changed")
    sources.update(original_runner=Path(original.__file__), shared_runner=Path(shared.__file__),
                   optimizer_runner=Path(__file__))
    protocol = dict(
        version=VERSION, experiment="posthoc development; previously viewed test is not a new confirmation set",
        inputs={key: dict(path=str(paths[key]), sha256=hashes[key]) for key in INPUT_KEYS},
        sources={key: dict(path=str(path), sha256=helper.sha256(path)) for key, path in sources.items()},
        baseline_validation={seed: dict(path=str(path), sha256=baseline_hashes[seed])
                             for seed, path in baseline_paths.items()},
        architecture="unchanged final single-shared-expert SSTHNet; 660 inputs, 7 x 3 outputs",
        candidates=list(CANDIDATES), maximum_epochs=20, screening_seeds=list(SCREEN_SEEDS),
        confirmation_seeds=list(CONFIRM_SEEDS), folds=list(original.FOLDS),
        selection_epochs=list(SELECT_EPOCHS), selection_metric="validation macro-AP",
        selection_rule="four-fold probability mean, then per-seed metrics, then seed mean; maximum macro-AP; "
                       "ties within 1e-12 use A/B/C order then earliest epoch; baseline retained if no positive gain",
        confirmation_rule="only selected configuration and same selected epoch; no reselection on seeds 3-5",
        optimizer="AdamW", base_lr=2e-4, minimum_cosine_lr=2e-5,
        schedule="epoch-level cosine: pi*(epoch-1)/19; no warm restarts",
        weight_decay=1e-4, batch_size=128, prediction_batch_size=256, gradient_clip=5.0,
        positive_weights="A/B: original per-fold clipped sqrt(negative/positive); C: ones",
        loss="unchanged masked BCE averaged over estimable target-horizon cells",
        seed_formula="20261884 + (seed_index - 1) * 1000 + fold",
        scaler="original training-subset-only per-fold standardisation",
        cpu_threads=2, cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        deterministic_algorithms="original warn_only=True; CUDA warnings retained in logs",
        metric_policy="raw probabilities; AP primary; AUROC, Brier, log loss and ECE diagnostic; no test reads",
        ece="ten equal-width bins; p=1 in last bin; observation-count-weighted absolute calibration gap",
        log_loss="natural log; probabilities clipped to [1e-7, 1-1e-7] for numerical stability",
        torch_version=torch.__version__, numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    fingerprint = shared.boot.canonical_hash(protocol)
    manifest = output / "protocol.json"
    if manifest.exists():
        require(json.loads(manifest.read_text()) == protocol, "Optimization protocol changed; use a new version")
    else:
        require(create, "Run preflight before starting jobs")
        output.mkdir(parents=True, exist_ok=True)
        require(not list(output.iterdir()), "New output directory must be empty")
        helper.save_json(manifest, protocol)
    x, groups = helper.development(paths)
    val_x, reference = helper.outer(paths, "validation")
    helper.save_json(output / "PREFLIGHT.json", dict(contract_hash=fingerprint,
        screening_jobs=24, maximum_confirmation_jobs=12, validation_rows=len(val_x),
        test_files_opened=False, created_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))) if create else None
    print("VALIDATION_ONLY_PREFLIGHT_COMPLETE", fingerprint, "screening_jobs=24 test_files_opened=0", flush=True)
    return loaded, x, groups, val_x, reference, protocol, fingerprint


def diagnostic_rows(helper, reference, probability):
    rows = helper.cell_metrics(reference, probability, "validation")
    for row in rows:
        task = shared.TASKS.index(row["task_id"])
        selected = (reference["horizons"] == row["horizon_minutes"]) & (reference["masks"][:, task] > 0)
        y = reference["labels"][selected, task].astype(np.float64)
        p = probability[selected, task].astype(np.float64)
        clipped = np.clip(p, 1e-7, 1 - 1e-7)
        row["log_loss"] = float(-np.mean(y * np.log(clipped) + (1-y) * np.log1p(-clipped)))
        bins = np.minimum((p * 10).astype(np.int64), 9)
        counts = np.bincount(bins, minlength=10)
        gaps = np.bincount(bins, weights=p-y, minlength=10)
        row["ece_10bin"] = float(np.abs(gaps[counts > 0]).sum() / len(y))
    return rows


def macro(rows):
    require(len(rows) == 21, "Expected all 21 target-horizon cells")
    return {key: float(np.mean([row[key] for row in rows])) for key in METRICS}


def predict(model, x, reference, device):
    model.eval()
    slots = np.searchsorted(np.array(shared.HORIZONS), reference["horizons"])
    result = np.empty(reference["labels"].shape, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(x), 256):
            end = min(start + 256, len(x))
            values = np.array(x[start:end], dtype=np.float32, copy=True).reshape(-1, 3, 22, 10)
            logits, _ = model(torch.as_tensor(values, device=device))
            p = torch.sigmoid(logits).cpu().numpy()
            result[start:end] = p[np.arange(end-start), :, slots[start:end]]
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "Invalid validation probability")
    return result


def verify_complete(helper, directory, fingerprint, job):
    marker = json.loads((directory / "COMPLETE.json").read_text())
    require(marker["contract_hash"] == fingerprint and marker["job"] == job, "Completed job identity changed")
    for name, expected in marker["files"].items():
        require(helper.sha256(directory / name) == expected, "Completed output changed: " + name)


def train_job(context, output, job, device):
    loaded, x, groups, val_x, reference, protocol, fingerprint = context
    helper, training = loaded["io"], loaded["training"]
    candidate, seed, fold, epochs = (job[key] for key in ("candidate", "seed", "fold", "epochs"))
    directory = job_dir(output, candidate, seed, fold)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "COMPLETE.json").exists():
        verify_complete(helper, directory, fingerprint, job)
        print("JOB_REUSED", job, flush=True)
        return
    with exclusive_lock(directory / "RUNNING.lock"):
        seed_value = 20261884 + (seed - 1) * 1000 + fold
        original.seed_everything(seed_value)
        model = original.make_model(loaded["models"], "ssthnet").to(device)
        require(sum(p.numel() for p in model.parameters()) == 155271, "Final-model architecture changed")
        train = np.flatnonzero(groups["group_fold"] != fold)
        mean, scale = training.fit_scaler(x, groups["representative_index"][train])
        require(np.isfinite(mean).all() and np.isfinite(scale).all(), "Invalid fold scaler")
        model.set_scaler(torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device))
        weights = class_weights(candidate, training.positive_weights(groups["labels"], groups["masks"], train))
        positive = torch.as_tensor(weights, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        rng = np.random.default_rng(seed_value)
        history, start_epoch = [], 1
        latest = directory / "latest.pt"
        if latest.exists():
            saved = original.load_checkpoint(latest)
            require(saved["contract_hash"] == fingerprint and saved["job"] == job, "Resume identity changed")
            require(saved["device_type"] == device.type, "Resume device type changed")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            rng.bit_generator.state = saved["numpy_generator_state"]
            original.random.setstate(saved["python_rng_state"])
            np.random.set_state(saved["numpy_rng_state"])
            torch.set_rng_state(saved["torch_rng_state"])
            if device.type == "cuda":
                torch.cuda.set_rng_state(saved["cuda_rng_state"], device)
            history, start_epoch = saved["history"], saved["epoch"] + 1
        for epoch in range(start_epoch, epochs + 1):
            rate = learning_rate(candidate, epoch)
            for group in optimizer.param_groups:
                group["lr"] = rate
            model.train()
            shuffled = rng.permutation(train)
            loss_sum, used, skipped = 0., 0, 0
            shared.sync(device)
            started = time.perf_counter()
            for start in range(0, len(shuffled), 128):
                selected = shuffled[start:start+128]
                labels, masks = shared.target_labels(groups, selected, None)
                if not masks.any():
                    skipped += 1
                    continue
                values = np.asarray(x[groups["representative_index"][selected]], dtype=np.float32).reshape(-1, 3, 22, 10)
                require(np.isfinite(values).all(), "Non-finite frozen feature")
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(torch.as_tensor(values, device=device))
                loss = shared.masked_loss(logits, torch.as_tensor(labels, device=device),
                                          torch.as_tensor(masks, device=device), positive)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                loss_sum += float(loss.detach()) * len(selected)
                used += len(selected)
                if used // 25000 != (used - len(selected)) // 25000:
                    print(f"TRAIN {candidate} seed={seed} fold={fold} epoch={epoch} groups={used}/{len(train)}", flush=True)
            require(used > 0, "No eligible training labels")
            shared.sync(device)
            training_seconds = time.perf_counter() - started
            probability = predict(model, val_x, reference, device)
            values = macro(diagnostic_rows(helper, reference, probability))
            prediction_path = directory / f"epoch{epoch:02d}_validation.npz"
            helper.save_npz(prediction_path, probability=probability, contract_hash=np.array(fingerprint),
                            validation_reference_sha256=np.array(protocol["inputs"]["validation_reference"]["sha256"]))
            history.append(dict(epoch=epoch, learning_rate=rate, mean_training_loss=loss_sum/used,
                                training_seconds=training_seconds, skipped_batches=skipped, **values))
            saved = dict(job=job, epoch=epoch, contract_hash=fingerprint, device_type=device.type,
                         device=str(device), model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                         history=history, numpy_generator_state=rng.bit_generator.state,
                         python_rng_state=original.random.getstate(), numpy_rng_state=np.random.get_state(),
                         torch_rng_state=torch.get_rng_state(),
                         cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
            if epoch in SELECT_EPOCHS:
                original.atomic_checkpoint(directory / f"epoch{epoch:02d}.pt", dict(
                    job=job, epoch=epoch, contract_hash=fingerprint, model_state=model.state_dict()))
            original.atomic_checkpoint(latest, saved)
            helper.save_csv(directory / "VALIDATION_HISTORY.csv", history)
            print(f"EPOCH_COMPLETE {candidate} seed={seed} fold={fold} epoch={epoch} "
                  f"loss={loss_sum/used:.6f} validation_AP={values['average_precision']:.6f} "
                  f"validation_AUROC={values['auroc']:.6f}", flush=True)
        helper.save_csv(directory / "VALIDATION_HISTORY.csv", history)
        files = {f"epoch{epoch:02d}_validation.npz": helper.sha256(directory / f"epoch{epoch:02d}_validation.npz")
                 for epoch in range(1, epochs+1)}
        files.update({f"epoch{epoch:02d}.pt": helper.sha256(directory / f"epoch{epoch:02d}.pt")
                      for epoch in SELECT_EPOCHS if epoch <= epochs})
        files["VALIDATION_HISTORY.csv"] = helper.sha256(directory / "VALIDATION_HISTORY.csv")
        helper.save_json(directory / "COMPLETE.json", dict(job=job, contract_hash=fingerprint, files=files,
                                                          device=str(device), test_files_opened=False))
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("JOB_COMPLETE", job, flush=True)


def average_folds(output, candidate, seed, epoch, fingerprint, reference_sha):
    total = None
    for fold in original.FOLDS:
        with np.load(job_dir(output, candidate, seed, fold) / f"epoch{epoch:02d}_validation.npz", allow_pickle=False) as z:
            require(str(z["contract_hash"]) == fingerprint and str(z["validation_reference_sha256"]) == reference_sha,
                    "Validation prediction identity changed")
            p = z["probability"].astype(np.float64)
            require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "Invalid fold probabilities")
            if total is None:
                total = p
            else:
                require(total.shape == p.shape, "Fold probability shape changed")
                total += p
    return (total / 4).astype(np.float32)


def choose(curves, baseline):
    eligible = [row for row in curves if row["epoch"] in SELECT_EPOCHS]
    require(len(eligible) == len(CANDIDATES)*len(SELECT_EPOCHS), "Incomplete candidate/epoch grid")
    maximum = max(row["average_precision"] for row in eligible)
    tied = [row for row in eligible if maximum - row["average_precision"] <= 1e-12]
    best = min(tied, key=lambda row: (CANDIDATES.index(row["candidate"]), row["epoch"]))
    return dict(candidate=best["candidate"], epoch=best["epoch"], validation_macro_ap=best["average_precision"],
                baseline_validation_macro_ap=baseline, validation_gain=best["average_precision"]-baseline,
                proceed_to_confirmation=best["average_precision"] > baseline + 1e-12,
                selection_seeds=list(SCREEN_SEEDS), diagnostic_not_independent_confirmation=True)


def aggregate(context, output, candidates, seeds, epochs, prefix):
    loaded, _, _, _, reference, protocol, fingerprint = context
    helper = loaded["io"]
    seed_rows, cell_rows, curves = [], [], []
    for candidate in candidates:
        for epoch in epochs:
            current = []
            for seed in seeds:
                probability = average_folds(output, candidate, seed, epoch, fingerprint,
                                            protocol["inputs"]["validation_reference"]["sha256"])
                rows = diagnostic_rows(helper, reference, probability)
                summary = dict(candidate=candidate, seed=seed, epoch=epoch, **macro(rows))
                seed_rows.append(summary)
                current.append(summary)
                cell_rows.extend(dict(candidate=candidate, seed=seed, epoch=epoch, **row) for row in rows)
            curves.append(dict(candidate=candidate, epoch=epoch, seeds=len(seeds),
                               **{key: float(np.mean([row[key] for row in current])) for key in METRICS}))
    helper.save_csv(output / f"{prefix}_SEED_CURVES.csv", seed_rows)
    helper.save_csv(output / f"{prefix}_21CELL_RESULTS.csv", cell_rows)
    helper.save_csv(output / f"{prefix}_CURVES.csv", curves)
    return curves


def baseline_metrics(context, output):
    loaded, _, _, _, reference, protocol, _ = context
    helper = loaded["io"]
    rows = []
    for seed in (*SCREEN_SEEDS, *CONFIRM_SEEDS):
        source = protocol["baseline_validation"][str(seed)]
        require(helper.sha256(Path(source["path"])) == source["sha256"], "Baseline validation changed")
        prediction = shared.boot.read_npz(Path(source["path"]))
        shared.check_reference(reference, prediction, "original validation")
        rows.append(dict(seed=seed, **macro(diagnostic_rows(helper, reference, prediction["probability"]))))
    helper.save_csv(output / "ORIGINAL_VALIDATION_SEEDS.csv", rows)
    return float(np.mean([row["average_precision"] for row in rows if row["seed"] in SCREEN_SEEDS]))


def jobs_for_stage(output, stage):
    if stage == "screening":
        return screening_jobs()
    selection = json.loads((output / "SELECTION.json").read_text())
    require(selection["proceed_to_confirmation"], "No validation gain; do not add training jobs")
    return confirmation_jobs(selection)


def run_workers(args, output, stage):
    processes, streams = [], []
    try:
        for lane, device in enumerate(args.devices):
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "worker",
                       "--project-root", str(args.project_root), "--stage", stage,
                       "--lane", str(lane), "--lanes", str(len(args.devices)), "--device", device]
            if args.allow_shared_gpu:
                command.append("--allow-shared-gpu")
            env = os.environ.copy()
            env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            stream = (output / f"{stage}_lane{lane}.log").open("ab")
            streams.append(stream)
            processes.append(subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                              stdin=subprocess.DEVNULL, cwd=args.project_root, env=env,
                                              start_new_session=True))
        helper = original.modules(args.project_root)[0]["io"]
        helper.save_json(output / f"{stage}_WORKERS.json", dict(
            workers=[dict(pid=p.pid, device=d) for p, d in zip(processes, args.devices)]))
        while any(p.poll() is None for p in processes):
            failed = [p for p in processes if p.poll() not in (None, 0)]
            require(not failed, f"Worker failed; inspect {stage}_lane*.log; no automatic retuning/retry")
            complete = sum((job_dir(output, j["candidate"], j["seed"], j["fold"]) / "COMPLETE.json").exists()
                           for j in jobs_for_stage(output, stage))
            helper.save_json(output / "STATUS.json", dict(status=stage.upper()+"_RUNNING", complete_jobs=complete,
                                                          planned_jobs=len(jobs_for_stage(output, stage)),
                                                          test_files_opened=False))
            time.sleep(15)
        require(all(p.returncode == 0 for p in processes), "A training worker failed")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for stream in streams:
            stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "worker"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--devices", nargs="+", default=["cuda:4", "cuda:5", "cuda:6"])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--stage", choices=("screening", "confirmation"), default="screening")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    output = args.project_root / "reports" / VERSION
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    context = prepare(args.project_root, output, create=args.command == "preflight")
    helper = context[0]["io"]
    if args.command == "preflight":
        baseline_metrics(context, output)
        return
    if args.command == "worker":
        require(0 <= args.lane < args.lanes <= 4, "Invalid worker lane")
        device = shared.choose_device(args.device, allow_shared=args.allow_shared_gpu)
        for job in jobs_for_stage(output, args.stage)[args.lane::args.lanes]:
            train_job(context, output, job, device)
        return
    require(1 <= len(args.devices) <= 4 and len(set(args.devices)) == len(args.devices), "Use 1-4 distinct GPUs")
    with exclusive_lock(output / "RUNNING.lock"):
        try:
            run_workers(args, output, "screening")
            curves = aggregate(context, output, CANDIDATES, SCREEN_SEEDS, range(1, 21), "SCREENING")
            selection = choose(curves, baseline_metrics(context, output))
            selection["contract_hash"] = context[-1]
            selection_path = output / "SELECTION.json"
            if selection_path.exists():
                require(json.loads(selection_path.read_text()) == selection, "Frozen selection changed")
            else:
                helper.save_json(selection_path, selection)
            print("FROZEN_SELECTION", json.dumps(selection), flush=True)
            if not selection["proceed_to_confirmation"]:
                status = dict(status="COMPLETE_NO_VALIDATION_GAIN_ORIGINAL_RETAINED", test_files_opened=False)
            else:
                run_workers(args, output, "confirmation")
                aggregate(context, output, [selection["candidate"]], (*SCREEN_SEEDS, *CONFIRM_SEEDS),
                          [selection["epoch"]], "FIVE_SEED_VALIDATION")
                status = dict(status="COMPLETE_FIVE_SEED_VALIDATION_TEST_NOT_RUN", test_files_opened=False,
                              configuration=selection["candidate"], epoch=selection["epoch"])
            helper.save_json(output / "STATUS.json", status)
            helper.save_json(output / "COMPLETE.json", dict(contract_hash=context[-1], **status))
        except BaseException as error:
            helper.save_json(output / "STATUS.json", dict(status="FAILED_OR_INTERRUPTED", error=repr(error),
                                                          test_files_opened=False))
            raise


if __name__ == "__main__":
    main()
