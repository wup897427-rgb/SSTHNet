#!/usr/bin/env python3
"""Prepare/run five-seed single-history-window ablations after model freeze.

The independent runner never modifies the active optimization experiment.
Training/validation logic is reused through a process-local model adapter.
Test-set evaluation is deliberately a separate, not implemented stage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssthnet_training_optimization as opt

require = opt.require
VERSION = "ssthnet_history_window_ablations_v1"
WINDOWS = (120, 300, 600)
SEEDS = (1, 2, 3, 4, 5)


def window_model_base(base, seconds):
    require(seconds in WINDOWS, "History length must be 120, 300 or 600 seconds")
    slot = WINDOWS.index(seconds)

    class SingleHistorySSTHNet(base.SSTHMoEAblation):
        def __init__(self, variant):
            require(variant == "single_expert", "Use the final single-expert SSTHNet")
            super().__init__(variant)
            self.history_seconds = seconds
            self.effective_input_features = 220
            self.physiology_token_count = 22

        def _physiology_tokens(self, values):
            require(tuple(values.shape[1:]) == (3, 22, 10), "Expected original frozen cache geometry")
            # Discard other histories BEFORE normalization/tokenization/attention.
            values = (values[:, slot:slot+1] - self.feature_mean[None, slot:slot+1]) / self.feature_scale[None, slot:slot+1]
            tokens = self.tokenizer(values)
            tokens = tokens + self.source_embedding[None, None] + self.scale_embedding[None, slot:slot+1, None]
            batch = tokens.shape[0]
            tokens = self.source_mixer(tokens.reshape(batch, 22, 64))
            tokens = tokens.reshape(batch, 1, 22, 64).permute(0, 2, 1, 3).contiguous()
            # Keep the same scale-processing layer, now with a length-one sequence.
            tokens = self.scale_mixer(tokens.reshape(batch * 22, 1, 64))
            return tokens.reshape(batch, 22, 64)

    return SimpleNamespace(SSTHMoEAblation=SingleHistorySSTHNet)


def window_context(context, seconds, fingerprint):
    loaded, x, groups, val_x, reference, protocol, _ = context
    local = dict(loaded)
    local["models"] = window_model_base(loaded["models"], seconds)
    return local, x, groups, val_x, reference, protocol, fingerprint


def plan_spec(parent_protocol_sha256):
    return dict(version=VERSION, parent_optimization_protocol_sha256=parent_protocol_sha256,
        windows_seconds=list(WINDOWS), reference="same frozen final three-history SSTHNet",
        seeds=list(SEEDS), folds=list(opt.original.FOLDS), new_training_jobs=60,
        effective_features_per_single_window=220, tokens_per_single_window=22,
        full_features=660, full_tokens=66, outputs="seven targets x three forecast horizons, unchanged",
        registered_parameters_each=155271, unused_single_window_scale_embedding_parameters=128,
        architecture="same modules/depth/width; discard excluded windows before normalization and attention; "
                     "retain length-one scale encoder and original three-row scale-embedding parameter for capacity matching",
        protocol_policy="inherit selected configuration and common evaluation epoch after optimization COMPLETE; "
                        "if no validation gain, original constant weighted protocol (train 6, evaluate epoch 5)",
        model_selection="none within window ablations; do not select separate hyperparameters/epochs per window",
        data_policy="same frozen rows, folds, target labels and masks; no additional complete-history filter",
        summary="within-seed four-fold probability mean; metrics then arithmetic mean across five seeds",
        interpretation="history-input ablation, not deletion of scale interaction; token count also changes",
        current_stage="preparation and matched validation; no test files opened; no manuscript edits")


def resolve_reference(parent, old):
    require((parent / "COMPLETE.json").exists(),
            "Training optimization has not finished; this ablation remains prepared, not started")
    state = json.loads((parent / "COMPLETE.json").read_text())
    selected = json.loads((parent / "SELECTION.json").read_text())
    require(state.get("test_files_opened") is False, "Unexpected parent evaluation policy")
    if state["status"] == "COMPLETE_NO_VALIDATION_GAIN_ORIGINAL_RETAINED":
        require(not selected["proceed_to_confirmation"], "Inconsistent parent selection")
        return dict(kind="original_epoch5", candidate=opt.CANDIDATES[0], training_epochs=6,
                    evaluation_epoch=5, root=str(old))
    require(state["status"] == "COMPLETE_FIVE_SEED_VALIDATION_TEST_NOT_RUN", "Parent optimization failed or is incomplete")
    require(selected["proceed_to_confirmation"] and selected["candidate"] == state["configuration"]
            and selected["epoch"] == state["epoch"], "Frozen parent selection differs")
    require(selected["candidate"] in opt.CANDIDATES and selected["epoch"] in opt.SELECT_EPOCHS,
            "Parent selection outside the predeclared grid")
    return dict(kind="optimized", candidate=selected["candidate"], training_epochs=selected["epoch"],
                evaluation_epoch=selected["epoch"], root=str(parent))


def plan(root, output):
    loaded, _ = opt.original.modules(root)
    helper = loaded["io"]
    parent = root / "reports" / opt.VERSION
    protocol_path = parent / "protocol.json"
    require(protocol_path.exists(), "Prepare the parent optimization protocol first")
    parent_protocol = json.loads(protocol_path.read_text())
    source_paths = dict(window_runner=Path(__file__), optimizer_runner=Path(opt.__file__),
                        original_runner=Path(opt.original.__file__), shared_runner=Path(opt.shared.__file__))
    for key, entry in parent_protocol["sources"].items():
        path = Path(entry["path"])
        require(helper.sha256(path) == entry["sha256"], "Active optimization source changed: " + key)
    specification = plan_spec(helper.sha256(protocol_path))
    specification["sources"] = {key: dict(path=str(path), sha256=helper.sha256(path)) for key, path in source_paths.items()}
    path = output / "PLAN.json"
    if path.exists():
        require(json.loads(path.read_text()) == specification, "Prepared design changed; use a new experiment version")
    else:
        output.mkdir(parents=True, exist_ok=True)
        require(not list(output.iterdir()), "New ablation output directory must be empty")
        helper.save_json(path, specification)
    return helper, parent


def full_reference_files(helper, parent, old, reference):
    result = {}
    if reference["kind"] == "original_epoch5":
        frozen = json.loads((old / "protocol.json").read_text())
        for seed in SEEDS:
            path = old / f"joint_seed{seed:02d}/VALIDATION_PREDICTIONS.npz"
            digest = helper.sha256(path)
            require(digest == frozen["joint_sources"][str(seed)]["sha256"]["validation"], "Original full-model predictions changed")
            result[str(seed)] = dict(path=str(path), sha256=digest)
    else:
        epoch = reference["evaluation_epoch"]
        for seed in SEEDS:
            for fold in opt.original.FOLDS:
                directory = opt.job_dir(parent, reference["candidate"], seed, fold)
                marker = json.loads((directory / "COMPLETE.json").read_text())
                for name in (f"epoch{epoch:02d}.pt", f"epoch{epoch:02d}_validation.npz"):
                    path = directory / name
                    digest = helper.sha256(path)
                    require(digest == marker["files"][name], "Optimized full-model output changed")
                    result[f"seed{seed:02d}/fold{fold}/{name}"] = dict(path=str(path), sha256=digest)
    return result


def prepare(root, output):
    helper, parent = plan(root, output)
    old = root / "reports" / opt.shared.VERSION
    reference = resolve_reference(parent, old)
    context = opt.prepare(root, parent)
    parent_state = json.loads((parent / "COMPLETE.json").read_text())
    require(parent_state["contract_hash"] == context[-1], "Parent completion contract changed")
    parent_protocol = context[-2]
    baseline = full_reference_files(helper, parent, old, reference)
    protocol = dict(plan=json.loads((output / "PLAN.json").read_text()), reference=reference,
        parent_contract_hash=context[-1], inputs=parent_protocol["inputs"],
        parent_completion_sha256=helper.sha256(parent / "COMPLETE.json"),
        selection_sha256=helper.sha256(parent / "SELECTION.json"), full_reference_files=baseline)
    path = output / "protocol.json"
    if path.exists():
        require(json.loads(path.read_text()) == protocol, "Window-ablation identity changed")
    else:
        helper.save_json(path, protocol)
    fingerprint = opt.shared.boot.canonical_hash(protocol)
    helper.save_json(output / "PREFLIGHT.json", dict(contract_hash=fingerprint, jobs=60,
        reference=reference, test_files_opened=False))
    print("WINDOW_ABLATION_PREFLIGHT_COMPLETE", fingerprint, reference, flush=True)
    return context, reference, fingerprint


def jobs(reference):
    return [dict(seconds=seconds, candidate=reference["candidate"], seed=seed, fold=fold,
                 epochs=reference["training_epochs"])
            for seed in SEEDS for fold in opt.original.FOLDS for seconds in WINDOWS]


def train_window_job(context, output, entry, fingerprint, device):
    seconds = entry["seconds"]
    local_hash = opt.shared.boot.canonical_hash(dict(parent=fingerprint, window_seconds=seconds))
    job = {key: entry[key] for key in ("candidate", "seed", "fold", "epochs")}
    opt.train_job(window_context(context, seconds, local_hash), output / f"window{seconds}", job, device)


def summarize(context, output, reference, fingerprint):
    helper, parent = context[0]["io"], Path(reference["root"])
    candidate, epoch = reference["candidate"], reference["evaluation_epoch"]
    summary, cells, seed_rows = [], [], []
    for seconds in WINDOWS:
        local_hash = opt.shared.boot.canonical_hash(dict(parent=fingerprint, window_seconds=seconds))
        local_context = window_context(context, seconds, local_hash)
        curves = opt.aggregate(local_context, output / f"window{seconds}", [candidate], SEEDS, [epoch], "VALIDATION")
        summary.append(dict(configuration=f"{seconds}s_only", inputs=220, tokens=22, **curves[0]))
        for row in opt.shared.boot.read_csv(output / f"window{seconds}/VALIDATION_21CELL_RESULTS.csv"):
            cells.append(dict(configuration=f"{seconds}s_only", **row))
        for row in opt.shared.boot.read_csv(output / f"window{seconds}/VALIDATION_SEED_CURVES.csv"):
            seed_rows.append(dict(configuration=f"{seconds}s_only", **row))
    full_metrics = []
    for seed in SEEDS:
        if reference["kind"] == "original_epoch5":
            prediction = opt.shared.boot.read_npz(parent / f"joint_seed{seed:02d}/VALIDATION_PREDICTIONS.npz")
            opt.shared.check_reference(context[4], prediction, "full-window reference")
            probability = prediction["probability"]
        else:
            probability = opt.average_folds(parent, candidate, seed, epoch, context[-1],
                                           context[-2]["inputs"]["validation_reference"]["sha256"])
        rows = opt.diagnostic_rows(helper, context[4], probability)
        values = opt.macro(rows)
        full_metrics.append(values)
        seed_rows.append(dict(configuration="120_300_600s", candidate=candidate, seed=seed, epoch=epoch, **values))
        cells.extend(dict(configuration="120_300_600s", candidate=candidate, seed=seed, epoch=epoch, **row) for row in rows)
    summary.insert(0, dict(configuration="120_300_600s", inputs=660, tokens=66, candidate=candidate, epoch=epoch, seeds=5,
                           **{key: float(np.mean([row[key] for row in full_metrics])) for key in opt.METRICS}))
    helper.save_csv(output / "VALIDATION_SUMMARY.csv", summary)
    helper.save_csv(output / "VALIDATION_21CELL_RESULTS.csv", cells)
    helper.save_csv(output / "VALIDATION_SEED_RESULTS.csv", seed_rows)
    differences = []
    for seconds in WINDOWS:
        for seed in SEEDS:
            row = next(r for r in seed_rows if r["configuration"] == f"{seconds}s_only" and int(r["seed"]) == seed)
            differences.append(dict(window_seconds=seconds, seed=seed,
                **{key+"_full_minus_single": full_metrics[seed-1][key]-float(row[key]) for key in opt.METRICS}))
    helper.save_csv(output / "VALIDATION_PAIRED_SEED_DIFFERENCES.csv", differences)
    helper.save_json(output / "COMPLETE.json", dict(status="COMPLETE_MATCHED_WINDOW_VALIDATION_TEST_NOT_RUN",
        contract_hash=fingerprint, new_training_jobs=60, test_files_opened=False,
        note="No test intervals or clinical superiority claim; test evaluation requires a separate frozen step"))


def run_workers(args, output, reference):
    processes, streams = [], []
    try:
        for lane, device in enumerate(args.devices):
            command = [sys.executable, "-u", "-B", str(Path(__file__).resolve()), "worker", "--project-root", str(args.project_root),
                       "--lane", str(lane), "--lanes", str(len(args.devices)), "--device", device]
            if args.allow_shared_gpu:
                command.append("--allow-shared-gpu")
            stream = (output / f"lane{lane}.log").open("ab")
            streams.append(stream)
            processes.append(subprocess.Popen(command, cwd=args.project_root, stdin=subprocess.DEVNULL,
                                              stdout=stream, stderr=subprocess.STDOUT, start_new_session=True))
        helper = opt.original.modules(args.project_root)[0]["io"]
        helper.save_json(output / "WORKERS.json", dict(workers=[dict(pid=p.pid, device=d) for p,d in zip(processes,args.devices)]))
        while any(p.poll() is None for p in processes):
            require(all(p.poll() in (None,0) for p in processes), "A window worker failed; inspect lane logs")
            complete = sum((opt.job_dir(output / f"window{j['seconds']}", j["candidate"], j["seed"], j["fold"])/"COMPLETE.json").exists()
                           for j in jobs(reference))
            helper.save_json(output / "STATUS.json", dict(status="TRAINING", complete_jobs=complete, planned_jobs=60,
                                                          test_files_opened=False))
            time.sleep(15)
        require(all(p.returncode == 0 for p in processes), "A window worker failed")
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
    parser.add_argument("command", choices=("plan", "preflight", "run", "worker"))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--devices", nargs="+", default=["cuda:4","cuda:5","cuda:6"])
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--lane", type=int, default=0)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    output = args.project_root / "reports" / VERSION
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    if args.command == "plan":
        helper, parent = plan(args.project_root, output)
        if not (output / "protocol.json").exists():
            helper.save_json(output / "STATUS.json", dict(status="PREPARED_WAITING_FOR_PARENT_FREEZE_NO_TRAINING",
                                                          planned_jobs=60, test_files_opened=False))
        print("WINDOW_ABLATION_PREPARED; 60 jobs; no training launched; parent completion required", flush=True)
        return
    context, reference, fingerprint = prepare(args.project_root, output)
    if args.command == "preflight":
        return
    if args.command == "worker":
        require(0 <= args.lane < args.lanes <= 4, "Invalid GPU lane")
        device = opt.shared.choose_device(args.device, allow_shared=args.allow_shared_gpu)
        for entry in jobs(reference)[args.lane::args.lanes]:
            train_window_job(context, output, entry, fingerprint, device)
        return
    require(1 <= len(args.devices) <= 4 and len(set(args.devices)) == len(args.devices), "Use 1-4 distinct GPUs")
    with opt.exclusive_lock(output / "RUNNING.lock"):
        try:
            run_workers(args, output, reference)
            summarize(context, output, reference, fingerprint)
            context[0]["io"].save_json(output / "STATUS.json", dict(status="COMPLETE_MATCHED_WINDOW_VALIDATION_TEST_NOT_RUN"))
        except BaseException as error:
            context[0]["io"].save_json(output / "STATUS.json", dict(status="FAILED_OR_INTERRUPTED", error=repr(error)))
            raise


if __name__ == "__main__":
    main()
