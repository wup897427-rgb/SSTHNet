#!/usr/bin/env python3
"""Matched one-factor ablations of the final single-shared-expert SSTHNet."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("ssthnet", "no_source_interaction", "no_scale_interaction", "independent_queries")
FOLDS = (0, 1, 2, 3)
FIXED_EPOCH = 5
TRAIN_EPOCHS = 6
VERSION = "ssthnet_single_expert_matched_ablations_v1"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(path, name):
    require(path.is_file(), f"Missing source: {path}")
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def modules(root):
    paths = {
        "io": root / "scripts/run_lightgbm660.py",
        "models": root / "experiments/model_runs/phase6ca84x_v7_extended_structural_ablations_v1/ca84x_models.py",
        "training": root / "experiments/model_runs/phase6ca84_v7_scale_source_task_horizon_sparse_moe_v1/2840_ca84_scale_source_task_horizon_sparse_moe.py",
        "metrics": root / "experiments/model_runs/phase6ca75_v7_locked_outer_test_benchmark_v1/2750_ca75_outer_test.py",
        "metric_protocol": root / "experiments/model_runs/phase6ca75_v7_locked_outer_test_benchmark_v1/ca75_protocol.py",
    }
    loaded = {key: load_module(paths[key], "ssth_ablation_" + key) for key in ("io", "models", "training", "metrics")}
    require(hasattr(loaded["io"], "validate_labels"), "Upload the corrected run_lightgbm660.py first")
    return loaded, paths


def make_model(base_models, variant):
    require(variant in VARIANTS, f"Unknown variant: {variant}")

    class FinalSSTHNetAblation(base_models.SSTHMoEAblation):
        def __init__(self):
            super().__init__("single_expert")
            self.ablation_variant = variant
            if variant == "no_source_interaction":
                self.source_mixer = nn.Identity()
            elif variant == "no_scale_interaction":
                self.scale_mixer = nn.Identity()
            elif variant == "independent_queries":
                # Preserve the RNG stream used by all unchanged modules and dropout.
                with torch.random.fork_rng(devices=[]):
                    independent = torch.empty(21, 64)
                    nn.init.normal_(independent, std=0.02)
                del self.task_query
                del self.horizon_query
                del self.interaction_query
                self.independent_query = nn.Parameter(independent)
            require(self.router is None and len(self.private_experts) == 0,
                    "A final-SSTHNet ablation must not contain routed private experts")

        def _queries(self):
            if self.ablation_variant == "independent_queries":
                return self.independent_query
            return super()._queries()

    return FinalSSTHNetAblation()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def gpu_inventory():
    result = subprocess.run([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    processes = subprocess.run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True)
    occupied = {line.split(",")[0].strip() for line in processes.stdout.splitlines() if "," in line}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    visible = None if visible is None else [item.strip() for item in visible.split(",")]
    rows = []
    for raw in csv.reader(result.stdout.splitlines()):
        physical, uuid, free, utilisation = [value.strip() for value in raw]
        logical = physical
        if visible is not None:
            matches = [i for i, item in enumerate(visible) if item == physical or uuid.startswith(item)]
            if not matches:
                continue
            logical = str(matches[0])
        rows.append(dict(device="cuda:" + logical, physical=physical,
                         free_mib=int(free), utilisation=int(utilisation), occupied=uuid in occupied))
    return rows


def choose_device(requested):
    if requested == "cpu":
        return torch.device("cpu")
    require(torch.cuda.is_available(), "CUDA unavailable; use the original Torch/CUDA environment")
    inventory = gpu_inventory()
    candidates = [row for row in inventory if not row["occupied"] and row["free_mib"] >= 6000 and row["utilisation"] <= 10]
    if requested != "auto":
        candidates = [row for row in candidates if row["device"] == requested]
    require(candidates, f"No idle GPU with >=6000 MiB free. Inventory: {inventory}")
    selected = max(candidates, key=lambda row: row["free_mib"])
    print("GPU_SELECTED", json.dumps(selected), flush=True)
    return torch.device(selected["device"])


def atomic_checkpoint(path, value):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Checkpoints are generated by this runner and contain optimizer and RNG state.
    return torch.load(path, map_location="cpu", weights_only=False)


def train_fold(args, loaded, x, groups, output, variant, seed_index, fold, device):
    helper, training = loaded["io"], loaded["training"]
    directory = output / f"{variant}_seed{seed_index:02d}" / f"fold{fold}"
    directory.mkdir(parents=True, exist_ok=True)
    complete = directory / "COMPLETE.json"
    if complete.exists():
        marker = json.loads(complete.read_text())
        require(marker["checkpoint_sha256"] == helper.sha256(directory / "epoch5.pt"), "Completed checkpoint changed")
        print(f"FOLD_REUSED {variant} seed={seed_index} fold={fold}", flush=True)
        return
    seed = 20261884 + (seed_index - 1) * 1000 + fold
    seed_everything(seed)
    model = make_model(loaded["models"], variant).to(device)
    train = np.flatnonzero(groups["group_fold"] != fold)
    mean, scale = training.fit_scaler(x, groups["representative_index"][train])
    require(np.isfinite(mean).all() and np.isfinite(scale).all(), "Non-finite original training scaler")
    model.set_scaler(torch.as_tensor(mean, device=device), torch.as_tensor(scale, device=device))
    positive_weights = torch.as_tensor(training.positive_weights(groups["labels"], groups["masks"], train), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    start_epoch = 1
    history = []
    latest = directory / "latest.pt"
    if latest.exists():
        checkpoint = load_checkpoint(latest)
        require((checkpoint["variant"], checkpoint["seed_index"], checkpoint["fold"]) == (variant, seed_index, fold), "Resume job changed")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        rng.bit_generator.state = checkpoint["numpy_generator_state"]
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(checkpoint["cuda_rng_state"], device)
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
    for epoch in range(start_epoch, TRAIN_EPOCHS + 1):
        model.train()
        shuffled = rng.permutation(train)
        loss_sum, processed = 0., 0
        for start in range(0, len(shuffled), 128):
            selected = shuffled[start:start + 128]
            values = np.asarray(x[groups["representative_index"][selected]], dtype=np.float32).reshape(-1, 3, 22, 10)
            require(np.isfinite(values).all(), "Non-finite feature in the original SSTHNet cache")
            mask_array = groups["masks"][selected]
            labels = torch.as_tensor(np.where(mask_array > 0, groups["labels"][selected], 0), dtype=torch.float32, device=device)
            masks = torch.as_tensor(mask_array, dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            logits, _auxiliary = model(torch.as_tensor(values, device=device))
            weights = torch.where(labels > .5, positive_weights.unsqueeze(0), torch.ones_like(labels))
            raw = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * weights * masks
            denominator = masks.sum(dim=0)
            estimable = denominator > 0
            require(bool(estimable.any()), "Batch contains no eligible outcomes")
            loss = (raw.sum(dim=0)[estimable] / denominator[estimable]).mean()
            require(bool(torch.isfinite(loss)), "Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
            processed += len(selected)
            if processed // 25000 != (processed - len(selected)) // 25000:
                print(f"TRAIN {variant} seed={seed_index} fold={fold} epoch={epoch} groups={processed}/{len(train)}", flush=True)
        history.append(dict(epoch=epoch, mean_training_loss=loss_sum / len(train)))
        checkpoint = dict(variant=variant, seed_index=seed_index, fold=fold, epoch=epoch,
                          model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), history=history,
                          numpy_generator_state=rng.bit_generator.state, python_rng_state=random.getstate(),
                          numpy_rng_state=np.random.get_state(), torch_rng_state=torch.get_rng_state(),
                          cuda_rng_state=torch.cuda.get_rng_state(device) if device.type == "cuda" else None)
        atomic_checkpoint(latest, checkpoint)
        if epoch == FIXED_EPOCH:
            atomic_checkpoint(directory / "epoch5.pt", dict(variant=variant, seed_index=seed_index,
                              fold=fold, epoch=epoch, model_state=model.state_dict()))
        helper.save_csv(directory / "training_history.csv", history)
        print(f"EPOCH_COMPLETE {variant} seed={seed_index} fold={fold} epoch={epoch} loss={history[-1]['mean_training_loss']:.6f}", flush=True)
    helper.save_json(complete, dict(variant=variant, seed_index=seed_index, fold=fold,
                                  training_epochs=TRAIN_EPOCHS, evaluation_epoch=FIXED_EPOCH,
                                  checkpoint_sha256=helper.sha256(directory / "epoch5.pt")))
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def predict_role(args, loaded, output, variant, seed_index, device, features, reference):
    count = len(reference["global_position"])
    slots = np.searchsorted(np.array([5, 10, 15]), reference["horizons"])
    probability = np.zeros((count, 7), dtype=np.float64)
    for fold in FOLDS:
        model = make_model(loaded["models"], variant).to(device).eval()
        checkpoint = load_checkpoint(output / f"{variant}_seed{seed_index:02d}" / f"fold{fold}/epoch5.pt")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        with torch.no_grad():
            for start in range(0, count, 256):
                end = min(start + 256, count)
                values = np.array(features[start:end], dtype=np.float32, copy=True).reshape(-1, 3, 22, 10)
                logits, _ = model(torch.as_tensor(values, device=device))
                scores = torch.sigmoid(logits).cpu().numpy()
                probability[start:end] += scores[np.arange(end - start), :, slots[start:end]]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return dict(**reference, probability=(probability / 4).astype(np.float32))


def run(args):
    root = args.project_root.resolve()
    loaded, source_files = modules(root)
    helper = loaded["io"]
    paths = helper.sources(args)
    x, groups = helper.development(paths)
    for name, path in paths.items():
        require(path.is_file(), f"Missing {name}: {path}")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    # Baseline identity is checked directly against the archived final model.
    seed_everything(20262884)
    original = loaded["models"].SSTHMoEAblation("single_expert").eval()
    seed_everything(20262884)
    baseline = make_model(loaded["models"], "ssthnet").eval()
    require(all(torch.equal(value, baseline.state_dict()[key]) for key, value in original.state_dict().items()), "Baseline initialization differs from archived SSTHNet")
    output = (args.out or root / "reports" / VERSION).resolve()
    require(not any(output == path or output in path.parents for path in paths.values()), "Output overlaps an input")
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "RUNNING.lock"
    with lock.open("x") as target:
        target.write(str(os.getpid()))
    try:
        print("Hashing frozen development inputs and source files...", flush=True)
        contract = dict(version=VERSION, script_sha256=helper.sha256(Path(__file__)),
                        source_sha256={key: helper.sha256(path) for key, path in source_files.items()},
                        input_sha256={key: helper.sha256(path) for key, path in paths.items()},
                        outer_paths={key: str(value) for key, value in paths.items() if not key.startswith("development")},
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        variants=list(VARIANTS), seed_indices=args.seed_indices, folds=list(FOLDS),
                        training_epochs=6, evaluation_epoch=5, batch_size=128, evaluation_batch_size=256,
                        learning_rate=2e-4, weight_decay=1e-4, expert="single shared; no private experts or router",
                        model_selection="none; retain every prespecified configuration",
                        test_role="posthoc evaluation on the previously observed internal benchmark")
        manifest = output / "protocol.json"
        if manifest.exists():
            require(json.loads(manifest.read_text()) == contract, "Configuration/source changed; use a separate output directory")
        else:
            helper.save_json(manifest, contract)
        specifications = []
        for variant in VARIANTS:
            model = make_model(loaded["models"], variant)
            specifications.append(dict(variant=variant, sources=22, scales=3, summaries_per_source_scale=10,
                                       source_interaction=not isinstance(model.source_mixer, nn.Identity),
                                       scale_interaction=not isinstance(model.scale_mixer, nn.Identity),
                                       shared_experts=1, private_experts=len(model.private_experts),
                                       learned_router=model.router is not None,
                                       shared_task_horizon_queries=variant != "independent_queries",
                                       outputs=21, trainable_parameters=sum(p.numel() for p in model.parameters())))
        helper.save_csv(output / "MODEL_SPECIFICATIONS.csv", specifications)
        for seed_index in args.seed_indices:
            for variant in VARIANTS:
                for fold in FOLDS:
                    train_fold(args, loaded, x, groups, output, variant, seed_index, fold, device)
        print("ALL_TRAINING_COMPLETE; evaluating fixed epoch5 ensembles", flush=True)
        validation_x, validation_reference = helper.outer(paths, "validation")
        validation_digest = {key: helper.sha256(paths[key]) for key in ("validation_features", "validation_reference")}
        cell_rows, summary_rows, thresholds_by_run = [], [], {}
        metrics = loaded["metrics"]
        for seed_index in args.seed_indices:
            for variant in VARIANTS:
                name = f"{variant}_seed{seed_index:02d}"
                directory = output / name
                validation = predict_role(args, loaded, output, variant, seed_index, device, validation_x, validation_reference)
                helper.save_npz(directory / "VALIDATION_PREDICTIONS.npz", **validation)
                threshold_rows = metrics.threshold_rows(validation, name)
                thresholds = {(row["task_id"], int(row["horizon_minutes"])): float(row["threshold"]) for row in threshold_rows}
                thresholds_by_run[name] = thresholds
                helper.save_csv(directory / "VALIDATION_THRESHOLDS.csv", threshold_rows)
                cells = metrics.metric_rows(validation, name, "validation", thresholds)
                summary = metrics.summary_row(cells, name, "validation", VERSION)
                summary_rows.append(dict(variant=variant, seed_index=seed_index, **summary))
                cell_rows.extend(dict(variant=variant, seed_index=seed_index, **row) for row in cells)
        helper.save_json(output / "VALIDATION_LOCK.json", dict(input_sha256=validation_digest,
                         thresholds_sha256={name: helper.sha256(output / name / "VALIDATION_THRESHOLDS.csv") for name in thresholds_by_run},
                         architecture_or_epoch_selection_performed=False))
        test_x, test_reference = helper.outer(paths, "test")
        test_digest = {key: helper.sha256(paths[key]) for key in ("test_features", "test_reference")}
        for seed_index in args.seed_indices:
            for variant in VARIANTS:
                name = f"{variant}_seed{seed_index:02d}"
                prediction = predict_role(args, loaded, output, variant, seed_index, device, test_x, test_reference)
                helper.save_npz(output / name / "TEST_PREDICTIONS.npz", **prediction)
                cells = metrics.metric_rows(prediction, name, "test", thresholds_by_run[name])
                summary = metrics.summary_row(cells, name, "test", VERSION)
                summary_rows.append(dict(variant=variant, seed_index=seed_index, **summary))
                cell_rows.extend(dict(variant=variant, seed_index=seed_index, **row) for row in cells)
                helper.save_csv(output / "ABLATION_RUN_RESULTS.csv", summary_rows)
                print("TEST_RESULT", variant, "seed", seed_index,
                      "AP", round(100 * summary["equal_cell_macro_auprc"], 4),
                      "AUROC", round(100 * summary["equal_cell_macro_auroc"], 4),
                      "F1", round(100 * summary["equal_cell_macro_f1"], 4), flush=True)
        helper.save_csv(output / "ABLATION_21CELL_RESULTS.csv", cell_rows)
        comparisons = []
        for role in ("validation", "test"):
            for variant in VARIANTS:
                rows = [row for row in summary_rows if row["role"] == role and row["variant"] == variant]
                baseline_rows = [row for row in summary_rows if row["role"] == role and row["variant"] == "ssthnet"]
                summary = dict(role=role, variant=variant, completed_seeds=len(rows))
                for metric in ("auprc", "auroc", "f1"):
                    values = np.array([row["equal_cell_macro_" + metric] for row in rows])
                    delta = np.array([row["equal_cell_macro_" + metric] - base["equal_cell_macro_" + metric] for row, base in zip(rows, baseline_rows)])
                    summary[metric + "_mean"] = float(values.mean())
                    summary[metric + "_sd"] = float(values.std(ddof=1)) if len(values) > 1 else None
                    summary[metric + "_delta_vs_ssthnet"] = float(delta.mean())
                comparisons.append(summary)
        helper.save_csv(output / "ABLATION_SUMMARY.csv", comparisons)
        helper.save_json(output / "COMPLETE.json", dict(status="COMPLETE_POSTHOC_FINAL_SSTHNET_ABLATIONS",
                         seed_indices=args.seed_indices, variants=list(VARIANTS), folds=4,
                         evaluation_epoch=5, test_input_sha256=test_digest, paired_intervals_computed=False))
        print("COMPLETE", output, flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run",))
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="auto", help="auto selects one idle GPU; or cuda:N")
    parser.add_argument("--seed-indices", type=int, nargs="+", default=[2])
    parser.add_argument("--cpu-threads", type=int, default=2)
    for name in ("development_features", "development_groups", "validation_features", "validation_reference", "test_features", "test_reference"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    args = parser.parse_args()
    require(args.cpu_threads > 0, "CPU thread count must be positive")
    require(len(args.seed_indices) == len(set(args.seed_indices)) and all(1 <= seed <= 5 for seed in args.seed_indices), "Choose distinct seed indices from 1..5")
    run(args)


if __name__ == "__main__":
    main()
