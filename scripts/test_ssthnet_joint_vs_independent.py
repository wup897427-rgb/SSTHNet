"""Synthetic contract tests; never read the real development or test caches."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ssthnet_joint_vs_independent as experiment

ROOT = Path(__file__).resolve().parents[1]


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.loaded, cls.sources = experiment.original.modules(ROOT)

    def test_default_device_selection_keeps_old_idle_policy(self):
        with patch.object(experiment.original, "choose_device", return_value=torch.device("cuda:2")) as choose:
            self.assertEqual(experiment.choose_device("auto"), torch.device("cuda:2"))
            choose.assert_called_once_with("auto")

    def test_shared_gpu_requires_explicit_device_but_allows_existing_processes(self):
        inventory = [dict(device="cuda:4", physical="4", free_mib=48092, utilisation=99, occupied=True)]
        with patch.object(torch.cuda, "is_available", return_value=True), \
                patch.object(experiment.original, "gpu_inventory", return_value=inventory), \
                patch.object(experiment.original, "choose_device", side_effect=AssertionError("must not change old selector")):
            self.assertEqual(experiment.choose_device("cuda:4", allow_shared=True), torch.device("cuda:4"))
            for invalid in ("auto", "cpu", "cuda", "cuda:-1"):
                with self.assertRaisesRegex(RuntimeError, "explicit logical device"):
                    experiment.choose_device(invalid, allow_shared=True)
            with self.assertRaisesRegex(RuntimeError, "not visible"):
                experiment.choose_device("cuda:7", allow_shared=True)
            with self.assertRaisesRegex(RuntimeError, "needs >= 49000"):
                experiment.choose_device("cuda:4", allow_shared=True, min_free_mib=49000)
            inventory[0]["free_mib"] = 5999
            with self.assertRaisesRegex(RuntimeError, "needs >= 6000"):
                experiment.choose_device("cuda:4", allow_shared=True)
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                experiment.choose_device("cuda:4", allow_shared=True)

    def test_shared_cli_validation_and_memory_floor(self):
        args = argparse.Namespace(cpu_threads=2, bootstrap_workers=2, replicates=5000,
                                  min_free_mib=6000, allow_shared_gpu=True, device="cuda:4", command="run")
        experiment.validate_args(args)
        args.device = "auto"
        with self.assertRaisesRegex(RuntimeError, "requires --device cuda:N"):
            experiment.validate_args(args)
        args.device, args.command = "cuda:4", "benchmark"
        with self.assertRaisesRegex(RuntimeError, "benchmark command must use an idle GPU"):
            experiment.validate_args(args)
        args.command, args.min_free_mib = "run", 5999
        with self.assertRaisesRegex(RuntimeError, "at least 6000"):
            experiment.validate_args(args)
        with self.assertRaisesRegex(RuntimeError, "At least 6000"):
            experiment.choose_device("cuda:4", allow_shared=True, min_free_mib=5999)

    def test_benchmark_rejects_other_gpu_processes_but_accepts_own_process(self):
        inventory = [dict(device="cuda:4", physical="4")]
        own = str(experiment.os.getpid())
        processes = SimpleNamespace(stdout=f"GPU-A, {own}\nGPU-B, 99999999\n")
        with patch.object(experiment.original, "gpu_inventory", return_value=inventory), \
                patch.object(experiment.original.subprocess, "run",
                             side_effect=[SimpleNamespace(stdout="GPU-A\n"), processes]) as command:
            experiment.check_benchmark_isolation(torch.device("cuda:4"))
            self.assertIn("--id=4", command.call_args_list[0].args[0])
        processes = SimpleNamespace(stdout=f"GPU-A, {own}\nGPU-A, 99999999\n")
        with patch.object(experiment.original, "gpu_inventory", return_value=inventory), \
                patch.object(experiment.original.subprocess, "run",
                             side_effect=[SimpleNamespace(stdout="GPU-A\n"), processes]):
            with self.assertRaisesRegex(RuntimeError, "other compute processes"):
                experiment.check_benchmark_isolation(torch.device("cuda:4"))

    def test_shared_run_defers_timing_and_later_benchmark_reuses_trained_models(self):
        helper = self.loaded["io"]
        shared = argparse.Namespace(allow_shared_gpu=True)
        exclusive = argparse.Namespace(allow_shared_gpu=False)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(experiment, "benchmark", side_effect=AssertionError("no shared timing")):
                status = experiment.benchmark_or_defer(shared, self.loaded, output, None,
                                                     torch.device("cuda:4"), {1: {}}, "synthetic")
            self.assertEqual(status["status"], "DEFERRED_SHARED_GPU")
            self.assertFalse((output / "FORWARD_BENCHMARK.csv").exists())
            for task in range(7):
                directory = output / experiment.job_name(task, 1) / "fold0"
                directory.mkdir(parents=True)
                checkpoint = directory / "epoch5.pt"
                checkpoint.write_bytes(b"synthetic checkpoint")
                helper.save_json(directory / "COMPLETE.json", dict(contract_hash="synthetic",
                                                                 checkpoint_sha256=helper.sha256(checkpoint)))

            def fake_benchmark(*unused):
                helper.save_csv(output / "FORWARD_BENCHMARK.csv", [dict(system="joint", median_ms=1.)])

            with patch.object(experiment, "benchmark", side_effect=fake_benchmark) as timing:
                status = experiment.benchmark_or_defer(exclusive, self.loaded, output, None,
                                                     torch.device("cpu"), {1: {}}, "synthetic")
                self.assertEqual(status["status"], "COMPLETE")
                self.assertEqual(timing.call_count, 1)
                status = experiment.benchmark_or_defer(shared, self.loaded, output, None,
                                                     torch.device("cuda:4"), {1: {}}, "synthetic")
                self.assertEqual(status["status"], "COMPLETE")
                self.assertEqual(timing.call_count, 1)
            (output / "FORWARD_BENCHMARK.csv").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "Benchmark CSV changed"):
                experiment.benchmark_or_defer(exclusive, self.loaded, output, None,
                                             torch.device("cpu"), {1: {}}, "synthetic")

    def test_shared_timing_and_unverified_csv_are_rejected(self):
        shared = argparse.Namespace(allow_shared_gpu=True)
        with self.assertRaisesRegex(RuntimeError, "Do not benchmark"):
            experiment.benchmark(shared, self.loaded, ROOT, None, torch.device("cpu"), {})
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "FORWARD_BENCHMARK.csv").write_bytes(b"unverified")
            with self.assertRaisesRegex(RuntimeError, "Unexpected unverified benchmark"):
                experiment.benchmark_or_defer(shared, self.loaded, output, None,
                                             torch.device("cpu"), {}, "synthetic")

    def test_all_seven_models_match_initial_joint_target_outputs(self):
        base = self.loaded["models"]
        experiment.original.seed_everything(20261884)
        joint = experiment.make_model(base).eval()
        self.assertEqual(sum(p.numel() for p in joint.parameters()), 155271)
        x = torch.randn(3, 3, 22, 10)
        with torch.no_grad():
            expected, _ = joint(x)
        for task in range(7):
            experiment.original.seed_everything(20261884)
            independent = experiment.make_model(base, task).eval()
            self.assertEqual(sum(p.numel() for p in independent.parameters()), 153345)
            self.assertEqual(len(independent.task_heads), 1)
            self.assertEqual(independent._queries().shape, (3, 64))
            self.assertIsNone(independent.router)
            self.assertEqual(len(independent.private_experts), 0)
            for key, value in joint.state_dict().items():
                if key in ("task_query", "interaction_query") or key.startswith("task_heads."):
                    continue
                torch.testing.assert_close(value, independent.state_dict()[key], rtol=0, atol=0)
            with torch.no_grad():
                actual, _ = independent(x)
            self.assertEqual(actual.shape, (3, 1, 3))
            torch.testing.assert_close(actual[:, 0], expected[:, task], rtol=1e-5, atol=2e-7)

    def test_only_selected_target_labels_contribute(self):
        groups = dict(labels=np.zeros((4, 7, 3)), masks=np.ones((4, 7, 3)))
        groups["labels"][:, 3] = 1
        a, mask = experiment.target_labels(groups, np.arange(4), 3)
        groups["labels"][:, [0, 1, 2, 4, 5, 6]] = -99
        b, _ = experiment.target_labels(groups, np.arange(4), 3)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (4, 1, 3))
        groups["labels"][0, 3, 0] = -1
        groups["masks"][0, 3, 0] = 0
        c, mask = experiment.target_labels(groups, np.arange(4), 3)
        self.assertEqual(c[0, 0, 0], 0)
        logits = torch.zeros((4, 1, 3), requires_grad=True)
        loss = experiment.masked_loss(logits, torch.tensor(c), torch.tensor(mask), torch.ones(1, 3))
        loss.backward()
        self.assertEqual(float(logits.grad[0, 0, 0]), 0)
        self.assertTrue(bool((logits.grad[1:] < 0).all()))
        self.assertIsNone(experiment.masked_loss(logits, torch.tensor(c), torch.zeros_like(logits), torch.ones(1, 3)))

    def test_single_target_receives_its_own_gradient_updates(self):
        model = experiment.make_model(self.loaded["models"], 5)
        before = model.tokenizer[0].weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        logits, _ = model(torch.randn(3, 3, 22, 10))
        loss = experiment.masked_loss(logits, torch.ones_like(logits), torch.ones_like(logits), torch.ones(1, 3))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.tokenizer[0].weight))
        self.assertIsNotNone(model.task_heads[0].weight.grad)

    def test_horizon_selection_and_four_fold_probability_average(self):
        class Fixed(torch.nn.Module):
            def __init__(self, fold, task):
                super().__init__()
                self.fold, self.task = fold, task

            def forward(self, x):
                tasks = 7 if self.task is None else 1
                return torch.arange(3, dtype=torch.float32)[None, None].expand(len(x), tasks, -1) + self.fold, {}

        reference = dict(horizons=np.array([15, 5, 10, 15, 5]))
        features = np.zeros((5, 3, 220), dtype=np.float32)
        for task in (None, 2):
            with patch.object(experiment, "checkpoint_model", side_effect=lambda loaded, out, seed, fold, t, device, sources: Fixed(fold, t)):
                probability = experiment.predict(self.loaded, ROOT, 1, task, torch.device("cpu"), features, reference, {})
            expected = np.array([np.mean([torch.sigmoid(torch.tensor(float(f + h))).item() for f in range(4)])
                                 for h in (2, 0, 1, 2, 0)], dtype=np.float32)
            np.testing.assert_allclose(probability[:, 0], expected, rtol=1e-7)
            self.assertEqual(probability.shape, (5, 7 if task is None else 1))

    def test_training_checkpoint_resume_and_manifest_guard(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(8, 3, 220)).astype(np.float32)
        groups = dict(representative_index=np.arange(8), group_fold=np.repeat(np.arange(4), 2),
                      labels=np.broadcast_to((np.arange(8) % 2)[:, None, None], (8, 7, 3)).copy(),
                      masks=np.ones((8, 7, 3), dtype=np.uint8))
        stats = {0: (np.zeros((3, 22, 10), dtype=np.float32), np.ones((3, 22, 10), dtype=np.float32),
                     np.ones((7, 3), dtype=np.float32))}
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            experiment.train_fold(self.loaded, x, groups, stats, output, 1, 0, 3, torch.device("cpu"), "synthetic")
            directory = output / "independent_target03_seed01/fold0"
            checkpoint = directory / "epoch5.pt"
            before = self.loaded["io"].sha256(checkpoint)
            marker = json.loads((directory / "COMPLETE.json").read_text())
            self.assertEqual(marker["training_epochs"], 6)
            self.assertEqual(marker["evaluation_epoch"], 5)
            saved = experiment.original.load_checkpoint(checkpoint)
            experiment.validate_checkpoint(saved, 1, 0, 3)
            self.assertEqual(saved["model_state"]["task_query"].shape, (1, 64))
            experiment.train_fold(self.loaded, x, groups, stats, output, 1, 0, 3, torch.device("cpu"), "synthetic")
            (directory / "COMPLETE.json").unlink()
            experiment.train_fold(self.loaded, x, groups, stats, output, 1, 0, 3, torch.device("cpu"), "synthetic")
            self.assertEqual(self.loaded["io"].sha256(checkpoint), before)
            with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                experiment.train_fold(self.loaded, x, groups, stats, output, 1, 0, 3, torch.device("cpu"), "different")

    @staticmethod
    def synthetic_reference():
        patients = np.repeat(np.arange(20), 3)
        labels = np.broadcast_to((patients % 2)[:, None], (60, 7)).copy()
        return dict(global_position=np.arange(60), case_id=patients + 100,
                    subject_id=patients + 1000, anchor_seconds=np.zeros(60),
                    horizons=np.tile([5, 10, 15], 20), labels=labels,
                    masks=np.ones((60, 7), dtype=np.uint8))

    def test_patient_identity_and_record_identity_guards(self):
        reference = self.synthetic_reference()
        inverse, count = experiment.boot.map_patients(reference, reference)
        self.assertEqual(count, 20)
        self.assertEqual(len(inverse), 60)
        wrong = dict(reference, probability=np.full((60, 7), .5))
        wrong["case_id"] = wrong["case_id"] + 1
        with self.assertRaises(ValueError):
            experiment.check_reference(reference, wrong, "wrong")
        missing = {k: v for k, v in reference.items() if k != "subject_id"}
        with self.assertRaisesRegex(ValueError, "do not substitute case_id"):
            experiment.boot.map_patients(reference, missing)

    def test_default_joint_reuse_checks_all_five_seeds_and_rejects_changed_inputs(self):
        helper = self.loaded["io"]
        reference = self.synthetic_reference()
        prediction = dict(reference, probability=np.where(reference["labels"] > 0, .8, .2).astype(np.float32))
        data = {role: (None, reference) for role in ("validation", "test")}
        source_hashes = {k: helper.sha256(v) for k, v in self.sources.items()}
        input_hashes = {k: str(i) for i, k in enumerate(("development_features", "development_groups", "validation_features",
                                                       "validation_reference", "test_features", "test_reference"))}
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            protocol = dict(script_sha256=helper.sha256(ROOT / "scripts/run_ssthnet_ablations.py"),
                            source_sha256=source_hashes, input_sha256=input_hashes, training_epochs=6,
                            evaluation_epoch=5, batch_size=128, learning_rate=2e-4, weight_decay=1e-4,
                            folds=list(experiment.FOLDS), seed_indices=list(experiment.SEEDS),
                            torch_version=torch.__version__, numpy_version=np.__version__)
            helper.save_json(folder / "protocol.json", protocol)
            thresholds_sha256 = {}
            model_state = experiment.make_model(self.loaded["models"]).state_dict()
            for seed in experiment.SEEDS:
                directory = folder / f"ssthnet_seed{seed:02d}"
                directory.mkdir()
                for role in ("validation", "test"):
                    helper.save_npz(directory / f"{role.upper()}_PREDICTIONS.npz", **prediction)
                helper.save_csv(directory / "VALIDATION_THRESHOLDS.csv", self.loaded["metrics"].threshold_rows(prediction, directory.name))
                thresholds_sha256[directory.name] = helper.sha256(directory / "VALIDATION_THRESHOLDS.csv")
                for fold in experiment.FOLDS:
                    run = directory / f"fold{fold}"
                    run.mkdir()
                    path = run / "epoch5.pt"
                    experiment.original.atomic_checkpoint(path, dict(variant="ssthnet", seed_index=seed, fold=fold,
                                                                     epoch=5, model_state=model_state))
                    helper.save_json(run / "COMPLETE.json", dict(checkpoint_sha256=helper.sha256(path)))
            helper.save_json(folder / "VALIDATION_LOCK.json", dict(thresholds_sha256=thresholds_sha256))
            args = argparse.Namespace(project_root=ROOT, joint_mode="reuse", joint_roots=[folder])
            result = experiment.audit_joint_sources(args, self.loaded, input_hashes, source_hashes, data)
            self.assertEqual(set(result), set(experiment.SEEDS))
            changed = dict(input_hashes, test_reference="changed")
            with self.assertRaisesRegex(RuntimeError, "input fingerprints differ"):
                experiment.audit_joint_sources(args, self.loaded, changed, source_hashes, data)
            wrong_thresholds = experiment.boot.read_csv(folder / "ssthnet_seed01/VALIDATION_THRESHOLDS.csv")
            wrong_thresholds[0]["threshold"] = .99
            helper.save_csv(folder / "ssthnet_seed01/VALIDATION_THRESHOLDS.csv", wrong_thresholds)
            with self.assertRaisesRegex(RuntimeError, "thresholds changed"):
                experiment.audit_joint_sources(args, self.loaded, input_hashes, source_hashes, data)

    def test_forward_cache_reuse_and_corruption_guard(self):
        reference = self.synthetic_reference()
        values = np.zeros((60, 3, 220), dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(experiment, "predict", return_value=np.full((60, 1), .5, dtype=np.float32)) as predict:
                for _ in range(2):
                    experiment.cached_prediction(self.loaded, output, 1, 0, torch.device("cpu"), values, reference, {}, "test", "synthetic")
                self.assertEqual(predict.call_count, 1)
            path = output / "independent_target00_seed01/TEST_FORWARD_CACHE.npz"
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "cache changed"):
                experiment.cached_prediction(self.loaded, output, 1, 0, torch.device("cpu"), values, reference, {}, "test", "synthetic")

    def test_bootstrap_is_paired_fixed_seed_principal_and_resumable(self):
        helper = self.loaded["io"]
        reference = self.synthetic_reference()
        inverse, patients = experiment.boot.map_patients(reference, reference)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            for seed in experiment.SEEDS:
                for system in experiment.SYSTEMS:
                    directory = output / f"{system}_seed{seed:02d}"
                    directory.mkdir()
                    rng = np.random.default_rng(seed)
                    p = rng.uniform(.1, .9, (60, 7)).astype(np.float32)
                    if system == "joint":
                        p = .7 * p + .3 * reference["labels"]
                    helper.save_npz(directory / "TEST_PREDICTIONS.npz", **reference, probability=p)
                    helper.save_csv(directory / "VALIDATION_THRESHOLDS.csv", [dict(task_id=t, horizon_minutes=h, threshold=.5)
                                                                            for t in experiment.TASKS for h in experiment.HORIZONS])
            args = argparse.Namespace(bootstrap_seed=20261018, bootstrap_workers=1, replicates=100)
            experiment.bootstrap_intervals(args, self.loaded, output, inverse, patients, "synthetic")
            before = helper.sha256(output / "PAIRED_BOOTSTRAP_INTERVALS.csv")
            experiment.bootstrap_intervals(args, self.loaded, output, inverse, patients, "synthetic")
            self.assertEqual(before, helper.sha256(output / "PAIRED_BOOTSTRAP_INTERVALS.csv"))
            rows = experiment.boot.read_csv(output / "PAIRED_BOOTSTRAP_INTERVALS.csv")
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[0]["design"], "patient_clustered_fixed_five_seeds")
            self.assertGreater(float(rows[0]["delta"]), 0)
            self.assertEqual(rows[0]["difference"], "joint_minus_independent_targets")
            self.assertEqual(int(rows[0]["estimable_replicates"]), 100)

    def test_evaluation_locks_validation_and_uses_seed_metric_means(self):
        helper = self.loaded["io"]
        reference = self.synthetic_reference()
        data = {role: (np.zeros((60, 3, 220), dtype=np.float32), copy.deepcopy(reference)) for role in ("validation", "test")}
        visited = []

        def synthetic_predict(loaded, output, seed, task, device, features, ref, sources, role, contract):
            if role == "test":
                self.assertTrue((output / "VALIDATION_LOCK.json").exists())
            visited.append((role, seed, task))
            p = np.random.default_rng(seed).uniform(.1, .9, (60, 7)).astype(np.float32)
            return p if task is None else p[:, task:task + 1]

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            with patch.object(experiment, "cached_prediction", side_effect=synthetic_predict):
                experiment.evaluate(self.loaded, output, data, torch.device("cpu"), {}, "synthetic")
            self.assertTrue(visited)
            rows = experiment.boot.read_csv(output / "FIVE_SEED_SUMMARY.csv")
            self.assertEqual(len(rows), 4)
            runs = experiment.boot.read_csv(output / "RUN_RESULTS.csv")
            selected = [float(r["equal_cell_macro_auprc"]) for r in runs if r["role"] == "test" and r["system"] == "joint"]
            summary = next(r for r in rows if r["role"] == "test" and r["system"] == "joint")
            self.assertAlmostEqual(float(summary["equal_cell_macro_auprc_mean"]), np.mean(selected))
            with patch.object(experiment, "cached_prediction", side_effect=AssertionError("must reuse assembled cache")):
                experiment.evaluate(self.loaded, output, data, torch.device("cpu"), {}, "synthetic")
            differences = experiment.boot.read_csv(output / "PAIRED_21CELL_RESULTS.csv")
            self.assertEqual(len(differences), 42)
            self.assertTrue(all(abs(float(r["joint_minus_independent_auprc"])) < 1e-10 for r in differences))

    def test_reused_joint_predictions_are_saved_to_output_without_touching_sources(self):
        helper = self.loaded["io"]
        reference = self.synthetic_reference()
        probability = np.where(reference["labels"] > 0, .8, .2).astype(np.float32)
        data = {role: (np.zeros((60, 3, 220), dtype=np.float32), copy.deepcopy(reference))
                for role in ("validation", "test")}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "new_results"
            output.mkdir()
            sources, original_files = {}, {}
            for seed in experiment.SEEDS:
                directory = root / f"original_seed{seed:02d}"
                directory.mkdir()
                source = dict(files={}, sha256={})
                for role in ("validation", "test"):
                    path = directory / f"{role.upper()}_PREDICTIONS.npz"
                    helper.save_npz(path, **reference, probability=probability,
                                    source_only_provenance=np.array("preserve original"))
                    source["files"][role] = str(path)
                    source["sha256"][role] = helper.sha256(path)
                    original_files[path] = (path.read_bytes(), path.stat().st_mtime_ns)
                thresholds = directory / "VALIDATION_THRESHOLDS.csv"
                helper.save_csv(thresholds, self.loaded["metrics"].threshold_rows(
                    dict(reference, probability=probability), directory.name))
                source["files"]["thresholds"] = str(thresholds)
                sources[seed] = source
            with patch.object(experiment, "cached_prediction",
                              return_value=probability[:, :1]):
                experiment.evaluate(self.loaded, output, data, torch.device("cpu"),
                                    sources, "synthetic")
            for path, (content, modified) in original_files.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, modified)
            for seed in experiment.SEEDS:
                for role in ("validation", "test"):
                    path = output / f"joint_seed{seed:02d}" / f"{role.upper()}_PREDICTIONS.npz"
                    self.assertTrue(path.is_file())
                    marker = json.loads(path.with_suffix(".json").read_text())
                    self.assertEqual(marker["sha256"], helper.sha256(path))
                    np.testing.assert_array_equal(experiment.boot.read_npz(path)["probability"],
                                                  probability)
            with patch.object(experiment, "cached_prediction",
                              side_effect=AssertionError("must reuse destination caches")):
                experiment.evaluate(self.loaded, output, data, torch.device("cpu"),
                                    sources, "synthetic")


if __name__ == "__main__":
    unittest.main(verbosity=2)
