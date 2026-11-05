"""Synthetic tests; no server data, training, or test outcomes required."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score

import evaluate_ssthnet_history_windows as run
import run_lightgbm660 as io


def reference():
    horizons = np.tile([5, 10, 15], 12)
    y = np.repeat(np.arange(12) % 2, 3)
    labels = np.tile(y[:, None], (1, 7)).astype(np.int8)
    masks = np.ones_like(labels)
    return dict(global_position=np.arange(36), case_id=np.repeat(np.arange(12), 3),
        anchor_seconds=np.zeros(36), horizons=horizons, labels=labels, masks=masks)


def cache_fixture():
    ref = reference()
    patients = np.repeat(np.arange(12), 3)
    support, selected = [], []
    for task in range(7):
        for horizon in (5, 10, 15):
            sel = ref["horizons"] == horizon
            y = ref["labels"][sel, task]
            selected.append((task, sel))
            support.append((np.bincount(patients[sel][y == 1], minlength=12),
                            np.bincount(patients[sel][y == 0], minlength=12)))
    rng = np.random.default_rng(99)
    cache = [[[] for _ in range(4)] for _ in range(5)]
    for s in range(5):
        for m in range(4):
            p = rng.choice(np.array([.1, .3, .5, .7, .9]), size=(36, 7))
            for task, sel in selected:
                cache[s][m].append(run.boot.prepare_cell(ref["labels"][sel, task], p[sel, task], patients[sel], .5))
    return cache, support, ref, patients


class EvaluationTests(unittest.TestCase):
    def test_four_fold_mean_is_float32_and_order_invariant(self):
        p = [np.full((3, 7), value, dtype=np.float32) for value in (.1, .2, .7, .8)]
        a = run.average_probabilities(p)
        np.testing.assert_allclose(a, .45, atol=1e-7)
        np.testing.assert_array_equal(a, run.average_probabilities(p[::-1]))
        self.assertEqual(a.dtype, np.float32)

    def test_fold_geometry_and_invalid_probability_rejected(self):
        p = [np.full((3, 7), .2) for _ in range(4)]
        for invalid in (np.full((2, 7), .2), np.full((3, 7), np.nan), np.full((3, 7), 1.1)):
            with self.assertRaises(RuntimeError):
                run.average_probabilities([*p[:3], invalid])
        with self.assertRaises(RuntimeError):
            run.average_probabilities(p[:3])

    def test_masked_nonbinary_label_is_not_rewritten(self):
        ref = reference()
        ref["labels"][0, 0], ref["masks"][0, 0] = -1, 0
        prediction = run.prediction_arrays(ref, np.full((36, 7), .5))
        self.assertEqual(prediction["labels"][0, 0], -1)
        self.assertEqual(prediction["masks"][0, 0], 0)

    def test_threshold_grid_exact_and_no_duplicates(self):
        rows = [dict(task_id=t, horizon_minutes=h, threshold=.5) for t in run.shared.TASKS for h in (5, 10, 15)]
        self.assertEqual(len(run.threshold_map(rows)), 21)
        for invalid in (rows[:-1], [*rows[:-1], rows[0]], [dict(r, threshold=float("nan")) for r in rows]):
            with self.assertRaises(RuntimeError):
                run.threshold_map(invalid)

    def test_exact_weighted_metrics_with_ties(self):
        y = np.array([0, 1, 0, 1, 1, 0])
        p = np.array([.2, .2, .7, .7, .9, .9])
        subjects = np.array([0, 0, 1, 1, 2, 2])
        counts = np.array([2, 1, 3])
        got = run.boot.weighted_metrics(run.boot.prepare_cell(y, p, subjects, .7), counts)
        w = counts[subjects]
        expected = [average_precision_score(y, p, sample_weight=w), roc_auc_score(y, p, sample_weight=w),
                    f1_score(y, p >= .7, sample_weight=w)]
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_primary_fixed_seed_mean_and_crossed_draw(self):
        cache, support, _, _ = cache_fixture()
        chunk = run.bootstrap_chunk(cache, support, 12, 0, 2, 1)
        for rep in range(2):
            metrics, slots, valid = run.boot.evaluate_replicate(cache, support, 12, run.BASE_SEED, rep)
            self.assertEqual(chunk["valid"][rep], valid)
            np.testing.assert_allclose(chunk["fixed"][rep], metrics.mean(axis=0), equal_nan=True)
            np.testing.assert_allclose(chunk["crossed"][rep], metrics[slots].mean(axis=0), equal_nan=True)

    def test_bootstrap_paired_identical_models_give_zero_difference(self):
        cache, support, _, _ = cache_fixture()
        for s in range(5):
            cache[s] = [copy.deepcopy(cache[s][0]) for _ in range(4)]
        chunk = run.bootstrap_chunk(cache, support, 12, 0, 3, 2)
        for key in ("fixed", "crossed"):
            for m in (1, 2, 3):
                np.testing.assert_allclose(chunk[key][:, 0], chunk[key][:, m], equal_nan=True)

    def test_unsupported_draw_remains_nan_without_redraw(self):
        cache, _, _, _ = cache_fixture()
        support = [(np.zeros(12), np.ones(12)) for _ in range(21)]
        chunk = run.bootstrap_chunk(cache, support, 12, 7, 9, 1)
        self.assertFalse(chunk["valid"].any())
        self.assertTrue(chunk["missing_cells"].all())
        self.assertTrue(np.isnan(chunk["fixed"]).all())
        np.testing.assert_array_equal(chunk["replicates"], [7, 8])

    def test_chunk_resume_rejects_wrong_contract_or_replicate_ids(self):
        cache, support, _, _ = cache_fixture()
        chunk = dict(run.bootstrap_chunk(cache, support, 12, 0, 2, 1), contract_hash=np.array("frozen"))
        run.validate_chunk(chunk, 0, 2, "frozen")
        with self.assertRaises(RuntimeError):
            run.validate_chunk(chunk, 0, 2, "different")
        chunk["replicates"][0] = 100
        with self.assertRaises(RuntimeError):
            run.validate_chunk(chunk, 0, 2, "frozen")

    def test_interval_direction_units_and_multiplicity(self):
        point = np.zeros((5, 4, 3))
        point[:, 0] = .7
        point[:, 1:] = .6
        values = np.tile(point.mean(axis=0), (100, 1, 1))
        rows = run.interval_rows(point, values, values, np.ones(100, dtype=bool))
        self.assertEqual(len(rows), 18)
        for r in rows:
            self.assertAlmostEqual(r["difference_pp"], 10)
            self.assertAlmostEqual(r["ci95_lower_pp"], 10)
            self.assertEqual(r["interval_support_sufficient"], True)
        self.assertAlmostEqual(rows[0]["ap_bonferroni_lower_pp"], 10)
        self.assertTrue(np.isnan(rows[1]["ap_bonferroni_lower_pp"]))
        self.assertFalse(run.interval_rows(point, values, values, np.arange(100) < 94)[0]["interval_support_sufficient"])

    def test_patient_mapping_does_not_treat_operations_as_patients(self):
        ref = reference()
        mapping = dict(ref, subject_id=np.repeat(np.arange(6), 6))
        _, count = run.boot.map_patients(ref, mapping)
        self.assertEqual(count, 6)
        with self.assertRaises(ValueError):
            run.boot.map_patients(ref, ref)

    def test_validation_freeze_before_any_test_load_and_corruption_guard(self):
        ref = reference()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "validation_reference.npz"
            io.save_npz(path, **ref)
            protocol = dict(inputs=dict(validation_reference=run.bind_file(io, path)), checkpoints={})
            for config in run.CONFIGS:
                for seed in run.SEEDS:
                    for fold in run.original.FOLDS:
                        vp = root / f"{config}_{seed}_{fold}.npz"
                        io.save_npz(vp, probability=np.full((36, 7), .5, dtype=np.float32),
                                    contract_hash=np.array("training"), validation_reference_sha256=np.array(io.sha256(path)))
                        protocol["checkpoints"][f"{config}/seed{seed:02d}/fold{fold}"] = {
                            "epoch15_validation.npz": run.bind_file(io, vp), "training_contract_hash": "training"}

            class Metrics:
                @staticmethod
                def threshold_rows(prediction, arm):
                    return [dict(task_id=t, horizon_minutes=h, threshold=.5)
                            for t in run.shared.TASKS for h in (5, 10, 15)]

            output = root / "evaluation"
            output.mkdir()
            loaded = dict(io=io, metrics=Metrics)
            with patch.object(run.boot, "read_npz", wraps=run.boot.read_npz) as load:
                lock = run.freeze_validation(loaded, output, protocol)
                self.assertEqual(len(lock["models"]), 20)
                self.assertEqual(load.call_count, 1)
                self.assertEqual(load.call_args.args[0], path)
            again = run.freeze_validation(loaded, output, protocol)
            self.assertEqual(lock, again)
            thresholds = Path(lock["models"]["120s_only/seed01"]["thresholds"]["path"])
            thresholds.write_text("corrupted", encoding="ascii")
            with self.assertRaises(RuntimeError):
                run.validation_lock(io, output, protocol)

    def test_test_access_blocked_without_threshold_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "Test access refused"):
                run.validation_lock(io, Path(tmp), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
