import tempfile
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from . import evaluate_ssthnet_final_clinical as analysis
except ImportError:
    import evaluate_ssthnet_final_clinical as analysis


class FinalClinicalTests(unittest.TestCase):
    def test_subset_alignment_and_unknown_ids(self):
        np.testing.assert_array_equal(analysis.subset_indices([8, 2, 6], [6, 8]), [2, 0])
        for source, target in (([1, 1], [1]), ([1, 2], [3]), ([1, 2], [1, 1])):
            with self.assertRaises(ValueError):
                analysis.subset_indices(source, target)

    def test_unavailable_state_is_not_normal(self):
        reference = dict(horizons=np.array([5, 5, 5, 10]), masks=np.ones((4, 1)))
        status = np.array([[0], [-1], [1], [0]])
        np.testing.assert_array_equal(analysis.eligible_selection(reference, 0, 5, status), [True, False, False, False])
        reference["masks"][0, 0] = 0
        self.assertFalse(analysis.eligible_selection(reference, 0, 5, status).any())

    def test_patient_mapping_requires_patient_identity(self):
        reference = dict(case_id=np.array([1, 2, 1]))
        rows = [dict(case_id="1", subject_id="17"), dict(case_id="2", subject_id="17")]
        np.testing.assert_array_equal(analysis.patient_names(reference, rows), ["17", "17", "17"])
        with self.assertRaises(ValueError):
            analysis.patient_names(reference, rows + rows[:1])

    def test_weighted_ranking_matches_sklearn_with_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        p = np.array([.2, .2, .7, .7, .9, .1])
        ids = np.array([0, 0, 1, 1, 2, 2])
        counts = np.array([2, 0, 3])
        cell = analysis.boot.prepare_cell(y, p, ids, .5)
        value = analysis.boot.weighted_metrics(cell, counts)
        weights = counts[ids]
        np.testing.assert_allclose(value[:2], [average_precision_score(y, p, sample_weight=weights),
                                              roc_auc_score(y, p, sample_weight=weights)], atol=1e-12)

    def test_bootstrap_seed_average_not_probability_ensemble(self):
        y = np.array([1, 0, 1, 0])
        p1, p2 = np.array([.9, .8, .1, .2]), np.array([.1, .2, .9, .8])
        mean_ap = np.mean([average_precision_score(y, p1), average_precision_score(y, p2)])
        ensemble_ap = average_precision_score(y, (p1 + p2) / 2)
        self.assertNotAlmostEqual(mean_ap, ensemble_ap)

    def test_validation_offset_does_not_fit_test_labels(self):
        yv = np.array([0, 0, 1, 0, 1, 0, 1, 0])
        pv = np.array([.1, .2, .2, .3, .4, .5, .6, .7])
        alpha = analysis.qc.offset_intercept(yv, analysis.qc.logits(pv))
        test_p = np.array([.3, .4, .7])
        adjusted = analysis.qc.expit(analysis.qc.logits(test_p) + alpha)
        self.assertTrue(np.all(np.diff(adjusted) > 0))
        self.assertAlmostEqual(float(analysis.qc.expit(analysis.qc.logits(pv) + alpha).sum()), float(yv.sum()), places=9)

    def test_uncertainty_gate_and_no_nan_subset_estimator(self):
        result = analysis.confidence_row(.6, np.r_[np.repeat(.6, 96), np.repeat(np.nan, 4)], 100)
        self.assertEqual(result["estimable_replicates"], 96)
        self.assertEqual(result["interval_status"], "not_estimable")
        result = analysis.confidence_row(.6, np.repeat(.6, 100), 100)
        self.assertEqual(result["interval_status"], "reported")
        self.assertTrue(np.isnan(np.array([.6, np.nan]).mean()))

    def test_replicate_uses_same_patient_weights_for_both_models(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([.2, .8, .7, .3])
        ids = np.array([0, 0, 1, 1])
        cell = analysis.boot.prepare_cell(y, p, ids, .5)
        cache = [[[cell] * 21], [[cell] * 21 for _ in range(5)]]
        context = dict(patients=2, cache={name: cache for name in analysis.POPULATIONS},
            support={name: [(np.ones(2), np.ones(2))] * 21 for name in analysis.POPULATIONS})
        a = analysis.ranking_replicate(context, 20261022, 1)
        b = analysis.ranking_replicate(context, 20261022, 1)
        np.testing.assert_array_equal(a["values"], b["values"])
        np.testing.assert_allclose(a["values"][:, 0], a["values"][:, 1], atol=1e-12)

    def test_missing_class_draw_is_not_redrawn_or_zero_filled(self):
        context = dict(patients=2, cache={}, support={name: [(np.zeros(2), np.ones(2))] for name in analysis.POPULATIONS})
        result = analysis.ranking_replicate(context, 20261022, 1)
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["values"]).all())

    def test_run_lock_excludes_second_process(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            with analysis.run_lock(path):
                with self.assertRaises(FileExistsError):
                    with analysis.run_lock(path):
                        pass
            with analysis.run_lock(path):
                pass

    def test_os_lock_released_after_abrupt_process_death(self):
        with tempfile.TemporaryDirectory() as folder:
            program = ("import sys; from pathlib import Path; "
                f"sys.path.insert(0, {str(Path(analysis.__file__).parent)!r}); "
                "import evaluate_ssthnet_final_clinical as a; "
                f"lock=a.run_lock(Path({folder!r})); lock.__enter__(); "
                "print('LOCKED',flush=True); import time; time.sleep(120)")
            child = subprocess.Popen([sys.executable, "-c", program], stdout=subprocess.PIPE, text=True)
            try:
                self.assertEqual(child.stdout.readline().strip(), "LOCKED")
                with self.assertRaises(FileExistsError):
                    with analysis.run_lock(Path(folder)):
                        pass
            finally:
                child.kill()
                child.wait(timeout=15)
                child.stdout.close()
            with analysis.run_lock(Path(folder)):
                pass

    def test_final_verification_includes_nested_distribution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bootstrap"
            path.mkdir()
            item = path / "DISTRIBUTIONS.npz"
            analysis.save_npz(item, values=np.ones(3))
            protocol = dict(example="frozen")
            completion = dict(contract_hash=analysis.boot.canonical_hash(protocol),
                              outputs={"bootstrap/DISTRIBUTIONS.npz": analysis.record(item)})
            analysis.verify_completion(completion, protocol)
            analysis.save_npz(item, values=np.zeros(3))
            with self.assertRaises(ValueError):
                analysis.verify_completion(completion, protocol)

    def test_resume_rejects_modified_chunk_with_unchanged_metadata(self):
        y, p = np.array([0, 1, 1, 0]), np.array([.2, .8, .7, .3])
        cell = analysis.boot.prepare_cell(y, p, np.array([0, 0, 1, 1]), .5)
        cache = [[[cell] * 21], [[cell] * 21 for _ in range(5)]]
        context = dict(patients=2, cache={name: cache for name in analysis.POPULATIONS},
            support={name: [(np.ones(2), np.ones(2))] * 21 for name in analysis.POPULATIONS}, point=np.ones((2, 2, 2)))
        protocol = dict(workers=1, replicates=2, chunk_size=2, base_seed=20261022)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            analysis.bootstrap_stage(context, protocol, path, "discrimination")
            chunk = path / "discrimination_bootstrap/CHUNK_000001_000002.npz"
            data = analysis.read_npz(chunk)
            data["values"][0, 0, 0, 0] -= .01
            analysis.save_npz(chunk, **data)
            with self.assertRaisesRegex(ValueError, "hash changed"):
                analysis.bootstrap_stage(context, protocol, path, "discrimination")

    def test_full_calibration_macro_is_not_finite_cell_average(self):
        result = analysis.qc.macro([dict(calibration_slope=1.)] * 20 + [dict(calibration_slope=np.nan)], ["calibration_slope"])
        self.assertTrue(np.isnan(result["calibration_slope"]))
        self.assertEqual(result["calibration_slope_estimable_cells"], 20)


if __name__ == "__main__":
    unittest.main()
