import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_ssthnet_ablations as original


class ExtractedHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded, _ = original.modules(ROOT)

    def test_thresholds_and_metrics(self):
        labels = np.tile(np.array([0, 1, 0, 1])[:, None], (3, 7))
        prediction = dict(horizons=np.repeat([5, 10, 15], 4), labels=labels,
                          masks=np.ones_like(labels), probability=.1 + .8 * labels)
        metrics = self.loaded["metrics"]
        rows = metrics.threshold_rows(prediction, "synthetic")
        self.assertEqual(len(rows), 21)
        thresholds = {(r["task_id"], r["horizon_minutes"]): r["threshold"] for r in rows}
        cells = metrics.metric_rows(prediction, "synthetic", "validation", thresholds)
        summary = metrics.summary_row(cells, "synthetic", "validation", "synthetic")
        self.assertEqual(summary["equal_cell_macro_auprc"], 1.0)
        self.assertEqual(summary["equal_cell_macro_auroc"], 1.0)

    def test_scaler_and_weights(self):
        training = self.loaded["training"]
        values = np.arange(4 * 3 * 220, dtype=np.float32).reshape(4, 3, 220)
        mean, scale = training.fit_scaler(values, np.array([0, 1]))
        np.testing.assert_allclose(mean, values[:2].reshape(2, 3, 22, 10).mean(axis=0))
        self.assertTrue((scale > 0).all())
        labels = np.tile(np.array([0, 1, 0, 1])[:, None, None], (1, 7, 3))
        weights = training.positive_weights(labels, np.ones_like(labels), np.arange(4))
        np.testing.assert_array_equal(weights, np.ones((7, 3)))

    def test_public_factory_matches_research_model(self):
        import torch
        from ssthnet import make_model
        torch.manual_seed(42)
        public = make_model()
        torch.manual_seed(42)
        archived = original.make_model(self.loaded["models"], "ssthnet")
        for key, value in public.state_dict().items():
            self.assertTrue(torch.equal(value, archived.state_dict()[key]), key)


if __name__ == "__main__":
    unittest.main()
