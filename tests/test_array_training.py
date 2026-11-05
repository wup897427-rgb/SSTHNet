import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("core_array_training", ROOT / "tools/train_from_arrays.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class ArrayTrainingTests(unittest.TestCase):
    def test_fixed_recipe_without_historical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((8, 3, 220), dtype=np.float32)
            np.save(root / "development.npy", values)
            np.savez(root / "groups.npz", representative_index=np.arange(8),
                     group_fold=np.repeat(np.arange(4), 2),
                     labels=np.tile(np.array([0, 1])[:, None, None], (4, 7, 3)),
                     masks=np.ones((8, 7, 3), dtype=np.uint8))
            np.save(root / "validation.npy", values[:6])
            np.savez(root / "reference.npz", global_position=np.arange(6), case_id=np.arange(6),
                     anchor_seconds=np.zeros(6), horizons=np.repeat([5, 10, 15], 2),
                     labels=np.tile(np.array([0, 1])[:, None], (3, 7)), masks=np.ones((6, 7)))
            args = ["--development-features", str(root / "development.npy"),
                    "--development-groups", str(root / "groups.npz"),
                    "--validation-features", str(root / "validation.npy"),
                    "--validation-reference", str(root / "reference.npz"),
                    "--seed", "1", "--fold", "0", "--out", str(root / "output")]
            with patch.object(entry.training, "train_job") as train:
                entry.main(args)
                context, _, job, device = train.call_args.args
                self.assertEqual(job, dict(candidate="A_constant_weighted", seed=1, fold=0, epochs=15))
                self.assertEqual(device.type, "cpu")
                self.assertNotIn("test_features", context[-2]["inputs"])
                self.assertTrue((root / "output/CORE_PROTOCOL_seed01_fold0.json").exists())
            with patch.object(entry.training, "train_job"):
                entry.main(args)
            changed = values.copy()
            changed[0, 0, 0] = 1
            np.save(root / "development.npy", changed)
            with self.assertRaisesRegex(ValueError, "changed"):
                entry.main(args)


if __name__ == "__main__":
    unittest.main()
