import unittest

import torch

from ssthnet import make_model


class ModelTests(unittest.TestCase):
    def test_final_identity(self):
        model = make_model()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 155271)
        self.assertIsNone(model.router)
        self.assertEqual(len(model.private_experts), 0)

    def test_output_and_bad_shape(self):
        torch.set_num_threads(1)
        model = make_model().eval()
        with torch.inference_mode():
            logits, _ = model(torch.zeros(2, 3, 22, 10))
        self.assertEqual(tuple(logits.shape), (2, 7, 3))
        self.assertTrue(torch.isfinite(logits).all())
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 3, 21, 10))

    def test_scaler_buffers_and_state_roundtrip(self):
        model = make_model()
        model.set_scaler(torch.ones(3, 22, 10), torch.full((3, 22, 10), 2.0))
        copied = make_model()
        copied.load_state_dict(model.state_dict(), strict=True)
        self.assertTrue(torch.equal(copied.feature_mean, model.feature_mean))
        self.assertTrue(torch.equal(copied.feature_scale, model.feature_scale))


if __name__ == "__main__":
    unittest.main()
