import unittest

import torch

from backend.latent_model import LatentDecodeState, LatentDecodingSpec


class LatentDecodeStateTest(unittest.TestCase):
    def test_monet_emits_fixed_latent_span(self):
        spec = LatentDecodingSpec("monet", 10, 11, 12, fixed_steps=3, max_steps=3)
        state = LatentDecodeState(spec)
        hidden = torch.ones(1, 4)

        self.assertEqual(state.advance(10, hidden), 10)
        self.assertTrue(state.active)
        self.assertEqual(state.advance(99, hidden * 2), 11)
        self.assertEqual(state.advance(99, hidden * 3), 11)
        self.assertEqual(state.advance(99, hidden * 4), 12)
        self.assertFalse(state.active)

    def test_lvr_uses_predicted_end(self):
        spec = LatentDecodingSpec("lvr", 20, 21, 22, max_steps=4)
        state = LatentDecodeState(spec)
        hidden = torch.ones(1, 4)

        self.assertEqual(state.advance(20, hidden), 20)
        self.assertEqual(state.advance(99, hidden), 21)
        self.assertEqual(state.advance(22, hidden), 22)
        self.assertFalse(state.active)

    def test_lvr_forces_end_at_safety_limit(self):
        spec = LatentDecodingSpec("lvr", 20, 21, 22, max_steps=2)
        state = LatentDecodeState(spec)
        hidden = torch.ones(1, 4)

        state.advance(20, hidden)
        self.assertEqual(state.advance(99, hidden), 21)
        self.assertEqual(state.advance(99, hidden), 22)


if __name__ == "__main__":
    unittest.main()
