from __future__ import annotations

import unittest

from hex_reconstruction.soft_policy import soft_policy_cross_entropy


class SoftPolicyTests(unittest.TestCase):
    def test_soft_targets_with_same_argmax_have_different_loss(self) -> None:
        logits = (4.0, 0.0)
        nearly_tied = (0.51, 0.49)
        nearly_one_hot = (0.99, 0.01)
        self.assertEqual(max(range(2), key=nearly_tied.__getitem__), 0)
        self.assertEqual(max(range(2), key=nearly_one_hot.__getitem__), 0)
        self.assertNotAlmostEqual(
            soft_policy_cross_entropy(logits, nearly_tied),
            soft_policy_cross_entropy(logits, nearly_one_hot),
        )


if __name__ == "__main__":
    unittest.main()

