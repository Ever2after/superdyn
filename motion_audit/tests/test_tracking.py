"""Numerical contracts: margin gates, cumulative travel, and gauge invariance."""

import unittest
import numpy as np
from qa_pipeline import tracking as t


class PairMetricTests(unittest.TestCase):
    def test_pair_requires_absolute_and_relative_change(self):
        a = np.zeros((3, 3))
        b = np.array([[2, 0, 0], [2.1, 0, 0], [2.2, 0, 0]])
        self.assertFalse(t.pair_metrics(a, b)[0])  # .2 absolute, but only 10%
        b = np.array([[0.1, 0, 0], [0.13, 0, 0], [0.16, 0, 0]])
        self.assertFalse(t.pair_metrics(a, b)[0])  # 60%, but .06 absolute
        b = np.array([[0.5, 0, 0], [0.6, 0, 0], [0.7, 0, 0]])
        self.assertTrue(t.pair_metrics(a, b)[0])

    def test_travel_is_cumulative_not_endpoint(self):
        a = np.array([[0, 0, 0], [0.3, 0, 0], [0, 0, 0]])
        b = np.tile([1.0, 0, 0], (3, 1))
        distance, travel, m = t.pair_metrics(a, b)
        self.assertFalse(distance)
        self.assertTrue(travel)
        self.assertAlmostEqual(m["travel_a_depths"], 0.6)

    def test_equal_motion_rejected(self):
        a = np.array([[0, 0, 0], [0.3, 0, 0], [0.6, 0, 0]])
        b = a + [1, 0, 0]
        self.assertEqual(t.pair_metrics(a, b)[:2], (False, False))

    def test_rigid_gauge_invariance(self):
        a = np.array([[0, 0, 0], [0.3, 0.1, 0], [0.7, 0.2, 0]])
        b = np.tile([1.0, 0, 0], (3, 1))
        theta = 0.67
        r = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        original = t.pair_metrics(a, b)
        changed = t.pair_metrics(a @ r + [3, -4, 5], b @ r + [3, -4, 5])
        self.assertEqual(original[:2], changed[:2])
        for key in original[2]:
            np.testing.assert_allclose(original[2][key], changed[2][key], atol=1e-12)

    def test_uncertainty_margin_can_veto(self):
        a = np.array([[0, 0, 0], [0.2, 0, 0], [0.5, 0, 0]])
        b = np.tile([1.0, 0, 0], (3, 1))
        ba = np.tile(a, (128, 1, 1))
        bb = np.tile(b, (128, 1, 1))
        self.assertEqual(t.pair_metrics(a, b, ba, bb)[:2], (True, True))
        ba[:64] = a[::-1]
        self.assertFalse(t.pair_metrics(a, b, ba, bb)[0])
        ba[:64] = b
        bb[:64] = a
        self.assertFalse(t.pair_metrics(a, b, ba, bb)[1])

    def test_ambiguous_axis_and_small_motion_rejected(self):
        self.assertIsNone(t.direction([0.13, 0.12, 0], 0.1))
        self.assertIsNone(t.direction([0.08, 0, 0], 0.1))
        self.assertEqual(t.direction([-0.25, 0.02, 0], 0.1), "left")


if __name__ == "__main__":
    unittest.main()
