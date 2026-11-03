"""Offline checks for the synthetic-split pilot: geometry contract and designs."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Content/Python"))

import pilot_qa  # noqa: E402
from pilot_design import ASSETS, FRAMES, designs  # noqa: E402
from superdyn_ue.math3d import compose  # noqa: E402
from qa_pipeline import tracking  # noqa: E402  (motion_audit on sys.path via pilot_qa)


class GeometryContract(unittest.TestCase):
    def camera(self, location, rpy):
        return compose(dict(location_cm=location, rotation_rpy_deg=rpy, scale=[1, 1, 1]))

    def test_identity_camera_axes(self):
        R, C = pilot_qa.camera_rh(self.camera([0, 0, 0], [0, 0, 0]))
        K = pilot_qa.intrinsics(90, 1280, 720)
        # Unreal +X forward, +Y right, +Z up.
        for ue, expect in [([100, 0, 0], (640, 360)), ([100, 50, 0], (960, 360)), ([100, 0, 50], (640, 40))]:
            uv, z = pilot_qa.project(K, R, C, pilot_qa.point_rh(ue))
            self.assertAlmostEqual(z, 1.0)
            np.testing.assert_allclose(uv, expect, atol=1e-6)

    def test_unproject_roundtrip(self):
        R, C = pilot_qa.camera_rh(self.camera([120, -340, 155], [2, -7, 33]))
        K = pilot_qa.intrinsics(75, 1280, 720)
        depth = np.full((720, 1280), 6.5)
        mask = np.zeros((720, 1280), bool)
        mask[400, 900] = True
        X = pilot_qa.unproject(K, R, C, depth, mask)[0]
        uv, z = pilot_qa.project(K, R, C, X)
        np.testing.assert_allclose(uv, [900.5, 400.5], atol=1e-6)
        self.assertAlmostEqual(z, 6.5)

    def test_yaw_turn_right_is_right(self):
        Ra, _ = pilot_qa.camera_rh(self.camera([0, 0, 0], [0, 0, 0]))
        Rb, _ = pilot_qa.camera_rh(self.camera([0, 0, 0], [0, 0, 30]))
        optical = Rb[2] @ Ra.T
        self.assertGreater(math.degrees(math.atan2(optical[0], optical[2])), 29)

    def test_direction_rule_matches_real_split(self):
        rng = np.random.default_rng(0)
        for v in rng.normal(scale=0.3, size=(2000, 3)):
            self.assertEqual(pilot_qa.direction(v, 0.10), tracking.direction(v, 0.10))


class Designs(unittest.TestCase):
    def test_designs_hold_on_planned_trajectories(self):
        bad = [r for r in pilot_qa.dry_run((8.0, 15.0, 25.0)) if not r["ok"]]
        self.assertEqual(bad, [])

    def test_assets_and_shapes(self):
        for clip in designs():
            self.assertEqual(len(clip["cam"]["f"]), FRAMES)
            labels = [a["label"] for a in clip["actors"] if a["label"]]
            self.assertEqual(len(labels), len(set(labels)))
            for spec in clip["actors"]:
                self.assertIn(spec["asset"], ASSETS)
                self.assertEqual(len(spec["f"]), FRAMES)
                if spec["gait"] not in ("idle", "fly"):
                    self.assertIn(spec["gait"], ASSETS[spec["asset"]]["clips"])

    def test_answer_balance(self):
        answers = {}
        for clip in designs():
            for d in clip["design"]:
                answers.setdefault(d["task"], set()).add(d["answer"])
        self.assertGreaterEqual(len(answers["camera_translation_direction"]), 5)
        self.assertGreaterEqual(len(answers["object_world_motion_direction"]), 5)


if __name__ == "__main__":
    unittest.main()
