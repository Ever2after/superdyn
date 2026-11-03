"""Verify visual fitting preserves the original root coordinate system."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Content/Python"))
from superdyn_ue.asset_layout import fit_visual, walking_play_rate


class OfficialVisualLayoutTests(unittest.TestCase):
    def test_vehicle_fits_original_footprint_without_changing_proportions(self):
        low, high = [-205, -110, -40], [245, 110, 160]
        pose = fit_visual(low, high, [-72.75, -58.3, 0.25], [72.75, 58.3, 124])
        scale = pose["scale"][0]
        self.assertEqual(pose["scale"], [scale] * 3)
        self.assertLessEqual((high[0] - low[0]) * scale, 145.5)
        self.assertLessEqual((high[1] - low[1]) * scale, 116.6)
        self.assertAlmostEqual(low[2] * scale + pose["location_cm"][2], 0.25)
        self.assertAlmostEqual((low[0] + high[0]) * scale / 2 + pose["location_cm"][0], 0)

    def test_person_height_ground_and_forward_axis_are_preserved(self):
        pose = fit_visual([-80, -20, -3], [80, 20, 197], [-26, -35, 0], [26, 35, 181], person=True)
        self.assertAlmostEqual(200 * pose["scale"][2], 181)
        self.assertAlmostEqual(-3 * pose["scale"][2] + pose["location_cm"][2], 0)
        # Rotating source +Y by -90 degrees produces logical +X.
        yaw = math.radians(pose["rotation_rpy_deg"][2])
        self.assertAlmostEqual(-math.sin(yaw), 1)
        self.assertAlmostEqual(math.cos(yaw), 0)

    def test_walking_speed_accounts_for_root_scale_and_frame_rate(self):
        poses = [dict(location_cm=[i * 5, 0, 0], scale=[0.8] * 3) for i in range(25)]
        self.assertAlmostEqual(walking_play_rate(poses, 24), 1)


if __name__ == "__main__":
    unittest.main()
