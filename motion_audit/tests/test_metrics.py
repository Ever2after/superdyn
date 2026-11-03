import unittest
import numpy as np
from qa_pipeline.metrics import (
    camera_rates,
    rigid_flow,
    summarize_intervals,
    summarize_amplitudes,
)


class PhysicsChecks(unittest.TestCase):
    def test_rigid_translation_sign_and_scale_invariance(self):
        e = np.stack([np.eye(4)[:3], np.eye(4)[:3]])
        e[1, 0, 3] = -1
        depth = np.full((16, 24), 10.0)
        k = np.array([[100.0, 0, 12], [0, 100, 8], [0, 0, 1.0]])
        flow, valid, _ = rigid_flow(depth, k, k, e[0], e[1])
        np.testing.assert_allclose(flow[..., 0], -10, atol=1e-5)
        np.testing.assert_allclose(flow[..., 1], 0, atol=1e-5)
        scaled = e.copy()
        scaled[:, :, 3] *= 3
        flow2, _, _ = rigid_flow(depth * 3, k, k, scaled[0], scaled[1])
        np.testing.assert_allclose(flow, flow2, atol=1e-5)
        trans, rot, c = camera_rates(e, np.array([0.0, 2.0]), 10.0)
        np.testing.assert_allclose(trans, [0.05])
        np.testing.assert_allclose(rot, [0.0])
        np.testing.assert_allclose(c[1], [1, 0, 0])

    def test_camera_and_object_motion_at_different_times_is_not_joint(self):
        rows = [
            dict(
                dt=1.0,
                quality_valid=True,
                camera_translation=0.1 if i < 2 else 0,
                camera_rotation_deg=0.0,
                dynamic_residual_p75=0.0 if i < 2 else 0.1,
                dynamic_area=0.2,
            )
            for i in range(4)
        ]
        result = summarize_intervals(rows)
        self.assertTrue(result["camera_high"])
        self.assertTrue(result["dynamic_high"])
        self.assertEqual(result["joint_high_duty"], 0.0)
        self.assertFalse(result["joint_high"])

    def test_failed_reconstruction_is_unknown_not_low(self):
        result = summarize_intervals([dict(dt=1.0, quality_valid=False)])
        self.assertFalse(result["eligible"])
        self.assertNotIn("joint_high", result)

    def test_amplitude_is_invariant_to_encoded_video_clock(self):
        rows = [
            dict(
                dt=0.5,
                quality_valid=True,
                camera_translation=0.15,
                camera_rotation_deg=15.0,
                dynamic_residual_p75=0.09,
                dynamic_area=0.2,
            )
            for _ in range(8)
        ]
        compressed = [
            r
            | dict(
                dt=r["dt"] / 10,
                camera_translation=r["camera_translation"] * 10,
                camera_rotation_deg=r["camera_rotation_deg"] * 10,
                dynamic_residual_p75=r["dynamic_residual_p75"] * 10,
            )
            for r in rows
        ]
        a = summarize_amplitudes(rows)
        b = summarize_amplitudes(compressed)
        for key in [
            "camera_translation_path_depths_valid",
            "camera_rotation_path_deg_valid",
            "dynamic_residual_path_diagonals",
            "joint_high_duty",
        ]:
            self.assertAlmostEqual(a[key], b[key])
        self.assertEqual(a["joint_high"], b["joint_high"])


if __name__ == "__main__":
    unittest.main()
