"""Offline tests only: true source geometry/math, not a mocked Unreal runtime."""

import json, math, random, sys, tempfile, unittest, zipfile, stat, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Content/Python"))
from bridge import engine_binary, command, validate_inputs, verify_frame_sequence
from installer import install_archive
from superdyn_ue.math3d import (
    point_to_unreal,
    point_to_blender,
    actor_matrix_to_unreal,
    camera_matrix_to_unreal,
    decompose,
    compose,
    rotation_matrix,
    project_unreal,
    unwrap_poses,
)
from superdyn_ue.io_utils import load_scene


class CoordinateTests(unittest.TestCase):
    def test_centimetres_and_handedness_are_explicit(self):
        self.assertEqual(point_to_unreal([1, 2, 3]), [100.0, -200.0, 300.0])
        self.assertEqual(point_to_blender([100, -200, 300]), [1.0, 2.0, 3.0])
        yaw = np.eye(4)
        yaw[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        self.assertAlmostEqual(
            decompose(actor_matrix_to_unreal(yaw))["rotation_rpy_deg"][2], -90.0
        )

    def test_camera_axes_preserve_a_known_optical_projection(self):
        c = camera_matrix_to_unreal(np.eye(4))
        K = [[500, 0, 640], [0, 500, 360], [0, 0, 1]]
        self.assertTrue(
            np.allclose(project_unreal(point_to_unreal([1, 2, -5]), c, K), [740, 160])
        )
        self.assertAlmostEqual(np.linalg.det(np.array(c)[:3, :3]), 1.0)

    def test_rotator_roundtrip_under_random_rotation_and_scale(self):
        rng = random.Random(741)
        for _ in range(200):
            p = dict(
                location_cm=[rng.uniform(-2000, 2000) for _ in range(3)],
                rotation_rpy_deg=[
                    rng.uniform(-170, 170),
                    rng.uniform(-85, 85),
                    rng.uniform(-170, 170),
                ],
                scale=[rng.uniform(0.2, 3.0) for _ in range(3)],
            )
            self.assertTrue(
                np.allclose(compose(decompose(compose(p))), compose(p), atol=1e-8)
            )

    def test_angle_unwrapping_avoids_a_full_spin(self):
        p = [
            dict(location_cm=[0, 0, 0], rotation_rpy_deg=[0, 0, y], scale=[1, 1, 1])
            for y in [178, 179, -179, -178]
        ]
        self.assertEqual(
            [x["rotation_rpy_deg"][2] for x in unwrap_poses(p)], [178, 179, 181, 182]
        )

    def test_four_real_exported_inputs_validate(self):
        report = validate_inputs()
        self.assertEqual(report["cases"], 4)
        self.assertEqual(report["frames"], 768)
        self.assertFalse(report["unreal_execution_verified"])
        self.assertLess(
            max(r["max_projection_error_px"] for r in report["results"]), 0.01
        )

    def test_rotating_drone_parts_are_not_frozen_into_the_main_mesh(self):
        p = load_scene(ROOT / "inputs/01_low_follow/scene.json.gz")
        parts = [m for m in p["meshes"] if m.get("animated")]
        self.assertEqual(len(parts), 4)
        for part in parts:
            poses = p["tracks"][part["key"]]
            self.assertEqual(part["parent"], "air")
            self.assertGreater(
                abs(poses[-1]["rotation_rpy_deg"][2] - poses[0]["rotation_rpy_deg"][2]),
                360,
            )


class FrameVerificationTests(unittest.TestCase):
    """Exercise image validation only; these test PNGs are not Unreal renders."""

    def test_preview_checks_the_three_expected_frames(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "rgb").mkdir()
            for frame in [0, 64, 128]:
                cv2.imwrite(
                    str(output / "rgb" / f"frame_{frame:06d}.png"),
                    np.zeros((18, 32, 3), np.uint8),
                )
            paths = verify_frame_sequence(
                output, dict(frames=192, width=32, height=18), preview=True
            )
            self.assertEqual(len(paths), 3)
            self.assertFalse((output / "verification.json").exists())

    def test_incomplete_sequence_cannot_be_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                verify_frame_sequence(
                    Path(directory), dict(frames=192, width=32, height=18)
                )

    def test_wrong_resolution_cannot_be_verified(self):
        import cv2

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "rgb").mkdir()
            cv2.imwrite(
                str(output / "rgb/frame_000000.png"), np.zeros((8, 8, 3), np.uint8)
            )
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                verify_frame_sequence(output, dict(frames=1, width=32, height=18))


class DriverTests(unittest.TestCase):
    def test_launcher_runs_real_editor_and_offscreen_not_null_rhi(self):
        args = command("/engine/UnrealEditor", ROOT / "request.json", 6)
        self.assertIn("-RenderOffscreen", args)
        self.assertIn("-vulkan", args)
        self.assertNotIn("-nullrhi", args)
        self.assertTrue(any(x.startswith("-ExecCmds=py ") for x in args))

    def test_install_preserves_executable_and_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive = d / "test.zip"
            exe = zipfile.ZipInfo("Engine/Binaries/Linux/UnrealEditor")
            exe.create_system = 3
            exe.external_attr = (stat.S_IFREG | 0o755) << 16
            link = zipfile.ZipInfo("Engine/Binaries/Linux/UnrealEditor-Cmd")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr(exe, b"not-a-real-engine")
                z.writestr(link, b"UnrealEditor")
            target = install_archive(archive, d / "install")
            self.assertTrue(os.access(target / exe.filename, os.X_OK))
            self.assertTrue((target / link.filename).is_symlink())
            self.assertEqual(engine_binary(str(target)), target / exe.filename)

    def test_archive_escape_is_rejected_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            archive = d / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape", b"bad")
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                install_archive(archive, d / "install")
            self.assertFalse((d / "escape").exists())

    def test_parallel_install_handles_shared_parent_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "parallel.zip"
            contents = {
                f"Engine/shared/{index}.bin": bytes([index]) * 8192
                for index in range(32)
            }
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_DEFLATED
            ) as package:
                for name, data in contents.items():
                    package.writestr(name, data)
            destination = install_archive(archive, directory / "install", workers=8)
            for name, data in contents.items():
                self.assertEqual((destination / name).read_bytes(), data)

    def test_resume_rechecks_crc_and_repairs_same_size_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "resume.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr("Engine/corrupt.bin", b"correct")
                package.writestr("Engine/keep.bin", b"keep")
            destination = directory / "install"
            (destination / "Engine").mkdir(parents=True)
            (destination / "Engine/corrupt.bin").write_bytes(b"incorrc")
            keep = destination / "Engine/keep.bin"
            keep.write_bytes(b"keep")
            modified = keep.stat().st_mtime_ns
            install_archive(archive, destination, workers=4, resume=True)
            self.assertEqual(
                (destination / "Engine/corrupt.bin").read_bytes(), b"correct"
            )
            self.assertEqual(keep.stat().st_mtime_ns, modified)

    def test_native_editor_scripts_parse_without_importing_unreal(self):
        import ast

        for p in (ROOT / "Content/Python").rglob("*.py"):
            ast.parse(p.read_text(), filename=str(p))


if __name__ == "__main__":
    unittest.main()
