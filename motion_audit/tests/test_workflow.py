"""Scope, non-destructive extraction and resumable generation contracts."""

import tempfile, time, unittest, zipfile
from pathlib import Path
from unittest.mock import patch
from qa_pipeline import generation
from qa_pipeline.config import TH, group
from qa_pipeline.storage import save, read
from qa_pipeline.inventory import discover, prepare_generation
from qa_pipeline.extraction import extract_one


class WorkflowTests(unittest.TestCase):
    def test_readiness_probes_a_running_http_model_server(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(dict(data=[dict(id=self.server.model)])).encode())

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.model = "temporal-object-qwen"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            self.assertTrue(generation.server_ready(server.server_port))
            server.model = "different-model"
            self.assertFalse(generation.server_ready(server.server_port))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_scope_excludes_spring_and_non_rgb_entries_remain_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dataset in [
                "DAVIS",
                "spring",
                "dynpose-100k",
                "point_odyssey",
                "dynamic_replica",
            ]:
                (root / dataset / "scene" / "rgb").mkdir(parents=True)
            (root / "youtube_vis" / "metadata_only").mkdir(parents=True)
            rows = discover(root)
            self.assertEqual({r["stratum"] for r in rows}, {"DAVIS", "youtube_vis"})
            self.assertEqual(
                next(r for r in rows if r["stratum"] == "youtube_vis")[
                    "rgb_directory_ready"
                ],
                False,
            )

    def test_missing_group_flags_are_unknown_instead_of_low(self):
        with self.assertRaises(ValueError):
            group(dict(translation_high=True, rotation_high=False))
        self.assertEqual(
            group(
                dict(translation_high="True", rotation_high=False, dynamic_high=True)
            ),
            "HLH",
        )

    def test_zip_resume_verifies_existing_files_and_preserves_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data.zip"
            state = root / "state.json"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("scene/rgb/000.png", b"original")
            extract_one(archive, state)
            extract_one(archive, state)
            self.assertEqual(read(state)["status"], "completed")
            self.assertTrue(archive.exists())
            target = root / "scene/rgb/000.png"
            target.write_bytes(b"conflict")
            with self.assertRaises(ValueError):
                extract_one(archive, state)
            self.assertEqual(target.read_bytes(), b"conflict")

    def test_zip_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape", b"bad")
            with self.assertRaises(ValueError):
                extract_one(archive, root / "state.json")
            self.assertFalse((root.parent / "escape").exists())

    def test_archive_run_exports_complete_progress(self):
        from qa_pipeline.extraction import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "SAV"
            dataset.mkdir()
            with zipfile.ZipFile(dataset / "data.zip", "w") as archive:
                archive.writestr("scene/rgb/000.png", b"frame")
            run(root, root / "status", ["SAV"], jobs=1)
            status = read(root / "status/run_status.json")
            self.assertEqual(status["status"], "completed")
            self.assertEqual(status["completed_files"], 1)
            self.assertEqual(status["completed_bytes"], 5)

    def test_background_paths_keep_callers_working_directory(self):
        import os
        from types import SimpleNamespace
        from qa_pipeline.cli import background
        from qa_pipeline.config import ROOT

        cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                with patch(
                    "qa_pipeline.cli.subprocess.Popen",
                    return_value=SimpleNamespace(pid=987654321),
                ) as spawn:
                    background(
                        [
                            "generate",
                            "--out",
                            "run",
                            "--manifest",
                            "clips.json",
                            "--background",
                        ]
                    )
                args = spawn.call_args.args[0]
                self.assertEqual(args[args.index("--out") + 1], str(Path(tmp) / "run"))
                self.assertEqual(
                    args[args.index("--manifest") + 1], str(Path(tmp) / "clips.json")
                )
                self.assertEqual(spawn.call_args.kwargs["cwd"], ROOT)
                self.assertNotIn("--background", args)
            finally:
                os.chdir(cwd)

    def test_completed_arbitrary_size_run_resumes_without_loading_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            selection = dict(planned=2, selected_groups={"LLL": 2}, videos=[])
            for gpu in [3, 7]:
                vid = f"video{gpu}"
                selection["videos"].append(dict(video_id=vid, worker_gpu=gpu))
                folder = out / "cases" / vid
                folder.mkdir(parents=True)
                save(
                    folder / "result.json",
                    dict(
                        video_id=vid,
                        gpu=gpu,
                        group="LLL",
                        preprocess_complete=True,
                        semantic_complete=True,
                        qa_count=0,
                        preprocess_seconds=1,
                    ),
                )
                save(folder / "selected_record.json", dict(video_id=vid))
                save(folder / "qa.json", [])
                save(folder / "qa_generated.json", [])
            save(out / "selection.json", selection)
            save(
                out / "config.json",
                dict(
                    qa_thresholds=TH,
                    source=str(out / "source"),
                    base_port=19920,
                    gpus=[3, 7],
                    qwen={"model": "unused"},
                ),
            )
            with patch.object(
                generation,
                "spawn_worker",
                side_effect=AssertionError("Unexpected model load"),
            ):
                generation.run(out)
            result = read(out / "run_status.json")
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["finished"], 2)
            self.assertEqual(result["videos_without_qa"], 2)

    def test_running_case_is_not_miscounted_as_zero_yield(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(
                root / "cases/video/result.json",
                dict(video_id="video", gpu=0, group="LLL", status="preprocessing"),
            )
            status = generation.aggregate(
                root,
                dict(
                    planned=1,
                    videos=[dict(video_id="video", worker_gpu=0)],
                    selected_groups={"LLL": 1},
                ),
                {0: dict(phase="preprocess", phase_started=time.time())},
                time.time(),
            )
            self.assertEqual(status["preprocessed"], 0)
            self.assertEqual(status["videos_without_qa"], 0)


if __name__ == "__main__":
    unittest.main()
