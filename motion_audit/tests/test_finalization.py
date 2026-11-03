"""Release selection and cleanup must not change the benchmark's meaning."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qa_pipeline.finalize import link_shared, select_final
from qa_pipeline.maintenance import cleanup
from qa_pipeline.storage import save


class FinalSelectionTests(unittest.TestCase):
    def examples(self):
        groups = ["HLH", "LHH", "HHH", "LLH", "HHL", "HHH"]
        rows = [dict(qa_id=str(i), video_id=f"v{i}", task="motion", group=g) for i, g in enumerate(groups)]
        decisions = [dict(q, status="ok", decision="keep") for q in rows]
        decisions[2]["decision"] = "exclude"
        decisions[5]["decision"] = "review"
        return rows, decisions

    def test_both_camera_and_dynamics_and_keep_are_required(self):
        rows, decisions = self.examples()
        selected = select_final(rows, decisions, ["HNH", "NHH", "HHH"])
        self.assertEqual([q["qa_id"] for q in selected], ["0", "1"])
        self.assertEqual(selected, rows[:2])
        self.assertEqual(decisions[2]["decision"], "exclude")

    def test_missing_duplicate_or_failed_audit_never_silently_keeps_qa(self):
        rows, decisions = self.examples()
        for bad in [decisions[:-1], decisions + [decisions[0]],
                    [dict(d, status="error") if i == 3 else d for i, d in enumerate(decisions)]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                select_final(rows, bad)

    def test_mismatched_group_metadata_is_rejected(self):
        rows, decisions = self.examples()
        decisions[0]["group"] = "LLH"
        with self.assertRaisesRegex(ValueError, "metadata mismatch"):
            select_final(rows, decisions)

    def test_relative_asset_links_resolve_and_cannot_replace_other_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.jpg"
            source.write_bytes(b"marked pixels")
            target = root / "release/media/frame.jpg"
            link_shared(source, target)
            link_shared(source, target)
            self.assertEqual(target.read_bytes(), b"marked pixels")
            self.assertFalse(Path(target.readlink()).is_absolute())
            other = root / "other.jpg"
            other.write_bytes(b"other")
            with self.assertRaises(ValueError):
                link_shared(other, target)


class CleanupTests(unittest.TestCase):
    def test_dry_run_and_cleanup_preserve_evidence_and_other_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["outputs/run/controller.log", "outputs/run/controller.pid", "outputs/run/qa.json",
                         "outputs/run/predictions/response.json", "outputs/run/cases/source.log",
                         ".venv/lib/keep.pyc", "vendor/model/keep.log", "__pycache__/module.pyc"]:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name)
            with patch("qa_pipeline.maintenance.active_controllers", return_value=[]):
                plan = cleanup(root)
                self.assertEqual(plan["files"], 3)
                self.assertTrue((root / "outputs/run/controller.log").exists())
                result = cleanup(root, apply=True)
            self.assertEqual(result["files"], 3)
            self.assertFalse((root / "outputs/run/controller.log").exists())
            for name in ["outputs/run/qa.json", "outputs/run/predictions/response.json",
                         "outputs/run/cases/source.log", ".venv/lib/keep.pyc", "vendor/model/keep.log"]:
                self.assertTrue((root / name).is_file(), name)

    def test_active_jobs_prevent_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "controller.log"
            path.write_text("still in use")
            with patch("qa_pipeline.maintenance.active_controllers", return_value=[dict(pid=123)]):
                with self.assertRaisesRegex(RuntimeError, "still active"):
                    cleanup(root, apply=True)
            self.assertTrue(path.exists())


class FrozenEvaluationTests(unittest.TestCase):
    def test_four_cohort_report_cannot_replace_final_benchmark(self):
        from qa_pipeline.evaluation.filtered_report import run

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "final"
            save(out / "manifest.json", dict(status="final"))
            with self.assertRaisesRegex(ValueError, "frozen benchmark"):
                run(root / "source", root / "audit.json", out)
            self.assertFalse((out / "summary.json").exists())

    def test_group_report_cannot_expand_frozen_qa_ids(self):
        from qa_pipeline.evaluation.reports import subset_report

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, out = root / "source", root / "final"
            save(source / "summary.json", dict(status="completed"))
            save(source / "models.json", [])
            save(source / "inputs.json", [dict(qa_id="extra", group="HHH")])
            save(out / "manifest.json", dict(selection=dict(qa_ids=["frozen"])))
            with self.assertRaisesRegex(ValueError, "frozen benchmark"):
                subset_report(["HHH"], out, source=source)
            self.assertFalse((out / "summary.json").exists())
