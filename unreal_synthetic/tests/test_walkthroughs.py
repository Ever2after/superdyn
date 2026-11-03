"""Regression checks for the cleaned delivery workflow and its failure guards."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from walkthroughs import DEFAULT_OUTPUT, verify_case
from plan_ab_walkthroughs import specifications
import render_ab_walkthroughs as renderer


class WalkthroughWorkflowTests(unittest.TestCase):
    def test_standalone_placements_resolve_to_installed_native_assets(self):
        specs = specifications()
        self.assertEqual(len(specs), 51)
        self.assertEqual(len({spec["id"] for spec in specs}), 51)
        self.assertEqual(len({spec["source_pack"] for spec in specs}), 14)
        for spec in specs:
            for value in [spec["mesh"], spec.get("animation")]:
                if not value:
                    continue
                relative = value.removeprefix("/Game/").split(".", 1)[0]
                self.assertTrue((ROOT / "Content" / (relative + ".uasset")).is_file(), value)

    def test_completed_preview_stops_before_launching_an_editor(self):
        with patch.object(renderer, "OUT", DEFAULT_OUTPUT), patch.object(renderer, "start", side_effect=AssertionError("Editor must not start")):
            with self.assertRaisesRegex(RuntimeError, "Completed delivery is preserved"):
                renderer.preview(7)

    def test_existing_passed_receipt_does_not_hide_a_corrupt_movie(self):
        original = DEFAULT_OUTPUT / "01_east_to_west"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / original.name
            folder.mkdir()
            for name in ["verification.json", "route.json", "native_validation.json"]:
                (folder / name).write_bytes((original / name).read_bytes())
            (folder / "sample.mp4").write_bytes(b"corrupt movie")
            with self.assertRaisesRegex(ValueError, "Movie hash mismatch"):
                verify_case(folder, DEFAULT_OUTPUT / "scene_manifest.json")

    def test_refactored_placement_values_match_the_original_render_recipe(self):
        snapshot = DEFAULT_OUTPUT / "source_snapshot/scripts/plan_ab_walkthroughs.py"
        tree = ast.parse(snapshot.read_text())
        namespace = {}
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "ANCHORS" for target in statement.targets):
                namespace["anchors"] = ast.literal_eval(statement.value)
        import plan_ab_walkthroughs
        self.assertEqual(plan_ab_walkthroughs.ANCHORS, namespace["anchors"])
        manifest = json.loads((DEFAULT_OUTPUT / "scene_manifest.json").read_text())
        specs = {item["id"]: item for item in specifications()}
        for record in manifest["objects"]:
            spec = specs[record["id"]]
            self.assertEqual(spec["mesh"], record["mesh"])
            self.assertEqual(spec["source_pack"], record["source_pack"])
            self.assertEqual(spec["anchor_cm"], record["requested_anchor_cm"])


if __name__ == "__main__":
    unittest.main()
