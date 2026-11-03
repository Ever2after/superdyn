"""Geometry and data-contract checks for the independent scene families."""

import json
import math
from pathlib import Path
import random
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scene_families import ASSETS, CONFIG, scaled_layout, inspection_route, mesh_record, context_mesh
from superdyn_ue.family_geometry import Ground, fits
from terrain.layout import height


class SceneFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.families = json.loads(CONFIG.read_text())["families"]
        cls.layouts = {f["id"]: scaled_layout(f["seed"], f["xy_scale"]) for f in cls.families}

    def test_all_families_have_distinct_connected_relief_and_reserved_ascent(self):
        fingerprints = set()
        for spec in self.families:
            layout, report = self.layouts[spec["id"]]
            self.assertEqual(report["status"], "passed")
            self.assertGreaterEqual(max(layout["levels"]) - min(layout["levels"]), 3)
            self.assertGreaterEqual(1 - report["flat_fraction"], 0.30)
            self.assertGreaterEqual(len(layout["ramps"]), 4)
            self.assertTrue(all(c["maximum_grade"] <= 0.6 + 1e-8 for c in report["ramp_connections"]))
            route = inspection_route(layout, 192)
            self.assertGreaterEqual(route["vertical_span_m"], 1.4)
            self.assertEqual(len(route["poses"]), 192)
            self.assertLess(route["support_query_error_m"], 1e-6)
            fingerprints.add(layout["generation"]["layout_sha256"])
        self.assertEqual(len(fingerprints), 3)

    def test_runtime_queries_match_source_mesh_triangles_at_edges_and_inside(self):
        rng = random.Random(112)
        for layout, _ in self.layouts.values():
            ground = Ground(layout)
            a, b, c, d = layout["bounds"]
            points = [(rng.uniform(a, b), rng.uniform(c, d)) for _ in range(500)]
            points += [(a, c), (b, d), (a, d), (b, c)]
            for ramp in layout["ramps"]:
                x0, x1, y0, y1 = ramp["bounds"]
                points += [(x0, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)]
            for x, y in points:
                self.assertAlmostEqual(ground.z(x, y), height(layout, x, y), places=7)

    def test_placement_rejects_cliffs_objects_and_reserved_paths(self):
        layout, _ = self.layouts["industrial"]
        ground = Ground(layout)
        rectangle = layout["generation"]["flat_rectangles"][0]
        a, b, c, d = rectangle["bounds"]
        xy = ((a + b) / 2, (c + d) / 2)
        self.assertIsNotNone(fits(ground, xy, [0.2, 0.2], [], []))
        self.assertIsNone(fits(ground, xy, [0.2, 0.2], [(*xy, 0.3, 0.3)], []))
        self.assertIsNone(fits(ground, xy, [0.2, 0.2], [], [xy]))
        self.assertIsNone(fits(ground, [layout["bounds"][0], 0], [1, 1], [], []))

    def test_export_preserves_surface_normals_and_centimetres(self):
        vertices = [(0, 0, 0), (1, 0, 0.5), (1, 1, 0.5), (0, 1, 0)]
        record = mesh_record(vertices, [[0, 1, 2, 3]], ["ramp"], "industrial")
        section = record["sections"][0]
        points = np.array(section["vertices_cm"])
        normals = np.array(section["normals"])
        self.assertEqual(points[:, 0].max(), 100)
        self.assertEqual(points[:, 2].max(), 50)
        for triangle in section["triangles"]:
            cross = np.cross(points[triangle[1]] - points[triangle[0]], points[triangle[2]] - points[triangle[0]])
            self.assertGreater(np.dot(cross, normals[triangle[0]]), 0)
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1))

    def test_recipes_use_installed_assets_and_different_object_vocabularies(self):
        vocabularies = []
        for spec in self.families:
            vocabulary = {g["asset"] for g in spec["asset_groups"]}
            vocabularies.append(vocabulary)
            for key in vocabulary:
                for path in ASSETS[key][1:]:
                    self.assertTrue((ROOT / "Content" / (path.removeprefix("/Game/") + ".uasset")).is_file(), path)
        self.assertIn("container", vocabularies[0])
        self.assertIn("lamp", vocabularies[1])
        self.assertIn("deer", vocabularies[2])
        self.assertNotIn("deer", vocabularies[0])

    def test_exterior_context_does_not_put_ground_under_the_core(self):
        layout, _ = self.layouts["industrial"]
        record = context_mesh(layout)
        a, b, c, d = layout["bounds"]
        boundary_x = set()
        for section in record["sections"]:
            points = np.array(section["vertices_cm"]) / 100
            triangles = points[np.array(section["triangles"])]
            centres = triangles.mean(axis=1)
            inside = (centres[:, 0] > a + 1e-8) & (centres[:, 0] < b - 1e-8) & (centres[:, 1] > c + 1e-8) & (centres[:, 1] < d - 1e-8)
            self.assertFalse(inside.any())
            boundary_x.update(round(p[0], 7) for p in points if abs(p[1] - c) < 1e-7)
        expected = np.linspace(a, b, round((b - a) / layout["grid_step"]) + 1)
        self.assertTrue(all(round(x, 7) in boundary_x for x in expected))
        closures = [s for s in record["sections"] if s["material"] == "wall"]
        self.assertTrue(closures)
        closing_points = np.array(closures[0]["vertices_cm"]) / 100
        self.assertLess(closing_points[:, 2].min(), layout["bottom_z"])
        self.assertTrue(all(abs(p[0] - a) < 1e-7 or abs(p[0] - b) < 1e-7 or abs(p[1] - c) < 1e-7 or abs(p[1] - d) < 1e-7 for p in closing_points))


if __name__ == "__main__":
    unittest.main()
