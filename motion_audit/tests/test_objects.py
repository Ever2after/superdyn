"""Contracts for the new localization ownership and geometry-first workflow."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from qa_pipeline import objects as g
from qa_pipeline import tracking as t


class GMOSPipelineTests(unittest.TestCase):
    def test_mask_bbox_keeps_last_pixel_and_empty_visibility(self):
        m = np.zeros((20, 40), bool)
        self.assertIsNone(g.bbox_from_mask(m))
        m[19, 39] = True
        self.assertEqual(g.bbox_from_mask(m), [975, 950, 1000, 1000])
        self.assertTrue(t.base.box_mask(g.bbox_from_mask(m), 20, 40)[m].all())

    def test_mask_bbox_covers_exact_rectangle_in_normalized_grid(self):
        m = np.zeros((100, 200), bool)
        m[20:80, 40:160] = True
        self.assertEqual(g.bbox_from_mask(m), [200, 200, 800, 800])
        np.testing.assert_array_equal(t.base.box_mask(g.bbox_from_mask(m), 100, 200), m)

    def test_camera_only_fallback_has_no_fabricated_entities(self):
        r = g.camera_record("test")
        self.assertEqual(r["object_ids"], [])
        self.assertFalse(r["pair_distance_pass"])
        self.assertFalse(r["travel_comparison_pass"])
        self.assertEqual(r["anchors_a"], [])
        self.assertEqual(r["anchors_b"], [])

    def test_semantic_prompt_excludes_localization_and_geometry_answers(self):
        self.assertIn(
            "Do not generate bounding boxes, masks, tracking IDs, 3D geometry, or QA answers.",
            g.SEMANTIC_PROMPT,
        )
        self.assertIn("single_entity", g.SEMANTIC_PROMPT)


if __name__ == "__main__":
    unittest.main()
