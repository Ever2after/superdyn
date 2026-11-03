"""Boundary walls enclose the real terrain without changing camera trajectories."""

import sys
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Content/Python"))
from superdyn_ue.enclosure import wall_records
from superdyn_ue.io_utils import load_scene


class BoundaryGeometryTests(unittest.TestCase):
    def test_real_walls_have_level_tops_above_highest_ground_and_outward_thickness(self):
        for case in ["01_low_follow","02_low_slalom","03_low_orbit","04_ramp_ascent"]:
            packet=load_scene(ROOT/"inputs"/case/"scene.json.gz")
            terrain=next(m for m in packet["meshes"] if m["key"]=="terrain")
            walls=wall_records(terrain)
            self.assertEqual(len(walls),4)
            low,high=walls[0]["terrain_bounds_cm"]
            expected_top=500 if case=="04_ramp_ascent" else 400
            self.assertLess(low[2],0)  # Mesh underside is not the lowest ground.
            for pose in packet["camera"]:
                for axis in [0,1]:
                    self.assertGreater(pose["location_cm"][axis],low[axis])
                    self.assertLess(pose["location_cm"][axis],high[axis])
            for wall in walls:
                p=wall["profile"];vertices=np.array(wall["sections"][0]["vertices_cm"])
                offset=(vertices[:,p["axis"]]-p["fixed"])*p["outward_sign"]
                self.assertGreaterEqual(offset.min(),-1e-6)
                self.assertLessEqual(offset.max(),30+1e-6)
                self.assertEqual(wall["ground_elevation_range_cm"],[0,expected_top-100])
                self.assertEqual(wall["top_z_cm"],expected_top)
                self.assertEqual(wall["height_above_lowest_ground_cm"],expected_top)
                self.assertEqual(wall["clearance_above_highest_ground_cm"],100)
                self.assertTrue(np.isin(vertices[:,2],[wall["bottom_z_cm"],expected_top]).all())
                for coordinate,_ in p["floor_profile_cm"]:
                    expected=np.zeros(3);expected[p["axis"]]=p["fixed"]
                    expected[1-p["axis"]]=coordinate;expected[2]=expected_top
                    self.assertLess(np.linalg.norm(vertices-expected,axis=1).min(),1e-6)

    def test_boundary_triangles_have_consistent_outward_source_normals(self):
        packet=load_scene(ROOT/"inputs/03_low_orbit/scene.json.gz")
        terrain=next(m for m in packet["meshes"] if m["key"]=="terrain")
        for wall in wall_records(terrain):
            section=wall["sections"][0]
            vertices=np.array(section["vertices_cm"]);normals=np.array(section["normals"])
            triangles=np.array(section["triangles"])
            cross=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
            self.assertTrue(np.isfinite(vertices).all())
            self.assertGreater(np.sum(cross*normals[triangles[:,0]],axis=1).min(),0)


if __name__=="__main__":
    unittest.main()
