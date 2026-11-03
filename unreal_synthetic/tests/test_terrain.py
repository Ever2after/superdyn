"""Reproducible partitions, full-surface connectivity and rejected bad layouts."""
from copy import deepcopy
import json,subprocess,sys,tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from terrain.sampling import generate,load_settings
from terrain.validation import validate


class TerracedSamplingTests(unittest.TestCase):
    def test_same_seed_reproduces_complete_geometry_and_context(self):
        self.assertEqual(generate(42),generate(42))
        self.assertNotEqual(generate(42)['generation']['layout_sha256'],generate(43)['generation']['layout_sha256'])

    def test_many_seeds_change_partition_and_height_topology(self):
        scenes=[generate(seed) for seed in range(32)]
        self.assertGreaterEqual(len({len(s['plateaus']) for s in scenes}),3)
        self.assertGreaterEqual(len({tuple(s['levels']) for s in scenes}),3)
        self.assertEqual({s['generation']['pattern'] for s in scenes},{'mixed_extrema'})
        self.assertEqual(len({s['generation']['layout_sha256'] for s in scenes}),len(scenes))
        settings=load_settings();settings['level_count_range']=[5,5];settings['region_count_range']=[9,10]
        self.assertEqual(len(generate(31,settings)['levels']),5)

    def test_all_flat_areas_and_ramps_remain_connected(self):
        for seed in [7,31,42,73,123,2026,31415,27182]:
            layout=generate(seed);report=validate(layout)
            self.assertEqual(report['status'],'passed',report)
            self.assertGreaterEqual(report['flat_fraction'],layout['generation']['settings']['minimum_flat_fraction'])
            self.assertLessEqual(report['flat_fraction'],layout['generation']['settings']['maximum_flat_fraction']+1e-9)
            self.assertTrue(all(r['area_m2']>=layout['generation']['settings']['minimum_flat_rectangle_area'] for r in report['flat_rectangles']))
            self.assertTrue(all(check['passed'] for check in report['checks']))

    def test_multiple_extrema_and_no_dominant_direction(self):
        for seed in range(32):
            report=validate(generate(seed));m=report['height_metrics']
            self.assertGreaterEqual(m['global_high_components'],2)
            self.assertGreaterEqual(m['global_low_components'],2)
            self.assertGreaterEqual(min(m['high_area_fraction'],m['low_area_fraction']),.12-1e-9)
            self.assertLessEqual(m['planar_trend_r2'],.5+1e-9)

    def test_monotonic_height_field_is_rejected_despite_forged_metadata(self):
        layout=generate(42)
        # Deliberately create an exact X gradient; stored sampler metrics are unchanged.
        for p in layout['plateaus']:p['height']=(p['bounds'][0]+p['bounds'][1])/2+20
        report=validate(layout);checks={c['name']:c['passed'] for c in report['checks']}
        self.assertFalse(checks['multiple_separated_high_and_low_zones'])
        self.assertFalse(checks['height_distribution_is_not_a_dominant_plane'])
        self.assertAlmostEqual(report['height_metrics']['planar_trend_r2'],1.)

    def test_archived_settings_keep_their_original_validation_contract(self):
        layout=generate(42);layout['generation']['settings'].pop('height_assignment')
        report=validate(layout)
        self.assertEqual(report['status'],'passed')
        self.assertEqual(report['height_metrics'],{})

    def test_compact_footprint_and_minimum_slope_coverage(self):
        for seed in [7,55,42,3,56,73]:
            layout=generate(seed);x0,x1,y0,y1=layout['bounds'];report=validate(layout)
            self.assertLessEqual((x1-x0)*(y1-y0),24*22)
            self.assertGreaterEqual(1-report['flat_fraction'],.30-1e-9)
            self.assertGreaterEqual(len(layout['ramps']),5)

    def test_lowering_flat_limit_rejects_an_otherwise_valid_surface(self):
        layout=generate(42);report=validate(layout)
        settings=deepcopy(layout['generation']['settings']);settings['maximum_flat_fraction']=report['flat_fraction']-.01
        checks={c['name']:c['passed'] for c in validate(layout,settings)['checks']}
        self.assertFalse(checks['sufficient_ramp_coverage'])
        # Old settings lacking this new cap retain their validation contract.
        settings.pop('maximum_flat_fraction')
        self.assertEqual(validate(layout,settings)['status'],'passed')

    def test_removing_ramps_disconnects_the_ground(self):
        layout=generate(42);layout['ramps']=[];report=validate(layout)
        self.assertEqual(report['status'],'failed')
        self.assertTrue(any(c['name']=='entire_walkable_surface_is_connected' and not c['passed'] for c in report['checks']))

    def test_wrong_end_height_is_detected_across_full_width(self):
        layout=generate(42);layout['ramps'][0]['z1']+=.5;report=validate(layout)
        self.assertEqual(report['status'],'failed')
        self.assertTrue(any(c['name']=='ramp_ends_join_across_full_width' and not c['passed'] for c in report['checks']))

    def test_invalid_grid_and_impossible_constraints_fail_boundedly(self):
        settings=load_settings();settings['mesh_grid_step']=.3
        with self.assertRaisesRegex(ValueError,'align'):generate(42,settings)
        settings=load_settings();settings.update(minimum_flat_fraction=.999,maximum_attempts=2);settings.pop('maximum_flat_fraction')
        with self.assertRaisesRegex(RuntimeError,'2 attempts'):generate(42,settings)

    def test_sample_cli_needs_no_blender_and_writes_valid_json(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plan.json'
            subprocess.run([sys.executable,str(ROOT/'manage.py'),'terrain','sample','--seed','42','--output',str(path)],cwd=folder,capture_output=True,check=True)
            layout=json.loads(path.read_text());self.assertEqual(layout['generation']['seed'],42)
            result=subprocess.run([sys.executable,str(ROOT/'manage.py'),'terrain','validate','--layout',str(path)],cwd=folder,capture_output=True,check=True,text=True)
            self.assertEqual(json.loads(result.stdout)['status'],'passed')
