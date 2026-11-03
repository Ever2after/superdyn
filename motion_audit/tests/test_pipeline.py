"""End-to-end scope screening and safe reuse of completed annotations."""
import tempfile
import unittest
from pathlib import Path
from qa_pipeline.config import TH
from qa_pipeline.pipeline import screen, reuse_cases
from qa_pipeline.storage import read, save
from qa_pipeline.inventory import cache_source, prepare_generation
import numpy as np
from PIL import Image


class PipelineTests(unittest.TestCase):
    def test_extracted_archive_cache_is_rebound_without_changing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rgb = root / 'scene/rgb'
            rgb.mkdir(parents=True)
            cache_root = root / 'geometry'
            cache = cache_root / 'cache/SAV__scene/predictions.npz'
            cache.parent.mkdir(parents=True)
            for i in range(9):
                image = Image.new('RGB', (2, 3), (i, 0, 0))
                image.save(rgb / f'{i}.png')
                image.save(cache.parent / f'{i * 4:04d}.png')
            np.savez(cache, times=np.arange(9) / 8)
            metrics = dict(translation_high=False, rotation_high=False, dynamic_high=True)
            result = dict(video_id='SAV__scene', stratum='SAV', scene_id='scene', status='ok',
                          source_path=str(root / 'data.zip'), source_kind='zip', zip_prefix='scene',
                          total_frames=9, selected_frame_indices=list(range(9)),
                          selected_frame_names=[f'{i}.png' for i in range(9)],
                          amplitude_summary=dict(eligible=True), unfiltered_amplitude_summary=metrics)
            original = cache_root / 'results/SAV__scene.json'
            save(original, result)
            self.assertEqual(cache_source(result, cache), root / 'scene')
            out = root / 'qa'
            selection = prepare_generation(out, cache_roots=[cache_root])
            self.assertEqual(selection['planned'], 1)
            normalized = read(out / 'source/results/SAV__scene.json')
            self.assertEqual(normalized['source_path'], str(root / 'scene'))
            self.assertEqual(normalized['source_kind'], 'directory')
            self.assertEqual(read(original), result)
            Image.new('RGB', (2, 3), (255, 0, 0)).save(rgb / '4.png')
            self.assertIsNone(cache_source(result, cache))

    def test_completed_geometry_worker_does_not_construct_engine(self):
        from qa_pipeline.preprocess import worker
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save(out / 'config.json', dict(gpus=[0]))
            save(out / 'selection.json', dict(videos=[dict(video_id='v')]))
            save(out / 'results/v.json', dict(status='ok'))
            cache = out / 'cache/v/predictions.npz'
            cache.parent.mkdir(parents=True)
            cache.touch()
            worker(out, 0)
            self.assertEqual(read(out / 'logs/gpu0_status.json')['reused'], 1)

    def test_screen_records_missing_and_short_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for name, count in [('valid', 9), ('short', 8), ('metadata', None)]:
                folder = root / name
                if count is not None:
                    (folder / 'rgb').mkdir(parents=True)
                    for i in range(count):
                        (folder / 'rgb' / f'{i}.png').touch()
                rows.append(dict(video_id=name, stratum='DAVIS', source_path=str(folder)))
            yes, no = screen(rows)
            self.assertEqual([r['video_id'] for r in yes], ['valid'])
            self.assertEqual({r['reason'] for r in no},
                             {'missing_rgb_directory', 'fewer_than_9_distinct_measurement_frames'})

    def test_reuse_requires_same_inputs_and_preserves_prior_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old, new = root / 'old', root / 'new'
            cache = root / 'cache/predictions.npz'
            cache.parent.mkdir()
            cache.touch()
            source = root / 'source'
            data = dict(intervals=[{'quality_valid': True}], selected_frame_indices=[0, 4],
                        selected_frame_names=['a.png', 'b.png'], source_path='/data/scene')
            save(source / 'results/v.json', data)
            save(root / 'new_result.json', data)
            save(old / 'config.json', dict(source=str(source), qa_thresholds=TH))
            save(old / 'selection.json', dict(videos=[dict(video_id='v', vggt_cache=str(cache))]))
            folder = old / 'cases/v'
            save(folder / 'result.json', dict(video_id='v', gpu=0, group='LLL',
                 preprocess_complete=True, semantic_complete=True, qa_count=1))
            save(folder / 'selected_record.json', dict(video_id='v'))
            save(folder / 'mask_complete.json', dict(backend='full GMOS MOS'))
            for name in ['qa_numeric.json', 'qa.json', 'qa_generated.json']:
                save(folder / name, [dict(video_id='v', group='LLL', thresholds=TH)])
            row = dict(video_id='v', worker_gpu=7, group='HHH', qa_split='development',
                       vggt_cache=str(cache), result_path=str(root / 'new_result.json'))
            reused = reuse_cases(new, dict(videos=[row]), [old])
            self.assertEqual(len(reused), 1)
            self.assertEqual(reuse_cases(new, dict(videos=[row]), [old]), reused)
            self.assertEqual(read(new / 'cases/v/result.json')['gpu'], 7)
            self.assertEqual(read(new / 'cases/v/qa.json')[0]['group'], 'HHH')
            self.assertEqual(read(old / 'cases/v/qa.json')[0]['group'], 'LLL')
            self.assertEqual(read(old / 'cases/v/result.json')['gpu'], 0)
            save(root / 'new_result.json', dict(data, intervals=[{'quality_valid': False}]))
            self.assertEqual(reuse_cases(root / 'mismatch', dict(videos=[row]), [old]), [])


if __name__ == '__main__':
    unittest.main()
