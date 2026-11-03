"""Holistic video filtering, uncertainty and balanced sampling contracts."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from qa_pipeline import framing_filter as ff
from qa_pipeline.storage import read, save
from qa_pipeline.framing_filter import balanced_sample, decide, validate_observation

def observation(exclude=False):
    return dict(decision='exclude' if exclude else 'keep')

class FramingFilterTests(unittest.TestCase):
    def test_decision_is_preserved_without_extra_postprocessing(self):
        self.assertEqual(decide(observation(True))['decision'],'exclude')
        self.assertEqual(decide(observation())['decision'],'keep')
        self.assertEqual(decide(dict(decision='review')),dict(decision='review'))

    def test_rationale_confidence_or_extra_output_fields_are_rejected(self):
        for field,value in [('reason','rationale'),('confidence',.95),('scene_summary','a scene')]:
            with self.assertRaises(ValueError):
                validate_observation(dict(observation(),**{field:value}))

    def test_unknown_observations_and_per_frame_outputs_do_not_silently_pass(self):
        r=observation()
        r['frames']=[{'frame_index':i} for i in range(9)]
        with self.assertRaises(ValueError):
            validate_observation(r)
        with self.assertRaises(ValueError):
            validate_observation(dict(decision='unknown'))

    def test_sampling_uses_unique_videos_and_balances_cohorts(self):
        qa=[dict(video_id=f'{g}__{i}',group=g) for g in ['LLL','LLH','HHL','HLH'] for i in range(30)]
        qa+=qa[:10]
        sample=balanced_sample(qa,100,11)
        self.assertEqual(len({r['video_id'] for r in sample}),100)
        self.assertEqual({g:sum(r['group4']==g for r in sample) for g in ['LL','LH','HL','HH']},dict(LL=25,LH=25,HL=25,HH=25))
        self.assertEqual(sample,balanced_sample(list(reversed(qa)),100,11))

    def test_all_mode_keeps_every_video_from_unequal_cohorts_once(self):
        qa=[dict(video_id=f'{g}__{i}',group=g) for g,n in [('LLL',2),('LLH',9),('HHH',17)] for i in range(n)]
        qa+=qa[:5]
        selected=balanced_sample(qa,None,11)
        self.assertEqual({r['video_id'] for r in selected},{q['video_id'] for q in qa})
        self.assertEqual(len(selected),28)
        self.assertEqual(sum(r['group4']=='HH' for r in selected),17)

    def test_one_video_is_one_request_containing_all_nine_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            image=out/'frame.jpg';image.write_bytes(b'fixture image bytes')
            save(out/'selection.json',[dict(video_id='example',worker_gpu=4,group='HHH',
                 group4='HH',source='fixture',qa_count=2,
                 frames=[dict(frame_index=i,image_path=str(image)) for i in range(9)])])
            config=dict(base_port=20000,seed=1,prompt_sha256='fixture',inference=dict(max_tokens=32))
            completion=dict(choices=[dict(finish_reason='stop',message=dict(content=__import__('json').dumps(observation())))])
            chats=[]
            def api(port,path,payload=None,timeout=600):
                if path=='models':
                    return dict(data=[dict(id=ff.MODEL_NAME)])
                chats.append(payload)
                return completion
            with patch.object(ff,'api_request',side_effect=api):
                ff.worker(out,config,4,Mock(poll=Mock(return_value=None)))
            self.assertEqual(len(chats),1)
            content=chats[0]['messages'][0]['content']
            self.assertEqual(sum(part['type']=='image_url' for part in content),9)
            self.assertEqual([part['text'] for part in content if part['type']=='text'][1:],
                             [f'Frame F{i}:' for i in range(9)])
            result=read(out/'predictions/example.json')
            self.assertEqual(result['decision'],'keep')
            self.assertNotIn('observation',result)
            self.assertNotIn('reason',result)
            self.assertNotIn('confidence',result)

