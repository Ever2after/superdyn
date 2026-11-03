"""Input/answer separation, labeled-frame ordering and balanced pilot coverage."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from collections import Counter

from qa_pipeline.config import TASKS
from qa_pipeline.qa_visual_filter import audit_text, payload, sample_qa, select_qa

class QAVisualFilterTests(unittest.TestCase):
    def test_full_scope_keeps_multiple_questions_from_each_video(self):
        qa=[dict(qa_id='v1_q1',video_id='v1',task=TASKS[0],group='LLL'),
            dict(qa_id='v1_q2',video_id='v1',task=TASKS[3],group='LLL'),
            dict(qa_id='v2_q1',video_id='v2',task=TASKS[2],group='HHH')]
        rows,meta=select_qa(qa,100,1,{},all_qa=True)
        self.assertEqual({r['qa_id'] for r in rows},{'v1_q1','v1_q2','v2_q1'})
        self.assertEqual(len(rows),3)
        self.assertTrue(meta['complete_dataset'])
        self.assertEqual(meta['group4_quota'],dict(LL=2,HH=1))

    def test_joint_request_preserves_question_options_and_target_identity_without_answers(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths=[]
            for i in range(9):
                f=Path(tmp)/f'F{i}.jpg';f.write_bytes(f'labeled frame {i}'.encode());paths.append(str(f))
            q=dict(question='From F2 to F5, which target traveled farther?',options=['B','A'],
                   requested_object_ids=['g2'],target_mapping={'A':'g1','B':'g2'},frame_paths=paths,
                   answer='SECRET_REFERENCE',evidence='SECRET_GEOMETRY',prompt='SECRET_BASELINE_PROMPT')
            request=payload(q,dict(seed=1))
            text=request['messages'][0]['content'][0]['text']
            self.assertIn(q['question'],text)
            self.assertIn('Requested visual targets: B (green)',text)
            self.assertIn('[1] Object B\n[2] Object A',text)
            self.assertNotIn('SECRET',json.dumps(request))
            images=[c for c in request['messages'][0]['content'] if c['type']=='image_url']
            self.assertEqual(len(images),9)
            for i,image in enumerate(images):
                self.assertEqual(base64.b64decode(image['image_url']['url'].split(',')[1]),f'labeled frame {i}'.encode())
            self.assertEqual(request['max_tokens'],32)
            self.assertFalse(request['chat_template_kwargs']['enable_thinking'])

    def test_camera_question_has_no_implied_object_target(self):
        text=audit_text(dict(question='Camera translation?',options=['left','right'],
                             requested_object_ids=[],target_mapping={'A':'g1'}))
        self.assertIn('Requested visual targets: none (camera-only question)',text)

    def test_sampling_covers_tasks_cohorts_and_known_diagnostics_without_repeating_videos(self):
        qa=[]
        for ti,task in enumerate(TASKS):
            for group in ['LLL','LLH','HHL','HHH']:
                for i in range(7):
                    key=f'fixture__{ti}_{group}_{i}'
                    qa.append(dict(qa_id=key,video_id=key,task=task,group=group,
                                   requested_object_ids=['g1'] if ti>=2 else []))
        prior={q['video_id']:'review' for q in qa[::10]}
        selected,meta=sample_qa(qa,100,7,prior)
        self.assertEqual(len(selected),100)
        self.assertEqual(len({q['video_id'] for q in selected}),100)
        self.assertTrue(set(prior)<={q['video_id'] for q in selected})
        self.assertEqual(dict(Counter(q['task'] for q in selected)),meta['task_quota'])
        self.assertEqual(meta['group4_quota'],dict(LL=25,LH=25,HL=25,HH=25))
