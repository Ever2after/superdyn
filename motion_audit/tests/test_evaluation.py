import unittest
from qa_pipeline.evaluation.runner import parse_answer,prompt,GROUP4

class EvaluationContractTests(unittest.TestCase):
 def test_object_ids_do_not_change_option_letter_semantics(self):
  self.assertEqual(parse_answer('A',['B','A']),0)
  self.assertEqual(parse_answer('Object A',['B','A']),1)
  self.assertEqual(parse_answer('A. Object B',['B','A']),0)
  self.assertEqual(parse_answer('<answer>B</answer>',['B','A']),1)
 def test_semantic_and_explicit_mcq_answers(self):
  self.assertEqual(parse_answer('increased',['increased','decreased']),0)
  self.assertEqual(parse_answer('The answer is C.',['left','right','forward']),2)
  self.assertEqual(parse_answer('(B)',['farther','closer']),1)
 def test_ambiguous_responses_are_not_guessed(self):
  self.assertIsNone(parse_answer('A or B',['A','B']))
  self.assertIsNone(parse_answer('The answer is A or B',['left','right']))
  self.assertIsNone(parse_answer('I cannot tell',['left','right']))
 def test_prompt_does_not_serialize_reference_or_geometry(self):
  example=dict(question='Which way did object A move?',options=['B','A'],answer='REFERENCE_SECRET',
   answer_index=123456789,evidence={'private':'GEOMETRY_SECRET'})
  value=prompt(example)
  self.assertNotIn('REFERENCE_SECRET',value);self.assertNotIn('GEOMETRY_SECRET',value);self.assertNotIn('123456789',value)
  self.assertIn('A. Object B',value);self.assertIn('B. Object A',value)
 def test_requested_four_group_mapping(self):
  self.assertEqual({key for key,value in GROUP4.items() if value=='HLL+LHL+HHL'},{'HLL','LHL','HHL'})
  self.assertEqual({key for key,value in GROUP4.items() if value=='HLH+LHH+HHH'},{'HLH','LHH','HHH'})

class VariableSizeEvaluationTests(unittest.TestCase):
 def test_reuse_checks_pixels_and_rescores_without_changing_old_result(self):
  import tempfile
  from pathlib import Path
  from qa_pipeline.evaluation import runner as ev
  before=ev.OUT
  with tempfile.TemporaryDirectory() as tmp:
   try:
    root=Path(tmp);old=root/'old';ev.OUT=root/'new'
    old_image=root/'old.jpg';new_image=root/'new.jpg'
    old_image.write_bytes(b'identical image');new_image.write_bytes(b'identical image')
    common=dict(qa_id='qa',video_id='v',task=ev.TASKS[0],options=['left','right'],prompt_sha256='same-prompt',target_mapping={},requested_object_ids=[])
    old_e=dict(common,group='LLL',group4='LLL',frame_paths=[str(old_image)]*9)
    new_e=dict(common,group='HHH',group4=ev.GROUP4['HHH'],frame_paths=[str(new_image)]*9)
    for folder,e in [(old,old_e),(ev.OUT,new_e)]:
     ev.save(folder/'inputs.json',[e]);ev.save(folder/'reference_answers.json',{'qa':dict(answer_index=1)})
     ev.save(folder/'protocol.json',dict(generation=dict(do_sample=False,max_new_tokens=64)))
     ev.save(folder/'models.json',[dict(key='test',revision='pinned',backend='qwen25')])
    src=old/'predictions/test/qa.json'
    ev.save(src,dict(common,status='ok',raw_output='A',correct=True,group='LLL',group4='LLL'))
    self.assertEqual(ev.reuse_predictions(old),dict(test=1))
    dest=ev.OUT/'predictions/test/qa.json';row=ev.read(dest)
    self.assertFalse(row['correct']);self.assertEqual(row['group'],'HHH')
    self.assertTrue(ev.read(src)['correct']);self.assertEqual(ev.read(src)['group'],'LLL')
    dest.unlink();new_image.write_bytes(b'changed pixels')
    self.assertEqual(ev.reuse_predictions(old),{})
    self.assertFalse(dest.exists())
   finally:ev.OUT=before

 def test_aggregate_uses_actual_qa_count_and_retains_empty_tasks(self):
  import tempfile,json
  from pathlib import Path
  from qa_pipeline.evaluation import runner as ev
  prior=ev.OUT
  with tempfile.TemporaryDirectory() as temp:
   try:
    ev.OUT=Path(temp)
    examples=[dict(qa_id='first',task=ev.TASKS[0],group='LLL',group4='LLL'),dict(qa_id='second',task=ev.TASKS[0],group='HHH',group4='HLH+LHH+HHH')]
    ev.save(ev.OUT/'inputs.json',examples);ev.save(ev.OUT/'models.json',[dict(key='test_model')])
    for i,e in enumerate(examples):ev.save(ev.OUT/'predictions/test_model'/f'{e["qa_id"]}.json',dict(e,status='ok',parsed=bool(i),correct=bool(i)))
    result=ev.aggregate(final=True);model=result['models']['test_model']
    self.assertEqual(result['planned_predictions'],2);self.assertEqual(result['status'],'completed')
    self.assertEqual(model['expected'],2);self.assertEqual(model['accuracy'],.5);self.assertEqual(model['invalid'],1)
    self.assertEqual(set(model['per_task']),set(ev.TASKS));self.assertIsNone(model['per_task'][ev.TASKS[1]]['accuracy'])
    self.assertEqual(model['per_group4']['LLH']['expected'],0)
   finally:ev.OUT=prior

if __name__=='__main__':unittest.main()
