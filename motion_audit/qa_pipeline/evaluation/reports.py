"""Validate complete predictions and produce a static evaluation report."""
import collections,csv,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from .runner import OUT,ROOT,read,save,parse_answer,prompt,digest,stamp

LABELS={'qwen25vl_3b':'Qwen2.5-VL 3B','qwen25vl_7b':'Qwen2.5-VL 7B','vgllm_qa_4b':'VG-LLM QA 4B','vgllm_qa_8b':'VG-LLM QA 8B','mllm4d_sft_8b':'MLLM-4D SFT 8B','4drgpt_8b':'4D-RGPT 8B'}
if (OUT/'models.json').exists():
 LABELS={m['key']:m.get('label',LABELS.get(m['key'],m['key'])) for m in read(OUT/'models.json')}
TASKS={'camera_translation_direction':'Camera translation','camera_view_direction_change':'Camera rotation','object_world_motion_direction':'Object motion direction','camera_object_distance_change':'Camera-object distance','object_pair_distance_change':'Object-object distance','object_travel_distance_comparison':'Object travel comparison'}

def report():
 s=read(OUT/'summary.json');assert s['status']=='completed',s['status']
 examples=read(OUT/'inputs.json');refs=read(OUT/'reference_answers.json');protocol=read(OUT/'protocol.json')
 ids={e['qa_id'] for e in examples};total=len(examples);assert len(ids)==total and total>0
 assert len({e['video_id'] for e in examples})==protocol['qa_video_count']
 for path,sha in protocol['source_hashes'].items():assert digest(Path(path))==sha
 images={p for e in examples for p in e['frame_paths']};manifest={p:digest(Path(p)) for p in sorted(images)}
 save(OUT/'media_sha256.json',manifest)
 for e in examples:
  assert len(e['frame_paths'])==9
  assert e['prompt']==prompt(e)
  assert hashlib.sha256(e['prompt'].encode()).hexdigest()==e['prompt_sha256']
  assert not ({'answer','answer_index','evidence','depth','camera_pose'}&set(e))
 rows=[]
 for key,label in LABELS.items():
  predictions=[read(p) for p in (OUT/'predictions'/key).glob('*.json')]
  assert {p['qa_id'] for p in predictions}==ids
  correct=invalid=0
  lookup={e['qa_id']:e for e in examples}
  for p in predictions:
   e=lookup[p['qa_id']];assert p['status']=='ok'
   assert p['prompt_sha256']==e['prompt_sha256']
   index=parse_answer(p['raw_output'],e['options']);assert index==p['predicted_option_index']
   hit=index is not None and index==refs[e['qa_id']]['answer_index'];assert hit==p['correct']
   assert p['usage']['frames']==9
   if key.startswith('vgllm'):assert p['usage']['geometry_inputs_supplied']
   correct+=hit;invalid+=index is None
  stat=s['models'][key];assert correct==stat['correct'];assert invalid==stat['invalid']
  assert {g:v['processed'] for g,v in stat['per_group4'].items()}=={g:sum(e['group4']==g for e in examples) for g in stat['per_group4']}
  rows.append(dict(model=label,key=key,correct=correct,total=total,accuracy_percent=100*correct/total,task_macro_percent=100*stat['task_macro_accuracy'],invalid=invalid,errors=stat['errors']))
 with (OUT/'leaderboard.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 families=[]
 for key in LABELS:
  for name,tasks in [('tasks_1_2',list(TASKS)[:2]),('tasks_3_4_5_6',list(TASKS)[2:])]:
   n=sum(s['models'][key]['per_task'][task]['expected'] for task in tasks)
   correct=sum(s['models'][key]['per_task'][task]['correct'] for task in tasks)
   families.append(dict(model=key,family=name,correct=correct,total=n,accuracy=correct/n if n else None,
    random_accuracy=float(np.mean([1/len(e['options']) for e in examples if e['task'] in tasks])) if n else None))
 with (OUT/'per_task_family.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(families[0]));w.writeheader();w.writerows(families)
 group8=[dict(model=key,group=g,**value) for key in LABELS for g,value in s['models'][key]['per_group8'].items()]
 with (OUT/'per_group8.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(group8[0]));w.writeheader();w.writerows(group8)
 save(OUT/'random_baseline.json',dict(overall=float(np.mean([1/len(e['options']) for e in examples])),
  per_task={task:float(np.mean([1/len(e['options']) for e in examples if e['task']==task])) if any(e['task']==task for e in examples) else None for task in TASKS},
  per_group4={g:float(np.mean([1/len(e['options']) for e in examples if e['group4']==g])) if any(e['group4']==g for e in examples) else None for g in set(e['group4'] for e in examples)}))
 native=OUT/'pilot_mllm_native_conv/predictions'
 agreement=[]
 for path in native.glob('*.json'):
  old=read(path);new=read(OUT/'predictions/mllm4d_sft_8b'/path.name)
  agreement.append(dict(qa_id=old['qa_id'],raw_equal=old['raw_output']==new['raw_output'],choice_equal=old['predicted_option_index']==new['predicted_option_index']))
 save(OUT/'mllm_projection_response_check.json',dict(native_comparisons=agreement,all_choices_equal=all(x['choice_equal'] for x in agreement) if agreement else None))
 save(OUT/'validation.json',dict(status='passed',checked_at=stamp(),models=len(LABELS),predictions=len(LABELS)*total,qa=total,qa_videos=len({e['video_id'] for e in examples}),unique_images=len(images),checks=['exact QA coverage','source hashes unchanged','prompt reconstruction and hashes','no gold or geometry in model-input fields','all frames exist','all raw responses reparsed and rescored','VG RGB geometry encoder enabled','four group denominators'],chance_accuracy=float(np.mean([1/len(e['options']) for e in examples]))))
 versions={}
 for name in ['VG-LLM','4D-RGPT','MLLM-4D']+(['DSR_Suite'] if 'dsr_suite_7b' in LABELS else []):
  folder=ROOT/'vendor'/name
  versions[name]=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip()
  patch=subprocess.check_output(['git','-C',str(folder),'diff'],text=True)
  if patch:(OUT/(name+'_compatibility.patch')).write_text(patch)
 save(OUT/'code_versions.json',dict(vendor_commits=versions,local_code_sha256={p:digest(Path(__file__).parent/p) for p in ['runner.py','protocol.py','backends.py','reports.py']}))
 draw(s,rows,examples)
 (OUT/'README.txt').write_text(f'''Temporal geometry evaluation

{total} QA / {len({e['video_id'] for e in examples})} videos.
All models receive the same nine marked frames, original question and options.
Invalid responses count as incorrect. References are automatic geometry estimates.
See models.json for pinned checkpoints, protocol.json for input/scoring settings,
summary.json and leaderboard.csv for results, and validation.json for checks.
No reference answer or geometry evidence is sent to an adapter.
''')
 print(json.dumps(rows,indent=2))

def draw(s,rows,examples,*,output=None,title='Temporal Geometry: Initial VLM Evaluation',groups=None,group_field='group4',group_labels=None,footer_extra='',group_title=None,save_pdf=False,model_labels=None):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
 labels=model_labels or LABELS
 keys=[r['key'] for r in rows];names=[labels[k] for k in keys];tasks=list(TASKS)
 groups=groups or ['LLL','LLH','HLL+LHL+HHL','HLH+LHH+HHH']
 group_labels=group_labels or {g:g for g in groups}
 metric='per_group8' if group_field=='group' else 'per_group4'
 fig=plt.figure(figsize=(16,12),facecolor='#f7f9fc');gs=fig.add_gridspec(3,1,height_ratios=[1.3,2,1.4],hspace=.64)
 fig.suptitle(title,fontsize=23,fontweight='bold',y=.985)
 fig.text(.5,.95,f'{len(examples)} QA / {len({e["video_id"] for e in examples})} videos / 6 tasks / same 9 marked RGB frames per question',ha='center',color='#465469')
 ax=fig.add_subplot(gs[0]);values=[r['accuracy_percent'] for r in rows]
 bars=ax.barh(names[::-1],values[::-1],color=['#2368a2' if k.startswith('vgllm') else '#68879f' for k in keys[::-1]])
 for b,v,r in zip(bars,values[::-1],rows[::-1]):ax.text(v+.6,b.get_y()+b.get_height()/2,f'{v:.2f}% ({r["correct"]}/{len(examples)})',va='center')
 chance=100*np.mean([1/len(e['options']) for e in examples]);ax.axvline(chance,color='#b07036',ls='--',label=f'Uniform random: {chance:.2f}%')
 ax.set(xlim=(0,max(65,max(values)+15)),xlabel='Overall accuracy (%)',title='Agreement with automatic geometry references');ax.legend(loc='lower right',fontsize=9)
 def heat(ax,data,labels,title):
  ax.imshow(data,vmin=0,vmax=100,cmap='Blues',aspect='auto')
  ax.set_xticks(range(len(labels)),labels,fontsize=10);ax.set_yticks(range(len(names)),names);ax.set_title(title,pad=16,fontweight='bold')
  for i in range(len(names)):
   for j in range(len(labels)):ax.text(j,i,f'{data[i,j]:.1f}%' if np.isfinite(data[i,j]) else 'n/a',ha='center',va='center',color='white' if data[i,j]>55 else '#142c41')
 heat(fig.add_subplot(gs[1]),np.array([[100*s['models'][k]['per_task'][t]['accuracy'] if s['models'][k]['per_task'][t]['accuracy'] is not None else np.nan for t in tasks] for k in keys]),[f'{TASKS[t]}\n(n={sum(e["task"]==t for e in examples)})' for t in tasks],'Accuracy by task')
 heat(fig.add_subplot(gs[2]),np.array([[100*s['models'][k][metric][g]['accuracy'] if s['models'][k][metric][g]['accuracy'] is not None else np.nan for g in groups] for k in keys]),[f'{group_labels[g]} (n={sum(e[group_field]==g for e in examples)})' for g in groups],group_title or 'Accuracy by requested motion groups (translation / rotation / dynamics)')
 fig.subplots_adjust(left=.17,right=.97,bottom=.095,top=.89)
 fig.text(.03,.025,'Provisional automatic labels; no human GT verification. Invalid answers count as wrong.\nNative per-model image preprocessing; adapted 9-image protocol, not paper benchmark reproduction.'+('\n'+footer_extra if footer_extra else ''),fontsize=10,color='#586578')
 target=output or OUT
 fig.savefig(target/'results.png',dpi=160)
 if save_pdf:fig.savefig(target/'results.pdf')
 plt.close(fig)

def subset_report(requested,output,*,source=None,qa_ids=None,title=None,footer=''):
 from .runner import stats
 source=Path(source or OUT).resolve();parent=read(source/'summary.json')
 labels_by_model={m['key']:m.get('label',LABELS.get(m['key'],m['key'])) for m in read(source/'models.json')}
 assert parent['status']=='completed','Subset reporting requires a complete parent evaluation'
 normalized=[g.upper().replace('N','L') for g in requested]
 if any(len(g)!=3 or set(g)-set('HL') for g in normalized) or len(set(normalized))!=len(normalized):
  raise ValueError('Use distinct three-letter H/L or H/N groups')
 output=Path(output).resolve()
 if output==source:raise ValueError('A subset must have a separate output directory')
 output.mkdir(parents=True,exist_ok=True)
 labels=dict(zip(normalized,[g.upper() for g in requested]))
 examples=[e for e in read(source/'inputs.json') if e['group'] in normalized and (qa_ids is None or e['qa_id'] in qa_ids)]
 if qa_ids is not None and {e['qa_id'] for e in examples}!=set(qa_ids):raise ValueError('Selection contains missing IDs or QA outside requested groups')
 if not examples:raise ValueError('No QA in requested groups')
 if (output/'manifest.json').exists() and {e['qa_id'] for e in examples}!=set(read(output/'manifest.json')['selection']['qa_ids']):
  raise ValueError('Cannot overwrite a frozen benchmark with a different QA selection')
 refs=read(source/'reference_answers.json');ids={e['qa_id'] for e in examples};lookup={e['qa_id']:e for e in examples}
 summary=dict(status='completed',source=str(source),groups=normalized,display_groups=list(labels.values()),qa=len(examples),videos=len({e['video_id'] for e in examples}),models={},created_at=stamp())
 leaderboard=[];task_rows=[];group_rows=[];families=[]
 def chance(subset):return float(np.mean([1/len(e['options']) for e in subset])) if subset else None
 random=dict(overall=chance(examples),per_task={t:chance([e for e in examples if e['task']==t]) for t in TASKS},per_group={g:chance([e for e in examples if e['group']==g]) for g in normalized})
 for key,label in labels_by_model.items():
  rows=[]
  for qa_id in sorted(ids):
   row=read(source/'predictions'/key/f'{qa_id}.json');e=lookup[qa_id]
   assert row['qa_id']==qa_id and row['model']==key and row['video_id']==e['video_id'] and row['task']==e['task']
   assert row['status']=='ok' and row['group']==e['group'] and row['prompt_sha256']==e['prompt_sha256'] and row['usage']['frames']==9
   index=parse_answer(row['raw_output'],e['options'])
   assert index==row['predicted_option_index'] and row['parsed']==(index is not None)
   assert row['correct']==(index is not None and index==refs[qa_id]['answer_index'])
   rows.append(row)
  model=stats(rows,len(examples))
  model['per_task']={t:stats([r for r in rows if r['task']==t],sum(e['task']==t for e in examples)) for t in TASKS}
  model['per_group8']={g:stats([r for r in rows if r['group']==g],sum(e['group']==g for e in examples)) for g in normalized}
  model['per_group4']={g:stats([r for r in rows if r['group4']==g],sum(e['group4']==g for e in examples)) for g in sorted({e['group4'] for e in examples})}
  model['status']='completed'
  model['task_macro_accuracy']=float(np.mean([v['accuracy'] for v in model['per_task'].values() if v['accuracy'] is not None]))
  if qa_ids is None:
   assert model['correct']==sum(parent['models'][key]['per_group8'][g]['correct'] for g in normalized)
   assert model['processed']==sum(parent['models'][key]['per_group8'][g]['processed'] for g in normalized)
  summary['models'][key]=model
  leaderboard.append(dict(model=label,key=key,correct=model['correct'],total=len(examples),accuracy_percent=100*model['accuracy'],task_macro_percent=100*model['task_macro_accuracy'],random_percent=100*random['overall'],gap_vs_random_pp=100*(model['accuracy']-random['overall']),invalid=model['invalid'],errors=model['errors']))
  task_rows.extend(dict(model=key,task=t,**v) for t,v in model['per_task'].items())
  group_rows.extend(dict(model=key,group=g,display_group=labels[g],**v) for g,v in model['per_group8'].items())
  for name,tasks in [('tasks_1_2',list(TASKS)[:2]),('tasks_3_4_5_6',list(TASKS)[2:])]:
   selected=[r for r in rows if r['task'] in tasks]
   families.append(dict(model=key,family=name,**stats(selected),random_accuracy=chance([e for e in examples if e['task'] in tasks])))
 for name,rows in [('leaderboard.csv',leaderboard),('per_task.csv',task_rows),('per_group8.csv',group_rows),('per_task_family.csv',families)]:
  with (output/name).open('w',newline='') as file:
   writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 save(output/'summary.json',summary);save(output/'random_baseline.json',random)
 save(output/'selection.json',dict(groups=normalized,display_groups=labels,qa_ids=sorted(ids),qa=len(ids),videos=summary['videos'],group_qa_counts=dict(collections.Counter(e['group'] for e in examples)),task_qa_counts=dict(collections.Counter(e['task'] for e in examples))))
 for filename,table in [
  ('accuracy_by_task.csv',[dict(model=labels_by_model[key],**{f'task_{i+1}':100*model['per_task'][t]['accuracy'] if model['per_task'][t]['accuracy'] is not None else None for i,t in enumerate(TASKS)},overall=100*model['accuracy']) for key,model in summary['models'].items()]),
  ('accuracy_by_group.csv',[dict(model=labels_by_model[key],**{labels[g]:100*model['per_group8'][g]['accuracy'] if model['per_group8'][g]['accuracy'] is not None else None for g in normalized},overall=100*model['accuracy']) for key,model in summary['models'].items()])]:
  with (output/filename).open('w',newline='') as file:
   writer=csv.DictWriter(file,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
 group_note='H = high; N = non-high / low (L in source metadata). Order: translation / rotation / scene dynamics.'
 draw(summary,leaderboard,examples,output=output,model_labels=labels_by_model,save_pdf=True,title=title or 'Temporal Geometry: '+' + '.join(labels.values()),groups=normalized,group_field='group',group_labels=labels,footer_extra=footer or group_note)
 save(output/'validation.json',dict(status='passed',checked_at=stamp(),qa=len(examples),videos=summary['videos'],models=len(labels_by_model),predictions=len(examples)*len(labels_by_model),source_inputs_sha256=digest(source/'inputs.json'),source_summary_sha256=digest(source/'summary.json'),checks=['Exact selected QA IDs and requested groups','Same denominator for every model','Raw answers reparsed and rescored'],new_inference=False))
 return summary


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--groups',nargs='+')
 parser.add_argument('--report-out',type=Path)
 args=parser.parse_args()
 if args.groups:
  if args.report_out is None:parser.error('--groups requires --report-out')
  subset_report(args.groups,args.report_out)
 else:report()
