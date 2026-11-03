"""Parallel evaluation of a prepared temporal geometry benchmark.

Model inputs contain only RGB target marks, frame labels, questions and options.
Reference answers and geometry evidence are kept outside the adapter interface.
"""
import argparse
import collections
import csv
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from ..config import ROOT, FINAL_QA, FINAL_EVAL
from .protocol import display_options, prompt, parse_answer, stats
OUT=Path(os.environ.get('TEMPORAL_VLM_OUT',str(FINAL_EVAL))).resolve()
SOURCES=[Path(p).resolve() for p in json.loads(os.environ['TEMPORAL_VLM_SOURCES'])] if os.environ.get('TEMPORAL_VLM_SOURCES') else [FINAL_QA]
from ..config import GROUP4, TASKS
JOBS=[]
if (OUT/'jobs_config.json').exists():
 JOBS=[(r['model'],r['gpu'],r['shard'],r['shards']) for r in json.loads((OUT/'jobs_config.json').read_text())]

MAX_TOKENS=64

def stamp():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(path.read_text())
def save(path,data):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def render_inputs(q,source,folder):
 folder.mkdir(parents=True,exist_ok=True)
 labels={oid:label for label,oid in q['target_mapping'].items() if oid in q['object_ids']}
 masks={}
 if labels:
  with np.load(source/'cases'/q['video_id']/'instance_masks.npz') as archive:masks={oid:archive[oid] for oid in labels}
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',17)
 paths=[]
 for i in range(9):
  im=Image.open(source/'cases'/q['video_id']/f'rgb_{i}.jpg').convert('RGB');draw=ImageDraw.Draw(im)
  for oid,label in labels.items():
   mask=masks[oid][i].astype('uint8');color=(242,112,30) if label=='A' else (10,200,125)
   contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
   for contour in contours:
    points=[tuple(p) for p in contour[:,0].tolist()]
    if len(points)>1:draw.line(points+[points[0]],fill=color,width=2)
   yy,xx=np.where(mask)
   if len(xx):
    x=max(0,min(int(np.median(xx)),im.width-25));y=max(25,min(int(yy.min())-24,im.height-25))
    draw.rectangle((x,y,x+24,y+23),fill=color);draw.text((x+5,y+1),label,font=font,fill='black')
  draw.rectangle((0,0,40,24),fill='black');draw.text((4,1),f'F{i}',font=font,fill='white')
  dest=folder/f'F{i}.jpg';im.save(dest,quality=95);paths.append(str(dest))
 return paths

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
 if (OUT/'inputs.json').exists():return read(OUT/'inputs.json')
 cv2.setNumThreads(1);examples=[];reference={};source_hashes={}
 for source in SOURCES:
  path=source/'qa.json';source_hashes[str(path)]=digest(path)
  for q in read(path):
   assert q['quality_status']=='provisional_automated_pass'
   key=q['qa_id'];assert key not in reference
   reference[key]=dict(answer=q['answer'],answer_index=q['answer_index'],video_id=q['video_id'],task=q['task'],group=q['group'])
   presentation=hashlib.sha256(json.dumps([q['video_id'],q['object_ids'],q['target_mapping']],sort_keys=True).encode()).hexdigest()[:20]
   folder=OUT/'media'/presentation
   paths=[str(folder/f'F{i}.jpg') for i in range(9)]
   if not all(Path(path).exists() for path in paths):paths=render_inputs(q,source,folder)
   example=dict(qa_id=key,video_id=q['video_id'],task=q['task'],group=q['group'],group4=GROUP4[q['group']],
    question=q['question'],options=q['options'],frame_paths=paths,frame_indices=list(range(9)),
    target_mapping=q['target_mapping'],requested_object_ids=q['object_ids'])
   example['prompt']=prompt(example);example['prompt_sha256']=hashlib.sha256(example['prompt'].encode()).hexdigest()
   examples.append(example)
 examples.sort(key=lambda example:example['qa_id']);assert examples,'No accepted QA to evaluate'
 save(OUT/'inputs.json',examples);save(OUT/'reference_answers.json',reference)
 save(OUT/'protocol.json',dict(examples=len(examples),source_video_count=sum(len({q['video_id'] for q in read(source/'qa.json')}) for source in SOURCES),qa_video_count=len({e['video_id'] for e in examples}),
  source_hashes=source_hashes,presentation='All nine measurement RGB frames as ordered images, with frame labels and requested target contours only.',
  targets='Original QA A/B mapping retained; model receives only visual marks and original question/descriptions.',
  excluded_from_model_input=['reference answer','answer index','depth','camera pose','geometry evidence','threshold-derived metrics'],
  generation=dict(do_sample=False,num_beams=1,max_new_tokens=MAX_TOKENS),group4_mapping=GROUP4,
  group4_counts=dict(collections.Counter(e['group4'] for e in examples)),scoring=f'Deterministic MCQ matching; invalid/error predictions count as incorrect; denominator {len(examples)} per model.',
  reference_status='Provisional model-estimated geometry labels; no human review.',
  model_input_policy='Same marked images, question and option order for all models; native per-model visual preprocessing.',
  scope='Initial evaluation at nine frames; not the papers original benchmark protocol or a 16/32-frame setting.',
  jobs=[dict(model=m,gpu=gpu,shard=s,shards=n) for m,gpu,s,n in JOBS],created_at=stamp()))
 return examples

def worker(key,gpu,shard,shards,limit=None):
 examples=read(OUT/'inputs.json')[shard::shards]
 if limit:examples=examples[:limit]
 if all((OUT/'predictions'/key/f'{e["qa_id"]}.json').exists() and read(OUT/'predictions'/key/f'{e["qa_id"]}.json').get('status')=='ok' for e in examples):
  save(OUT/'jobs'/f'{key}_s{shard}.json',dict(status='completed',model=key,gpu=gpu,pid=os.getpid(),reused_all=True,updated_at=stamp()));return
 import torch
 from .backends import load_backend
 torch.set_num_threads(2);torch.manual_seed(20261020);cv2.setNumThreads(1)
 specs={m['key']:m for m in read(OUT/'models.json')};spec=specs[key]
 job=f'{key}_s{shard}';state=OUT/'jobs'/f'{job}.json'
 save(state,dict(status='waiting_checkpoint',model=key,gpu=gpu,pid=os.getpid(),updated_at=stamp()))
 ready=Path(spec.get('ready_path',str(Path('/tmp/superdyn_vlm_eval255/models')/key/'READY')));wait_start=time.time()
 while not ready.exists():
  download=OUT/'downloads'/f'{key}.json'
  if download.exists() and read(download)['status']=='failed':raise RuntimeError('Checkpoint download failed: '+read(download).get('error',''))
  if time.time()-wait_start>7200:raise TimeoutError('Checkpoint not ready after two hours')
  time.sleep(5)
 assert ready.read_text()==spec['revision']
 start=time.perf_counter();save(state,dict(status='loading',model=key,gpu=gpu,pid=os.getpid(),updated_at=stamp()))
 backend=load_backend(spec)
 load=time.perf_counter()-start
 save(OUT/'loading'/f'{job}.json',dict(model=key,gpu=gpu,load_seconds=load,**backend.provenance))
 examples=read(OUT/'inputs.json')[shard::shards]
 if limit:examples=examples[:limit]
 references=read(OUT/'reference_answers.json');consecutive_errors=0
 for index,e in enumerate(examples):
  path=OUT/'predictions'/key/f'{e["qa_id"]}.json'
  if path.exists() and read(path)['status']=='ok':continue
  save(state,dict(status='running',model=key,gpu=gpu,pid=os.getpid(),index=index,planned=len(examples),active_qa=e['qa_id'],updated_at=stamp()))
  start=time.perf_counter();raw='';result=dict(qa_id=e['qa_id'],model=key,video_id=e['video_id'],task=e['task'],group=e['group'],group4=e['group4'],gpu=gpu,prompt_sha256=e['prompt_sha256'])
  try:
   raw,usage=backend.generate(e['prompt'],e['frame_paths'])
   parsed=parse_answer(raw,e['options']);correct=parsed==references[e['qa_id']]['answer_index'] if parsed is not None else False
   result.update(status='ok',raw_output=raw,predicted_option_index=parsed,parsed=parsed is not None,correct=bool(correct),usage=usage)
   consecutive_errors=0
  except Exception as exc:
   traceback.print_exc();result.update(status='error',error=f'{type(exc).__name__}: {exc}',raw_output=raw,parsed=False,correct=False)
   consecutive_errors+=1;torch.cuda.empty_cache()
  result.update(elapsed_seconds=time.perf_counter()-start,finished_at=stamp());save(path,result)
  print('PREDICTION',key,index,e['qa_id'],result['status'],result['correct'],round(result['elapsed_seconds'],2),flush=True)
  if consecutive_errors>=3:raise RuntimeError('Three consecutive inference errors; stopping this worker for diagnosis')
 save(state,dict(status='completed',model=key,gpu=gpu,pid=os.getpid(),updated_at=stamp()))

def reuse_predictions(prior):
 """Reuse inference only for identical prompts, images, references and revisions."""
 prior=Path(prior).resolve();old_inputs={e['qa_id']:e for e in read(prior/'inputs.json')}
 old_refs=read(prior/'reference_answers.json');refs=read(OUT/'reference_answers.json')
 old_models={m['key']:m for m in read(prior/'models.json')};models=read(OUT/'models.json')
 assert read(prior/'protocol.json')['generation']==read(OUT/'protocol.json')['generation']
 matched=[];hashes={}
 def sha(path):
  if path not in hashes:hashes[path]=digest(Path(path))
  return hashes[path]
 for e in read(OUT/'inputs.json'):
  old=old_inputs.get(e['qa_id'])
  if old is None or any(e[k]!=old[k] for k in ['prompt_sha256','options','target_mapping','requested_object_ids']):continue
  if refs[e['qa_id']]['answer_index']!=old_refs[e['qa_id']]['answer_index']:continue
  if len(old['frame_paths'])!=len(e['frame_paths']) or any(sha(a)!=sha(b) for a,b in zip(e['frame_paths'],old['frame_paths'])):continue
  for spec in models:
   key=spec['key'];old_spec=old_models.get(key)
   if old_spec is None or any(spec[k]!=old_spec[k] for k in ['revision','backend']):continue
   src=prior/'predictions'/key/f'{e["qa_id"]}.json';dest=OUT/'predictions'/key/src.name
   if not src.exists() or dest.exists():continue
   row=read(src)
   if row['status']!='ok' or row['prompt_sha256']!=e['prompt_sha256']:continue
   index=parse_answer(row['raw_output'],e['options'])
   row.update(group=e['group'],group4=e['group4'],task=e['task'],predicted_option_index=index,parsed=index is not None,
    correct=index is not None and index==refs[e['qa_id']]['answer_index'],reused_prediction=True,reused_from=str(src))
   save(dest,row);matched.append(dict(model=key,qa_id=e['qa_id']))
 path=OUT/'reused_predictions.json';ledger=read(path) if path.exists() else []
 save(path,ledger+matched)
 return dict(collections.Counter(r['model'] for r in matched))

def aggregate(states=None,started=None,final=False):
 inputs=read(OUT/'inputs.json');expected=len(inputs);models=read(OUT/'models.json');summaries={};task_rows=[];group_rows=[];group8_rows=[];ids={e['qa_id'] for e in inputs};model_predictions={}
 for spec in models:
  key=spec['key'];rows=[read(path) for path in sorted((OUT/'predictions'/key).glob('*.json'))]
  model_predictions[key]={r['qa_id']:r for r in rows}
  assert len({r['qa_id'] for r in rows})==len(rows)
  assert all(r['qa_id'] in ids for r in rows),'Unexpected QA prediction'
  summary=stats(rows,expected);summary['status']='completed' if len(rows)==expected and not summary['errors'] else 'completed_with_errors' if len(rows)==expected else 'running'
  per_task={task:stats([r for r in rows if r['task']==task],sum(e['task']==task for e in inputs)) for task in TASKS}
  per_group={group:stats([r for r in rows if r['group4']==group],sum(e['group4']==group for e in inputs)) for group in sorted(set(GROUP4.values()))}
  summary.update(per_task=per_task,per_group4=per_group,per_group8={group:stats([r for r in rows if r['group']==group],sum(e['group']==group for e in inputs)) for group in sorted(set(GROUP4))})
  summary['task_macro_accuracy']=float(np.mean([value['accuracy'] for value in per_task.values() if value['accuracy'] is not None])) if rows else None
  summaries[key]=summary
  task_rows.extend(dict(model=key,task=task,**value) for task,value in per_task.items())
  group_rows.extend(dict(model=key,group=group,**value) for group,value in per_group.items())
  group8_rows.extend(dict(model=key,group=group,**value) for group,value in summary['per_group8'].items())
 elapsed=time.time()-started if started else None
 eta_shards=[]
 for job in states or []:
  key=job['model'];shards=next(n for model,gpu,shard,n in JOBS if model==key and shard==job['shard'])
  shard_ids=[e['qa_id'] for e in inputs[job['shard']::shards]]
  predictions=model_predictions[key]
  pending=sum(qa_id not in predictions or predictions[qa_id]['status']!='ok' for qa_id in shard_ids)
  times=[r['elapsed_seconds'] for r in predictions.values() if r.get('status')=='ok' and not r.get('reused_prediction') and 'elapsed_seconds' in r]
  prior_times=[r['elapsed_seconds'] for r in predictions.values() if r.get('status')=='ok' and 'elapsed_seconds' in r]
  mean=float(np.mean(times)) if times else float(np.mean(prior_times))*1.5 if prior_times else 5.0
  phase=read(OUT/'jobs'/f'{key}_s{job["shard"]}.json').get('status') if (OUT/'jobs'/f'{key}_s{job["shard"]}.json').exists() else 'starting'
  eta_shards.append(dict(model=key,shard=job['shard'],gpu=job['gpu'],remaining=pending,mean_seconds=mean,measured=len(times),phase=phase,seconds=pending*mean+(180 if phase in ['loading','starting','waiting_checkpoint'] else 0)))
 eta=max((j['seconds'] for j in eta_shards),default=0)
 info=dict(status='completed' if all(v['status']=='completed' for v in summaries.values()) else 'completed_with_errors' if final else 'running',
  planned_models=len(models),planned_predictions=len(models)*expected,finished_predictions=sum(v['processed'] for v in summaries.values()),
  completed_models=sum(v['status']=='completed' for v in summaries.values()),updated_at=stamp(),elapsed_seconds=elapsed,models=summaries,
  jobs=states or [],reference_status='automatic geometry labels, not visually reviewed',
  eta=dict(remaining_seconds=0 if final else round(eta),range_seconds=[round(eta*.7),round(eta*1.8)],shards=eta_shards,
   method='Maximum remaining shard times observed generation service time; prior timings only before new observations. Checkpoint download wait is additional.'))
 save(OUT/'run_status.json',info);save(OUT/'summary.json',info)
 for name,rows in [('per_task.csv',task_rows),('per_group4.csv',group_rows),('per_group8.csv',group8_rows)]:
  with (OUT/name).open('w',newline='') as file:
   writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 return info

def spawn_job(key,gpu,shard,shards):
 spec=next(m for m in read(OUT/'models.json') if m['key']==key)
 executable=Path(spec['python']) if spec.get('python') else ROOT/('.venv_eval_vg/bin/python' if key.startswith(('vgllm','qwen25vl')) else '.venv_eval_rgpt/bin/python' if key=='4drgpt_8b' else '.venv_qwen/bin/python')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',HF_HUB_DISABLE_XET='1',TOKENIZERS_PARALLELISM='false')
 with (OUT/'logs'/f'{key}_s{shard}.log').open('a') as log:
  command=[str(executable),'-u','-m','qa_pipeline.evaluation.runner','--worker',key,'--gpu',str(gpu),'--shard',str(shard),'--shards',str(shards)]
  return subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)

def run():
 prepare();lock=(OUT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 (OUT/'controller.pid').write_text(str(os.getpid()));started=time.time();processes=[];attempts={};retry_after={}
 try:
  for key,gpu,shard,shards in JOBS:
   process=spawn_job(key,gpu,shard,shards);attempts[key,shard]=1
   processes.append((key,gpu,shard,process))
  while True:
   for i,(key,gpu,shard,process) in enumerate(processes):
    if process.poll() is not None and process.returncode!=0 and attempts[key,shard]<3:
     token=(key,shard)
     if token not in retry_after:retry_after[token]=time.time()+15
     if time.time()>=retry_after[token]:
      shards=next(n for model,_,s,n in JOBS if model==key and s==shard)
      processes[i]=(key,gpu,shard,spawn_job(key,gpu,shard,shards));attempts[token]+=1;retry_after.pop(token,None)
      print('RETRY_WORKER',key,shard,attempts[token],flush=True)
   final=all(process.poll() is not None for *_,process in processes)
   if any(process.returncode and attempts[key,shard]<3 for key,_,shard,process in processes if process.poll() is not None):final=False
   states=[dict(model=key,gpu=gpu,shard=shard,pid=process.pid,exit_code=process.poll(),attempts=attempts[key,shard]) for key,gpu,shard,process in processes]
   info=aggregate(states,started,final)
   print(stamp(),info['status'],info['finished_predictions'],'/',info['planned_predictions'],flush=True)
   if final:
    if info['status']=='completed':
     subprocess.run([str(ROOT/'.venv/bin/python'),'-m','qa_pipeline.evaluation.reports'],cwd=ROOT,check=True)
    break
   time.sleep(5)
 except BaseException as exc:
  save(OUT/'controller_error.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}',updated_at=stamp()));raise
 finally:
  for *_,process in processes:
   if process.poll() is None:process.send_signal(signal.SIGINT)

def main(argv=None):
 global OUT,SOURCES,JOBS
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--source',type=Path,action='append');ap.add_argument('--models',type=Path);ap.add_argument('--jobs-config',type=Path);ap.add_argument('--prepare','--prepare-only',action='store_true');ap.add_argument('--launch','--background',action='store_true');ap.add_argument('--run',action='store_true')
 ap.add_argument('--worker');ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=1);ap.add_argument('--limit',type=int)
 ap.add_argument('--reuse-evaluation',type=Path,action='append',default=[])
 args=ap.parse_args(argv)
 OUT=args.out.resolve();os.environ['TEMPORAL_VLM_OUT']=str(OUT)
 if args.source:SOURCES=[p.resolve() for p in args.source]
 os.environ['TEMPORAL_VLM_SOURCES']=json.dumps([str(p) for p in SOURCES])
 OUT.mkdir(parents=True,exist_ok=True)
 for flag,name in [(args.models,'models.json'),(args.jobs_config,'jobs_config.json')]:
  if flag and flag.resolve()!=OUT/name:
   if (OUT/'manifest.json').exists() and read(flag)!=read(OUT/name):raise ValueError('Use a new output directory to change a frozen evaluation configuration')
   save(OUT/name,read(flag))
 if (OUT/'jobs_config.json').exists():JOBS=[(r['model'],r['gpu'],r['shard'],r['shards']) for r in read(OUT/'jobs_config.json')]
 if not (OUT/'models.json').exists():ap.error('Provide --models with pinned checkpoint revisions and paths')
 if (args.launch or args.run) and {j[0] for j in JOBS}!={m['key'] for m in read(OUT/'models.json')}:ap.error('jobs_config.json must schedule every model exactly once across its shards')
 if args.worker:worker(args.worker,args.gpu,args.shard,args.shards,args.limit);return
 prepare()
 for prior in args.reuse_evaluation:print('REUSED_PREDICTIONS',json.dumps(reuse_predictions(prior)))
 if args.launch:
  lock=(OUT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
  with (OUT/'logs/controller.log').open('a') as log:
   process=subprocess.Popen([str(ROOT/'.venv/bin/python'),'-u','-m','qa_pipeline.evaluation.runner','--run'],cwd=ROOT,
    env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  (OUT/'controller.pid').write_text(str(process.pid));print(json.dumps(dict(status='launched',pid=process.pid,models=len(read(OUT/'models.json')),predictions=len(read(OUT/'models.json'))*len(read(OUT/'inputs.json')),gpus=list(range(8)))))
 elif args.run:run()
 else:print('Prepared',len(read(OUT/'inputs.json')),'marked-image MCQ inputs')

if __name__=='__main__':main()
