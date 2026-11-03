"""QA-conditioned audit using existing labeled evaluation frames, without answers."""
import base64
import collections
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

from . import framing_filter as runtime
from .config import QWEN_MODEL, TASKS
from .storage import read, save, write_csv

PROMPT_VERSION = 'qa_visual_decision_v1'
PROMPT = """Review the question and options against all nine ordered video frames TOGETHER.
Orange contours mark target A; green contours mark target B. Use the targets,
frame interval and coordinate convention specified in the question.
Exclude the QA if a requested target is only an unidentifiable or severely
cropped fragment, so its intended object-level motion cannot be assessed;
if target labels are missing, misleading or change identity within the relevant
interval; or if the wording cannot be mapped unambiguously to the visible
targets and intended temporal relationship. A visible hand, limb or surface
patch must not be mistaken for an entire clearly observable person or object.
Keep a partial target when its identity and relevant motion remain clear.
Brief occlusion or irrelevant frames outside the question interval alone do
not invalidate a QA. Camera-only questions do not require object labels.
Do not solve the QA, choose an option, infer the reference answer, or reject
a question merely because 3D reasoning or camera-motion compensation is hard.
Choose review for genuinely borderline target visibility or question grounding.
Return ONLY {"decision":"keep"}, {"decision":"exclude"}, or
{"decision":"review"}. No explanation or other fields.
"""

def prior_labels(folder):
    labels={}
    if folder is not None:
        for label in ['exclude','review']:
            path=folder/label/'manifest.json'
            for row in read(path)['cases']:
                labels[row['video_id']]=label
    return labels

def sample_qa(qa, count, seed, prior):
    """Exact task/cohort quotas; each video at most once; known stress cases included.

    Binary sampling constraints, rather than model/answer scores, determine
    eligibility. Random costs resolve ties reproducibly; this is not a random
    population sample and is not used to estimate population exclusion rates.
    """
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    qa=sorted(qa,key=lambda q:q['qa_id'])
    if count<6 or len({q['video_id'] for q in qa})<count:
        raise ValueError('Need at least six QA and enough unique videos')
    videos=sorted({q['video_id'] for q in qa});vi={v:i for i,v in enumerate(videos)}
    unknown=set(prior)-set(videos)
    if unknown:
        raise ValueError(f'Prior cases absent from QA: {unknown}')
    if len(prior)>count:
        raise ValueError('Pilot size smaller than prior diagnostic set')
    groups=['LL','LH','HL','HH'];n=len(videos)
    # Give object tasks the remainder, retaining camera-only controls.
    task_quota={t:count//6 for t in TASKS}
    for t in [*TASKS[2:],*TASKS[:2]][:count%6]:task_quota[t]+=1
    group_quota={g:count//4+(i<count%4) for i,g in enumerate(groups)}
    row_ids=[];cols=[]
    for j,q in enumerate(qa):
        row_ids.extend([vi[q['video_id']],n+TASKS.index(q['task']),n+6+groups.index(runtime.cohort(q['group']))])
        cols.extend([j,j,j])
    matrix=coo_matrix((np.ones(len(cols)),(row_ids,cols)),shape=(n+10,len(qa))).tocsc()
    lower=np.r_[[1. if v in prior else 0. for v in videos],list(task_quota.values()),list(group_quota.values())]
    upper=np.r_[np.ones(n),list(task_quota.values()),list(group_quota.values())]
    costs=np.random.default_rng(seed).uniform(size=len(qa))
    # Prefer object QA for diagnostic videos where such a task exists.
    costs+=np.array([2. if q['video_id'] in prior and not q['requested_object_ids'] else 0. for q in qa])
    result=milp(costs,integrality=np.ones(len(qa)),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,lower,upper),options=dict(time_limit=45,mip_rel_gap=0))
    if result.x is None:
        raise RuntimeError(f'No feasible balanced QA sample: {result.message}')
    selected=[dict(q) for q,x in zip(qa,result.x) if x>.5]
    assert len(selected)==count and len({q['video_id'] for q in selected})==count
    assert set(prior)<={q['video_id'] for q in selected}
    assert dict(collections.Counter(q['task'] for q in selected))==task_quota
    assert dict(collections.Counter(runtime.cohort(q['group']) for q in selected))==group_quota
    np.random.default_rng(seed).shuffle(selected)
    return selected,dict(task_quota=task_quota,group4_quota=group_quota,solver_message=result.message)

def audit_text(q):
    requested=set(q['requested_object_ids'])
    targets=[f'{label} ({"orange" if label=="A" else "green"})' for label,oid in q['target_mapping'].items() if oid in requested]
    options='\n'.join(f'[{i+1}] '+('Object '+v if v in ['A','B'] else v) for i,v in enumerate(q['options']))
    return (PROMPT+'\nRequested visual targets: '+(', '.join(targets) if targets else 'none (camera-only question)')
            +'\nQuestion: '+q['question']+'\nOptions:\n'+options)

def select_qa(qa,count,seed,prior,all_qa=False):
    if not all_qa:
        return sample_qa(qa,count,seed,prior)
    selected=[dict(q) for q in sorted(qa,key=lambda q:q['qa_id'])]
    if len({q['qa_id'] for q in selected})!=len(selected):
        raise ValueError('Duplicate QA IDs in evaluation inputs')
    random.Random(seed).shuffle(selected)
    return selected,dict(task_quota=dict(collections.Counter(q['task'] for q in selected)),
                         group4_quota=dict(collections.Counter(runtime.cohort(q['group']) for q in selected)),
                         complete_dataset=True)

def payload(q,config):
    content=[dict(type='text',text=audit_text(q))]
    for i,path in enumerate(q['frame_paths']):
        content.append(dict(type='text',text=f'Frame F{i}:'))
        data=base64.b64encode(Path(path).read_bytes()).decode()
        content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+data)))
    assert len(q['frame_paths'])==9
    return dict(model=runtime.MODEL_NAME,messages=[dict(role='user',content=content)],
                temperature=0,max_tokens=32,seed=config['seed'],
                chat_template_kwargs=dict(enable_thinking=False),response_format=dict(type='json_object'))

def prepare(out,args):
    out.mkdir(parents=True,exist_ok=True)
    prompt_hash=hashlib.sha256(PROMPT.encode()).hexdigest()
    all_qa=getattr(args,'all_qa',False)
    qa=read(args.eval_out/'inputs.json')
    expected=dict(eval_out=str(args.eval_out.resolve()),prior_filter=str(args.prior_filter.resolve()) if args.prior_filter else None,
                  count=len(qa) if all_qa else args.count,all_qa=all_qa,seed=args.seed,gpus=args.gpus,base_port=args.base_port,
                  model=str(Path(args.qwen_model or QWEN_MODEL).resolve()),prompt_sha256=prompt_hash)
    if (out/'config.json').exists():
        config=read(out/'config.json')
        if any(config.get(k,False if k=='all_qa' else None)!=v for k,v in expected.items()):
            raise ValueError('Cannot resume a different QA audit configuration')
        for q in read(out/'selection.json'):
            assert all(runtime.digest(Path(p))==h for p,h in zip(q['frame_paths'],q['frame_sha256']))
        return config
    for name in ['logs','workers','predictions','raw']:(out/name).mkdir(exist_ok=True)
    prior=prior_labels(args.prior_filter)
    rows,sampling=select_qa(qa,args.count,args.seed,prior,all_qa=all_qa)
    allowed=['qa_id','video_id','task','group','question','options','frame_paths','target_mapping','requested_object_ids']
    selected=[];frame_hashes={}
    for i,q in enumerate(rows):
        row={k:q[k] for k in allowed}
        assert Path(row['qa_id']).name==row['qa_id']
        for p in q['frame_paths']:
            if p not in frame_hashes:frame_hashes[p]=runtime.digest(Path(p))
        row.update(group4=runtime.cohort(q['group']),worker_gpu=args.gpus[i%len(args.gpus)],
                   frame_sha256=[frame_hashes[p] for p in q['frame_paths']],
                   prior_video_decision=prior.get(q['video_id']))
        selected.append(row)
    model=Path(expected['model']);index=read(model/'model.safetensors.index.json')
    weights=sorted(set(index['weight_map'].values()))
    assert all((model/w).is_file() for w in weights)
    config=dict(**expected,created_at=runtime.stamp(),prompt_version=PROMPT_VERSION,
                policy=dict(unit='one QA and nine jointly presented target-labeled frames',decision_only=True,
                            reference_answers_supplied=False,geometry_supplied=False,baseline_predictions_supplied=False,
                            per_frame_predictions=False,hard_qa_is_not_invalid_qa=True),
                sampling=dict(**sampling,unique_videos=len({q['video_id'] for q in rows}),prior_diagnostic_videos=len(prior),
                              population_representative=all_qa,
                              method='All evaluation QA; no subsampling or per-video limit' if all_qa else 'Random-cost constrained task/cohort sampling'),
                inference=dict(temperature=0,max_tokens=32,enable_thinking=False,gpu_memory_utilization=.4,kv_cache_bytes=3*1024**3),
                model_provenance=dict(config_sha256=runtime.digest(model/'config.json'),
                                      architectures=read(model/'config.json')['architectures'],
                                      weight_files=[dict(name=w,bytes=(model/w).stat().st_size) for w in weights]),
                inputs_sha256=runtime.digest(args.eval_out/'inputs.json'),original_qa_modified=False)
    save(out/'selection.json',selected);save(out/'config.json',config)
    (out/'prompt.txt').write_text(PROMPT)
    print('Prepared',len(selected),'QA from',len({q['video_id'] for q in selected}),'videos; diagnostic videos',len(prior),flush=True)
    return config

def worker(out,config,gpu,server):
    port=config['base_port']+gpu;state=out/'workers'/f'gpu{gpu}.json'
    selected=[q for q in read(out/'selection.json') if q['worker_gpu']==gpu]
    save(state,dict(gpu=gpu,phase='loading_model',planned=len(selected),updated_at=runtime.stamp()))
    deadline=time.monotonic()+1500
    while True:
        if server.poll() is not None:raise RuntimeError(f'GPU {gpu} server exited: {server.returncode}')
        try:
            info=runtime.api_request(port,'models',timeout=1)
            if any(m['id']==runtime.MODEL_NAME for m in info['data']):break
        except (OSError,ValueError,KeyError,RuntimeError):pass
        if time.monotonic()>deadline:raise RuntimeError(f'GPU {gpu} load timeout')
        time.sleep(3)
    save(out/'workers'/f'gpu{gpu}_ready.json',dict(gpu=gpu,ready_at=runtime.stamp()))
    for q in selected:
        key=q['qa_id'];dest=out/'predictions'/f'{key}.json'
        if dest.exists() and read(dest).get('status')=='ok':continue
        save(state,dict(gpu=gpu,phase='inference',qa_id=key,updated_at=runtime.stamp()))
        started=time.monotonic();request=payload(q,config);error=None
        for attempt in range(2):
            try:
                raw=runtime.api_request(port,'chat/completions',request)
                save(out/'raw'/f'{key}__attempt{attempt}.json',raw)
                choice=raw['choices'][0]
                if choice['finish_reason']!='stop':raise ValueError('Response truncated')
                decision=runtime.decide(json.loads(choice['message']['content']))['decision']
                result={k:q[k] for k in ['qa_id','video_id','task','group','group4','prior_video_decision']}
                result.update(status='ok',decision=decision,gpu=gpu,usage=raw.get('usage'),attempts=attempt+1,
                              elapsed_seconds=time.monotonic()-started,prompt_sha256=config['prompt_sha256'],finished_at=runtime.stamp())
                save(dest,result);error=None;break
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                save(out/'raw'/f'{key}__attempt{attempt}_error.json',dict(error=error,at=runtime.stamp()))
                time.sleep(1)
        if error:
            result={k:q[k] for k in ['qa_id','video_id','task','group','group4','prior_video_decision']}
            result.update(status='error',error=error,gpu=gpu,elapsed_seconds=time.monotonic()-started)
            save(dest,result)
    save(state,dict(gpu=gpu,phase='completed',updated_at=runtime.stamp()))

def aggregate(out,config,started,errors,final=False):
    selected=read(out/'selection.json');results=[]
    for q in selected:
        path=out/'predictions'/f'{q["qa_id"]}.json'
        if path.exists():results.append(read(path))
    good=[r for r in results if r['status']=='ok'];counts=collections.Counter(r['decision'] for r in good)
    states=[read(p) for p in sorted((out/'workers').glob('gpu[0-9].json'))]
    eta=[]
    for gpu in config['gpus']:
        done=[r for r in good if r['gpu']==gpu];remaining=sum(q['worker_gpu']==gpu for q in selected)-len(done)
        if remaining and not done:eta=[];break
        if done:eta.append(remaining*sum(r['elapsed_seconds'] for r in done)/len(done))
    summary=dict(status=('completed' if len(good)==len(selected) and not errors else 'failed') if final else 'running',
                 phase='inference' if good or any(s['phase']=='inference' for s in states) else 'loading_models',
                 planned=len(selected),finished=len(results),valid=len(good),errors=len(results)-len(good),
                 decisions={d:counts[d] for d in ['keep','exclude','review']},workers=states,worker_errors=errors,
                 elapsed_seconds=time.monotonic()-started,eta_seconds=max(eta) if eta else None,
                 updated_at=runtime.stamp(),gpus=config['gpus'],unit='QA',original_qa_modified=False)
    if final:summary['phase']=summary['status']
    save(out/'run_status.json',summary)
    if final:
        save(out/'results.json',results);save(out/'summary.json',summary)
        write_csv(out/'results.csv',[{k:v for k,v in r.items() if k!='usage'} for r in results])
    return summary

def export(out):
    """Retain labeled frames only for exclude/review; no HTML report."""
    selection={q['qa_id']:q for q in read(out/'selection.json')};results=read(out/'results.json')
    stats=[]
    for field in ['task','group4','prior_video_decision']:
        values=sorted({r.get(field) or 'not_previously_flagged' for r in results})
        for value in values:
            rows=[r for r in results if (r.get(field) or 'not_previously_flagged')==value]
            stats.append(dict(dimension=field,value=value,qa=len(rows),
                              **{d:sum(r.get('decision')==d for r in rows) for d in ['keep','exclude','review']}))
    save(out/'counts.json',stats);write_csv(out/'counts.csv',stats)
    for decision in ['exclude','review']:
        folder=out/decision;folder.mkdir(exist_ok=True);cases=[]
        for r in results:
            if r.get('decision')!=decision:continue
            q=dict(selection[r['qa_id']]);case=folder/r['qa_id'];case.mkdir(exist_ok=True)
            frames=[]
            for i,p in enumerate(q['frame_paths']):
                dest=case/f'F{i}.jpg'
                if not dest.exists():os.link(p,dest)
                frames.append(str(dest))
            q.update(frame_paths=frames,decision=decision)
            save(case/'qa.json',q);save(case/'prediction.json',r);cases.append(q)
        save(folder/'manifest.json',dict(decision=decision,qa=len(cases),cases=cases))
    good=[r for r in results if r['status']=='ok']
    for q in selection.values():
        assert all(runtime.digest(Path(p))==h for p,h in zip(q['frame_paths'],q['frame_sha256']))
    config=read(out/'config.json')
    assert runtime.digest(Path(config['eval_out'])/'inputs.json')==config['inputs_sha256']
    save(out/'validation.json',dict(passed=len(good)==len(selection),qa=len(selection),
                                   unique_videos=len({q['video_id'] for q in selection.values()}),
                                   marked_frames_validated=len(selection)*9,reference_answers_supplied=False,
                                   decisions_only=True,existing_evaluation_inputs_unchanged=True,
                                   html_reports_created=False,manual_ground_truth_validation=False))

def run(out,args):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare(out,args);started=time.monotonic();servers={};errors={}
        aggregate(out,config,started,errors)
        try:
            selected=read(out/'selection.json')
            needed=[g for g in config['gpus'] if any(q['worker_gpu']==g and
                    (not (out/'predictions'/f'{q["qa_id"]}.json').exists() or
                     read(out/'predictions'/f'{q["qa_id"]}.json')['status']!='ok') for q in selected)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(needed))) as pool:
                jobs={}
                for gpu in needed:
                    servers[gpu]=runtime.serve(out,config,gpu)
                    jobs[pool.submit(worker,out,config,gpu,servers[gpu])]=gpu
                while jobs:
                    done,_=concurrent.futures.wait(jobs,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                    for job in done:
                        gpu=jobs.pop(job)
                        try:job.result()
                        except Exception:
                            errors[str(gpu)]=traceback.format_exc()
                            save(out/'workers'/f'gpu{gpu}.json',dict(gpu=gpu,phase='failed',error=errors[str(gpu)]))
                        runtime.stop_owned_server(servers[gpu])
                    aggregate(out,config,started,errors)
        finally:
            for process in servers.values():runtime.stop_owned_server(process)
            summary=aggregate(out,config,started,errors,final=True)
            if summary['status']=='completed':export(out)
        if summary['status']!='completed':raise RuntimeError('QA visual audit incomplete; inspect run_status.json')
