"""Resumable Qwen framing audit; original videos and QA are never modified.

Nine frames are presented together in one multimodal request. Qwen makes
one holistic clip-level decision; no per-frame labels or frame-count voting.
"""
import base64
import collections
import concurrent.futures
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone

from PIL import Image, ImageOps
from .config import ROOT, QWEN_MODEL
from .storage import read, save, write_csv

MODEL_NAME = "framing-filter-qwen"
PROMPT_VERSION = "framing_decision_v3"
PROMPT = """Inspect all nine chronological frames together as ONE video.
Exclude overly close egocentric, selfie, or object-closeup footage where the
main people or objects are mostly outside the frame and only small body parts
or object fragments are visible. Judge the overall sequence, not one brief
close pass. Keep videos with largely visible subjects and usable spatial
layout, including wide first-person views and complete small objects filmed
nearby. Choose review if the overall framing is ambiguous or genuinely mixed.
Return only {"decision":"keep"}, {"decision":"exclude"}, or
{"decision":"review"}. No explanation or other fields.
"""

def stamp():
    return datetime.now(timezone.utc).isoformat()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def cohort(group):
    if len(group) != 3 or set(group) - {'L', 'H'}:
        raise ValueError(f"Invalid motion group {group}")
    return ('H' if 'H' in group[:2] else 'L') + group[2]

def validate_observation(result):
    if not isinstance(result,dict) or set(result)!={'decision'}:
        raise ValueError('Return only one decision field, with no explanation or per-frame labels')
    if result['decision'] not in {'keep','exclude','review'}:
        raise ValueError('Invalid framing decision')
    return result

def decide(result):
    validate_observation(result)
    return dict(decision=result['decision'])

def balanced_sample(qa, count, seed):
    by_video={}
    for q in qa:
        prior=by_video.setdefault(q['video_id'],q['group'])
        if prior != q['group']:
            raise ValueError('One video has inconsistent motion groups')
    if count is None:
        selected=[dict(video_id=v,group=g,group4=cohort(g)) for v,g in sorted(by_video.items())]
        random.Random(seed).shuffle(selected)
        return selected
    if count < 4 or count > len(by_video):
        raise ValueError('Sample count must be between four and the QA video count')
    rng=random.Random(seed)
    selected=[]
    for index, c in enumerate(['LL','LH','HL','HH']):
        pool=sorted(v for v,g in by_video.items() if cohort(g)==c)
        n=count//4+(index<count%4)
        if n>len(pool):
            raise ValueError(f'Not enough {c} videos for balanced sampling')
        selected.extend(dict(video_id=v,group=by_video[v],group4=c) for v in rng.sample(pool,n))
    return selected

def prepare(out, args):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():
        config=read(out/'config.json')
        expected=dict(eval_out=str(args.eval_out.resolve()),qa_run=str(args.qa_run.resolve()),
                      requested_count=args.count,all_videos=args.all_videos,seed=args.seed,gpus=args.gpus,model=str(Path(args.qwen_model or QWEN_MODEL).resolve()),
                      base_port=args.base_port,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest())
        if any(config.get(k)!=v for k,v in expected.items()):
            raise ValueError('Resume configuration differs; use a new output directory')
        for row in read(out/'selection.json'):
            for f in row['frames']:
                if digest(Path(f['image_path'])) != f['image_sha256']:
                    raise ValueError('Prepared image changed before resume')
        return config
    eval_out=args.eval_out.resolve();qa_run=args.qa_run.resolve()
    qa=read(eval_out/'inputs.json')
    selected=balanced_sample(qa,None if args.all_videos else args.count,args.seed)
    source_rows={r['video_id']:r for r in read(qa_run/'qa/selection.json')['videos']}
    for folder in ['media','predictions','raw','logs','workers']:
        (out/folder).mkdir(exist_ok=True)
    def prepare_video(index_and_row):
        i,row=index_and_row
        vid=row['video_id'];source=source_rows[vid]
        if source['source_kind']!='directory':
            raise ValueError('Framing filter requires extracted original RGB directories')
        geometry=read(Path(source['result_path']))
        filenames=geometry['selected_frame_names']
        indices=geometry['selected_frame_indices']
        if len(filenames)!=9 or len(indices)!=9 or len(set(indices))!=9:
            raise ValueError('Nine distinct sampled source frames required')
        media=out/'media'/vid;media.mkdir(exist_ok=True)
        row['frames']=[]
        for j,(name,frame_index) in enumerate(zip(filenames,indices)):
            original=Path(source['source_path'])/'rgb'/name
            with Image.open(original) as opened:
                im=ImageOps.exif_transpose(opened).convert('RGB')
                original_size=list(im.size)
                im.thumbnail((896,896),Image.Resampling.LANCZOS)
                target=media/f'F{j}.jpg';im.save(target,quality=92)
            row['frames'].append(dict(frame_index=j,released_frame_index=frame_index,source_path=str(original),
                                     original_size=original_size,image_size=list(im.size),
                                     image_path=str(target),image_sha256=digest(target)))
        row['source']=vid.split('__',1)[0]
        row['qa_ids']=[q['qa_id'] for q in qa if q['video_id']==vid]
        row['qa_count']=len(row['qa_ids'])
        row['worker_gpu']=args.gpus[i%len(args.gpus)]
        return row
    save(out/'run_status.json',dict(status='running',phase='preparing_frames',planned=len(selected),
                                  prepared=0,finished=0,updated_at=stamp(),gpus=args.gpus))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for prepared,_ in enumerate(pool.map(prepare_video,enumerate(selected)),1):
            if prepared%20==0 or prepared==len(selected):
                save(out/'run_status.json',dict(status='running',phase='preparing_frames',planned=len(selected),
                                              prepared=prepared,finished=0,updated_at=stamp(),gpus=args.gpus))
    model=Path(args.qwen_model or QWEN_MODEL).resolve()
    model_config=read(model/'config.json')
    index=read(model/'model.safetensors.index.json')
    weights=sorted(set(index['weight_map'].values()))
    if any(not (model/w).is_file() for w in weights):
        raise ValueError('Model weights incomplete')
    config=dict(created_at=stamp(),eval_out=str(eval_out),qa_run=str(qa_run),count=len(selected),requested_count=args.count,all_videos=args.all_videos,seed=args.seed,
                gpus=args.gpus,base_port=args.base_port,model=str(model),served_model=MODEL_NAME,
                prompt_version=PROMPT_VERSION,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
                sampling=('All unique videos represented in the evaluation QA; no group balancing or subsampling' if args.all_videos else 'Uniform random unique videos within each of four balanced cohorts'),
                frame_policy='Nine original unmarked RGB frames at the existing QA sampling indices; aspect-preserving resize, no spatial crop',
                policy=dict(unit='one holistic video-level judgment from all nine frames jointly',
                            per_frame_predictions=False,frame_count_voting=False,decision_only=True,
                            allowed_decisions=['keep','exclude','review'],postprocessing='strict format validation only; no confidence or consistency overrides'),
                inference=dict(temperature=0,seed=args.seed,max_tokens=32,enable_thinking=False,
                               response_format='json_object',max_model_len=16384,
                               gpu_memory_utilization=.40,kv_cache_bytes=3*1024**3),
                model_provenance=dict(architectures=model_config.get('architectures'),
                                      model_type=model_config.get('model_type'),config_sha256=digest(model/'config.json'),
                                      safetensors_index_sha256=digest(model/'model.safetensors.index.json'),
                                      weight_files=[dict(name=w,bytes=(model/w).stat().st_size) for w in weights]),
                inputs_sha256=digest(eval_out/'inputs.json'),manual_review=False,existing_qa_modified=False)
    save(out/'selection.json',selected);save(out/'config.json',config)
    (out/'prompt.txt').write_text(PROMPT)
    return config

def api_request(port, path, payload=None, timeout=600):
    req=urllib.request.Request(f'http://127.0.0.1:{port}/v1/{path}',
                               data=json.dumps(payload).encode() if payload is not None else None,
                               headers={'Content-Type':'application/json'})
    try:
        with urllib.request.urlopen(req,timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'HTTP {exc.code}: {exc.read().decode()[:3000]}') from exc

def serve(out, config, gpu):
    port=config['base_port']+gpu
    # Refuse to attach to or overwrite another process' service.
    import socket
    with socket.socket() as sock:
        # Permit rebinding TIME_WAIT sockets after our previous server exited;
        # an actively listening server still makes bind fail.
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))
    command=[str(ROOT/'.venv_vllm/bin/vllm'),'serve',config['model'],
             '--served-model-name',MODEL_NAME,'--host','127.0.0.1','--port',str(port),
             '--tensor-parallel-size','1','--dtype','bfloat16',
             '--gpu-memory-utilization',str(config['inference']['gpu_memory_utilization']),
             '--kv-cache-memory-bytes',str(config['inference']['kv_cache_bytes']),
             '--enforce-eager','--max-model-len','16384','--max-num-seqs','2',
             '--max-num-batched-tokens','4096','--limit-mm-per-prompt','{"image":9,"video":0}']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='2',LD_PRELOAD='/opt/conda/lib/libstdc++.so.6',VLLM_NO_USAGE_STATS='1')
    with (out/'logs'/f'qwen_gpu{gpu}.log').open('a') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                                 stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(out/'workers'/f'gpu{gpu}_server.json',dict(pid=process.pid,gpu=gpu,port=port,command=command,started_at=stamp()))
    return process

def worker(out, config, gpu, server):
    port=config['base_port']+gpu
    selected=[r for r in read(out/'selection.json') if r['worker_gpu']==gpu]
    state=out/'workers'/f'gpu{gpu}.json'
    save(state,dict(gpu=gpu,phase='loading_model',planned=len(selected),updated_at=stamp()))
    deadline=time.monotonic()+1500
    while True:
        if server.poll() is not None:
            raise RuntimeError(f'GPU {gpu} Qwen server exited with code {server.returncode}')
        try:
            info=api_request(port,'models',timeout=1)
            if any(m['id']==MODEL_NAME for m in info['data']):
                break
        except (OSError,ValueError,KeyError,RuntimeError):
            pass
        if time.monotonic()>deadline:
            raise RuntimeError(f'GPU {gpu} model load timeout')
        time.sleep(3)
    save(out/'workers'/f'gpu{gpu}_ready.json',dict(gpu=gpu,ready_at=stamp(),models=info))
    for row in selected:
        vid=row['video_id'];dest=out/'predictions'/f'{vid}.json'
        if dest.exists() and read(dest).get('status')=='ok':
            continue
        save(state,dict(gpu=gpu,phase='inference',video_id=vid,updated_at=stamp()))
        start=time.monotonic();error=None
        content=[dict(type='text',text=PROMPT)]
        for f in row['frames']:
            content.append(dict(type='text',text=f'Frame F{f["frame_index"]}:'))
            data=base64.b64encode(Path(f['image_path']).read_bytes()).decode()
            content.append(dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+data)))
        payload=dict(model=MODEL_NAME,messages=[dict(role='user',content=content)],temperature=0,
                     max_tokens=config['inference']['max_tokens'],seed=config['seed'],
                     chat_template_kwargs=dict(enable_thinking=False),response_format=dict(type='json_object'))
        for attempt in range(2):
            try:
                raw=api_request(port,'chat/completions',payload)
                save(out/'raw'/f'{vid}__attempt{attempt}.json',raw)
                if raw['choices'][0]['finish_reason']!='stop':
                    raise ValueError(f'Incomplete response: {raw["choices"][0]["finish_reason"]}')
                observation=json.loads(raw['choices'][0]['message']['content'])
                decision=decide(observation)
                result=dict(video_id=vid,status='ok',gpu=gpu,group=row['group'],group4=row['group4'],
                            source=row['source'],qa_count=row['qa_count'],
                            **decision,usage=raw.get('usage'),attempts=attempt+1,
                            elapsed_seconds=time.monotonic()-start,prompt_sha256=config['prompt_sha256'],finished_at=stamp())
                save(dest,result);error=None;break
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
                save(out/'raw'/f'{vid}__attempt{attempt}_error.json',dict(error=error,at=stamp()))
                time.sleep(1)
        if error is not None:
            save(dest,dict(video_id=vid,status='error',gpu=gpu,group=row['group'],group4=row['group4'],
                           source=row['source'],qa_count=row['qa_count'],error=error,
                           elapsed_seconds=time.monotonic()-start,finished_at=stamp()))
    save(state,dict(gpu=gpu,phase='completed',updated_at=stamp()))

def aggregate(out, config, started, final=False, worker_errors=None):
    selected=read(out/'selection.json');results=[]
    for r in selected:
        path=out/'predictions'/f'{r["video_id"]}.json'
        if path.exists():
            results.append(read(path))
    good=[r for r in results if r['status']=='ok']
    failures=[r for r in results if r['status']!='ok']
    counts=collections.Counter(r['decision'] for r in good)
    elapsed=time.monotonic()-started
    worker_status=[read(p) for p in sorted((out/'workers').glob('gpu[0-9].json'))]
    phase='inference' if any(r.get('phase')=='inference' for r in worker_status) else 'loading_models'
    status=('completed' if len(good)==len(selected) and not worker_errors else 'failed') if final else 'running'
    eta=None
    rates=[]
    for gpu in config['gpus']:
        done=[r for r in good if r['gpu']==gpu]
        remaining=sum(r['worker_gpu']==gpu for r in selected)-sum(r.get('gpu')==gpu for r in results)
        if done:
            rates.append(remaining*sum(r['elapsed_seconds'] for r in done)/len(done))
        elif remaining:
            rates=[];break
    if rates:
        eta=max(rates)
    summary=dict(status=status,phase='completed' if final and status=='completed' else phase,
                 planned=len(selected),finished=len(results),valid=len(good),errors=len(failures),
                 decisions={k:counts[k] for k in ['keep','review','exclude']},
                 selected_groups=dict(collections.Counter(r['group4'] for r in selected)),
                 elapsed_seconds=elapsed,eta_seconds=eta,updated_at=stamp(),gpus=config['gpus'],
                 workers=worker_status,worker_errors=worker_errors or {},manual_review=False,
                 existing_qa_modified=False)
    save(out/'run_status.json',summary)
    if final:
        save(out/'results.json',results)
        for label in ['keep','review','exclude']:
            save(out/f'{label}_videos.json',[r['video_id'] for r in good if r['decision']==label])
        save(out/'summary.json',summary)
        rows=[dict(video_id=r['video_id'],source=r['source'],group=r['group'],group4=r['group4'],
                   status=r['status'],decision=r.get('decision'),qa_count=r['qa_count'],
                   elapsed_seconds=r.get('elapsed_seconds')) for r in results]
        write_csv(out/'results.csv',rows)
    return summary

def run(out, args):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        config=prepare(out,args);started=time.monotonic()
        servers={};errors={}
        aggregate(out,config,started)
        try:
            selected=read(out/'selection.json')
            needed=[]
            for gpu in config['gpus']:
                if any(r['worker_gpu']==gpu and
                       (not (out/'predictions'/f'{r["video_id"]}.json').exists()
                        or read(out/'predictions'/f'{r["video_id"]}.json').get('status')!='ok') for r in selected):
                    needed.append(gpu)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(needed))) as pool:
                jobs={}
                for gpu in needed:
                    servers[gpu]=serve(out,config,gpu)
                    jobs[pool.submit(worker,out,config,gpu,servers[gpu])]=gpu
                while jobs:
                    done,_=concurrent.futures.wait(jobs,timeout=5,return_when=concurrent.futures.FIRST_COMPLETED)
                    for job in done:
                        gpu=jobs.pop(job)
                        try:
                            job.result()
                        except Exception:
                            errors[str(gpu)]=traceback.format_exc()
                            save(out/'workers'/f'gpu{gpu}.json',dict(gpu=gpu,phase='failed',error=errors[str(gpu)],updated_at=stamp()))
                        # Free each completed replica immediately, including its KV cache.
                        stop_owned_server(servers[gpu])
                    aggregate(out,config,started,worker_errors=errors)
        finally:
            for process in servers.values():
                stop_owned_server(process)
            summary=aggregate(out,config,started,final=True,worker_errors=errors)
            if summary['status']=='completed':
                report(out)
        if summary['status']!='completed':
            raise RuntimeError('Framing audit incomplete; see run_status.json and per-GPU logs')

def stop_owned_server(process):
    # Signal only the process group started by this controller, never unrelated jobs.
    try:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=10)
    except ProcessLookupError:
        pass

def report(out):
    """Export aggregate decision counts without HTML galleries."""
    config=read(out/'config.json');selection=read(out/'selection.json');results=read(out/'results.json')
    mapping={r['video_id']:r for r in selection}
    good=[r for r in results if r['status']=='ok']
    counts=[]
    for dimension in ['group4','source']:
        for value in sorted({r[dimension] for r in good}):
            rows=[r for r in good if r[dimension]==value]
            counts.append(dict(dimension=dimension,value=value,videos=len(rows),
                               **{d:sum(r['decision']==d for r in rows) for d in ['keep','review','exclude']},
                               qa=sum(r['qa_count'] for r in rows),
                               qa_on_excluded_videos=sum(r['qa_count'] for r in rows if r['decision']=='exclude')))
    save(out/'counts.json',counts);write_csv(out/'counts.csv',counts)
