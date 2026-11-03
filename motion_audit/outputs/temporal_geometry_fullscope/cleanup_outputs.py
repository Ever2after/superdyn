"""Authorized output cleanup, preserving QA/eval data and the live filter job."""
import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
FINAL = ROOT/'temporal_geometry_fullscope'
EVAL = ROOT/'temporal_vlm_eval_fullscope'
GEO = FINAL/'geometry'
QA = FINAL/'qa'
KEEP = {FINAL.name,EVAL.name}
ACTIVE = ROOT/'framing_filter_fullscope_decision'
DEST = QA/'framing_filter_decision'
assert ROOT == Path('/home/jovyan/data-1/jusang/superdyn/motion_audit/outputs')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(path.read_text())

def save(path,data):
    tmp=path.with_suffix(path.suffix+'.cleanup_tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');tmp.replace(path)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def scan(root):
    todo=[root];files=[];links=[]
    while todo:
        folder=todo.pop()
        with os.scandir(folder) as entries:
            for e in entries:
                if e.is_symlink():
                    links.append((Path(e.path),os.path.abspath(os.path.join(folder,os.readlink(e.path)))))
                elif e.is_dir(follow_symlinks=False):
                    todo.append(Path(e.path))
                else:
                    files.append(str(Path(e.path).relative_to(root)))
    return files,links

protected=[QA/'qa.json',QA/'qa.jsonl',QA/'qa_generated.json',QA/'qa_held.json',
           EVAL/'inputs.json',EVAL/'reference_answers.json',EVAL/'summary.json',
           EVAL/'leaderboard.csv',EVAL/'per_task.csv',EVAL/'per_group4.csv',EVAL/'results.png']
protected=[p for p in protected if p.exists()]
before_hashes={str(p):sha(p) for p in protected}
eval_files_before,eval_links_before=scan(EVAL)
assert not eval_links_before
selection=read(QA/'selection.json')
rows=selection['videos']
assert len(rows)==4457 and len(read(QA/'qa.json'))==2387
plan=dict(started_at=stamp(),retained=[str(FINAL),str(EVAL)],protected_hashes=before_hashes,
          deleted=[],moved_caches=[],moved_results=[],updated_links=[],live_filter_destination=str(DEST))
save(FINAL/'output_cleanup.json',plan)

# Bring cached prerequisites into the retained tree. Directory renames preserve
# all bytes and existing hard links, without copying gigabytes of predictions.
for i,row in enumerate(rows):
    vid=row['video_id']
    cache=GEO/'cache'/vid
    result=GEO/'results'/f'{vid}.json'
    if not cache.exists():
        origin=Path(row['vggt_cache']).parent.resolve(strict=True)
        assert origin.is_relative_to(ROOT) and not origin.is_relative_to(FINAL)
        origin.rename(cache)
        plan['moved_caches'].append(dict(video_id=vid,source=str(origin),destination=str(cache)))
    assert (cache/'predictions.npz').is_file()
    if not result.exists():
        origin=Path(row['result_path']).resolve(strict=True)
        assert origin.is_relative_to(ROOT)
        os.link(origin,result)
        plan['moved_results'].append(dict(video_id=vid,source=str(origin),destination=str(result)))
    for link,target in [(QA/'source/cache'/vid,cache),(QA/'source/results'/f'{vid}.json',result)]:
        if link.is_symlink():
            prior=os.readlink(link)
            if prior!=str(target):
                temp=link.with_name(link.name+'.cleanup_link')
                temp.symlink_to(target,target_is_directory=target.is_dir());temp.replace(link)
                plan['updated_links'].append(dict(path=str(link),previous_target=prior,target=str(target)))
    row.update(vggt_cache=str(cache/'predictions.npz'),result_path=str(result),cache_origin=str(GEO))
    if (i+1)%500==0:
        print('Cache dependency check',i+1,'/',len(rows),'relocated',len(plan['moved_caches']),flush=True)
save(QA/'selection.json',selection)
for path in [QA/'config.json',FINAL/'pipeline_config.json']:
    cfg=read(path);cfg['cache_roots']=[str(GEO)]
    if 'reuse_runs' in cfg:
        cfg['reuse_runs']=[]
    save(path,cfg)

# No retained symbolic link may still depend on a directory slated for removal.
_,links=scan(FINAL)
external=[(str(p),t) for p,t in links if not any(t==str(ROOT/k) or t.startswith(str(ROOT/k)+'/') for k in KEEP)]
assert not external,external[:5]
broken=[str(p) for p,_ in links if not p.exists()]
assert not broken,broken[:5]
print('Dependencies preserved:',len(plan['moved_caches']),'cache directories;',len(links),'valid retained links',flush=True)

# The controller keeps absolute paths in memory. Pause only our controller,
# relocate its output, then retain a temporary alias until it exits. GPU model
# servers keep their open log descriptors and continue without a reload.
if ACTIVE.exists() and not ACTIVE.is_symlink():
    assert not DEST.exists()
    pid=int((ACTIVE/'controller.pid').read_text())
    cmdfile=Path(f'/proc/{pid}/cmdline')
    paused=False
    if cmdfile.exists():
        cmd=cmdfile.read_bytes()
        assert str(ACTIVE).encode() in cmd and b'filter-framing' in cmd
        os.kill(pid,signal.SIGSTOP);paused=True
    try:
        ACTIVE.rename(DEST)
        ACTIVE.symlink_to(DEST,target_is_directory=True)
    finally:
        if paused:
            os.kill(pid,signal.SIGCONT)
    plan['live_filter_pid']=pid
    print('Live filter relocated:',DEST,flush=True)
    watcher=DEST/'finalize_location.py'
    watcher.write_text('''import json,os,time
from pathlib import Path
old=Path('''+repr(str(ACTIVE))+''')
new=Path('''+repr(str(DEST))+''')
pid='''+str(pid)+'''
marker=new/'location_finalization.json'
marker.write_text(json.dumps(dict(status='waiting_for_controller_exit',controller_pid=pid))+'\\n')
while True:
    proc=Path(f'/proc/{pid}/cmdline')
    if not proc.exists() or not proc.read_bytes():
        break
    command=proc.read_bytes()
    if b'filter-framing' not in command or str(old).encode() not in command:
        break
    time.sleep(5)
def rewrite(value):
    if isinstance(value,str):
        return str(new)+value[len(str(old)):] if value==str(old) or value.startswith(str(old)+'/') else value
    if isinstance(value,list):return [rewrite(v) for v in value]
    if isinstance(value,dict):return {k:rewrite(v) for k,v in value.items()}
    return value
for name in ['selection.json','config.json','report_manifest.json','coverage_plan.json']:
    path=new/name
    if path.exists():
        value=json.loads(path.read_text());updated=rewrite(value)
        if value!=updated:
            temp=path.with_suffix('.relocation_tmp')
            temp.write_text(json.dumps(updated,indent=2)+'\\n');temp.replace(path)
if old.is_symlink() and os.readlink(old)==str(new):
    old.unlink()
marker.write_text(json.dumps(dict(status='completed',temporary_alias_removed=not old.exists(),destination=str(new)))+'\\n')
print('Temporary output alias removed',flush=True)
''')
    with (DEST/'relocation.log').open('a') as log:
        process=subprocess.Popen([str(ROOT.parent/'.venv/bin/python'),'-u',str(watcher)],
                                 cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,
                                 stderr=subprocess.STDOUT,start_new_session=True)
    plan['relocation_watcher_pid']=process.pid

# Stop only the obsolete server for the old scene-scale report being deleted.
obsolete_pid=3577329
cmdfile=Path(f'/proc/{obsolete_pid}/cmdline')
if cmdfile.exists():
    cmd=cmdfile.read_bytes()
    if str(ROOT.parent).encode() in cmd and b'serve_scene_scale.py' in cmd:
        os.kill(obsolete_pid,signal.SIGTERM)
        plan['stopped_obsolete_report_server']=obsolete_pid

save(FINAL/'output_cleanup.json',plan)
targets=[p for p in ROOT.iterdir() if p.name not in KEEP and p!=ACTIVE]
plan['delete_plan']=[str(p) for p in targets]
save(FINAL/'output_cleanup.json',plan)
for path in sorted(targets,key=lambda p:p.name):
    assert path.parent==ROOT and path.name not in KEEP and path!=ACTIVE
    directory=path.is_dir() and not path.is_symlink()
    print('Deleting',path.name,flush=True)
    if directory:
        shutil.rmtree(path)
    else:
        path.unlink()
    plan['deleted'].append(dict(path=str(path),kind='directory' if directory else 'file_or_link'))
    save(FINAL/'output_cleanup.json',plan)

after_hashes={str(p):sha(p) for p in protected}
assert before_hashes==after_hashes,'Protected QA/evaluation bytes changed'
eval_files_after,_=scan(EVAL)
assert set(eval_files_after)==set(eval_files_before),'Full evaluation lost files'
assert len(read(QA/'qa.json'))==2387
qa_videos={q['video_id'] for q in read(QA/'qa.json')}
for vid in qa_videos:
    assert (QA/'cases'/vid/'rgb_0.jpg').is_file()
    assert (QA/'source/cache'/vid/'predictions.npz').is_file()
    assert (QA/'source/results'/f'{vid}.json').is_file()
_,links=scan(FINAL)
assert all(p.exists() for p,_ in links),'A retained link is broken after cleanup'
remaining=[p.name for p in ROOT.iterdir()]
assert set(remaining)<=KEEP|{ACTIVE.name}
plan.update(status='completed',finished_at=stamp(),remaining_output_entries=remaining,
            deleted_directories=sum(r['kind']=='directory' for r in plan['deleted']),
            deleted_files=sum(r['kind']!='directory' for r in plan['deleted']),
            protected_hashes_unchanged=True,full_eval_files_preserved=len(eval_files_after),
            final_qa=2387,final_qa_videos=len(qa_videos),valid_retained_links=len(links),
            temporary_alias_present=ACTIVE.is_symlink())
save(FINAL/'output_cleanup.json',plan)
print(json.dumps({k:v for k,v in plan.items() if k not in {'moved_caches','moved_results','updated_links','protected_hashes','delete_plan','deleted'}},indent=2),flush=True)
