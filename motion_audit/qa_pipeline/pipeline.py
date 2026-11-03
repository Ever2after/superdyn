"""Resumable geometry -> multi-object annotation -> QA orchestration."""

import collections
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .config import ROOT, DATASETS, EXCLUDED, TH
from .inventory import discover, manifest, prepare_generation, cache_source
from .storage import read, save


def stamp():
    return datetime.now(timezone.utc).isoformat()


def screen(rows):
    """Reject only missing RGB or insufficient frames, with an explicit ledger."""
    accepted, rejected = [], []
    for row in rows:
        if row['stratum'] not in DATASETS:
            rejected.append(dict(row, reason='excluded_source'))
            continue
        rgb = Path(row['source_path']) / 'rgb'
        if not rgb.is_dir():
            rejected.append(dict(row, reason='missing_rgb_directory'))
            continue
        count = sum(e.is_file() and Path(e.name).suffix.lower() in {'.png', '.jpg', '.jpeg'}
                    for e in os.scandir(rgb))
        row = dict(row, released_rgb_frames=count)
        if count < 9:
            rejected.append(dict(row, reason='fewer_than_9_distinct_measurement_frames'))
        else:
            accepted.append(row)
    ids = [r['video_id'] for r in accepted]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate video IDs')
    return accepted, rejected


def cached_ids(roots):
    """Fast cache inventory; numerical group metrics are computed at QA setup."""
    found = set()
    for root in roots:
        for path in (Path(root) / 'results').glob('*.json'):
            row = read(path)
            vid = row.get('video_id')
            if (row.get('stratum') in DATASETS and row.get('status') == 'ok'
                    and (Path(root) / 'cache' / str(vid) / 'predictions.npz').is_file()
                    and cache_source(row, Path(root) / 'cache' / str(vid) / 'predictions.npz') is not None):
                found.add(vid)
    return found


def reuse_cases(out, selection, runs):
    """Reuse only completed full-GMOS cases with identical geometry inputs/gates."""
    ledger = out / 'reused_cases.json'
    reused = read(ledger) if ledger.exists() else []
    by_id = {r['video_id']: r for r in selection['videos']}
    for run in runs:
        run = Path(run).resolve()
        config_path = run / 'config.json'
        if not config_path.exists() or not (run / 'selection.json').exists():
            continue
        config = read(config_path)
        if config.get('qa_thresholds') != TH:
            continue
        source = Path(config['source'])
        for old in read(run / 'selection.json')['videos']:
            vid = old['video_id']
            if vid not in by_id:
                continue
            row = by_id[vid]
            folder = out / 'cases' / vid
            if (folder / 'result.json').exists():
                continue
            prior = run / 'cases' / vid
            required = ['result.json', 'selected_record.json', 'qa_numeric.json',
                        'qa.json', 'qa_generated.json', 'mask_complete.json']
            if not all((prior / name).is_file() for name in required):
                continue
            result = read(prior / 'result.json')
            if (not result.get('preprocess_complete') or not result.get('semantic_complete')
                    or result.get('error') or result.get('semantic_error')):
                continue
            if 'full GMOS' not in read(prior / 'mask_complete.json').get('backend', ''):
                continue
            old_cache = Path(old.get('vggt_cache', source / 'cache' / vid / 'predictions.npz'))
            if old_cache.resolve() != Path(row['vggt_cache']).resolve():
                continue
            old_input = read(source / 'results' / (vid + '.json'))
            new_input = read(Path(row['result_path']))
            keys = ['intervals', 'selected_frame_indices', 'selected_frame_names', 'source_path']
            if any(old_input.get(k) != new_input.get(k) for k in keys):
                continue
            if any(q.get('thresholds') != TH for q in read(prior / 'qa_generated.json')):
                continue

            def copy_asset(src, dst):
                if Path(src).suffix == '.npz':
                    try:
                        return os.link(src, dst)
                    except OSError:
                        pass
                return shutil.copy2(src, dst)

            staging = folder.with_name(folder.name + '.reuse_tmp')
            if staging.exists():
                shutil.rmtree(staging)
            shutil.copytree(prior, staging, copy_function=copy_asset)
            staging.rename(folder)
            for name in ['qa.json', 'qa_generated.json', 'qa_numeric.json']:
                qs = read(folder / name)
                for q in qs:
                    q.update(group=row['group'], split=row['qa_split'], manual_review=False)
                save(folder / name, qs)
            result.update(gpu=row['worker_gpu'], group=row['group'],
                          reused_completed_case=True, reused_from=str(prior))
            save(folder / 'result.json', result)
            reused.append(dict(video_id=vid, source=str(prior), qa=result.get('qa_count', 0)))
    save(out / 'reused_cases.json', reused)
    return reused


def run(out, args):
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    started = time.time()
    status = dict(status='running', stage='inventory', started_at=stamp(),
                  gpus=args.gpus, excluded_sources=sorted(EXCLUDED), manual_review=False)
    process = None

    def update(**values):
        status.update(values, updated_at=stamp(), elapsed_seconds=time.time() - started)
        save(out / 'run_status.json', status)

    def stage(command, stage_out, extra):
        nonlocal process
        update(stage=command)
        with (out / (command + '.log')).open('a') as log:
            process = subprocess.Popen(
                [str(ROOT / '.venv/bin/python'), '-u', str(ROOT / 'qa.py'), command,
                 '--out', str(stage_out), *extra], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        update(stage_controller_pid=process.pid)
        while True:
            path = stage_out / 'run_status.json'
            detail = read(path) if path.exists() else {}
            update(stage_progress=detail)
            code = process.poll()
            if code is not None:
                process = None
                return code
            time.sleep(5)

    try:
        update()
        config_path = out / 'pipeline_config.json'
        if config_path.exists():
            config = read(config_path)
            rows = read(out / 'inventory.json')
            accepted = read(out / 'eligible_videos.json')
            rejected = read(out / 'unusable_videos.json')
        else:
            rows = manifest(args.manifest) if args.manifest else discover(args.data_root)
            accepted, rejected = screen(rows)
            if not accepted:
                raise ValueError('No videos with 9 released RGB frames')
            roots = args.cache_root or [p for p in [ROOT / 'outputs/temporal_geometry_fullscope/geometry',
                                                   *sorted((ROOT / 'outputs').glob('dynamicverse*'))]
                                       if (p / 'cache').is_dir() and (p / 'results').is_dir()]
            config = dict(gpus=args.gpus, cache_roots=[str(Path(p).resolve()) for p in roots],
                          reuse_runs=[str(p.resolve()) for p in args.reuse_run],
                          base_port=args.base_port, qwen_model=args.qwen_model,
                          quality_filter=False, qa_thresholds=TH)
            save(out / 'inventory.json', rows)
            save(out / 'eligible_videos.json', accepted)
            save(out / 'unusable_videos.json', rejected)
            save(config_path, config)
        roots = [Path(p) for p in config['cache_roots']]
        existing = cached_ids(roots)
        missing = [r for r in accepted if r['video_id'] not in existing]
        update(inventoried=len(rows), valid_videos=len(accepted), unusable_videos=len(rejected),
               reused_geometry=len(accepted) - len(missing), new_geometry=len(missing),
               source_counts=dict(collections.Counter(r['stratum'] for r in accepted)))
        save(out / 'missing_geometry.json', missing)
        gpu_arg = ','.join(str(g) for g in config['gpus'])
        geometry = out / 'geometry'
        if missing:
            code = stage('preprocess', geometry,
                         ['--manifest', str(out / 'missing_geometry.json'), '--gpus', gpu_arg])
            # Per-video errors are recorded and do not prevent successful videos reaching QA.
            if code and not (geometry / 'run_status.json').exists():
                raise RuntimeError('Geometry controller failed before recording progress')
            roots.append(geometry)
        available = cached_ids(roots)
        ready = [r for r in accepted if r['video_id'] in available]
        geometry_failures = [dict(r, result_path=str(geometry / 'results' / (r['video_id'] + '.json')))
                             for r in accepted if r['video_id'] not in available]
        save(out / 'geometry_failures.json', geometry_failures)
        save(out / 'qa_manifest.json', ready)
        update(stage='prepare_qa', geometry_successful=len(ready), geometry_failures=len(geometry_failures))
        if not ready:
            raise RuntimeError('No successful geometry caches')
        qa = out / 'qa'
        selection = prepare_generation(qa, manifest_path=out / 'qa_manifest.json', cache_roots=roots,
                                       gpus=config['gpus'], base_port=config['base_port'],
                                       qwen_model=config['qwen_model'])
        reused = reuse_cases(qa, selection, config['reuse_runs'])
        update(reused_completed_qa_cases=len(reused), reused_qa=sum(r['qa'] for r in reused))
        code = stage('generate', qa, ['--gpus', gpu_arg])
        final = read(qa / 'run_status.json')
        from .cli import summarize
        summarize(qa)
        complete = code == 0 and final['status'] == 'completed' and not geometry_failures
        update(status='completed' if complete else 'completed_with_errors', stage='finished',
               qa_count=final.get('qa_count', 0), finished_qa_videos=final.get('finished', 0),
               qa_failures=final.get('failures', 0), completed_at=stamp())
    except BaseException as exc:
        update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
        lock.close()
