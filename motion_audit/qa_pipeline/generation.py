"""Resumable per-GPU QA workers and semantic model lifecycle."""

from datetime import datetime, timezone, timedelta
import collections, csv, fcntl, hashlib, json, os, random, signal, subprocess, sys, time, traceback
import urllib.request
from pathlib import Path
import numpy as np
from . import objects as g, tracking as t
from .config import ROOT, SEED, TH, TASKS, GROUPS, QWEN_MODEL, group
from .storage import read, save
from .flow import DenseFlowRunner, reuse_flow

GPUS = list(range(8))
BASE_PORT = 18620


def stamp():
    return datetime.now(timezone.utc).isoformat()


def child_command(out, gpu, stage):
    return [
        str(ROOT / ".venv/bin/python"),
        "-u",
        str(ROOT / "qa.py"),
        "generate",
        "--out",
        str(out),
        "--worker-stage",
        stage,
        "--gpu",
        str(gpu),
    ]


def configure(out):
    global GPUS, BASE_PORT, QWEN_MODEL
    config = read(out / "config.json")
    GPUS = config["gpus"]
    BASE_PORT = config["base_port"]
    QWEN_MODEL = config["qwen"]["model"]
    t.base.SOURCE = Path(config["source"])


def prepare(out):
    selection = read(out / "selection.json")
    assert len(selection["videos"]) == selection["planned"]
    assert (
        read(out / "config.json")["qa_thresholds"] == TH
    ), "Resume must preserve QA thresholds"
    configure(out)
    return selection


def worker_preprocess(out, gpu):
    import cv2
    import torch

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    work = out / "workers" / f"gpu{gpu}"
    g.configure(work)
    rows = read(work / "selection.json")["videos"]
    status = dict(status="loading_models", gpu=gpu, pid=os.getpid(), started_at=stamp())
    save(work / "worker_status.json", status)
    load_start = time.perf_counter()
    needs_geometry = any(
        not (out / "cases" / row["video_id"] / "result.json").exists()
        or not read(out / "cases" / row["video_id"] / "result.json").get("preprocess_complete")
        for row in rows
    )
    runner = g.GMOSRunner(4) if needs_geometry else None
    if needs_geometry:
        g.sync()
    load = time.perf_counter() - load_start
    flow_runner = None
    status.update(status="preprocessing", model_load_seconds=load)
    save(work / "worker_status.json", status)
    records = []
    numeric = []
    geo_rows = []
    for index, row in enumerate(rows):
        vid = row["video_id"]
        folder = out / "cases" / vid
        folder.mkdir(exist_ok=True)
        previous = (
            read(folder / "result.json") if (folder / "result.json").exists() else {}
        )
        if previous.get("preprocess_complete"):
            records.append(read(folder / "selected_record.json"))
            numeric.extend(read(folder / "qa_numeric.json"))
            continue
        status.update(active_video=vid, shard_index=index, updated_at=stamp())
        save(work / "worker_status.json", status)
        started = time.perf_counter()
        timing = {}
        error = None
        for attempt in range(2):
            try:
                save(
                    folder / "result.json",
                    dict(
                        video_id=vid,
                        group=row["group"],
                        gpu=gpu,
                        status="preprocessing",
                        started_at=stamp(),
                        manual_review=False,
                    ),
                )
                if not (folder / "mask_complete.json").exists():
                    g.mask_stage([vid], runner=runner)
                    timing.update(read(work / "mask_timings.json")["per_video"][0])
                    save(
                        folder / "mask_complete.json",
                        dict(completed_at=stamp(), backend="full GMOS MOS"),
                    )
                else:
                    timing.update(
                        read(folder / "timings.json")
                        if (folder / "timings.json").exists()
                        else {}
                    )
                save(folder / "timings.json", timing)
                frames, indices = g.context_rgb(vid)
                reused = reuse_flow(vid, folder, indices, frames.shape[1:3])
                timing["flow_reused"] = reused
                if not reused:
                    if flow_runner is None:
                        flow_runner = DenseFlowRunner()
                    timing["flow_seconds"] = flow_runner.run(frames, indices, folder)
                del frames
                g.geometry_stage([vid])
                r = read(work / "selected_records.json")[0]
                qs = read(work / "qa_before_semantics.json")
                geo = read(work / "geometry_timings.json")["per_video"][0]
                timing.update(
                    geometry_seconds=geo["geometry_seconds"],
                    qa_seconds=geo["qa_seconds"],
                )
                timing["preprocess_elapsed_seconds"] = time.perf_counter() - started
                save(folder / "timings.json", timing)
                save(folder / "selected_record.json", r)
                save(folder / "qa_numeric.json", qs)
                result = dict(
                    video_id=vid,
                    group=row["group"],
                    gpu=gpu,
                    status="geometry_complete",
                    preprocess_complete=True,
                    numeric_qa=len(qs),
                    objects_for_semantics=geo["objects_for_semantics"],
                    preprocess_seconds=timing["preprocess_elapsed_seconds"],
                    attempts=attempt + 1,
                    manual_review=False,
                    updated_at=stamp(),
                )
                save(folder / "result.json", result)
                records.append(r)
                numeric.extend(qs)
                geo_rows.append(geo)
                error = None
                print("PREPROCESS_COMPLETE", json.dumps(result), flush=True)
                break
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
                torch.cuda.empty_cache()
        if error:
            save(
                folder / "result.json",
                dict(
                    video_id=vid,
                    group=row["group"],
                    gpu=gpu,
                    status="preprocess_error",
                    error=error,
                    preprocess_complete=False,
                    qa_count=0,
                    manual_review=False,
                    updated_at=stamp(),
                ),
            )
    save(work / "selected_records.json", records)
    save(work / "qa_before_semantics.json", numeric)
    save(work / "geometry_timings.json", dict(status="completed", per_video=geo_rows))
    status.update(
        status="preprocess_finished",
        finished_at=stamp(),
        successful_videos=len(records),
        failed_videos=len(rows) - len(records),
    )
    save(work / "worker_status.json", status)
    print("SHARD_PREPROCESS_FINISHED", gpu, len(records), flush=True)


def decorate(q, row):
    q.update(
        group=row["group"],
        split=row.get("qa_split", "development"),
        manual_review=False,
    )
    assert (
        q["answer"] == q["options"][q["answer_index"]]
        and q["task"] in t.TASKS
        and q["thresholds"] == t.TH
    )
    return q


def worker_semantics(out, gpu):
    work = out / "workers" / f"gpu{gpu}"
    g.configure(work)
    rows = {r["video_id"]: r for r in read(work / "selection.json")["videos"]}
    records = read(work / "selected_records.json")
    ids = [r["video_id"] for r in records]
    ids = [
        vid
        for vid in ids
        if not read(out / "cases" / vid / "result.json").get("semantic_complete")
    ]

    def completed(row, raw, passed):
        vid = row["video_id"]
        folder = out / "cases" / vid
        prior = read(folder / "result.json")
        raw = [decorate(q, rows[vid]) for q in raw]
        passed = [decorate(q, rows[vid]) for q in passed]
        save(folder / "qa_generated.json", raw)
        save(folder / "qa.json", passed)
        save(folder / "semantic_timing.json", row)
        prior.update(
            status=(
                "semantic_error"
                if row.get("error")
                else "ok" if passed else "no_eligible_qa"
            ),
            semantic_complete=True,
            qa_count=len(passed),
            held_qa=len(raw) - len(passed),
            semantic_error=row.get("error"),
            updated_at=stamp(),
        )
        save(folder / "result.json", prior)

    if ids:
        g.semantic_stage(
            ids, BASE_PORT + gpu, continue_on_error=True, on_case_complete=completed
        )
    save(
        work / "worker_status.json",
        dict(status="completed", gpu=gpu, finished_at=stamp(), manual_review=False),
    )


def server_ready(port):
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=0.7
        ) as response:
            models = json.load(response)["data"]
        return any(m["id"] == "temporal-object-qwen" for m in models)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def spawn_worker(out, gpu, stage):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        TORCH_HOME=str(ROOT / "checkpoints/torch"),
    )
    with (out / "logs" / f"{stage}_gpu{gpu}.log").open("a") as log:
        process = subprocess.Popen(
            child_command(out, gpu, stage),
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        out / "workers" / f"gpu{gpu}" / f"{stage}_process.json",
        dict(pid=process.pid, gpu=gpu, stage=stage, started_at=stamp()),
    )
    return process


def spawn_server(out, gpu):
    port = BASE_PORT + gpu
    memory_utilization = (
        read(out / "config.json").get("qwen", {}).get("gpu_memory_utilization", 0.38)
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        LD_PRELOAD="/opt/conda/lib/libstdc++.so.6",
        VLLM_NO_USAGE_STATS="1",
    )
    command = [
        str(ROOT / ".venv_vllm/bin/vllm"),
        "serve",
        QWEN_MODEL,
        "--served-model-name",
        "temporal-object-qwen",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(memory_utilization),
        "--kv-cache-memory-bytes",
        str(2 * 1024**3),
        "--enforce-eager",
        "--max-model-len",
        "16384",
        "--max-num-seqs",
        "2",
        "--max-num-batched-tokens",
        "4096",
        "--limit-mm-per-prompt",
        '{"image":3,"video":0}',
    ]
    with (out / "logs" / f"qwen_gpu{gpu}.log").open("a") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    save(
        out / "workers" / f"gpu{gpu}" / "server.json",
        dict(pid=process.pid, gpu=gpu, port=port, command=command, started_at=stamp()),
    )
    return process


def aggregate(out, selection, states, started, final=False):
    planned = selection.get("planned", len(selection["videos"]))
    results = []
    raw = []
    qa = []
    records = []
    for row in selection["videos"]:
        folder = out / "cases" / row["video_id"]
        path = folder / "result.json"
        if not path.exists():
            continue
        result = read(path)
        results.append(result)
        if result.get("preprocess_complete"):
            records.append(read(folder / "selected_record.json"))
        if not result.get("semantic_complete"):
            continue
        raw.extend(read(folder / "qa_generated.json"))
        qa.extend(read(folder / "qa.json"))
    assert len({q["qa_id"] for q in qa}) == len(qa)
    save(out / "qa.json", qa)
    save(out / "qa_generated.json", raw)
    held = [q for q in raw if q["quality_status"] != "provisional_automated_pass"]
    save(out / "qa_held.json", held)
    tmp = out / "qa.jsonl.tmp"
    tmp.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qa))
    tmp.replace(out / "qa.jsonl")
    save(out / "selected_records.json", records)
    failures = [r for r in results if r.get("error") or r.get("semantic_error")]
    save(out / "failures.json", failures)
    worker_failures = [dict(gpu=gpu, error=state.get("error"))
                       for gpu, state in states.items() if state["phase"] == "failed"]
    save(out / "worker_failures.json", worker_failures)
    processed = sum(bool(r.get("preprocess_complete")) for r in results)
    finished = sum(bool(r.get("semantic_complete")) for r in results)
    observed = [
        r["preprocess_seconds"]
        for r in results
        if r.get("preprocess_complete") and not r.get("reused_completed_case")
    ]
    mean_pre = float(np.mean(observed)) if observed else 19.0
    semantic_times = [
        read(out / "cases" / r["video_id"] / "semantic_timing.json")["qwen_seconds"]
        for r in results
        if r.get("semantic_complete")
        and (out / "cases" / r["video_id"] / "semantic_timing.json").exists()
    ]
    positive = [v for v in semantic_times if v > 0]
    mean_sem = float(np.mean(positive)) if positive else 7.0
    remaining = []
    workers = []
    now = time.time()
    for gpu, state in states.items():
        shard = [r for r in selection["videos"] if r["worker_gpu"] == gpu]
        owned = [r for r in results if r["gpu"] == gpu]
        pre = sum(
            bool(r.get("preprocess_complete")) or r.get("status") == "preprocess_error"
            for r in owned
        )
        done = sum(bool(r.get("semantic_complete")) for r in owned)
        phase = state["phase"]
        since = now - state["phase_started"]
        if phase == "preprocess":
            eta = (
                max(0, (len(shard) - pre) * mean_pre - min(since, mean_pre))
                + 120
                + len(shard) * mean_sem / 2
                + 15
            )
        elif phase == "server_starting":
            eta = max(0, 120 - since) + max(0, len(shard) - done) * mean_sem / 2 + 15
        elif phase == "semantics":
            eta = max(5, (len(shard) - done) * mean_sem / 2 - since * 0.3) + 10
        else:
            eta = 0.0
        remaining.append(eta)
        workers.append(
            dict(
                gpu=gpu,
                phase=phase,
                preprocessed=pre,
                finished=done,
                planned=len(shard),
                pid=state.get("worker_pid"),
                server_pid=state.get("server_pid"),
                error=state.get("error"),
            )
        )
    eta = 0.0 if final else max(remaining, default=0.0)
    status = (
        "completed_with_errors"
        if final and (failures or worker_failures or finished < planned)
        else "completed" if final else "running"
    )
    info = dict(
        status=status,
        planned=planned,
        finished=finished,
        preprocessed=processed,
        qa_count=len(qa),
        held_qa_count=len(held),
        failures=len(failures),
        worker_failures=worker_failures,
        videos_with_qa=len({q["video_id"] for q in qa}),
        videos_without_qa=sum(
            bool(r.get("semantic_complete")) and r.get("qa_count", 0) == 0
            for r in results
        ),
        selected_groups=selection["selected_groups"],
        finished_groups=dict(
            collections.Counter(
                r["group"] for r in results if r.get("semantic_complete")
            )
        ),
        qa_groups=dict(collections.Counter(q["group"] for q in qa)),
        task_counts=dict(collections.Counter(q["task"] for q in qa)),
        elapsed_seconds=now - started,
        updated_at=stamp(),
        manual_review=False,
        workers=workers,
        eta=dict(
            remaining_seconds=round(eta),
            range_seconds=[round(eta * 0.8), round(eta * 1.6)],
            estimated_completion_utc=(
                datetime.now(timezone.utc) + timedelta(seconds=eta)
            ).isoformat(),
            measured_preprocess_videos=len(observed),
            mean_preprocess_seconds_per_video=mean_pre,
            measured_semantic_videos=len(positive),
            mean_semantic_request_seconds=mean_sem,
            method="Maximum remaining shard pipeline; observed preprocess service times, estimated 120s server load and two semantic requests per GPU; contention uncertainty included.",
        ),
    )
    save(out / "run_status.json", info)
    save(out / "eta.json", info["eta"])
    save(out / "summary.json", info)
    return info


def run(out):
    selection = prepare(out)
    lock = (out / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (out / "controller.pid").write_text(str(os.getpid()))
    states = {}
    started = time.time()
    try:
        for gpu in GPUS:
            shard = [r for r in selection["videos"] if r["worker_gpu"] == gpu]
            if all(
                (out / "cases" / r["video_id"] / "result.json").exists()
                and read(out / "cases" / r["video_id"] / "result.json").get(
                    "semantic_complete"
                )
                for r in shard
            ):
                states[gpu] = dict(phase="completed", phase_started=time.time())
                continue
            process = spawn_worker(out, gpu, "preprocess")
            states[gpu] = dict(
                phase="preprocess",
                phase_started=time.time(),
                worker=process,
                worker_pid=process.pid,
            )
        while True:
            for gpu, state in states.items():
                phase = state["phase"]
                if phase == "preprocess" and state["worker"].poll() is not None:
                    code = state["worker"].returncode
                    if code:
                        state.update(
                            phase="failed", error=f"preprocess worker exited {code}"
                        )
                        continue
                    records = read(
                        out / "workers" / f"gpu{gpu}" / "selected_records.json"
                    )
                    before = read(
                        out / "workers" / f"gpu{gpu}" / "qa_before_semantics.json"
                    )
                    if any(q["object_ids"] for q in before):
                        server = spawn_server(out, gpu)
                        state.update(
                            phase="server_starting",
                            server=server,
                            server_pid=server.pid,
                            phase_started=time.time(),
                        )
                    else:
                        # Camera-only shards require no model call, but the common
                        # semantic exporter still checks API readiness; handle an
                        # empty-QA shard directly without starting a 27B replica.
                        for r in records:
                            folder = out / "cases" / r["video_id"]
                            qs = [q for q in before if q["video_id"] == r["video_id"]]
                            for q in qs:
                                q.update(
                                    quality_status="provisional_automated_pass",
                                    annotation_status="model_estimated_automated_checks_only_needs_visual_review",
                                )
                                decorate(
                                    q,
                                    next(
                                        row
                                        for row in selection["videos"]
                                        if row["video_id"] == r["video_id"]
                                    ),
                                )
                            save(folder / "qa_generated.json", qs)
                            save(folder / "qa.json", qs)
                            result = read(folder / "result.json")
                            result.update(
                                status="ok" if qs else "no_eligible_qa",
                                semantic_complete=True,
                                qa_count=len(qs),
                            )
                            save(folder / "result.json", result)
                        state.update(phase="completed")
                elif phase == "server_starting":
                    if state["server"].poll() is not None:
                        state.update(
                            phase="failed",
                            error=f'Qwen server exited {state["server"].returncode}',
                        )
                        continue
                    if server_ready(BASE_PORT + gpu):
                        process = spawn_worker(out, gpu, "semantics")
                        state.update(
                            phase="semantics",
                            phase_started=time.time(),
                            worker=process,
                            worker_pid=process.pid,
                        )
                    elif time.time() - state["phase_started"] > 600:
                        state["server"].send_signal(signal.SIGINT)
                        state.update(
                            phase="failed", error="Qwen startup timed out after 600s"
                        )
                elif phase == "semantics" and state["worker"].poll() is not None:
                    code = state["worker"].returncode
                    if state["server"].poll() is None:
                        state["server"].send_signal(signal.SIGINT)
                    state.update(
                        phase="completed" if code == 0 else "failed",
                        error=None if code == 0 else f"semantic worker exited {code}",
                    )
            final = all(
                state["phase"] in ["completed", "failed"] for state in states.values()
            )
            info = aggregate(out, selection, states, started, final)
            print(
                stamp(),
                info["status"],
                "preprocessed",
                info["preprocessed"],
                "finished",
                info["finished"],
                "QA",
                info["qa_count"],
                "ETA",
                info["eta"]["remaining_seconds"],
                flush=True,
            )
            if final:
                break
            time.sleep(5)
    except BaseException as exc:
        path = out / "run_status.json"
        info = read(path) if path.exists() else {}
        info.update(
            status="controller_failed",
            error=f"{type(exc).__name__}: {exc}",
            updated_at=stamp(),
        )
        save(path, info)
        raise
    finally:
        for state in states.values():
            for key in ["server", "worker"]:
                process = state.get(key)
                if process and process.poll() is None:
                    process.send_signal(signal.SIGINT)
        lock.close()
