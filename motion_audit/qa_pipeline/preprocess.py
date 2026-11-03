"""Explicit VGGT cache generation; never runs as a side effect of QA generation."""

import fcntl, hashlib, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path
from .config import ROOT, DATASETS, EXCLUDED
from .storage import read, save, write_csv
from .frames import frames_for
from .metrics import interval_statistics, summarize_amplitudes, summarize_intervals


def worker(out, gpu):
    config = read(out / "config.json")
    rows = read(out / "selection.json")["videos"]
    gpus = config["gpus"]
    rows = [r for i, r in enumerate(rows) if gpus[i % len(gpus)] == gpu]
    worker_status = out / "logs" / f"gpu{gpu}_status.json"
    pending = [r for r in rows if not (
        (out / "results" / f'{r["video_id"]}.json').is_file()
        and read(out / "results" / f'{r["video_id"]}.json').get("status") == "ok"
        and (out / "cache" / r["video_id"] / "predictions.npz").is_file())]
    if not pending:
        save(worker_status, dict(status="finished", gpu=gpu, pid=os.getpid(),
                                 planned=len(rows), reused=len(rows)))
        return
    import numpy as np
    import torch, cv2
    from .inference import Engine

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    save(worker_status, dict(status="loading_models", gpu=gpu, pid=os.getpid(), planned=len(rows)))
    engine = Engine()
    for index, row in enumerate(rows):
        vid = row["video_id"]
        path = out / "results" / f"{vid}.json"
        cache = out / "cache" / vid
        if (
            path.exists()
            and read(path).get("status") == "ok"
            and (cache / "predictions.npz").is_file()
        ):
            continue
        save(worker_status, dict(status="processing", gpu=gpu, pid=os.getpid(),
                                 planned=len(rows), shard_index=index, active_video=vid))
        started = time.perf_counter()
        result = dict(row)
        try:
            frames, mids, times, meta = frames_for(row)
            if (cache / "predictions.npz").is_file():
                with np.load(cache / "predictions.npz") as f:
                    pred = {k: f[k] for k in f.files}
                assert len(times) == pred["depth"].shape[0] and np.allclose(
                    pred["times"], times
                ), "Cache sampling differs"
                kwargs = {k: v for k, v in pred.items() if k != "times"}
            else:
                kwargs = engine.predict(frames, mids, cache)
                tmp = cache / "predictions.tmp.npz"
                np.savez_compressed(tmp, **kwargs, times=times)
                tmp.replace(cache / "predictions.npz")
            intervals, trajectory = interval_statistics(**kwargs, times=times)
            raw, _ = interval_statistics(
                **kwargs, times=times, pixel_quality_filter=False
            )
            result.update(
                status="ok",
                **meta,
                **trajectory,
                intervals=intervals,
                summary=summarize_intervals(intervals),
                amplitude_summary=summarize_amplitudes(intervals),
                unfiltered_amplitude_summary=summarize_amplitudes(
                    [i | {"quality_valid": True} for i in raw]
                ),
            )
        except Exception as exc:
            traceback.print_exc()
            result.update(status="error", error=f"{type(exc).__name__}: {exc}")
            torch.cuda.empty_cache()
        result.update(gpu=gpu, elapsed_seconds=time.perf_counter() - started)
        save(path, result)
        print(vid, result["status"], flush=True)
    save(worker_status, dict(status="finished", gpu=gpu, pid=os.getpid(), planned=len(rows)))


def run(out):
    lock = (out / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = read(out / "config.json")
    selection = read(out / "selection.json")
    processes = []
    started = time.time()
    try:
        for gpu in config["gpus"]:
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                OMP_NUM_THREADS="2",
                OPENBLAS_NUM_THREADS="1",
                MKL_NUM_THREADS="2",
                TORCH_HOME=str(ROOT / "checkpoints/torch"),
            )
            with (out / "logs" / f"gpu{gpu}.log").open("a") as log:
                processes.append(
                    subprocess.Popen(
                        [
                            str(ROOT / ".venv/bin/python"),
                            "-u",
                            str(ROOT / "qa.py"),
                            "preprocess",
                            "--out",
                            str(out),
                            "--worker-gpu",
                            str(gpu),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                )
        while True:
            results = [
                read(out / "results" / f'{r["video_id"]}.json')
                for r in selection["videos"]
                if (out / "results" / f'{r["video_id"]}.json').exists()
            ]
            final = all(p.poll() is not None for p in processes)
            successful = sum(r["status"] == "ok" for r in results)
            status = (
                "completed"
                if final and successful == selection["planned"]
                else "completed_with_errors" if final else "running"
            )
            observed = [r["elapsed_seconds"] for r in results if r.get("status") == "ok"]
            mean_seconds = sum(observed) / len(observed) if observed else None
            remaining_shard = max((sum(i % len(config["gpus"]) == n and not
                                      (out / "results" / f'{row["video_id"]}.json').exists()
                                      for i, row in enumerate(selection["videos"]))
                                   for n in range(len(config["gpus"]))), default=0)
            eta = remaining_shard * mean_seconds if mean_seconds is not None else None
            save(
                out / "run_status.json",
                dict(
                    status=status,
                    planned=selection["planned"],
                    finished=len(results),
                    successful=successful,
                    failures=sum(r["status"] != "ok" for r in results),
                    elapsed_seconds=time.time() - started,
                    eta=dict(remaining_seconds=0 if final else eta,
                             measured_videos=len(observed), mean_video_seconds=mean_seconds,
                             scope="geometry stage only; GMOS/QA follows",
                             method="Maximum remaining shard times observed mean service time"),
                    workers=[dict(pid=p.pid, exit_code=p.poll()) for p in processes],
                ),
            )
            if final:
                flat = [
                    {
                        k: v
                        for k, v in r.items()
                        if k
                        not in [
                            "intervals",
                            "summary",
                            "amplitude_summary",
                            "unfiltered_amplitude_summary",
                        ]
                    }
                    | r.get("amplitude_summary", {})
                    for r in results
                ]
                if flat:
                    write_csv(out / "per_video.csv", flat)
                raw = [
                    dict(
                        video_id=r["video_id"],
                        stratum=r["stratum"],
                        **r["unfiltered_amplitude_summary"],
                    )
                    for r in results
                    if r["status"] == "ok"
                ]
                if raw:
                    write_csv(out / "per_video_unfiltered.csv", raw)
                if status != "completed":
                    raise RuntimeError(
                        "Preprocessing incomplete; see run_status.json and logs/"
                    )
                return
            time.sleep(5)
    finally:
        for p in processes:
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        lock.close()


def prepare(out, rows, gpus):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "selection.json").exists():
        return
    rows = [r for r in rows if r["stratum"] in DATASETS]
    if not rows:
        raise ValueError("Empty active-scope manifest")
    if len({r["video_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate video IDs")
    for name in ["cache", "results", "logs"]:
        (out / name).mkdir(exist_ok=True)
    save(out / "selection.json", dict(planned=len(rows), videos=rows))
    save(
        out / "config.json",
        dict(
            gpus=list(gpus),
            excluded_sources=sorted(EXCLUDED),
            sampling="33 uniform context frames / 9 measurement frames; normalized released-frame index",
            code_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT / "qa_pipeline").glob("*.py")
            },
        ),
    )
