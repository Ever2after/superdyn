"""One entry point for inventory, extraction, geometry caches and six-task QA."""

import argparse, collections, json, os, subprocess, sys
from pathlib import Path
from .config import ROOT, DATA_ROOT, DATASETS, TASKS, GROUP4, QA_RUN, EVAL_RUN, FINAL_QA, FINAL_EVAL
from .storage import read, save, write_csv


def gpu_list(value):
    try:
        values = [int(x) for x in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("Use comma-separated GPU indices")
    if not values or any(x < 0 for x in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU indices must be distinct and nonnegative")
    return values


def background(args):
    args = [x for x in args if x != "--background"]
    path_flags = {"--out", "--manifest", "--cache-root", "--data-root", "--reuse-run", "--eval-out", "--qa-run", "--prior-filter"}
    for i, arg in enumerate(args):
        if arg in path_flags and i + 1 < len(args):
            args[i + 1] = str(Path(args[i + 1]).resolve())
        elif "=" in arg and arg.split("=", 1)[0] in path_flags:
            flag, value = arg.split("=", 1)
            args[i] = flag + "=" + str(Path(value).resolve())
    for i, arg in enumerate(args):
        if arg == "--qwen-model" and i + 1 < len(args) and Path(args[i + 1]).exists():
            args[i + 1] = str(Path(args[i + 1]).resolve())
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--out", type=Path, required=True)
    known, _ = p.parse_known_args(args)
    out = known.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "controller.pid").exists():
        pid = int((out / "controller.pid").read_text())
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists() and str(ROOT / "qa.py").encode() in cmd.read_bytes():
            raise RuntimeError(f"Controller {pid} is already running")
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="2",
        TORCH_HOME=str(ROOT / "checkpoints/torch"),
    )
    with (out / "controller.log").open("a") as log:
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "qa.py"), *args],
            cwd=ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (out / "controller.pid").write_text(str(process.pid))
    print(json.dumps(dict(status="launched", pid=process.pid, out=str(out))))


def summarize(out):
    rows = read(out / "qa.json")
    from .finalize import counts

    summary = counts(rows)
    write_csv(out / "qa_counts.csv", summary)
    save(out / "qa_counts.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "evaluate":
        from .evaluation.runner import main as evaluate

        return evaluate(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser(
        "inventory", help="List released scenes within the current dataset scope"
    )
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser(
        "extract", help="Unpack ZIP archives with CRC verification and safe resume"
    )
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=["MVS-Synth", "SAV", "VOST"]
    )
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--background", action="store_true")
    p = sub.add_parser(
        "preprocess", help="Explicitly generate VGGT/audit caches for a manifest"
    )
    p.add_argument("--manifest", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=gpu_list, default=[0])
    p.add_argument("--background", action="store_true")
    p.add_argument("--worker-gpu", type=int, help=argparse.SUPPRESS)
    p = sub.add_parser("run", help="Run missing geometry caches, then GMOS annotation and QA")
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--cache-root", type=Path, action="append")
    p.add_argument("--reuse-run", type=Path, action="append", default=[])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=gpu_list, default=list(range(8)))
    p.add_argument("--base-port", type=int, default=19620)
    p.add_argument("--qwen-model")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser(
        "generate", help="GMOS + RAFT + six geometry tasks + semantic-only Qwen"
    )
    p.add_argument("--manifest", type=Path)
    p.add_argument("--cache-root", type=Path, action="append")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpus", type=gpu_list, default=[0])
    p.add_argument("--base-port", type=int, default=18620)
    p.add_argument("--qwen-model")
    p.add_argument("--quality-filter", action="store_true")
    p.add_argument("--skip-missing-cache", action="store_true")
    p.add_argument("--background", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument(
        "--worker-stage", choices=["preprocess", "semantics"], help=argparse.SUPPRESS
    )
    p.add_argument("--gpu", type=int, help=argparse.SUPPRESS)
    p = sub.add_parser("filter-framing", help="Audit overly close / fragmented framing using Qwen; preserve original QA")
    p.add_argument("--eval-out", type=Path, required=True)
    p.add_argument("--qa-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--all", dest="all_videos", action="store_true", help="Process every unique video represented in the evaluation QA")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--gpus", type=gpu_list, default=[4,5,6,7])
    p.add_argument("--base-port", type=int, default=20640)
    p.add_argument("--qwen-model")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser("filter-qa-visual", help="QA-conditioned target visibility and question grounding audit")
    p.add_argument("--eval-out", type=Path, required=True)
    p.add_argument("--prior-filter", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--all", dest="all_qa", action="store_true", help="Audit every evaluation QA, including multiple QA from the same video")
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--gpus", type=gpu_list, default=[4,5,6,7])
    p.add_argument("--base-port", type=int, default=21640)
    p.add_argument("--qwen-model")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser("finalize", help="Freeze keep QA in the high-camera/high-dynamics groups and rescore baselines")
    p.add_argument("--qa-source", type=Path, default=QA_RUN)
    p.add_argument("--evaluation", type=Path, default=EVAL_RUN)
    p.add_argument("--filter-results", type=Path)
    p.add_argument("--out", type=Path, default=FINAL_QA)
    p.add_argument("--eval-out", type=Path, default=FINAL_EVAL)
    p.add_argument("--groups", nargs="+", default=["HNH", "NHH", "HHH"])
    p.add_argument("--expected-qa", type=int, default=896)
    sub.add_parser("evaluate", help="Run pinned model adapters; see evaluate --help")
    p = sub.add_parser("report", help="Rescore existing predictions and render PNG/PDF, without inference")
    p.add_argument("--evaluation", type=Path, default=EVAL_RUN)
    p.add_argument("--selection", type=Path)
    p.add_argument("--filter-results", type=Path)
    p.add_argument("--groups", nargs="+")
    p.add_argument("--out", type=Path, default=FINAL_EVAL)
    p = sub.add_parser("cleanup", help="Remove inactive runtime logs, PID/lock/temp files and Python bytecode")
    p.add_argument("--out", type=Path, default=ROOT)
    p.add_argument("--apply", action="store_true", help="Delete planned runtime files; default is dry run")
    for name in ["status", "summarize"]:
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=FINAL_EVAL if name == "status" else FINAL_QA)
    args = parser.parse_args(argv)
    if getattr(args, "background", False):
        background(argv)
        return
    out = args.out.resolve()
    if args.command == "inventory":
        from .inventory import discover

        rows = discover(args.data_root, args.datasets)
        save(out, rows)
        print(
            json.dumps(
                dict(
                    videos=len(rows),
                    rgb_ready=sum(r["rgb_directory_ready"] for r in rows),
                    source_counts=dict(collections.Counter(r["stratum"] for r in rows)),
                    manifest=str(out),
                )
            )
        )
    elif args.command == "extract":
        if args.jobs < 1:
            parser.error("--jobs must be positive")
        from .extraction import run

        run(args.data_root, out, args.datasets, args.jobs)
    elif args.command == "preprocess":
        if args.worker_gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.worker_gpu)
            from .preprocess import worker

            worker(out, args.worker_gpu)
        else:
            if not args.manifest and not (out / "selection.json").exists():
                parser.error("preprocess requires --manifest for a new run")
            from .inventory import manifest
            from .preprocess import prepare, run

            if not (out / "selection.json").exists():
                prepare(out, manifest(args.manifest), args.gpus)
            run(out)
    elif args.command == "run":
        from .pipeline import run

        run(out, args)
    elif args.command == "generate":
        if args.worker_stage:
            if args.gpu is None:
                parser.error("--worker-stage requires --gpu")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
            from . import generation

            generation.configure(out)
            (
                generation.worker_preprocess
                if args.worker_stage == "preprocess"
                else generation.worker_semantics
            )(out, args.gpu)
        else:
            from .inventory import prepare_generation

            selection = prepare_generation(
                out,
                manifest_path=args.manifest,
                cache_roots=args.cache_root,
                gpus=args.gpus,
                base_port=args.base_port,
                qwen_model=args.qwen_model,
                quality_filter=args.quality_filter,
                skip_missing=args.skip_missing_cache,
            )
            if args.prepare_only:
                print(
                    json.dumps(
                        dict(
                            planned=selection["planned"],
                            groups=selection["selected_groups"],
                        )
                    )
                )
                return
            from .generation import run

            run(out)
            if read(out / "run_status.json")["status"] != "completed":
                raise RuntimeError("QA generation incomplete; see run_status.json")
            summarize(out)
    elif args.command == "filter-framing":
        from .framing_filter import run

        run(out, args)
    elif args.command == "filter-qa-visual":
        from .qa_visual_filter import run

        run(out, args)
    elif args.command == "finalize":
        from .finalize import run

        result = run(args.qa_source, args.evaluation, args.filter_results, out,
                     args.eval_out, args.groups, args.expected_qa)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "report":
        if args.filter_results:
            if args.selection or args.groups:
                parser.error("--filter-results uses four-cohort reporting; do not combine with --selection/--groups")
            from .evaluation.filtered_report import run

            run(args.evaluation, args.filter_results, out)
        else:
            from .evaluation.reports import subset_report
            from .finalize import digest

            selection = read(args.selection or FINAL_QA / "selection.json") if args.selection or not args.groups else None
            frozen = read(out / "manifest.json") if (out / "manifest.json").exists() else None
            export_validation = read(out / "validation.json") if frozen and (out / "validation.json").exists() else None
            if frozen:
                ids = set(selection["qa_ids"]) if selection else None
                if ids != set(frozen["selection"]["qa_ids"]):
                    raise ValueError("Use a separate output for a different selection")
                if any(digest(Path(p)) != sha for p, sha in frozen["source_sha256"].items()):
                    raise ValueError("Frozen benchmark sources changed")
            requested_groups = args.groups or selection["groups"]
            if frozen:
                requested_groups = [g.replace("L", "N") for g in requested_groups]
            result = subset_report(requested_groups, out,
                                   source=args.evaluation,
                                   qa_ids=set(selection["qa_ids"]) if selection else None,
                                   title="Temporal Geometry: Final Benchmark" if frozen else None,
                                   footer="Qwen QA audit = keep; camera translation OR rotation high, and scene dynamics high." if frozen else "")
            if frozen:
                result.update(final=True, qa_directory=str(FINAL_QA), selection_rule="QA audit decision=keep AND group in HLH/LHH/HHH (N=L)")
                save(out / "summary.json", result)
                save(out / "run_status.json", result)
                save(out / "selection.json", selection)
                if export_validation:
                    export_validation["last_report"] = read(out / "validation.json")
                    save(out / "validation.json", export_validation)
            print(json.dumps(dict(out=str(out), qa=result["qa"], videos=result["videos"])))
    elif args.command == "cleanup":
        from .maintenance import cleanup

        result = cleanup(out, apply=args.apply)
        if args.apply:
            result = {k: v for k, v in result.items() if k != "removed"}
            result["manifest"] = str(out / "maintenance.json")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "status":
        print(json.dumps(read(out / "run_status.json"), ensure_ascii=False, indent=2))
    elif args.command == "summarize":
        summarize(out)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        return _main(args)
    except Exception as exc:
        import traceback

        if (
            args
            and args[0] in {"run", "generate", "preprocess", "extract", "filter-framing", "filter-qa-visual"}
            and not any(x in args for x in ["--worker-stage", "--worker-gpu"])
        ):
            for i, token in enumerate(args):
                if token == "--out" and i + 1 < len(args):
                    out = Path(args[i + 1]).resolve()
                    path = out / "run_status.json"
                    status = read(path) if path.exists() else {}
                    status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    save(path, status)
                    break
        raise
