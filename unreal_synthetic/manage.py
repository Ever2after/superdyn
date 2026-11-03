#!/usr/bin/env python3
"""Host CLI. Run with unreal_synthetic/.venv/bin/python; Unreal uses its own Python."""

import argparse, json, subprocess, sys
from pathlib import Path
from bridge import ROOT, doctor, validate_inputs, launch, engine_binary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    q = sub.add_parser("doctor")
    q.add_argument("--engine")
    sub.add_parser("validate")
    sub.add_parser("test")
    q = sub.add_parser("install-engine")
    q.add_argument("--archive", type=Path, required=True)
    q.add_argument("--destination", type=Path, default=ROOT / ".engine")
    for action in ["build", "render"]:
        q = sub.add_parser(action)
        q.add_argument("--engine")
        q.add_argument("--case", required=True)
        q.add_argument("--output", type=Path, required=True)
        q.add_argument("--gpu", type=int, default=6)
        q.add_argument("--preview", action="store_true")
        q.add_argument("--exr", action="store_true")
        q.add_argument(
            "--samples",
            type=int,
            default=1,
            help="Spatial samples per frame (verified baseline: 1)",
        )
        q.add_argument(
            "--rebuild",
            action="store_true",
            help="Recreate this case's generated map, sequence and meshes",
        )
    q = sub.add_parser("tours", help="Final A/B walkthrough workflow")
    q.add_argument("tour_action", choices=["verify", "report", "render", "preview", "retry_preview"])
    q.add_argument("--output", type=Path, default=ROOT / "outputs/ab_walkthroughs")
    q.add_argument("--gpu", type=int, default=7)
    q.add_argument("--case")
    q = sub.add_parser("scenes", help="Seeded industrial, urban and park scene families")
    q.add_argument("scene_action", choices=["plan", "render", "verify", "report"])
    q.add_argument("--output", type=Path, default=ROOT / "outputs/main_scene_families_v3")
    q.add_argument("--family", action="append", choices=["industrial", "urban", "park"])
    q.add_argument("--seed-offset", type=int, default=0)
    q.add_argument("--gpu", type=int, default=5)
    q = sub.add_parser("terrain", help="Seeded terraced terrain: sample / validate / audit (no rendering)")
    q.add_argument("terrain_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.action == "terrain":
        from terrain.cli import main as terrain_main
        sys.exit(terrain_main(args.terrain_args))
    if args.action == "scenes":
        cmd = [sys.executable, str(ROOT / "scripts/render_scene_families.py"), args.scene_action,
               "--output", str(args.output.resolve()), "--gpu", str(args.gpu), "--seed-offset", str(args.seed_offset)]
        for family in args.family or []:
            cmd += ["--family", family]
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if args.action == "tours":
        if args.tour_action in ["verify", "report"]:
            cmd = [sys.executable, str(ROOT / "scripts/finalize_ab_walkthroughs.py"),
                   "--output", str(args.output)]
            if args.tour_action == "verify":
                cmd.append("--verify-only")
        else:
            cmd = [sys.executable, str(ROOT / "scripts/render_ab_walkthroughs.py"),
                   args.tour_action, "--output", str(args.output), "--gpu", str(args.gpu)]
            if args.case:
                cmd += ["--case", args.case]
        subprocess.run(cmd, cwd=ROOT, check=True)
        return
    if args.action == "doctor":
        print(json.dumps(doctor(args.engine), indent=2))
        return
    if args.action == "validate":
        print(json.dumps(validate_inputs(), indent=2))
        return
    if args.action == "test":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(ROOT / "tests"),
                "-v",
            ],
            cwd=ROOT,
            check=True,
        )
        return
    if args.action == "install-engine":
        from installer import install_archive

        dest = install_archive(args.archive, args.destination)
        executable = engine_binary(str(dest))
        if executable is None:
            raise RuntimeError("Archive extracted but UnrealEditor was not found")
        print(executable)
        return
    print(
        json.dumps(
            launch(
                args.case,
                args.output,
                args.engine,
                args.gpu,
                args.preview,
                args.action,
                args.exr,
                args.rebuild,
                args.samples,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
