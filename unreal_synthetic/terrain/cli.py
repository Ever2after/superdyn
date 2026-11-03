"""Terrain CLI: sample one seeded layout, validate a layout file, or audit a seed range."""
import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path

from terrain.sampling import generate, load_settings
from terrain.validation import validate

PACKAGE = Path(__file__).resolve().parent


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(seeds, settings):
    started = time.monotonic()
    records = []
    for seed in seeds:
        try:
            cfg = generate(seed, settings)
            g = cfg["generation"]
            report = validate(cfg, settings)
            records.append(dict(seed=seed, status=report["status"], regions=len(cfg["plateaus"]), levels=cfg["levels"],
                                ramps=len(cfg["ramps"]), area_m2=report["ground_area_m2"], flat_fraction=report["flat_fraction"],
                                ramp_fraction=1 - report["flat_fraction"], height_metrics=report.get("height_metrics", {}),
                                height_repair_changed_regions=g.get("height_repair_changed_regions", 0), attempts=g["attempt"],
                                layout_sha256=g["layout_sha256"], pattern=g["pattern"], rejections=g["rejections"]))
        except Exception as error:
            records.append(dict(seed=seed, status="failed", error=str(error)))
    passed = [r for r in records if r["status"] == "passed"]

    def span(key):
        return [min(r[key] for r in passed), max(r[key] for r in passed)] if passed else []

    report = dict(settings_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
                  source_sha256={name: file_sha256(PACKAGE / name) for name in ["sampling.py", "validation.py", "height_patterns.py"]},
                  status="passed" if len(passed) == len(records) else "failed", requested=len(records), passed=len(passed),
                  failed=[r for r in records if r["status"] != "passed"], unique_layouts=len({r["layout_sha256"] for r in passed}),
                  region_count_distribution=dict(Counter(r["regions"] for r in passed)),
                  level_count_distribution=dict(Counter(len(r["levels"]) for r in passed)),
                  pattern_distribution=dict(Counter(r["pattern"] for r in passed)),
                  flat_fraction_range=span("flat_fraction"), ground_area_range_m2=span("area_m2"), ramp_fraction_range=span("ramp_fraction"),
                  maximum_attempts=max((r["attempts"] for r in passed), default=0), seconds=time.monotonic() - started, results=records)
    mixed = [r["height_metrics"] for r in passed if r["height_metrics"]]
    if mixed:
        report["height_pattern_summary"] = dict(
            minimum_separated_high_zones=min(m["global_high_components"] for m in mixed),
            minimum_separated_low_zones=min(m["global_low_components"] for m in mixed),
            planar_trend_r2_range=[min(m["planar_trend_r2"] for m in mixed), max(m["planar_trend_r2"] for m in mixed)],
            scenes_repaired=sum(r["height_repair_changed_regions"] > 0 for r in passed))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="manage.py terrain", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--settings", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("validate")
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p = sub.add_parser("audit")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--settings", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "validate":
        report = validate(json.loads(args.layout.read_text()))
        if args.output:
            save_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
        return 0 if report["status"] == "passed" else 1
    settings = load_settings(args.settings)
    if args.action == "sample":
        layout = generate(args.seed, settings)
        save_json(args.output, layout)
        print(json.dumps(dict(seed=args.seed, output=str(args.output), regions=len(layout["plateaus"]), levels=layout["levels"],
                              ramps=len(layout["ramps"]), attempts=layout["generation"]["attempt"]), indent=2))
        return 0
    if args.count < 1:
        parser.error("--count must be positive")
    report = audit(range(args.seed, args.seed + args.count), settings)
    save_json(args.output, report)
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    return 0 if report["status"] == "passed" else 1
