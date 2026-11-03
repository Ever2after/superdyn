"""Build the Hugging Face release of the real split.

Scope: the 2,262 QAs kept by the QA-conditioned visual audit (all motion groups) and the
1,247 source videos they use. The final 896-QA HH benchmark is listed as a subset.

Layout (paths inside QA records resolve relative to the release root):
  qa/qa.jsonl, qa/qa.json         unchanged QA records
  qa/visual_audit.jsonl           audit decision per QA (all keep)
  subsets/*.txt                   QA IDs: all_keep_2262, final_hh_896
  cases/<video_id>/               F0-F8 RGB, instance masks, object descriptions, segmentation and 3D anchor evidence
  media/<id>/F0..F8.jpg           model-input frames with A/B target contours
  eval/inputs.jsonl, eval/reference_answers.jsonl, eval/protocol.json
  videos/<source>/<video_id>.zip  released RGB frames (<video_id>/rgb/*.jpg|png, stored without recompression)
  videos/index.csv                video_id, source, frame count, group, QA count, zip path

Run: .venv/bin/python release_hf.py --out /path/to/empty/folder [--workers 16]
"""
import argparse, csv, json, shutil, zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QA_RUN = ROOT / "outputs/temporal_geometry_fullscope/qa"
EVAL = ROOT / "outputs/temporal_vlm_eval_fullscope"
INVENTORY = ROOT / "outputs/temporal_geometry_fullscope/inventory.json"
CASE_FILES = ["objects.json", "instance_masks.npz", "segmentation.json", "selected_record.json"] + [f"rgb_{i}.jpg" for i in range(9)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def zip_video(source_dir, target, video_id):
    # Same frame rule as qa_pipeline/frames.py: SA-V and MVS-Synth release PNG, the others JPEG.
    frames = sorted(p for p in (source_dir / "rgb").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not frames:
        raise RuntimeError("No RGB frames for " + video_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for frame in frames:
            archive.write(frame, f"{video_id}/rgb/{frame.name}")
    partial.rename(target)
    return len(frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true", help="Rebuild metadata in a previous output; finished zips are reused")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise RuntimeError("Use an empty output folder, or --resume a previous build")
    out.mkdir(parents=True, exist_ok=True)

    qa = json.loads((QA_RUN / "qa.json").read_text())
    audit = {r["qa_id"]: r for r in json.loads((QA_RUN / "qa_visual_filter_fullscope/results.json").read_text())}
    keep = [q for q in qa if audit[q["qa_id"]]["decision"] == "keep"]
    final_ids = [q["qa_id"] for q in json.loads((QA_RUN / "final/qa.json").read_text())]
    keep_ids = {q["qa_id"] for q in keep}
    videos = sorted({q["video_id"] for q in keep})
    if (len(keep), len(videos), len(final_ids)) != (2262, 1247, 896) or not set(final_ids) <= keep_ids:
        raise RuntimeError(f"Unexpected release scope: {len(keep)} QA, {len(videos)} videos, {len(final_ids)} final")

    write_jsonl(out / "qa/qa.jsonl", keep)
    (out / "qa/qa.json").write_text(json.dumps(keep, ensure_ascii=False, indent=1))
    write_jsonl(out / "qa/visual_audit.jsonl", [{k: audit[q["qa_id"]][k] for k in ["qa_id", "video_id", "task", "group", "decision"]} for q in keep])
    (out / "subsets").mkdir(exist_ok=True)
    (out / "subsets/all_keep_2262.txt").write_text("\n".join(q["qa_id"] for q in keep) + "\n")
    (out / "subsets/final_hh_896.txt").write_text("\n".join(final_ids) + "\n")

    inputs = {r["qa_id"]: r for r in json.loads((EVAL / "inputs.json").read_text())}
    references = json.loads((EVAL / "reference_answers.json").read_text())
    rows = []
    for q in keep:
        row = dict(inputs[q["qa_id"]])
        relative = []
        for path in row["frame_paths"]:
            path = Path(path)
            target = out / "media" / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            relative.append(f"media/{path.parent.name}/{path.name}")
        row["frame_paths"] = relative
        rows.append(row)
    write_jsonl(out / "eval/inputs.jsonl", rows)
    write_jsonl(out / "eval/reference_answers.jsonl", [dict(qa_id=i, **references[i]) for i in (q["qa_id"] for q in keep)])
    shutil.copy2(EVAL / "final/protocol.json", out / "eval/protocol.json")

    for video in videos:
        case = QA_RUN / "cases" / video
        for name in CASE_FILES:
            (out / "cases" / video).mkdir(parents=True, exist_ok=True)
            shutil.copy2(case / name, out / "cases" / video / name)

    inventory = {r["video_id"]: r for r in json.loads(INVENTORY.read_text())}
    qa_count = Counter(q["video_id"] for q in keep)
    group = {q["video_id"]: q["group"] for q in keep}

    def job(video):
        record = inventory[video]
        target = out / "videos" / record["stratum"] / f"{video}.zip"
        if target.exists():
            with zipfile.ZipFile(target) as archive:
                return video, len(archive.namelist())
        return video, zip_video(Path(record["source_path"]), target, video)

    with ThreadPoolExecutor(args.workers) as pool:
        frames = dict(pool.map(job, videos))
    with (out / "videos/index.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["video_id", "source", "scene_id", "num_frames", "group", "num_qa", "zip"])
        for video in videos:
            record = inventory[video]
            writer.writerow([video, record["stratum"], record["scene_id"], frames[video], group[video], qa_count[video],
                             f"videos/{record['stratum']}/{video}.zip"])
    summary = dict(qa=len(keep), videos=len(videos), final_hh_qa=len(final_ids), frames=sum(frames.values()),
                   tasks=dict(Counter(q["task"] for q in keep)), groups=dict(Counter(q["group"] for q in keep)),
                   sources=dict(Counter(inventory[v]["stratum"] for v in videos)))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
