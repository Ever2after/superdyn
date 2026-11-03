"""Freeze the audited high-camera/high-dynamics benchmark and its evaluation."""

import collections
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .config import EVAL_RUN, FINAL_EVAL, FINAL_GROUPS, FINAL_QA, GROUP4, QA_RUN, ROOT, TASKS
from .storage import read, save, write_csv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_rows(rows, name):
    lookup = {row["qa_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError(f"Duplicate QA IDs in {name}")
    return lookup


def select_final(qa, audit, groups=FINAL_GROUPS):
    """Select only successful keep decisions, with exact audit coverage."""
    source = unique_rows(qa, "source QA")
    decisions = unique_rows(audit, "visual audit")
    if set(source) != set(decisions):
        raise ValueError("The visual audit must cover exactly the source QA")
    normalized = tuple(g.upper().replace("N", "L") for g in groups)
    if not normalized or len(set(normalized)) != len(normalized) or any(
        len(g) != 3 or set(g) - {"H", "L"} for g in normalized
    ):
        raise ValueError("Use distinct three-letter H/L or H/N motion groups")
    selected = []
    for key, row in source.items():
        decision = decisions[key]
        if decision.get("status") != "ok" or decision.get("decision") not in {
            "keep", "exclude", "review"
        }:
            raise ValueError(f"Incomplete visual audit: {key}")
        for field in ("video_id", "task", "group"):
            if decision[field] != row[field]:
                raise ValueError(f"Audit metadata mismatch for {key}: {field}")
        if row["group"] in normalized and decision["decision"] == "keep":
            selected.append(row)
    if not selected:
        raise ValueError("Final selection is empty")
    return sorted(selected, key=lambda row: row["qa_id"])


def counts(rows):
    result = []
    for dimension, values in [
        ("task", TASKS),
        ("group", sorted({r["group"] for r in rows})),
        ("group4", sorted({GROUP4[r["group"]] for r in rows})),
        ("source", sorted({r["video_id"].split("__")[0] for r in rows})),
    ]:
        for value in values:
            def label(row):
                if dimension == "source":
                    return row["video_id"].split("__")[0]
                if dimension == "group4":
                    return GROUP4[row["group"]]
                return row[dimension]

            selected = [r for r in rows if label(r) == value]
            result.append(dict(dimension=dimension, value=value, qa=len(selected),
                               videos=len({r["video_id"] for r in selected})))
    result.append(dict(dimension="all", value="all", qa=len(rows),
                       videos=len({r["video_id"] for r in rows})))
    return result


def link_shared(source, target):
    """Use a relative link and refuse to overwrite unrelated assets."""
    source, target = Path(source).resolve(), Path(target)
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() != source:
            raise ValueError(f"Existing link points to a different asset: {target}")
        return
    target.symlink_to(os.path.relpath(source, target.parent), target_is_directory=source.is_dir())


def run(qa_source=QA_RUN, evaluation=EVAL_RUN, filter_results=None,
        out=FINAL_QA, eval_out=FINAL_EVAL, groups=FINAL_GROUPS, expected_qa=None):
    """Export a stable subset; source QA, references and predictions are unchanged."""
    qa_source, evaluation, out, eval_out = [
        Path(p).resolve() for p in (qa_source, evaluation, out, eval_out)
    ]
    filter_results = Path(filter_results or qa_source / "qa_visual_filter_fullscope/results.json").resolve()
    protected_dirs = (qa_source, evaluation, filter_results.parent)
    for target in (out, eval_out):
        if any(target == p or p.is_relative_to(target) for p in protected_dirs):
            raise ValueError("Final output must not overwrite a source directory")
    if out == eval_out or out.is_relative_to(eval_out) or eval_out.is_relative_to(out):
        raise ValueError("QA and evaluation need separate output directories")

    source_qa = read(qa_source / "qa.json")
    audit = read(filter_results)
    selected = select_final(source_qa, audit, groups)
    if expected_qa is not None and len(selected) != expected_qa:
        raise ValueError(f"Expected {expected_qa} final QA, found {len(selected)}")
    ids = {q["qa_id"] for q in selected}
    inputs = read(evaluation / "inputs.json")
    input_lookup = unique_rows(inputs, "evaluation inputs")
    if set(input_lookup) != {q["qa_id"] for q in source_qa}:
        raise ValueError("Source QA and evaluation have different coverage")
    config = read(filter_results.parent / "config.json")
    if config["inputs_sha256"] != digest(evaluation / "inputs.json"):
        raise ValueError("The audit was run on different evaluation inputs")
    refs = read(evaluation / "reference_answers.json")
    examples = [e for e in inputs if e["qa_id"] in ids]
    for q in selected:
        e, ref = input_lookup[q["qa_id"]], refs[q["qa_id"]]
        for field in ("video_id", "task", "group", "question", "options", "target_mapping"):
            if q[field] != e[field]:
                raise ValueError(f"Source/input mismatch: {q['qa_id']} {field}")
        if q["object_ids"] != e["requested_object_ids"] or any(
            q[field] != ref[field] for field in ("answer", "answer_index", "video_id", "task", "group")
        ):
            raise ValueError(f"Reference or target mismatch: {q['qa_id']}")
        if len(e["frame_paths"]) != 9 or set(e) & {
            "answer", "answer_index", "evidence", "depth", "camera_pose"
        }:
            raise ValueError(f"Invalid model input: {q['qa_id']}")
    from .evaluation.protocol import prompt

    for e in examples:
        if e["prompt"] != prompt(e) or hashlib.sha256(e["prompt"].encode()).hexdigest() != e["prompt_sha256"]:
            raise ValueError(f"Prompt mismatch: {e['qa_id']}")

    protected = [qa_source / "qa.json", filter_results, filter_results.parent / "config.json"]
    protected += [evaluation / name for name in (
        "inputs.json", "reference_answers.json", "models.json", "protocol.json", "summary.json", "leaderboard.csv"
    )]
    hashes = {str(p): digest(p) for p in protected}
    normalized = list(dict.fromkeys(g.upper().replace("N", "L") for g in groups))
    videos = sorted({q["video_id"] for q in selected})
    selection = dict(qa_ids=sorted(ids), qa=len(ids), videos=len(videos), video_ids=videos,
                     planned=len(videos), groups=normalized, display_groups=[g.replace("L", "N") for g in normalized],
                     retained_decisions=["keep"],
                     group_qa_counts=dict(collections.Counter(q["group"] for q in selected)),
                     task_qa_counts=dict(collections.Counter(q["task"] for q in selected)))
    # A frozen release can be regenerated only from the same source and rule.
    for target in (out, eval_out):
        manifest = target / "manifest.json"
        if manifest.exists():
            previous = read(manifest)
            if previous["source_sha256"] != hashes or previous["selection"]["qa_ids"] != sorted(ids):
                raise ValueError(f"Frozen release sources changed; use a new output: {target}")
    out.mkdir(parents=True, exist_ok=True)
    eval_out.mkdir(parents=True, exist_ok=True)

    # Relative QA frame references continue to resolve under the release folder.
    for video in videos:
        link_shared(qa_source / "cases" / video, out / "cases" / video)
    save(out / "qa.json", selected)
    (out / "qa.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False, allow_nan=False) + "\n" for q in selected))
    save(out / "selection.json", selection)
    save(out / "qa_counts.json", counts(selected))
    write_csv(out / "qa_counts.csv", counts(selected))
    save(eval_out / "inputs.json", examples)
    save(eval_out / "reference_answers.json", {key: refs[key] for key in sorted(ids)})
    for name in ("models.json", "jobs_config.json"):
        if (evaluation / name).exists():
            save(eval_out / name, read(evaluation / name))
    protocol = read(evaluation / "protocol.json")
    protocol.update(examples=len(ids), source_video_count=len(videos), qa_video_count=len(videos),
                    source_hashes={str(out / "qa.json"): digest(out / "qa.json")},
                    group4_counts=dict(collections.Counter(e["group4"] for e in examples)),
                    scoring=f"Deterministic MCQ matching; invalid/error predictions count as incorrect; denominator {len(ids)} per model.",
                    selection_rule="QA audit decision=keep AND group in HLH/LHH/HHH (N=L)",
                    reused_evaluation=str(evaluation))
    save(eval_out / "protocol.json", protocol)
    models = read(evaluation / "models.json")
    prediction_hashes = {}
    for model in models:
        for key in sorted(ids):
            src = evaluation / "predictions" / model["key"] / f"{key}.json"
            prediction_hashes[str(src)] = digest(src)
            link_shared(src, eval_out / "predictions" / model["key"] / src.name)

    # Freeze and validate the exact pixels that the original inference used.
    expected_media = read(evaluation / "media_sha256.json")
    media = sorted({path for e in examples for path in e["frame_paths"]})
    media_hashes = {}
    for path in media:
        sha = digest(path)
        if sha != expected_media[path]:
            raise ValueError(f"Evaluation image changed: {path}")
        media_hashes[path] = sha
        path_obj = Path(path)
        link_shared(path_obj.parent, eval_out / "media" / path_obj.parent.name)
    save(eval_out / "media_sha256.json", media_hashes)

    from .evaluation.reports import subset_report

    summary = subset_report([g.replace("L", "N") for g in normalized], eval_out,
                            source=evaluation, qa_ids=ids,
                            title="Temporal Geometry: Final Benchmark",
                            footer="Qwen QA audit = keep; camera translation OR rotation high, and scene dynamics high.")
    summary.update(final=True, selection_rule=protocol["selection_rule"], qa_directory=str(out))
    save(eval_out / "summary.json", summary)
    save(eval_out / "selection.json", selection)
    save(eval_out / "run_status.json", summary)
    if any(digest(Path(path)) != value for path, value in {**hashes, **prediction_hashes}.items()):
        raise RuntimeError("Source data changed during finalization")
    for q in selected:
        for ref in q["frame_references"]:
            if not (out / ref["image_path"]).is_file():
                raise FileNotFoundError(out / ref["image_path"])
    validation = dict(status="passed", qa=len(ids), videos=len(videos), models=len(models),
                      predictions=len(ids) * len(models), unique_images=len(media), new_inference=False,
                      source_sha256=hashes, source_predictions_unchanged=True,
                      checks=["exact audit coverage", "keep AND motion-group intersection", "matching QA and references",
                              "same model prompts and option order", "no reference fields in inputs",
                              "same marked pixels", "raw responses reparsed and rescored",
                              "all QA frame references resolve", "source hashes unchanged"])
    manifest = dict(status="final", created_at=datetime.now(timezone.utc).isoformat(),
                    selection=selection, source_sha256=hashes, qa_file=str(out / "qa.json"),
                    qa_sha256=digest(out / "qa.json"), evaluation=str(eval_out),
                    audit_prompt_version=config["prompt_version"],
                    reference_status="Automatic geometry estimates; QA interpretability audited by Qwen, not human ground truth.",
                    shared_assets="Relative symlinks to retained source cases, marked media and raw predictions; retain both fullscope directories.")
    for target in (out, eval_out):
        save(target / "manifest.json", manifest)
        save(target / "validation.json", validation)
    (out / "README.txt").write_text(
        f"Final temporal geometry benchmark: {len(ids)} QA / {len(videos)} videos.\n"
        "Rule: QA-conditioned Qwen audit decision=keep AND group in HLH/LHH/HHH (HNH/NHH/HHH).\n"
        "qa.json/jsonl: exact source questions, choices, geometry references and evidence.\n"
        "selection.json: fixed QA/video IDs. qa_counts.csv: task/group/source counts.\n"
        f"Evaluation and graphs: {eval_out}\n"
        "cases/ links to retained source evidence; keep both fullscope directories together.\n"
        "Automatic geometry references remain provisional; the audit checks visual interpretability.\n"
    )
    (eval_out / "README.txt").write_text(
        f"Final benchmark evaluation: {len(ids)} QA / {len(videos)} videos / {len(models)} models.\n"
        "Original raw responses rescored on the exact final QA; no new inference.\n"
        "leaderboard.csv, per_task.csv, per_group8.csv, per_task_family.csv: accuracy tables.\n"
        "results.png/pdf: overall, six-task and HNH/NHH/HHH group accuracy.\n"
        "inputs.json contains model inputs only; reference_answers.json is separate.\n"
        "predictions/ and media/ link to retained parent assets; keep the parent directories.\n"
        "manifest.json fixes selection/provenance; validation.json records consistency checks.\n"
    )
    if out == FINAL_QA and eval_out == FINAL_EVAL:
        save(ROOT / "benchmark.json", dict(status="final", qa=len(ids), videos=len(videos),
             groups=normalized, qa_file=str((out / "qa.json").relative_to(ROOT)),
             evaluation=str(eval_out.relative_to(ROOT)), manifest=str((out / "manifest.json").relative_to(ROOT))))
    return dict(status="final", qa=len(ids), videos=len(videos), qa_directory=str(out), evaluation=str(eval_out))
