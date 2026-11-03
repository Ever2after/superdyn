"""Pilot contact sheets (RGB + GT) and REPORT.md."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2
import numpy as np

from pilot_design import MEASURE

COLORS = {"A": (30, 112, 242), "B": (125, 200, 10)}  # BGR orange / green


def sheet(output, clip):
    vid = "synth_pilot__" + clip["id"]
    case = output / "qa" / "cases" / vid
    gt = np.load(output / "qa" / "gt" / (vid + ".npz"))
    masks = np.load(case / "instance_masks.npz")
    tiles_rgb, tiles_depth = [], []
    for i in range(9):
        image = cv2.imread(str(case / f"rgb_{i}.jpg"))
        for a in clip["actors"]:
            key = f'{clip["id"]}__{a["id"]}'
            color = COLORS.get(a["label"], (60, 60, 230))
            contours, _ = cv2.findContours(masks[key][i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, color, 3 if a["label"] else 1)
        cv2.putText(image, f"F{i}", (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 3)
        tiles_rgb.append(cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA))
        depth = gt["depth_m"][i].astype(np.float32)
        sky = ~(depth < 200)
        vis = np.clip(np.log(np.clip(depth, 0.5, 200)) / np.log(200), 0, 1)
        vis = cv2.applyColorMap((255 * (1 - vis)).astype(np.uint8), cv2.COLORMAP_TURBO)
        vis[sky] = (40, 40, 40)
        vis[gt["instance_ids"][i] > 0] = (255, 255, 255)
        tiles_depth.append(cv2.resize(vis, (320, 180), interpolation=cv2.INTER_NEAREST))
    board = np.vstack([np.hstack(tiles_rgb), np.hstack(tiles_depth)])
    path = output / "clips" / clip["id"] / "contact_sheet.jpg"
    cv2.imwrite(str(path), board, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return path


def collect_timing(output):
    """MRQ job durations from every render session log, keyed by job name."""
    import re
    jobs = {}
    for attempt in sorted((output / "native").glob("render_*")):
        request, log = attempt / "request.json", attempt / "editor.log"
        if not request.exists() or not log.exists() or "interrupted" in attempt.name or "stopped" in attempt.name:
            continue
        names = [j["name"] for j in json.loads(request.read_text()).get("jobs", [])]
        found = re.findall(r"Movie Pipeline completed\. Duration: \+(\d+):(\d+):([\d.]+)", log.read_text(errors="replace"))
        for name, (h, m, sec) in zip(names, found):
            if name.endswith("_rgb") or name.endswith("_gt"):
                jobs.setdefault(name, []).append(dict(attempt=attempt.name, seconds=int(h) * 3600 + int(m) * 60 + float(sec)))
    return jobs


def write(output):
    output = Path(output)
    plan = json.loads((output / "plan.json").read_text())
    qa = json.loads((output / "qa/qa.json").read_text())
    audit = json.loads((output / "qa/design_audit.json").read_text())
    clips = {c["id"]: c for c in json.loads((output / "qa/clips.json").read_text())}
    timing = collect_timing(output)
    if not timing and (output / "render_timing.json").exists():
        timing = json.loads((output / "render_timing.json").read_text())  # editor logs were cleaned up
    for clip in plan["clips"]:
        sheet(output, clip)
    overview = []
    for clip in plan["clips"]:
        image = cv2.imread(str(output / "qa/cases" / ("synth_pilot__" + clip["id"]) / "rgb_4.jpg"))
        image = cv2.resize(image, (480, 270), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (480, 30), (20, 20, 20), -1)
        cv2.putText(image, clip["id"] + "  [" + clip["env"] + "]", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        overview.append(image)
    while len(overview) % 4:
        overview.append(np.zeros_like(overview[0]))
    rows = [np.hstack(overview[i:i + 4]) for i in range(0, len(overview), 4)]
    cv2.imwrite(str(output / "overview.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])

    kept = Counter(a["task"] for a in audit if all(a["gates"].values()))
    designed = Counter(a["task"] for a in audit)
    fails = Counter(g for a in audit for g, ok in a["gates"].items() if not ok)
    answers = defaultdict(Counter)
    for q in qa:
        answers[q["task"]][q["answer"]] += 1
    cues = Counter((q["task"], q["cue"]) for q in qa if q["cue"])
    lines = ["# Synthetic split 파일럿 결과", "",
             f"[사양 초안](../../docs/SYNTHETIC_SPLIT_SPEC.md) · 클립 {len(plan['clips'])}개 · 환경 {len({c['env'] for c in plan['clips']})}개 · QA {len(qa)}개", "",
             "![대표 프레임](overview.jpg)", "",
             "## 설계 적중률", "",
             "| Task | 설계 | 채택 |", "|---|---:|---:|"]
    for task in designed:
        lines.append(f"| {task} | {designed[task]} | {kept[task]} |")
    lines += [f"| 전체 | {sum(designed.values())} | {sum(kept.values())} ({sum(kept.values()) / max(1, sum(designed.values())):.0%}) |", "",
              "탈락 사유(중복 가능): " + ", ".join(f"{k} {v}" for k, v in fails.most_common()) if fails else "탈락 없음", "",
              "## 정답 분포", ""]
    for task, counter in answers.items():
        lines.append(f"- {task}: " + ", ".join(f"{k} {v}" for k, v in sorted(counter.items())))
    lines += ["", "Cue(2D 단서와 3D 정답의 관계): " + ", ".join(f"{t}/{c} {n}" for (t, c), n in sorted(cues.items())), "",
              "## 클립별", "", "| 클립 | 환경 | split | 카메라 | 그룹 | 배경 depth s | center∈mask | 충돌 | 시야 차단 | 설계→채택 | 시트 |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for clip in plan["clips"]:
        info = clips[clip["id"]]
        mine = [a for a in audit if a["clip"] == clip["id"]]
        ok = sum(all(a["gates"].values()) for a in mine)
        lines.append(f'| {clip["id"]} | {clip["env"]} | {clip["split"]} | {clip["camera"]["primitive"]} | {info["group"]} | {info["scale_m"]:.1f}m | '
                     f'{info["center_in_mask"][0]}/{info["center_in_mask"][1]} | {info["collisions"]} | {info["line_of_sight_blocked"]} | {len(mine)}→{ok} | '
                     f'[보기](clips/{clip["id"]}/contact_sheet.jpg) |')
    lines += ["", "## 설계 QA 상세", "", "| 클립 | Task | 객체 | 설계 | GT(center) | GT(visible) | cue 측정 | 통과 못한 gate |", "|---|---|---|---|---|---|---|---|"]
    for a in audit:
        bad = [g for g, ok in a["gates"].items() if not ok]
        lines.append(f'| {a["clip"]} | {a["task"]} | {",".join(a["objects"]) or "-"} | {a["design_answer"]} | {a["gt_answer"]} | {a["visible_anchor_answer"]} | {a["measured_cue"]} | {", ".join(bad) or "-"} |')
    if timing:
        def last(name):
            runs = [r for r in timing.get(name, []) if r["seconds"] > 10]
            return runs[-1]["seconds"] if runs else None
        rows = []
        for clip in plan["clips"]:
            rgb, gt = last(clip["id"] + "_rgb"), last(clip["id"] + "_gt")
            rows.append(f'| {clip["id"]} | {clip["env"]} | {rgb and round(rgb)} | {rgb and round(rgb / 129, 1)} | {gt and round(gt)} | {gt and round(gt / 33, 1)} |')
        lines += ["", "## 렌더 비용 (공유 GPU, 세션 병렬)", "",
                  "| 클립 | 환경 | RGB 초 | RGB 초/프레임 | GT 초 | GT 초/프레임 |", "|---|---|---:|---:|---:|---:|"] + rows
        save_json_path = output / "render_timing.json"
        save_json_path.write_text(json.dumps(timing, indent=1))
    try:
        from pilot_qa import framing
        from pilot_design import designs
        lines += ["", "## 계획 단계 framing 검사 (렌더 전 계산 가능, 이번 배치에는 미적용)", "",
                  "| 클립 | 타깃 | 화면 안 비율 | 최소 투영 높이(px) |", "|---|---|---:|---:|"]
        for d in designs():
            for label, v in framing(d).items():
                lines.append(f'| {d["id"]} | {label} | {v["in_frame"]:.2f} | {v["min_height_px"]:.0f} |')
    except Exception as error:
        lines += ["", "framing 검사 실패: " + str(error)]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return dict(status="reported", qa=len(qa), designed=sum(designed.values()), kept=sum(kept.values()))
