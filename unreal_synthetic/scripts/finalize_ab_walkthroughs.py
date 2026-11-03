"""Verify walkthroughs and rebuild Markdown without altering original render evidence."""

import argparse
from collections import Counter
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from walkthroughs import CASE_IDS, DEFAULT_OUTPUT, read_json, verify_case, verify_delivery
from superdyn_ue.io_utils import save_json, sha256


def create_manifest(output):
    """Snapshot a new completed render once; existing manifests stay immutable."""
    scene = read_json(output / "scene_manifest.json")
    results = [verify_case(output / case, output / "scene_manifest.json") for case in CASE_IDS]
    files = [ROOT / "bridge.py", ROOT / "walkthroughs.py", ROOT / "manage.py",
             *sorted((ROOT / "Content/Python").rglob("*.py")),
             *[ROOT / "scripts" / name for name in ["plan_ab_walkthroughs.py", "render_ab_walkthroughs.py", "finalize_ab_walkthroughs.py"]],
             *sorted((ROOT / "inputs/ab_walkthroughs").glob("*.json"))]
    hashes = {}
    for source in files:
        relative = str(source.relative_to(ROOT))
        destination = output / "source_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        hashes[relative] = sha256(source)
    namespace = scene["map_path"].rsplit("/Maps/", 1)[0]
    asset_root = ROOT / "Content" / namespace.removeprefix("/Game/")
    save_json(output / "manifest.json", dict(
        status="passed", cases=results, total_frames=sum(case["frames"] for case in results),
        logical_placements=len(scene["objects"]), mesh_actors=len(scene["objects"]) + sum(bool(o.get("double_face")) for o in scene["objects"]),
        distinct_mesh_assets=len({o["mesh"] for o in scene["objects"]}), source_packs=len(scene["source_packs"]),
        category_counts=dict(Counter(o["category"] for o in scene["objects"])),
        source_counts=dict(Counter(o["source_pack"] for o in scene["objects"])),
        quadrant_counts={label: sum((o["anchor_cm"][0] >= 0) == east and (o["anchor_cm"][1] >= 0) == north for o in scene["objects"]) for label, east, north in [("NE", True, True), ("NW", False, True), ("SW", False, False), ("SE", True, False)]},
        source_hashes=hashes, excluded=scene["excluded"],
        saved_scene_assets_sha256={str(p.relative_to(ROOT)): sha256(p) for p in asset_root.rglob("*") if p.suffix in [".umap", ".uasset"]}))


def make_images(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(3, 3, figsize=(18, 10))
    fig.patch.set_facecolor("#151b23")
    for row, case in zip(axes, CASE_IDS):
        folder = output / case
        for ax, frame in zip(row, [0, 384, 768]):
            ax.imshow(mpimg.imread(folder / "rgb" / f"frame_{frame:06d}.png"))
            ax.set_axis_off()
            ax.set_title(f"{case} / {frame / 24:.1f}s", color="white", fontsize=9)
        fig2, axes2 = plt.subplots(2, 3, figsize=(15, 6))
        fig2.patch.set_facecolor("#151b23")
        for ax, frame in zip(axes2.flat, [0, 192, 384, 576, 768, 959]):
            ax.imshow(mpimg.imread(folder / "rgb" / f"frame_{frame:06d}.png"))
            ax.set_axis_off()
            ax.set_title(f"{frame / 24:.1f}s", color="white")
        fig2.tight_layout()
        fig2.savefig(folder / "contact_sheet.jpg", dpi=110, facecolor=fig2.get_facecolor())
        plt.close(fig2)
    fig.tight_layout()
    fig.savefig(output / "overview.jpg", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_report(output, verification):
    manifest = read_json(output / "manifest.json")
    scene = read_json(output / "scene_manifest.json")
    titles = {s["id"]: s["title"] for s in read_json(ROOT / "inputs/fab_selected_review/selection.json")["selected_sources"]}
    labels = ["동측 → 서측", "서측 → 동측", "전체 지형 순환"]
    lines = ["# A·B 에셋 지형의 1인칭 영상", "",
             "A·B 소스 **14개**, 서로 다른 메시 **29개**로 **51개 배치 요소·53개 메시 액터**를 구성했다. C 등급 사용은 제외했다.", "",
             "세 영상은 각각 **40초·960프레임·1280×720·24fps**다.", "",
             "![최종 영상 대표 프레임](overview.jpg)", "",
             "| 영상 | MP4 | 수평 이동 거리 | 대표 프레임 |", "| --- | --- | ---: | --- |"]
    for label, case in zip(labels, verification["cases"]):
        cid = case["id"]
        lines.append(f'| {label} | [영상]({cid}/sample.mp4) | {case["path_length_m"]:.1f}m | [6장]({cid}/contact_sheet.jpg) |')
    lines += ["", "## 배치", "",
              "차량 4대, 사람 4명, 동물 모델 6개, 나무 4개, 소품 25개, 구조물 부품 8개다. 북동 12개·북서 16개·남서 12개·남동 11개로 분산하고 통행 공간·경사로를 비웠다.", "",
              "목조 쉼터·카페·도시 외벽·기둥·벽돌 패널과 벤치·팔레트·상자·드럼통·케이블 릴을 배치했다. 얇은 벽돌 패널 두 곳은 반대 방향 메시를 추가해 양면에서 보이게 했다.", "",
              "![배치와 동선](layout_routes.png)", "",
              "| 소스 | 배치 요소 수 |", "| --- | ---: |"]
    for source, count in manifest["source_counts"].items():
        lines.append(f"| {titles[source]} | {count} |")
    lines += ["", "## 촬영과 확인 범위", "",
              "- 실제 지형 충돌면 위 165cm, 수평 화각 80°, 롤 0의 시퀀서 카메라다. 플레이어 손·HUD는 없다.",
              "- 지면 0–300cm와 수평 벽 상단 400cm, Poly Haven Kloofendal 하늘 HDRI를 유지했다.",
              f'- 전체 2,880개 RGB 해시·영상 디코딩을 확인했다. 최대 프레임 간 높이 변화는 {max(c["max_vertical_step_cm"] for c in verification["cases"]):.2f}cm다.',
              "- 차량은 정차하며 사람·동물은 원본 대기 애니메이션을 재생한다. 차량 주행 물리·발 IK·학습용 마스크·추적 라벨은 포함하지 않았다.",
              "- `scene_manifest.json`의 transform은 초기 배치 기록이다. 지면·발 위치 보정을 포함한 최종 자세는 저장된 Unreal 맵을 따른다.",
              "- Unfinished Building과 City Park·Rural Cabins의 전체 원본 맵은 제외했다. 후자 두 팩의 B 등급 개별 메시만 사용했다.", "",
              "## 저장 장면과 재현", "",
              f'- 맵: `{scene["map_path"]}`',
              "- 각 영상 폴더: `sample.mp4`, `rgb/`, `route.json`, `native_validation.json`, `verification.json`.",
              "- [렌더 당시 매니페스트](manifest.json) · [배치 기록](scene_manifest.json) · [발 보정](foot_alignment.json).",
              "- `source_snapshot/`은 렌더 당시 코드다. 리팩토링 이후 실행 코드는 프로젝트의 `scripts/`·`Content/Python/`를 사용한다. 원본 스냅샷과 영상·장면 해시는 유지한다.",
              "- 이전 출력·개발 시도·프리뷰는 정리했다. [실행 안내](../../docs/WORKFLOW.md) · [에셋 표](../../docs/ASSETS.md) · [정리 기록](../../docs/CLEANUP.md).", ""]
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--rebuild-images", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not (output / "manifest.json").exists():
        if args.verify_only:
            raise RuntimeError("Final manifest does not exist")
        create_manifest(output)
    verification = verify_delivery(output)
    save_json(ROOT / "reports/walkthrough_verification.json", verification)
    if not args.verify_only:
        if args.rebuild_images or not (output / "overview.jpg").exists():
            make_images(output)
        write_report(output, verification)
    print(f'Passed: {verification["movies"]} movies, {verification["decoded_frames"]} decoded frames, {verification["image_hashes_checked"]} frame hashes; original render evidence preserved')


if __name__ == "__main__":
    main()
