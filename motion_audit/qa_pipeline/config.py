"""Shared paths, six task definitions, thresholds and active dataset scope."""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
QA_RUN = ROOT / "outputs/temporal_geometry_fullscope/qa"
EVAL_RUN = ROOT / "outputs/temporal_vlm_eval_fullscope"
FINAL_QA = QA_RUN / "final"
FINAL_EVAL = EVAL_RUN / "final"
FINAL_GROUPS = ("HLH", "LHH", "HHH")
DATA_ROOT = Path(
    os.environ.get("DYNAMICVERSE_ROOT", "/home/jovyan/data-1/jusang/dynamicVerse")
)
EXCLUDED = {"spring", "point_odyssey", "dynamic_replica", "dynpose-100k"}
DATASETS = ("DAVIS", "MOSE", "MVS-Synth", "SAV", "VOST", "uvo", "youtube_vis")
QWEN_MODEL = os.environ.get(
    "QA_QWEN_MODEL", "/home/jovyan/data-1/jusang/freesee/models/Qwen3.8-27B"
)
SEED = 20261020
TH = dict(
    camera_translation=0.10,
    object_translation=0.10,
    direction_dominance=2.0,
    direction_gap=0.05,
    camera_turn_deg=15.0,
    camera_turn_gap_deg=8.0,
    camera_object_range_abs=0.10,
    camera_object_range_relative=0.20,
    pair_range_abs=0.10,
    pair_range_relative=0.20,
    pair_initial_separation=0.08,
    travel_gap=0.10,
    travel_ratio=1.5,
    travel_max_min=0.20,
    bootstrap_gap=0.05,
    min_time_span=0.20,
    min_intervals=2,
    min_tracks=32,
    min_track_survival=0.15,
    sam_score=0.80,
    min_mask_pixels=150,
    min_forward_coverage=0.25,
    max_fb_error_px=2.0,
    max_pair_mask_overlap=0.05,
    seed_count=256,
    bootstrap_replicates=128,
    bootstrap_direction_agreement=0.95,
)
TASKS = [
    "camera_translation_direction",
    "camera_view_direction_change",
    "object_world_motion_direction",
    "camera_object_distance_change",
    "object_pair_distance_change",
    "object_travel_distance_comparison",
]
GROUPS = ["LLL", "LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
GROUP4 = {
    "LLL": "LLL",
    "LLH": "LLH",
    "HLL": "HLL+LHL+HHL",
    "LHL": "HLL+LHL+HHL",
    "HHL": "HLL+LHL+HHL",
    "HLH": "HLH+LHH+HHH",
    "LHH": "HLH+LHH+HHH",
    "HHH": "HLH+LHH+HHH",
}


def group(row):
    keys = ["translation_high", "rotation_high", "dynamic_high"]
    if any(str(row.get(k)).lower() not in {"true", "false"} for k in keys):
        raise ValueError("Missing motion-group flags")
    return "".join("H" if str(row[k]).lower() == "true" else "L" for k in keys)
