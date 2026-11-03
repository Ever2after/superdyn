"""Seeded terraced terrain generator (pure NumPy; no Blender).

sampling: partition, height assignment and ramp placement; validation: walkability checks;
layout: height queries and meshes; paths: grid routes on the walkable surface.
"""
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "terrain"
