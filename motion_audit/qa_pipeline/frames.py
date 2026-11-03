"""Released-frame sampling; frame indices are not physical timestamps."""

import io, os, zipfile
from pathlib import Path


def frames_for(row):
    import numpy as np
    from PIL import Image
    from contextlib import nullcontext

    source = Path(row["source_path"])
    context = (
        zipfile.ZipFile(source) if row["source_kind"] == "zip" else nullcontext(None)
    )
    with context as archive:
        if archive:
            prefix = row["zip_prefix"] + "/rgb/"
            files = sorted(
                n
                for n in archive.namelist()
                if n.startswith(prefix)
                and Path(n).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            camera_available = row["zip_prefix"] + "/camera.npz" in archive.namelist()
            depth_available = any(
                n.startswith(row["zip_prefix"] + "/depths/") and not n.endswith("/")
                for n in archive.namelist()
            )
        else:
            files = sorted(
                e.path
                for e in os.scandir(source / "rgb")
                if e.is_file()
                and Path(e.name).suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            camera_available = (source / "camera.npz").exists()
            depth_available = (source / "depths").is_dir()
        n = len(files)
        if n < 9:
            raise ValueError(
                f"Only {n} released RGB frames; need 9 distinct measurement frames"
            )
        indices = np.rint(np.linspace(0, n - 1, 33)).astype(int)
        mids = np.arange(0, 33, 4)
        if np.any(np.diff(indices[mids]) <= 0):
            raise ValueError("Insufficient distinct measurement frames")
        decoded = {}
        for idx in sorted(set(indices)):
            if archive:
                with Image.open(io.BytesIO(archive.read(files[idx]))) as im:
                    decoded[idx] = np.asarray(im.convert("RGB"))
            else:
                with Image.open(files[idx]) as im:
                    decoded[idx] = np.asarray(im.convert("RGB"))
        frames = [decoded[idx] for idx in indices]
        times = indices[mids].astype(float) / (n - 1)
        meta = dict(
            total_frames=n,
            selected_frame_indices=indices[mids].tolist(),
            selected_frame_names=[Path(files[idx]).name for idx in indices[mids]],
            measurement_frames=9,
            context_frames=33,
            window_start=0.0,
            window_end=1.0,
            fps=None,
            video_duration=None,
            timeline="uniform_released_frame_index",
            time_units="normalized_full_released_sequence_NOT_seconds",
            window_policy="entire_released_frame_sequence",
            released_camera_available=camera_available,
            released_depth_available=depth_available,
            metric_source="VGGT-Omega + GMOS-S + RAFT-small; consistent estimator",
        )
        return frames, mids, times, meta
