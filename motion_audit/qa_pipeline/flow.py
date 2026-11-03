"""Dense RAFT-large on the same 33 context frames as GMOS."""

import os, time
from .config import ROOT
from .storage import read, save
import numpy as np


class DenseFlowRunner:
    def __init__(self):
        import torch
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

        self.torch = torch
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights).eval().cuda()

    def run(self, frames, indices, folder):
        torch = self.torch
        start = time.perf_counter()
        h, w = frames.shape[1:3]
        inputs = torch.from_numpy(frames).permute(0, 3, 1, 2).float().cuda() / 127.5 - 1
        forward = []
        backward = []
        with torch.inference_mode():
            for i in range(0, 32, 2):
                first = torch.cat([inputs[i : i + 2], inputs[i + 1 : i + 3]])
                second = torch.cat([inputs[i + 1 : i + 3], inputs[i : i + 2]])
                flow = (
                    self.model(first, second, num_flow_updates=16)[-1]
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                forward.extend(flow[:2])
                backward.extend(flow[2:])
        tmp = folder / "dense_flow.tmp.npz"
        np.savez_compressed(
            tmp,
            dense_forward=np.array(forward, dtype="float16"),
            dense_backward=np.array(backward, dtype="float16"),
            source_indices=indices,
        )
        tmp.replace(folder / "dense_flow.npz")
        elapsed = time.perf_counter() - start
        save(
            folder / "dense_flow_provenance.json",
            dict(
                model="torchvision RAFT-large",
                weights=self.weights.name,
                updates=16,
                context_frames=33,
                geometry_frames=9,
                shape=[h, w],
                source_indices=indices.tolist(),
                elapsed_seconds=elapsed,
            ),
        )
        return elapsed


def reuse_flow(vid, folder, indices, shape):
    if (folder / "dense_flow.npz").is_file():
        return True
    for cases in sorted((ROOT / "outputs").glob("*/cases")):
        source = cases / vid / "dense_flow.npz"
        if not source.is_file() or source.parent.resolve() == folder.resolve():
            continue
        with np.load(source) as archive:
            if not np.array_equal(archive["source_indices"], indices):
                continue
            if archive["dense_forward"].shape != (32, *shape, 2):
                continue
        os.link(source, folder / "dense_flow.npz")
        save(
            folder / "dense_flow_provenance.json",
            dict(
                reused=True,
                source=str(source),
                context_frames=33,
                source_indices=indices.tolist(),
                matching_geometry_grid=True,
                model="torchvision RAFT-large",
                updates=16,
            ),
        )
        return True
    return False
