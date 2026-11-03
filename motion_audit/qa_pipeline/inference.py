"""VGGT-Ω + GMOS-S + RAFT-small, fixed 4-second windows with real timestamps."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .config import ROOT

os.environ.setdefault("TORCH_HOME", str(ROOT / "checkpoints/torch"))
sys.path.insert(0, str(ROOT / "vendor/vggt-omega"))
sys.path.insert(0, str(ROOT / "vendor/gmos"))
sys.path.insert(0, str(ROOT / "vendor/gmos/core/sam2"))
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images
from vggt_omega.utils.pose_enc import encoding_to_camera
from core.pi3_encoder import Pi3Encoder
from core.sam2_encoder import SAM2Encoder
from core.gmos_s import GMOS_S
from core.sam2.sam2.utils.transforms import SAM2Transforms
from data.utils import pil_image_to_tensor_pi3
from utils import encode_pi3_with_cache
from safetensors.torch import load_file
from .metrics import interval_statistics, summarize_intervals, summarize_amplitudes


class Engine:
    def __init__(self):
        self.device = "cuda"
        self.omega = VGGTOmega().eval().cuda()
        self.omega.load_state_dict(
            torch.load(
                ROOT / "checkpoints/omega/vggt_omega_1b_416_reproduce.pt",
                map_location="cpu",
            )
        )
        self.pi3 = Pi3Encoder().eval().cuda()
        missing, unexpected = self.pi3.load_state_dict(
            load_file(str(ROOT / "checkpoints/gmos/pi3/model.safetensors")),
            strict=False,
        )
        if missing:
            raise ValueError(f"Pi3 encoder missing parameters: {missing[:5]}")
        self.sam = SAM2Encoder().eval().cuda()
        sam = torch.load(
            ROOT / "checkpoints/gmos/sam2.1_hiera_large.pt", map_location="cpu"
        )["model"]
        filtered = {
            k: v
            for k, v in sam.items()
            if "image_encoder" in k
            or "sam_mask_decoder.conv_s0" in k
            or "sam_mask_decoder.conv_s1" in k
            or "no_mem_embed" in k
        }
        missing, unexpected = self.sam.load_state_dict(filtered, strict=False)
        if missing:
            raise ValueError(f"SAM encoder missing parameters: {missing[:5]}")
        self.gmos = GMOS_S().eval().cuda()
        weights = torch.load(
            ROOT / "checkpoints/gmos/gmos_s.pth", map_location="cpu", weights_only=False
        )["model_state_dict"]
        weights = {k.removeprefix("module."): v for k, v in weights.items()}
        missing, unexpected = self.gmos.load_state_dict(weights, strict=False)
        if missing or set(unexpected) - {
            "pe_layer.positional_encoding_gaussian_matrix"
        }:
            raise ValueError(f"GMOS checkpoint mismatch: {missing}, {unexpected}")
        self.sam_transform = SAM2Transforms(
            resolution=1024,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).eval().cuda()

    @torch.inference_mode()
    def predict(self, frames, mids, cache):
        cache.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in mids:
            p = cache / f"{i:04d}.png"
            Image.fromarray(frames[i]).save(p)
            paths.append(str(p))
        images = load_and_preprocess_images(paths, image_resolution=416).cuda()
        prediction = self.omega(images)
        e, k = encoding_to_camera(prediction["pose_enc"], images.shape[-2:])
        depth = prediction["depth"][0, ..., 0].float().cpu().numpy()
        conf = prediction["depth_conf"][0].float().cpu().numpy()
        h, w = images.shape[-2:]
        masks = []
        pi_cache = {}
        # GMOS-S uses its official 5-frame Pi3 context and SAM image preprocessing.
        for offset in range(0, len(mids), 4):
            mid_batch = mids[offset : offset + 4]
            idx = np.array(
                [
                    [np.clip(i + j, 0, len(frames) - 1) for j in range(-2, 3)]
                    for i in mid_batch
                ]
            )
            pi_inputs = torch.stack(
                [
                    pil_image_to_tensor_pi3(
                        [Image.fromarray(frames[j]) for j in window]
                    )
                    for window in idx
                ]
            ).cuda()
            sam_inputs = torch.stack(
                [self.sam_transform(Image.fromarray(frames[i])) for i in mid_batch]
            ).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden = encode_pi3_with_cache(
                    self.pi3, pi_inputs, torch.from_numpy(idx).cuda(), pi_cache
                )
                hidden = hidden[:, 2, ..., -1024:].float()
                backbone = self.sam.forward_image(sam_inputs)
                _, vision, _, _ = self.sam._prepare_backbone_features(backbone)
                btn, hi = self.sam._get_vision_features(vision)
            seg = self.gmos(hidden, btn.float(), [v.float() for v in hi])["fg_mask"]
            # Match VGGT crop (aspect-ratio [0.5,2]), then resize to geometry grid.
            oh, ow = frames[0].shape[:2]
            seg = F.interpolate(
                seg, size=(oh, ow), mode="bilinear", align_corners=False
            )
            if oh / ow < 0.5:
                crop = min(ow, round(oh / 0.5))
                left = (ow - crop) // 2
                seg = seg[:, :, :, left : left + crop]
            elif oh / ow > 2:
                crop = min(oh, round(ow * 2))
                top = (oh - crop) // 2
                seg = seg[:, :, top : top + crop, :]
            seg = F.interpolate(
                seg, size=(h, w), mode="bilinear", align_corners=False
            ).sigmoid()
            masks.extend(seg[:, 0].cpu().numpy())
        ff = []
        bb = []
        for offset in range(0, len(mids) - 1, 4):
            a = images[offset : offset + 4]
            b = images[offset + 1 : offset + 5]
            count = min(len(a), len(b))
            a = a[:count]
            b = b[:count]
            flow = self.raft(
                torch.cat([a, b]) * 2 - 1,
                torch.cat([b, a]) * 2 - 1,
                num_flow_updates=12,
            )[-1]
            ff.extend(flow[:count].permute(0, 2, 3, 1).cpu().numpy())
            bb.extend(flow[count:].permute(0, 2, 3, 1).cpu().numpy())
        return dict(
            depth=depth,
            conf=conf,
            extrinsics=e[0].float().cpu().numpy(),
            intrinsics=k[0].float().cpu().numpy(),
            masks=np.array(masks),
            forward=np.array(ff),
            backward=np.array(bb),
        )
