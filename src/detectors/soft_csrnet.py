from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import OrderedDict
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ----------------- Official CSRNet (Li et al., CVPR 2018) -----------------

class CSRNet(nn.Module):
    """
    Official CSRNet:
      - Front-end: VGG-16 (NO batch norm), features[:23] (up to conv4_3), total stride = 8
      - Back-end: 6 dilated conv layers (dilation=2)
      - Output: 1x density map (sum ≈ count)
    """
    def __init__(self, load_vgg_weights: bool = True):
        super().__init__()
        # IMPORTANT: plain VGG16 (no BN)
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_FEATURES if load_vgg_weights else None)

        # Up to conv4_3 => features[:23]
        # (VGG16 features indices: conv1_1..pool1 (0..4),
        #  conv2_1..pool2 (5..9), conv3_1..pool3 (10..16),
        #  conv4_1..conv4_3 (17..23)  -> stop before pool4)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])

        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64,  3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return torch.relu(x)


# --------------------------- Inference wrapper ---------------------------

@dataclass
class CSRResult:
    count: float
    heatmap: np.ndarray        # float32 [0..1], HxW
    raw_density: np.ndarray    # float32, HxW
    annotated: Optional[np.ndarray]


class CSRNetCounter:
    """
    CSRNet video inference with robust weight loading (PyTorch 2.6+), ROI, EMA smoothing.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        tile: bool = True,
        scale_factor: float = 1.0,
        blur_sigma: int = 9,
        ema_decay: float = 0.5
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CSRNet(load_vgg_weights=True).to(self.device).eval()

        if weights_path:
            resolved = self._resolve(weights_path)
            if resolved is None:
                print(f"[CSRNet] Warning: weights not found at '{weights_path}'. Running with ImageNet front-end only.")
            else:
                try:
                    sd = torch.load(resolved, map_location=self.device, weights_only=False)  # PyTorch 2.6+
                except TypeError:
                    sd = torch.load(resolved, map_location=self.device)  # older torch

                # unwrap common formats
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]

                # strip DDP prefixes
                if isinstance(sd, dict):
                    cleaned = OrderedDict()
                    for k, v in sd.items():
                        if k.startswith("module."): k = k[7:]
                        if k.startswith("model."):  k = k[6:]
                        cleaned[k] = v
                    sd = cleaned

                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    if missing or unexpected:
                        print(f"[CSRNet] non-strict load → missing: {len(missing)}, unexpected: {len(unexpected)}")

        self.tile = bool(tile)
        self.scale = float(scale_factor)
        self.blur = int(max(0, blur_sigma))
        self.ema = float(ema_decay)
        self._ema_count = None

        # ImageNet mean/std (VGG)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1,3,1,1)

    @torch.inference_mode()
    def process(
        self,
        frame_bgr: np.ndarray,
        rois: Optional[List[List[Tuple[int,int]]]] = None,
        return_visualized: bool = True,
        heat_alpha: float = 0.55
    ) -> CSRResult:
        H, W = frame_bgr.shape[:2]
        mask = self._roi_mask((H, W), rois)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float().to(self.device) / 255.0
        img = img.permute(2,0,1).unsqueeze(0)
        img = (img - self.mean) / self.std

        d_small = self._infer_tiled(img) if (self.tile and max(H,W) > 1080) else self.model(img)
        d_small = torch.relu(d_small)  # [1,1,h8,w8]

        raw_sum = float(d_small.sum().cpu().item())
        count = raw_sum * self.scale

        if self.ema > 0:
            self._ema_count = count if self._ema_count is None else (1-self.ema)*count + self.ema*self._ema_count
            count_disp = self._ema_count
        else:
            count_disp = count

        d_up = F.interpolate(d_small, size=(H, W), mode="bicubic", align_corners=False)[0,0].cpu().numpy()
        d_up = np.maximum(d_up, 0.0)
        if mask is not None:
            d_up *= (mask.astype(np.float32) / 255.0)

        heat = d_up.copy()
        if self.blur > 0:
            k = int(max(3, self.blur*3) // 2 * 2 + 1)
            heat = cv2.GaussianBlur(heat, (k, k), self.blur)
        if heat.max() > 0:
            heat /= (heat.max() + 1e-8)

        annotated = None
        if return_visualized:
            cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame_bgr, 1.0 - heat_alpha, cmap, heat_alpha, 0.0)
            if rois:
                for poly in rois:
                    if len(poly) >= 3:
                        cv2.polylines(overlay, [np.array(poly, np.int32)], True, (0,255,255), 2)
            cv2.putText(overlay, f"Count≈{count_disp:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            annotated = overlay

        return CSRResult(float(count_disp), heat.astype(np.float32), d_up.astype(np.float32), annotated)

    # --------------------------- helpers ---------------------------

    def _infer_tiled(self, img: torch.Tensor, tile=640, overlap=64) -> torch.Tensor:
        _, _, H, W = img.shape
        out = torch.zeros((1,1,H//8,W//8), device=img.device)
        stride = tile - overlap
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1, x1 = min(H, y+tile), min(W, x+tile)
                patch = img[:,:,y:y1, x:x1]
                pred = self.model(patch)
                oy, ox = y//8, x//8
                h8, w8 = pred.shape[-2], pred.shape[-1]
                out[:,:,oy:oy+h8, ox:ox+w8] += pred
        return out

    def _roi_mask(self, hw: Tuple[int,int], rois: Optional[List[List[Tuple[int,int]]]]):
        if not rois: return None
        h, w = hw
        m = np.zeros((h,w), np.uint8)
        for poly in rois:
            if len(poly) >= 3:
                cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
        return m

    def _resolve(self, path: str) -> Optional[str]:
        if os.path.isabs(path) and os.path.exists(path): return path
        if os.path.exists(path): return path
        cand1 = os.path.join(os.path.dirname(__file__), path)
        if os.path.exists(cand1): return cand1
        cand2 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", path))
        if os.path.exists(cand2): return cand2
        return None
