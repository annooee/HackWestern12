# src/detectors/soft_csrnet.py
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


# ----------------------------- model blocks -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, dilation=d, bias=not bn)
        self.bn = nn.BatchNorm2d(out_ch) if bn else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SoftSpatialAttention(nn.Module):
    """
    Lightweight spatial attention: avg+max pooling -> conv -> sigmoid.
    Emphasizes head-like blobs without hard gating.
    """
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=True)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * att


class SoftCSRNet(nn.Module):
    """
    Soft-CSRNet: VGG-like front-end + dilated back-end + soft spatial attention.
    Output density at 1/8 resolution (sum ≈ count when trained).
    """
    def __init__(self):
        super().__init__()
        # Front-end (VGG-ish)
        self.f1 = nn.Sequential(
            ConvBNReLU(3, 64), ConvBNReLU(64, 64), nn.MaxPool2d(2)      # 1/2
        )
        self.f2 = nn.Sequential(
            ConvBNReLU(64, 128), ConvBNReLU(128, 128), nn.MaxPool2d(2)  # 1/4
        )
        self.f3 = nn.Sequential(
            ConvBNReLU(128, 256), ConvBNReLU(256, 256), ConvBNReLU(256, 256), nn.MaxPool2d(2)  # 1/8
        )
        # Back-end (dilated)
        self.b = nn.Sequential(
            ConvBNReLU(256, 512, d=2, p=2),
            ConvBNReLU(512, 512, d=2, p=2),
            ConvBNReLU(512, 512, d=2, p=2),
            ConvBNReLU(512, 256, d=2, p=2),
            ConvBNReLU(256, 128, d=2, p=2),
        )
        # Attention + head
        self.att = SoftSpatialAttention(k=7)
        self.head = nn.Conv2d(128, 1, kernel_size=1)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)  # 1/8 spatial
        x = self.b(x)
        x = self.att(x)
        x = self.head(x)
        x = torch.relu(x)  # density is non-negative
        return x  # [B,1,H/8,W/8]


# ----------------------------- API wrapper -----------------------------

@dataclass
class SoftCSRNetResult:
    count: float
    heatmap: np.ndarray        # float32 [0..1], HxW
    raw_density: np.ndarray    # float32, HxW (upsampled density)
    annotated: Optional[np.ndarray]


class SoftCSRNetCounter:
    """
    Soft-CSRNet inference wrapper with tiling, smoothing, ROI & homography support.
    Includes robust checkpoint loader for PyTorch 2.6+ (weights_only change).
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: Optional[str] = None,
        tile: bool = True,
        scale_factor: float = 1.0,   # multiply sum(density) to calibrate counts
        blur_sigma: int = 9,         # for visualization heatmap
        ema_decay: float = 0.6       # EMA smoothing for displayed count (0 disables)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SoftCSRNet().to(self.device).eval()

        # ---- Robust weights loading (handles PyTorch 2.6, nested dicts, prefixes) ----
        if weights_path:
            resolved = self._resolve_weights_path(weights_path)
            if resolved is None:
                print(f"[SoftCSRNet] Warning: weights not found at '{weights_path}'. Running without weights.")
            else:
                try:
                    # PyTorch 2.6 default weights_only=True rejects pickled checkpoints.
                    # We explicitly set weights_only=False for legacy checkpoints (only for trusted files).
                    sd = torch.load(resolved, map_location=self.device, weights_only=False)
                except TypeError:
                    # Older torch without weights_only kwarg
                    sd = torch.load(resolved, map_location=self.device)

                # Extract inner state_dict if present
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
                    sd = sd["model"]

                # Strip common prefixes like "module." or "model."
                if isinstance(sd, dict):
                    cleaned = OrderedDict()
                    for k, v in sd.items():
                        nk = k
                        if nk.startswith("module."):
                            nk = nk[7:]
                        if nk.startswith("model."):
                            nk = nk[6:]
                        cleaned[nk] = v
                    sd = cleaned

                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    if missing or unexpected:
                        print(f"[SoftCSRNet] Loaded non-strict. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # -----------------------------------------------------------------------------

        self.tile = bool(tile)
        self.scale_factor = float(scale_factor)
        self.blur_sigma = int(max(0, blur_sigma))
        self._ema_decay = float(ema_decay)
        self._ema_count = None

        # imagenet-ish norm
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self.device).view(1,3,1,1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self.device).view(1,3,1,1)

    # ----------------------------- public API -----------------------------

    @torch.inference_mode()
    def process(
        self,
        frame_bgr: np.ndarray,
        rois: Optional[List[List[Tuple[int, int]]]] = None,
        H: Optional[np.ndarray] = None,
        return_visualized: bool = True,
        heat_alpha: float = 0.55
    ) -> SoftCSRNetResult:
        H_img, W_img = frame_bgr.shape[:2]

        # ROI mask
        roi_mask = self._build_roi_mask((H_img, W_img), rois)

        # Prep tensor
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(rgb).float().to(self.device) / 255.0  # [H,W,3]
        img = img.permute(2, 0, 1).unsqueeze(0)                      # [1,3,H,W]
        img = (img - self.mean) / self.std

        # Inference (tiled for big frames)
        if self.tile and max(H_img, W_img) > 1080:
            d_small = self._infer_tiled(img, tile=640, overlap=64)   # [1,1,h8,w8]
        else:
            d_small = self.model(img)
        d_small = torch.relu(d_small)

        # Count from small map
        raw_sum = float(d_small.sum().cpu().item())
        count = raw_sum * self.scale_factor

        # EMA smoothing for display
        count_disp = count
        if self._ema_decay > 0:
            self._ema_count = count if self._ema_count is None else \
                (1 - self._ema_decay) * count + self._ema_decay * self._ema_count
            count_disp = self._ema_count

        # Upsample for viz + apply ROI mask
        d_up = F.interpolate(d_small, size=(H_img, W_img), mode="bicubic", align_corners=False)[0, 0].cpu().numpy()
        d_up = np.maximum(d_up, 0.0)
        if roi_mask is not None:
            d_up = d_up * (roi_mask.astype(np.float32) / 255.0)

        # Heatmap normalization
        heat = d_up.copy()
        if self.blur_sigma > 0:
            k = int(max(3, self.blur_sigma * 3) // 2 * 2 + 1)
            heat = cv2.GaussianBlur(heat, (k, k), self.blur_sigma)
        if heat.max() > 0:
            heat = heat / (heat.max() + 1e-8)

        annotated = None
        if return_visualized:
            cmap = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(frame_bgr, 1.0 - heat_alpha, cmap, heat_alpha, 0.0)
            if rois:
                for poly in rois:
                    if len(poly) >= 3:
                        cv2.polylines(overlay, [np.array(poly, np.int32)], True, (0, 255, 255), 2)
            cv2.putText(overlay, f"Count≈{count_disp:.1f}", (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            annotated = overlay

        return SoftCSRNetResult(
            count=float(count_disp),
            heatmap=heat.astype(np.float32),
            raw_density=d_up.astype(np.float32),
            annotated=annotated
        )

    # ----------------------------- helpers -----------------------------

    def _resolve_weights_path(self, weights_path: str) -> Optional[str]:
        """Resolve relative paths so both 'models/x.pth' and 'src/models/x.pth' can work."""
        if os.path.isabs(weights_path) and os.path.exists(weights_path):
            return weights_path
        # try as given (relative to CWD)
        if os.path.exists(weights_path):
            return weights_path
        # try relative to this file (…/detectors/)
        cand1 = os.path.join(os.path.dirname(__file__), weights_path)
        if os.path.exists(cand1):
            return cand1
        # try one level up from this file (…/src/)
        cand2 = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", weights_path))
        if os.path.exists(cand2):
            return cand2
        return None

    def _build_roi_mask(self, hw: Tuple[int, int], rois: Optional[List[List[Tuple[int, int]]]]) -> Optional[np.ndarray]:
        if not rois:
            return None
        h, w = hw
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in rois:
            if len(poly) >= 3:
                cv2.fillPoly(mask, [np.array(poly, dtype=np.int32)], 255)
        return mask

    @torch.inference_mode()
    def _infer_tiled(self, img: torch.Tensor, tile=640, overlap=64) -> torch.Tensor:
        """Run model on overlapping tiles and stitch outputs at 1/8 resolution."""
        _, _, H, W = img.shape
        out_H, out_W = H // 8, W // 8
        out = torch.zeros((1, 1, out_H, out_W), device=img.device)

        stride = tile - overlap
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y1 = min(H, y + tile)
                x1 = min(W, x + tile)
                patch = img[:, :, y:y1, x:x1]
                pred = self.model(patch)  # [1,1,h8,w8]
                oy, ox = y // 8, x // 8
                h8, w8 = pred.shape[-2], pred.shape[-1]
                out[:, :, oy:oy + h8, ox:ox + w8] += pred
        return out
