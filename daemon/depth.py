"""AR hologram Tier-1 (2.5D depth): monocular depth estimation + per-clip normalization.

Pure functions (numpy / cv2 / onnxruntime), no ComfyUI. Used by `_execute_ar_hologram` when
the carrier's flavor is "2.5d_depth": estimate a relative depth map per frame, then normalize
the whole clip into a stable 0..255 luma map (bright = near) packed as a third region.

Model: Depth Anything V2 (small), ONNX. Tensor names + input size are read from the loaded
session so we're robust to export differences. Output is relative inverse-depth where LARGER
values are NEARER — we keep that convention (bright = near) and let the shader push bright
vertices toward the viewer.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

# ImageNet normalization (Depth Anything preprocessing).
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_DEFAULT_INPUT = 518  # Depth Anything V2 default square input

_SESSIONS: dict[str, object] = {}


def ensure_depth_model(path: str, url: str) -> str:
    """Return the depth ONNX path, downloading it (~100MB) on first use if missing.

    Downloads to a temp name then renames atomically — an interrupted download must not
    leave a truncated file at `path` (existence is the only cache check).
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import urllib.request

        tmp = f"{path}.tmp"
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _session(model_path: str):
    if model_path not in _SESSIONS:
        import onnxruntime as ort

        # Prefer GPU; onnxruntime silently falls back to CPU when CUDA isn't available.
        _SESSIONS[model_path] = ort.InferenceSession(
            model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
    return _SESSIONS[model_path]


def _input_size(sess) -> tuple[int, int]:
    shape = sess.get_inputs()[0].shape  # [N, 3, H, W] — dims may be strings if dynamic
    h = shape[2] if isinstance(shape[2], int) else _DEFAULT_INPUT
    w = shape[3] if isinstance(shape[3], int) else _DEFAULT_INPUT
    return int(h), int(w)


def estimate_depth(
    frame_rgb: np.ndarray, model_path: str, alpha: np.ndarray | None = None
) -> np.ndarray:
    """Relative depth (float32 HxW, larger = nearer) at the frame's native resolution.

    When `alpha` (the matte) is given, background pixels are replaced with mid-gray before
    estimation: a flat chroma plate is a degenerate scene for the model and compresses the
    subject's own disparity range. The frame is letterboxed (not squashed) into the model's
    square input so subject proportions — the geometry the relief comes from — are preserved.
    """
    sess = _session(model_path)
    inp = sess.get_inputs()[0]
    ih, iw = _input_size(sess)
    H, W = frame_rgb.shape[:2]

    src = frame_rgb
    if alpha is not None:
        src = src.copy()
        src[alpha <= 0.5] = 128

    scale = min(iw / W, ih / H)
    rw, rh = max(1, round(W * scale)), max(1, round(H * scale))
    resized = cv2.resize(src, (rw, rh), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((ih, iw, 3), 128, dtype=np.uint8)
    ox, oy = (iw - rw) // 2, (ih - rh) // 2
    canvas[oy:oy + rh, ox:ox + rw] = resized

    img = canvas.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    x = img.transpose(2, 0, 1)[None].astype(np.float32)  # [1,3,ih,iw]

    out = np.asarray(sess.run(None, {inp.name: x})[0]).squeeze()  # -> HxW (model resolution)
    oh, ow = out.shape[:2]
    sy, sx = oh / ih, ow / iw  # output may not be at input resolution
    out = out[int(oy * sy):int(round((oy + rh) * sy)), int(ox * sx):int(round((ox + rw) * sx))]
    return cv2.resize(out.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def normalize_depth_clip(
    depths: list[np.ndarray],
    alphas: list[np.ndarray],
    crop_rect: tuple[int, int, int, int],
    ema: float = 0.85,
) -> list[np.ndarray]:
    """Crop each depth to the subject bbox and normalize the whole clip to uint8 (bright = near).

    Uses one robust per-clip range (2nd/98th percentile over subject pixels, alpha>0.5) so the
    surface doesn't breathe frame-to-frame, pins background to far (0), and applies a light
    temporal EMA to damp residual flicker.
    """
    x, y, cw, ch = crop_rect
    cd = [d[y:y + ch, x:x + cw] for d in depths]
    ca = [a[y:y + ch, x:x + cw] for a in alphas]

    subj = [c[m > 0.5] for c, m in zip(cd, ca) if (m > 0.5).any()]
    if subj:
        allv = np.concatenate([s.ravel() for s in subj])
        lo, hi = float(np.percentile(allv, 2)), float(np.percentile(allv, 98))
    else:
        lo, hi = 0.0, 1.0
    rng = max(hi - lo, 1e-6)

    out: list[np.ndarray] = []
    prev: np.ndarray | None = None
    for c, m in zip(cd, ca):
        n = np.clip((c - lo) / rng, 0.0, 1.0)  # larger raw depth -> brighter -> nearer
        if prev is not None:
            # Light EMA only: a heavier blend makes depth visibly lag the color frames,
            # which reads as waves rippling across the subject during motion.
            n = ema * n + (1.0 - ema) * prev
        prev = n
        # Spatial smoothing kills model speckle + quantization banding that the player's
        # depth-gradient shading would otherwise amplify into contour streaks.
        n = cv2.GaussianBlur(n, (5, 5), 1.2)
        # Pin background AFTER the EMA: pinning first leaks (1-ema) of the previous frame's
        # subject depth into background pixels, displacing a ghost skirt around motion.
        n = np.where(m > 0.5, n, 0.0)
        out.append((n * 255.0).astype(np.uint8))
    return out
