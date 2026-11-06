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
from typing import Callable, Iterable

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
        from daemon.hologram import ort_session

        # Prefer GPU; onnxruntime silently falls back to CPU when CUDA isn't available.
        _SESSIONS[model_path] = ort_session(
            model_path, ["CUDAExecutionProvider", "CPUExecutionProvider"]
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


def depth_range(
    chunks: Callable[[], Iterable[np.ndarray]], q_lo: float = 2.0, q_hi: float = 98.0,
    bins: int = 4096,
) -> tuple[float, float]:
    """The clip's robust depth range: the 2nd/98th percentile of its subject pixels (alpha>0.5).

    Exactly np.percentile's (linear) answer over all of them concatenated, without ever
    concatenating them -- that is ~0.8 MB a frame at 720x1056, held at once (wanly-gpu-docker#228).
    `chunks` is called once per pass and yields each frame's subject values: pass 1 counts and
    finds the extremes, pass 2 histograms them to find which bin each wanted rank falls in,
    pass 3 keeps only the values in those bins and reads the ranks off them.
    """
    n, lo, hi = 0, np.inf, -np.inf
    for c in chunks():
        if c.size:
            n += c.size
            lo, hi = min(lo, float(c.min())), max(hi, float(c.max()))
    if n == 0:
        return 0.0, 1.0
    if hi == lo:
        return lo, hi

    def bin_of(c: np.ndarray) -> np.ndarray:
        return np.minimum(((c.astype(np.float64) - lo) / (hi - lo) * bins).astype(np.int64),
                          bins - 1)

    counts = np.zeros(bins, dtype=np.int64)
    for c in chunks():
        if c.size:
            counts += np.bincount(bin_of(c), minlength=bins)
    cum = np.cumsum(counts)

    positions = [q / 100.0 * (n - 1) for q in (q_lo, q_hi)]
    ranks = sorted({r for pos in positions for r in (int(np.floor(pos)), int(np.ceil(pos)))})
    rank_bin = {r: int(np.searchsorted(cum, r, side="right")) for r in ranks}
    kept: dict[int, list[np.ndarray]] = {b: [] for b in rank_bin.values()}
    for c in chunks():
        if c.size:
            cb = bin_of(c)
            for b in kept:
                kept[b].append(c[cb == b])
    in_bin = {b: np.sort(np.concatenate(v)) for b, v in kept.items()}

    def at(r: int) -> float:
        b = rank_bin[r]
        return float(in_bin[b][r - (int(cum[b - 1]) if b else 0)])

    out = []
    for pos in positions:
        f, c = int(np.floor(pos)), int(np.ceil(pos))
        out.append(at(f) + (at(c) - at(f)) * (pos - f))
    return out[0], out[1]


class DepthNormalizer:
    """Turns one cropped raw depth at a time into uint8 (bright = near), in clip order.

    One robust per-clip range (depth_range) so the surface doesn't breathe frame-to-frame,
    background pinned to far (0), and a light temporal EMA carried between frames to damp
    residual flicker -- so one instance per clip, fed in order. Per frame so the streaming
    hologram pass never holds every depth map at once (wanly-gpu-docker#228).
    """

    def __init__(self, lo: float, hi: float, ema: float = 0.85):
        self.lo = lo
        self.rng = max(hi - lo, 1e-6)
        self.ema = ema
        self._prev: np.ndarray | None = None

    def step(self, depth_crop: np.ndarray, alpha_crop: np.ndarray) -> np.ndarray:
        n = np.clip((depth_crop - self.lo) / self.rng, 0.0, 1.0)  # larger raw -> brighter -> nearer
        if self._prev is not None:
            # Light EMA only: a heavier blend makes depth visibly lag the color frames,
            # which reads as waves rippling across the subject during motion.
            n = self.ema * n + (1.0 - self.ema) * self._prev
        self._prev = n
        # Spatial smoothing kills model speckle + quantization banding that the player's
        # depth-gradient shading would otherwise amplify into contour streaks.
        n = cv2.GaussianBlur(n, (5, 5), 1.2)
        # Pin background AFTER the EMA: pinning first leaks (1-ema) of the previous frame's
        # subject depth into background pixels, displacing a ghost skirt around motion.
        n = np.where(alpha_crop > 0.5, n, 0.0)
        return (n * 255.0).astype(np.uint8)
