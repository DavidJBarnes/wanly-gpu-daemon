"""AR hologram (Tier-0) matte + pack helpers.

Pure functions (numpy / cv2 / PIL), no ComfyUI. The daemon's `_execute_ar_hologram`
orchestrates these over the frames of a finalized clip to produce a packed color+alpha
mp4 + manifest + transparent poster.

v1 is chroma-key only (green screen). RVM matting for arbitrary backgrounds is a deferred
drop-in: implement `rvm_matte(frames) -> alphas` and branch in the executor.
"""
from __future__ import annotations

import cv2
import numpy as np

# Green-excess soft-key thresholds (on the 0..1 green-excess scale). Tunable.
KEY_T_LOW = 0.04   # excess <= this -> fully subject (alpha 1)
KEY_T_HIGH = 0.24  # excess >= this -> fully background (alpha 0)
GUARD_PX = 8       # black gutter between color and alpha regions in the packed frame


def parse_key_color(key_color: str | None) -> tuple[int, int, int]:
    """'0x00b140' / '#00b140' / 'green' / 'magenta' -> (r, g, b) 0..255."""
    if not key_color:
        return (0, 177, 64)
    s = key_color.strip().lower()
    if s == "green":
        return (0, 177, 64)
    if s in ("magenta", "pink"):
        return (177, 0, 177)
    s = s.replace("0x", "").replace("#", "")
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except Exception:
        return (0, 177, 64)


def _excess(f: np.ndarray, key_rgb: tuple[int, int, int]) -> np.ndarray:
    """Key-channel excess (high on the key background, ~0 on the subject). f is float RGB 0..1."""
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    kr, kg, kb = key_rgb
    if kg >= kr and kg >= kb:            # green key
        return g - np.maximum(r, b)
    return np.minimum(r, b) - g          # magenta / other (key is r+b high, g low)


def detect_greenscreen(frame_rgb: np.ndarray, key_color: str = "green", frac: float = 0.6) -> bool:
    """True if the outer border ring is dominantly the key color (a clean chroma plate)."""
    h, w = frame_rgb.shape[:2]
    b = max(2, int(min(h, w) * 0.04))
    ring = np.concatenate([
        frame_rgb[:b, :, :].reshape(-1, 3),
        frame_rgb[-b:, :, :].reshape(-1, 3),
        frame_rgb[:, :b, :].reshape(-1, 3),
        frame_rgb[:, -b:, :].reshape(-1, 3),
    ]).astype(np.float32) / 255.0
    excess = _excess(ring.reshape(1, -1, 3), parse_key_color(key_color)).reshape(-1)
    return float((excess > 0.15).mean()) >= frac


def chroma_key_despill(frame_rgb: np.ndarray, key_color: str) -> tuple[np.ndarray, np.ndarray]:
    """Green-excess soft key + despill.

    Returns (despilled RGB uint8, straight alpha float32 0..1). Soft edges (not a hard
    threshold) are where matte quality is won; despill clamps the key channel toward the
    other two to kill colored fringing on hair/edges.
    """
    key_rgb = parse_key_color(key_color)
    f = frame_rgb.astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    excess = _excess(f, key_rgb)
    alpha = np.clip((KEY_T_HIGH - excess) / (KEY_T_HIGH - KEY_T_LOW), 0.0, 1.0).astype(np.float32)

    kr, kg, kb = key_rgb
    if kg >= kr and kg >= kb:            # despill green fringe
        f[..., 1] = np.minimum(g, np.maximum(r, b))
    else:                               # despill magenta fringe
        m = g
        f[..., 0] = np.minimum(r, np.maximum(g, b)) if kr >= kg else r
        f[..., 2] = np.minimum(b, np.maximum(g, r)) if kb >= kg else b
        _ = m
    rgb = (np.clip(f, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb, alpha


def union_alpha_bbox(alphas: list[np.ndarray], thresh: float = 0.5, pad_frac: float = 0.04) -> tuple[int, int, int, int]:
    """One static crop rect (x, y, w, h) covering the subject across ALL frames.

    A single static crop keeps the textured quad stable in 3D; per-frame crops would make
    it swim. Padded and snapped to even dimensions for h264.
    """
    h, w = alphas[0].shape[:2]
    x0, y0, x1, y1 = w, h, 0, 0
    for a in alphas:
        ys, xs = np.where(a > thresh)
        if xs.size == 0:
            continue
        x0, x1 = min(x0, int(xs.min())), max(x1, int(xs.max()))
        y0, y1 = min(y0, int(ys.min())), max(y1, int(ys.max()))
    if x1 <= x0 or y1 <= y0:            # nothing matted — fall back to full even frame
        return (0, 0, w - (w % 2), h - (h % 2))
    pad = int(max(h, w) * pad_frac)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    cw, ch = (x1 - x0 + 1), (y1 - y0 + 1)
    cw -= cw % 2
    ch -= ch % 2
    return (x0, y0, cw, ch)


def edge_extend_color(rgb: np.ndarray, alpha: np.ndarray, radius: int = 6) -> np.ndarray:
    """Fill the background (alpha<0.5) with edge-extended subject color.

    Straight alpha means the color under alpha=0 is otherwise garbage; extending the
    subject's edge color outward stops 4:2:0 chroma subsampling from bleeding a hard
    background color across the matte edge (dark/colored halos). Shader premultiplies later.
    """
    bg_mask = (alpha < 0.5).astype(np.uint8) * 255
    if int(bg_mask.max()) == 0:
        return rgb
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, bg_mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def pack_sbs(rgb: np.ndarray, alpha: np.ndarray, guard_px: int = GUARD_PX) -> np.ndarray:
    """Pack one frame side-by-side: [ color | black guard | alpha-in-luma ]. Even width."""
    h = rgb.shape[0]
    a8 = (np.clip(alpha, 0.0, 1.0) * 255).astype(np.uint8)
    a3 = np.repeat(a8[:, :, None], 3, axis=2)
    guard = np.zeros((h, guard_px, 3), dtype=np.uint8)
    packed = np.concatenate([rgb, guard, a3], axis=1)
    if packed.shape[1] % 2:
        packed = np.concatenate([packed, np.zeros((h, 1, 3), dtype=np.uint8)], axis=1)
    return packed


def build_manifest(packed_w: int, packed_h: int, color_w: int, guard_px: int, fps: float,
                   crop_rect: tuple[int, int, int, int], subject_height_m: float) -> dict:
    """Player manifest. UV rects (top-left origin, normalized) so the shader is resolution-agnostic."""
    total = float(packed_w)
    return {
        "tier": 0,
        "layout": "sbs_color_alpha",
        "codec": "h264",
        "fps": round(float(fps), 3),
        "video_width": int(packed_w),
        "video_height": int(packed_h),
        "region_color_uv": {"x": 0.0, "y": 0.0, "w": color_w / total, "h": 1.0},
        "region_alpha_uv": {"x": (color_w + guard_px) / total, "y": 0.0, "w": color_w / total, "h": 1.0},
        "guard_px": int(guard_px),
        "crop_rect": {"x": int(crop_rect[0]), "y": int(crop_rect[1]), "w": int(crop_rect[2]), "h": int(crop_rect[3])},
        "subject_px_height": int(crop_rect[3]),
        "subject_height_m": float(subject_height_m),
        "premultiplied": False,
        "alpha_encoding": "luma",
    }
