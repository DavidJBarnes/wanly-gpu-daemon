"""The hologram frame pass streams from disk instead of holding the clip (wanly-gpu-docker#228).

It used to hold every frame, its matted copy, its alpha and its depth at once -- ~9 MB a frame,
4.4 GiB for 482 frames -- which OOM-killed ComfyUI on 3090b (#192) and would not fit beside the
captioner in the 2070's scene container. These tests pin two things: the packed output is what
the whole-clip version produced (that version is kept below, verbatim, as the reference), and
memory no longer grows with the clip.
"""
import io
import os
import tracemalloc

import cv2
import numpy as np
import pytest
from PIL import Image

from daemon import executor
from daemon.depth import depth_range
from daemon.hologram import (
    GUARD_PX, build_manifest, chroma_key_despill, detect_greenscreen, edge_extend_color,
    pack_sbs, pack_sbs_depth,
)

GREEN = (0, 177, 64)


# ---------------------------------------------------------------- the reference (origin/main)

def _ref_union_alpha_bbox(alphas, thresh=0.5, pad_frac=0.04):
    h, w = alphas[0].shape[:2]
    x0, y0, x1, y1 = w, h, 0, 0
    for a in alphas:
        ys, xs = np.where(a > thresh)
        if xs.size == 0:
            continue
        x0, x1 = min(x0, int(xs.min())), max(x1, int(xs.max()))
        y0, y1 = min(y0, int(ys.min())), max(y1, int(ys.max()))
    if x1 <= x0 or y1 <= y0:
        return (0, 0, w - (w % 2), h - (h % 2))
    pad = int(max(h, w) * pad_frac)
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w - 1, x1 + pad), min(h - 1, y1 + pad)
    cw, ch = (x1 - x0 + 1), (y1 - y0 + 1)
    cw -= cw % 2
    ch -= ch % 2
    return (x0, y0, cw, ch)


def _ref_normalize_depth_clip(depths, alphas, crop_rect, ema=0.85):
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
    out, prev = [], None
    for c, m in zip(cd, ca):
        n = np.clip((c - lo) / rng, 0.0, 1.0)
        if prev is not None:
            n = ema * n + (1.0 - ema) * prev
        prev = n
        n = cv2.GaussianBlur(n, (5, 5), 1.2)
        n = np.where(m > 0.5, n, 0.0)
        out.append((n * 255.0).astype(np.uint8))
    return out


def _reference(frame_files, packed_dir, key_color, subject_height_m, fps, flavor, depth_fn):
    """origin/main's _holo_process_frames_sync, chroma path (RVM needs a model download)."""
    arrs = [np.asarray(Image.open(ff).convert("RGB")) for ff in frame_files]
    assert detect_greenscreen(arrs[0], key_color)
    mode = "chroma"
    rgbs, alphas = [], []
    for arr in arrs:
        rgb, alpha = chroma_key_despill(arr, key_color)
        rgbs.append(rgb)
        alphas.append(alpha)
    x, y, cw, ch = _ref_union_alpha_bbox(alphas)
    depth_u8 = []
    if flavor == "2.5d_depth":
        mode += "+depth"
        raw = [depth_fn(arr, "", alpha=a) for arr, a in zip(arrs, alphas)]
        depth_u8 = _ref_normalize_depth_clip(raw, alphas, (x, y, cw, ch))
    packed_w = packed_h = 0
    for i, (rgb, alpha) in enumerate(zip(rgbs, alphas)):
        crgb = edge_extend_color(rgb[y:y + ch, x:x + cw], alpha[y:y + ch, x:x + cw])
        acrop = alpha[y:y + ch, x:x + cw]
        packed = (pack_sbs_depth(crgb, acrop, depth_u8[i]) if flavor == "2.5d_depth"
                  else pack_sbs(crgb, acrop))
        if packed_w == 0:
            packed_h, packed_w = packed.shape[0], packed.shape[1]
        Image.fromarray(packed).save(os.path.join(packed_dir, f"{i:05d}.png"))
    pal = (np.clip(alphas[0][y:y + ch, x:x + cw], 0.0, 1.0) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.dstack([rgbs[0][y:y + ch, x:x + cw], pal]), "RGBA").save(buf, format="PNG")
    manifest = build_manifest(packed_w, packed_h, cw, GUARD_PX, fps, (x, y, cw, ch),
                              subject_height_m, flavor=flavor, depth_scale_m=0.3, has_audio=False)
    return mode, packed_w, packed_h, buf.getvalue(), manifest


# ---------------------------------------------------------------- fixtures

def _fake_depth(arr, _path, alpha=None):
    """Deterministic, frame-dependent, with real spread: a radial bowl plus the frame's red."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bowl = -((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (w * h)
    return (bowl + arr[..., 0].astype(np.float32) / 255.0).astype(np.float32)


def _clip(dirpath, n=12, h=96, w=128, touch_bottom=False):
    """A soft-edged blob moving across a green screen. With touch_bottom the subject runs off
    the bottom edge -- a body cut off by the frame, the common case -- so the crop clamps."""
    os.makedirs(dirpath, exist_ok=True)
    files = []
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for i in range(n):
        cx, cy = 30 + 5 * i, (h - 10 if touch_bottom else h / 2)
        r = np.sqrt((xx - cx) ** 2 + ((yy - cy) * 0.7) ** 2)
        a = np.clip((22 - r) / 6, 0, 1)[..., None]
        subj = np.stack([150 + 4 * i + 0 * xx, 90 + xx / 4, 60 + yy / 3], -1)
        frame = (a * subj + (1 - a) * np.array(GREEN, np.float32)).astype(np.uint8)
        f = os.path.join(dirpath, f"{i + 1:05d}.png")
        Image.fromarray(frame).save(f)
        files.append(f)
    return files


def _run_both(tmp_path, monkeypatch, flavor, **clip):
    monkeypatch.setattr(executor, "estimate_depth", _fake_depth)
    monkeypatch.setattr(executor, "ensure_depth_model", lambda p, u: p)
    files = _clip(str(tmp_path / "frames"), **clip)
    new_dir, ref_dir = tmp_path / "new", tmp_path / "ref"
    new_dir.mkdir()
    ref_dir.mkdir()
    new = executor._holo_process_frames_sync(
        files, str(new_dir), "green", 1.7, "unused.onnx", 24.0, flavor, "", "", 0.3, False)
    ref = _reference(files, str(ref_dir), "green", 1.7, 24.0, flavor, _fake_depth)
    frames = [(np.asarray(Image.open(new_dir / f)), np.asarray(Image.open(ref_dir / f)))
              for f in sorted(os.listdir(ref_dir))]
    return new, ref, frames


# ---------------------------------------------------------------- equivalence

@pytest.mark.parametrize("touch_bottom", [False, True])
def test_2d_matte_output_is_identical(tmp_path, monkeypatch, touch_bottom):
    new, ref, frames = _run_both(tmp_path, monkeypatch, "2d_matte", touch_bottom=touch_bottom)
    assert new[:3] == ref[:3]                   # mode, packed_w, packed_h
    assert new[3] == ref[3]                     # poster bytes
    assert new[4] == ref[4]                     # manifest, crop rect included
    assert len(frames) == 12
    for a, b in frames:
        assert np.array_equal(a, b)


@pytest.mark.parametrize("touch_bottom", [False, True])
def test_25d_depth_output_is_identical(tmp_path, monkeypatch, touch_bottom):
    """The depth range is computed in passes over disk now, and must still be np.percentile's."""
    new, ref, frames = _run_both(tmp_path, monkeypatch, "2.5d_depth", touch_bottom=touch_bottom)
    assert new[0] == ref[0] == "chroma+depth"
    assert new[1:] == ref[1:]
    for a, b in frames:
        assert np.array_equal(a, b)


def test_the_spill_directory_is_removed(tmp_path, monkeypatch):
    _run_both(tmp_path, monkeypatch, "2.5d_depth")
    assert not [p for p in os.listdir(tmp_path) if p.startswith("matted_")]


@pytest.mark.parametrize("seed", range(5))
def test_depth_range_is_np_percentile(seed):
    rng = np.random.default_rng(seed)
    chunks = [rng.normal(size=rng.integers(0, 5000)).astype(np.float32) for _ in range(20)]
    chunks[3] = np.full(700, 0.25, np.float32)              # ties across a bin
    want = np.percentile(np.concatenate(chunks), [2, 98])
    got = depth_range(lambda: iter(chunks))
    assert got == pytest.approx(tuple(want), abs=1e-6)


def test_depth_range_degenerate():
    assert depth_range(lambda: iter([np.array([], np.float32)])) == (0.0, 1.0)
    assert depth_range(lambda: iter([np.full(9, 3.0, np.float32)])) == (3.0, 3.0)


# ---------------------------------------------------------------- memory

def _peak_mb(tmp_path, n):
    files = _clip(str(tmp_path / f"f{n}"), n=n, h=256, w=256)
    out = tmp_path / f"o{n}"
    out.mkdir()
    tracemalloc.start()
    try:
        executor._holo_process_frames_sync(
            files, str(out), "green", 1.7, "unused.onnx", 24.0, "2.5d_depth", "", "", 0.3, False)
        return tracemalloc.get_traced_memory()[1] / 1e6
    finally:
        tracemalloc.stop()


def test_memory_does_not_grow_with_the_clip(tmp_path, monkeypatch):
    """Four times the frames, about the same peak. The whole-clip version held ~10 bytes a
    pixel per frame (0.65 MB a frame here), so 60 frames would have peaked ~30 MB above 15."""
    monkeypatch.setattr(executor, "estimate_depth", _fake_depth)
    monkeypatch.setattr(executor, "ensure_depth_model", lambda p, u: p)
    short, long = _peak_mb(tmp_path, 15), _peak_mb(tmp_path, 60)
    assert long < short + 5, (short, long)
