"""Best-face crop from a clip — the canonical VACE identity reference.

Pure/CPU (cv2 Haar + ffmpeg), no ComfyUI. Used to pull one clean, frontal face crop out of
seg0 so every downstream VACE continuation anchors identity to the *same* face (no drift).
cv2's bundled Haar frontal detector keeps this dependency-free; if it finds nothing the caller
falls back to the job's start image.
"""
from __future__ import annotations

import glob
import io
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
from PIL import Image

_CASCADE = None


def _cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _CASCADE


def extract_best_face(
    video_bytes: bytes, pad_frac: float = 0.4, sample_fps: int = 4, max_frames: int = 24
) -> bytes | None:
    """Return a padded PNG crop of the best frontal face in the clip, or None if none found.

    Samples frames at `sample_fps`, detects frontal faces, and scores each by area * sqrt(sharpness)
    (Laplacian variance) so it prefers a large, in-focus, front-facing face. Crops with `pad_frac`
    headroom around the detection box.
    """
    tmpdir = tempfile.mkdtemp(prefix="face_")
    try:
        vpath = os.path.join(tmpdir, "v.mp4")
        with open(vpath, "wb") as f:
            f.write(video_bytes)
        frames_dir = os.path.join(tmpdir, "f")
        os.makedirs(frames_dir)
        subprocess.run(
            ["ffmpeg", "-i", vpath, "-vf", f"fps={sample_fps}", "-vsync", "0",
             os.path.join(frames_dir, "%04d.png")],
            check=True, capture_output=True,
        )
        files = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        if len(files) > max_frames:
            files = files[:: max(1, len(files) // max_frames)][:max_frames]

        cas = _cascade()
        best_score = -1.0
        best_crop: np.ndarray | None = None
        for fp in files:
            img = cv2.imread(fp)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))
            for (x, y, w, h) in faces:
                sharp = float(cv2.Laplacian(gray[y:y + h, x:x + w], cv2.CV_64F).var())
                score = (w * h) * (sharp ** 0.5)
                if score > best_score:
                    px, py = int(w * pad_frac), int(h * pad_frac)
                    x0, y0 = max(0, x - px), max(0, y - py)
                    x1, y1 = min(img.shape[1], x + w + px), min(img.shape[0], y + h + py)
                    best_score = score
                    best_crop = img[y0:y1, x0:x1].copy()

        if best_crop is None:
            return None
        rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        return buf.getvalue()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
