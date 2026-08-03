"""Score a generated segment for facial identity, per frame.

Answers two different questions, because collapsing them into one number has already sent
this project down a wrong path:

  mean_cos_start  cosine vs the START FRAME  -> how far did THIS generation drift
  mean_cos_ref    cosine vs the identity ref -> is it the character at all
  slope           cosine regressed on frame index -> does it decay, and how fast

A low mean with a flat slope is a weak-identity problem (dataset / LoRA / bucket size).
A good mean with a steep slope is temporal drift (segment boundaries, seed re-anchor).
The fixes are completely different, so both are reported.

Reference point from manual analysis of experiment/real_CTRL.mp4, used as the ground-truth
check on this module: 73 frames, NO_FACE 0, 0.811 vs start frame, 0.489 vs kellydw.

CPU only - the GPU is busy generating. Never raises: a scoring bug must not fail a
30-minute generation.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Beyond this many frames we stride instead of reading every frame, so a long stitched
# segment cannot quietly add minutes to every generation.
MAX_FRAMES_DENSE = 300
# insightface's detector input. 640 matches what the gate and all prior manual analysis
# used; changing it would make new numbers incomparable with the existing measurements.
DET_SIZE = (640, 640)

_app = None


def _get_app():
    """Load buffalo_l once per process. ~300MB on first ever run (downloads to
    ~/.insightface), then cached on disk."""
    global _app
    if _app is None:
        from insightface.app import FaceAnalysis
        logger.info("identity: loading buffalo_l (first run downloads ~300MB)")
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=DET_SIZE)
        _app = app
    return _app


def prewarm() -> bool:
    """Load the model at daemon start so the first scored segment isn't mysteriously slow."""
    try:
        _get_app()
        return True
    except ModuleNotFoundError as e:
        logger.warning(
            "identity: %s is not installed in the daemon venv - scoring will be SKIPPED for "
            "every segment. Fix: venv/bin/pip install insightface==0.7.3", e.name,
        )
        return False
    except Exception as e:
        logger.warning("identity: prewarm failed (%s) - scoring will be skipped", e)
        return False


@dataclass
class IdentityScore:
    frames: int                      # frames examined
    faces_detected: int
    no_face: int                     # counted, never silently dropped
    mean_cos_start: Optional[float]
    mean_cos_ref: Optional[float]
    min_cos_start: Optional[float]
    slope: Optional[float]           # cosine per frame, vs whichever ref was available
    face_px_p50: Optional[float]
    yaw_max: Optional[float]
    metrics: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        bits = []
        if self.mean_cos_start is not None:
            bits.append(f"{self.mean_cos_start:.3f} vs start")
        if self.mean_cos_ref is not None:
            bits.append(f"{self.mean_cos_ref:.3f} vs ref")
        if self.slope is not None:
            bits.append(f"drift {self.slope * max(self.frames - 1, 1):+.3f} over {self.frames}f")
        if self.no_face:
            bits.append(f"NO_FACE {self.no_face}")
        return ", ".join(bits) or "no measurable frames"


def _embed(app, img) -> list:
    """All faces in a frame, as (embedding, face) pairs."""
    try:
        return [(f.normed_embedding, f) for f in app.get(img)]
    except Exception:
        return []


def _cos(a, b) -> float:
    return float(np.dot(a, b))          # insightface embeddings are already L2-normalised


def _pick_reference_face(app, image_bytes: bytes, label: str):
    """Largest face in a reference still. Reference images are single-subject by
    construction, so largest is correct here - unlike video frames."""
    import cv2
    arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        logger.info("identity: %s could not be decoded", label)
        return None
    faces = _embed(app, arr)
    if not faces:
        logger.info("identity: no face found in %s", label)
        return None
    return max(faces, key=lambda t: t[1].bbox[2] - t[1].bbox[0])[0]


def score_video(
    video_bytes: bytes,
    start_frame_bytes: Optional[bytes] = None,
    reference_bytes: Optional[bytes] = None,
) -> tuple[Optional[IdentityScore], str]:
    """Score a segment.

    Returns (score, reason). `reason` is "" on success and otherwise says WHY scoring did
    not run - a missing dependency, an unreadable reference and an empty video are very
    different problems, and collapsing them into a single "not scored" message sent us
    looking for a missing reference when insightface simply was not installed.

    Never raises.
    """
    if not video_bytes:
        return None, "no video data"
    if not start_frame_bytes and not reference_bytes:
        logger.info("identity: no reference available - skipping")
        return None, "no reference image on the segment"

    tmp = None
    try:
        import cv2
        app = _get_app()

        emb_start = _pick_reference_face(app, start_frame_bytes, "start frame") if start_frame_bytes else None
        emb_ref = _pick_reference_face(app, reference_bytes, "identity reference") if reference_bytes else None
        if emb_start is None and emb_ref is None:
            logger.info("identity: no usable reference face - skipping")
            return None, "no face detected in the reference image"
        # the embedding used to disambiguate WHICH face is the subject on multi-face frames
        anchor = emb_ref if emb_ref is not None else emb_start

        fd, tmp = tempfile.mkstemp(suffix=".mp4", prefix="idscore_")
        with os.fdopen(fd, "wb") as fh:
            fh.write(video_bytes)

        cap = cv2.VideoCapture(tmp)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, -(-total // MAX_FRAMES_DENSE)) if total > MAX_FRAMES_DENSE else 1

        idx = 0
        examined = no_face = 0
        cs_start: list[float] = []
        cs_ref: list[float] = []
        slope_x: list[int] = []
        slope_y: list[float] = []
        face_px: list[float] = []
        yaws: list[float] = []
        multi_face = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            examined += 1
            faces = _embed(app, frame)
            if not faces:
                no_face += 1
                idx += 1
                continue
            if len(faces) > 1:
                multi_face += 1
            # Pick the face that matches the subject, NOT the largest. On a two-person
            # frame the largest face is frequently the other person.
            emb, f = max(faces, key=lambda t: _cos(t[0], anchor))
            if emb_start is not None:
                cs_start.append(_cos(emb, emb_start))
            if emb_ref is not None:
                cs_ref.append(_cos(emb, emb_ref))
            primary = cs_start[-1] if emb_start is not None else cs_ref[-1]
            slope_x.append(idx)
            slope_y.append(primary)
            face_px.append(float(f.bbox[2] - f.bbox[0]))
            yaws.append(abs(float(f.pose[1])))
            idx += 1
        cap.release()

        detected = examined - no_face
        if detected == 0:
            logger.info("identity: no face in any of %d frames examined", examined)
            return IdentityScore(
                frames=examined, faces_detected=0, no_face=no_face,
                mean_cos_start=None, mean_cos_ref=None, min_cos_start=None,
                slope=None, face_px_p50=None, yaw_max=None,
                metrics={"stride": stride, "total_frames": total},
            ), ""

        slope = None
        if len(slope_x) >= 3:
            slope = float(np.polyfit(np.array(slope_x, float), np.array(slope_y, float), 1)[0])

        # cosine by |yaw| band - identity legitimately falls off with pose, so a low overall
        # mean on a profile-heavy clip means something different than on a frontal one
        bands: dict[str, dict[str, float]] = {}
        ys = np.array(yaws)
        prim = np.array(slope_y)
        for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90)):
            sel = prim[(ys >= lo) & (ys < hi)]
            if sel.size:
                bands[f"{lo}-{hi}"] = {"n": int(sel.size), "mean": float(sel.mean()),
                                       "min": float(sel.min())}

        return IdentityScore(
            frames=examined,
            faces_detected=detected,
            no_face=no_face,
            mean_cos_start=float(np.mean(cs_start)) if cs_start else None,
            mean_cos_ref=float(np.mean(cs_ref)) if cs_ref else None,
            min_cos_start=float(np.min(cs_start)) if cs_start else None,
            slope=slope,
            face_px_p50=float(np.percentile(face_px, 50)),
            yaw_max=float(np.max(ys)),
            metrics={
                "stride": stride,
                "total_frames": total,
                "multi_face_frames": multi_face,
                "face_px_min": float(np.min(face_px)),
                "face_px_max": float(np.max(face_px)),
                "yaw_bands": bands,
                "series": [round(v, 4) for v in slope_y[:600]],
            },
        ), ""
    except ModuleNotFoundError as e:
        # The one that actually bit: insightface missing from the daemon venv. Say so
        # plainly instead of blaming the reference image.
        logger.warning("identity: dependency missing (%s) - scoring skipped", e)
        return None, f"dependency missing: {e.name}"
    except Exception as e:
        logger.warning("identity: scoring failed (%s) - segment unaffected", e, exc_info=True)
        return None, f"error: {type(e).__name__}: {e}"
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def as_result_fields(score: Optional[IdentityScore]) -> dict[str, Any]:
    """Flatten into the SegmentResult field names, or empty when scoring did not run."""
    if score is None:
        return {}
    d = asdict(score)
    return {
        "identity_mean_cos": d["mean_cos_start"],
        "identity_mean_cos_ref": d["mean_cos_ref"],
        "identity_min_cos": d["min_cos_start"],
        "identity_slope": d["slope"],
        "identity_frames": d["frames"],
        "identity_no_face": d["no_face"],
        "identity_face_px_p50": d["face_px_p50"],
        "identity_yaw_max": d["yaw_max"],
        "identity_metrics": d["metrics"],
    }
