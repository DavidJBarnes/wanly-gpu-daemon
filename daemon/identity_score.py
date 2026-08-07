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
        from .config import settings

        # Scoring 289 frames of detection + recognition was 255-309s on CPU: 23-29% of every
        # job, and larger than the video encode, the upload and the motion analysis combined.
        #
        # It was pinned to CPU for a good reason — the container's CUDA execution provider could
        # not load at all (wanly-gpu-docker #32, #33), and onnxruntime does not raise when that
        # happens, it just silently uses CPU. Both defects are fixed and verified, so the pin is
        # no longer buying anything on a box where the daemon owns the card.
        #
        # ctx_id matters as much as the providers list: insightface treats -1 as CPU regardless
        # of what onnxruntime was offered, so setting only one of the two changes nothing.
        gpu = settings.identity_scoring_gpu
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if gpu else ["CPUExecutionProvider"]
        )
        logger.info(
            "identity: loading buffalo_l on %s (first run downloads ~300MB)",
            "GPU (CPU fallback)" if gpu else "CPU",
        )
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=0 if gpu else -1, det_size=DET_SIZE)

        # Say which provider it ACTUALLY got. onnxruntime falls back to CPU without raising, so
        # "we asked for CUDA" and "we are running on CUDA" are different claims — and the gap
        # between them is exactly what cost a day on the container.
        try:
            actual = app.models["recognition"].session.get_providers()[0]
            logger.info("identity: recognition running on %s", actual)
            if gpu and actual != "CUDAExecutionProvider":
                import onnxruntime as ort

                have_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
                logger.warning(
                    "identity: asked for CUDA but got %s — scoring stays on CPU. %s",
                    actual,
                    (
                        # Two different causes, and they need different fixes. The 3090's daemon
                        # venv has the CPU-only `onnxruntime` package, which does not contain a
                        # CUDA provider at all; the container has onnxruntime-gpu, where the
                        # failure mode instead is the provider being present but unable to load
                        # its libraries.
                        "The CUDA provider failed to load — check LD_LIBRARY_PATH and that the "
                        "onnxruntime-gpu wheel matches this image's CUDA major."
                        if have_cuda
                        else "This venv has the CPU-only `onnxruntime` package. Install "
                        "`onnxruntime-gpu` (and remove `onnxruntime`, they provide the same "
                        "module) or set IDENTITY_SCORING_GPU=false to silence this."
                    ),
                )
        except Exception:  # pragma: no cover - diagnostic only, never fail scoring over it
            pass

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
    # Endpoints vs the ground truth (the job's start image, identical for every segment).
    # The MEAN blurs the trajectory: a clip going 0.95 -> 0.65 averages about the same as one
    # sitting flat at 0.80. Loss across a segment is start_cos_ref - end_cos_ref, and because
    # a continuation begins where the previous segment ended, these chain across the job.
    start_cos_ref: Optional[float]
    end_cos_ref: Optional[float]
    min_cos_start: Optional[float]
    slope: Optional[float]           # cosine per frame, vs whichever ref was available
    face_px_p50: Optional[float]
    yaw_max: Optional[float]
    # Face DETAIL, which cosine cannot see: ArcFace is largely blur-invariant, so a face can
    # stay unmistakably her while going soft. Measured on a real 2x5s chain, detail fell 233 ->
    # 79 (66%) while identity moved only 0.901 -> 0.870 -- David could see it, the metric could
    # not. Laplacian variance of the face crop, resized to a fixed 128x128 so this tracks
    # sharpness rather than how big the face happens to be.
    face_sharp_mean: Optional[float] = None
    face_sharp_start: Optional[float] = None
    face_sharp_end: Optional[float] = None
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def loss(self) -> Optional[float]:
        """Identity lost across this segment, measured against the job's ground truth."""
        if self.start_cos_ref is None or self.end_cos_ref is None:
            return None
        return self.start_cos_ref - self.end_cos_ref

    def summary(self) -> str:
        bits = []
        if self.start_cos_ref is not None and self.end_cos_ref is not None:
            bits.append(f"{self.start_cos_ref:.3f}->{self.end_cos_ref:.3f} vs truth "
                        f"(loss {self.loss:+.3f})")
        if self.mean_cos_start is not None:
            bits.append(f"{self.mean_cos_start:.3f} vs start")
        if self.mean_cos_ref is not None:
            bits.append(f"{self.mean_cos_ref:.3f} vs ref")
        if self.face_sharp_start is not None and self.face_sharp_end is not None:
            # Surfaced in the one-line log because cosine cannot see it: a face can hold
            # identity while going visibly soft, and that only shows up here.
            bits.append(f"detail {self.face_sharp_start:.0f}->{self.face_sharp_end:.0f}")
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
        sharps: list[float] = []
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
            # Prefer the GROUND TRUTH (the job's start image) over this segment's own start
            # frame. They are the same image on segment 0, but on a continuation the own-start
            # reference is the already-drifted last frame -- so a segment could read as stable
            # while identity had collapsed since the job began. The headline trajectory is vs
            # ground truth, so the series, the slope and the yaw bands must be too or the panel
            # contradicts the numbers printed above it.
            primary = cs_ref[-1] if emb_ref is not None else cs_start[-1]
            slope_x.append(idx)
            slope_y.append(primary)
            face_px.append(float(f.bbox[2] - f.bbox[0]))
            x1, y1, x2, y2 = (max(0, int(v)) for v in f.bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size and (x2 - x1) >= 40:
                g = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (128, 128))
                sharps.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
            yaws.append(abs(float(f.pose[1])))
            idx += 1
        cap.release()

        detected = examined - no_face
        if detected == 0:
            logger.info("identity: no face in any of %d frames examined", examined)
            return IdentityScore(
                frames=examined, faces_detected=0, no_face=no_face,
                mean_cos_start=None, mean_cos_ref=None, min_cos_start=None,
                start_cos_ref=None, end_cos_ref=None,
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
            start_cos_ref=float(cs_ref[0]) if cs_ref else None,
            end_cos_ref=float(cs_ref[-1]) if cs_ref else None,
            min_cos_start=float(np.min(cs_start)) if cs_start else None,
            slope=slope,
            face_px_p50=float(np.percentile(face_px, 50)),
            yaw_max=float(np.max(ys)),
            face_sharp_mean=float(np.mean(sharps)) if sharps else None,
            face_sharp_start=float(np.mean(sharps[:3])) if len(sharps) >= 3 else None,
            face_sharp_end=float(np.mean(sharps[-3:])) if len(sharps) >= 3 else None,
            metrics={
                "stride": stride,
                "total_frames": total,
                "multi_face_frames": multi_face,
                "face_px_min": float(np.min(face_px)),
                "face_px_max": float(np.max(face_px)),
                # Rides in metrics rather than its own columns: identity_metrics is already
                # persisted as JSON, so this needs no migration and no API change.
                "face_sharp_mean": round(float(np.mean(sharps)), 1) if sharps else None,
                "face_sharp_start": round(float(np.mean(sharps[:3])), 1) if len(sharps) >= 3 else None,
                "face_sharp_end": round(float(np.mean(sharps[-3:])), 1) if len(sharps) >= 3 else None,
                "yaw_bands": bands,
                # Which reference `series`, `slope` and `yaw_bands` are measured against.
                # Older clips predate this field and are all vs the segment's own start frame.
                "series_ref": "ground_truth" if emb_ref is not None else "own_start",
                "series": [round(v, 4) for v in slope_y[:600]],
                # Within-segment drift, for contrast: how far the face moved from where THIS
                # segment began, independent of what it inherited.
                "series_own_start": [round(v, 4) for v in cs_start[:600]] if cs_start else None,
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
        "identity_start_cos_ref": d["start_cos_ref"],
        "identity_end_cos_ref": d["end_cos_ref"],
        "identity_metrics": d["metrics"],
    }
