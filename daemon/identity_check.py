"""Identity QA for Lynx renders: how much does the output still look like the subject?

Samples frames from a finished render, embeds each detected face with InsightFace, and
cosine-compares them against the subject reference embedding. Measurement only — nothing
here gates, retries, or fails a job; the scores are logged and persisted so the
ip_scale/ref_scale calibration can be argued from numbers rather than impressions.

Note on embedding spaces: Lynx *conditions* on facexlib's ArcFace IR-SE50, while we
*measure* with InsightFace buffalo_l. Different spaces, so absolute values are not
comparable to the model card's; they are only meaningful relative to each other across
A/B arms, which is exactly how we use them.
"""

import logging
from typing import Any

import numpy as np

from daemon.config import settings
from daemon.stage_log import log_event

logger = logging.getLogger(__name__)

# Lazily-initialised InsightFace app: model load is ~1s and we reuse it across segments.
_face_app: Any | None = None


def _get_face_app() -> Any | None:
    """Return a prepared InsightFace FaceAnalysis, or None if unavailable.

    insightface is present in the RunPod image but not in every local daemon venv, so an
    import failure downgrades identity QA to "not measured" instead of failing the job.
    """
    global _face_app
    if _face_app is not None:
        return _face_app
    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        logger.warning("insightface unavailable, skipping identity QA: %s", exc)
        return None
    app = FaceAnalysis()
    app.prepare(ctx_id=0, det_size=(640, 640))
    _face_app = app
    return _face_app


def _largest_face_embedding(app: Any, image: "np.ndarray[Any, Any]") -> "np.ndarray[Any, Any] | None":
    """Embed the largest detected face, or None when the frame has no face."""
    faces = app.get(image)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding: "np.ndarray[Any, Any]" = np.asarray(face.embedding, dtype=np.float32)
    return embedding


def cosine_similarity(a: "np.ndarray[Any, Any]", b: "np.ndarray[Any, Any]") -> float:
    """Cosine similarity between two embeddings; 0.0 if either has no magnitude."""
    denom = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def sample_frames(video_path: str, count: int) -> list["np.ndarray[Any, Any]"]:
    """Read ``count`` frames spread evenly across the video (excluding the very ends)."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            logger.warning("identity QA: could not open %s", video_path)
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or count <= 0:
            return []
        # Interior sampling: the first/last frames are the least representative of a clip.
        step = total / (count + 1)
        frames: list["np.ndarray[Any, Any]"] = []
        for i in range(1, count + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(step * i), total - 1))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        return frames
    finally:
        cap.release()


def measure_identity(
    video_path: str,
    subject_image_path: str,
    correlation_id: str,
    sample_count: int | None = None,
) -> dict[str, Any] | None:
    """Cosine-compare sampled output frames against the subject reference.

    Returns a summary dict (also suitable for the job's JSON column), or None when the
    measurement could not run at all (no insightface, unreadable video, faceless subject).
    Never raises: identity QA must not be able to fail an otherwise good render.
    """
    count = settings.lynx_identity_sample_frames if sample_count is None else sample_count
    try:
        app = _get_face_app()
        if app is None:
            return None

        import cv2

        subject_img = cv2.imread(subject_image_path)
        if subject_img is None:
            logger.warning("identity QA: could not read subject image %s", subject_image_path)
            return None
        subject_embedding = _largest_face_embedding(app, subject_img)
        if subject_embedding is None:
            logger.warning("identity QA: no face in subject image %s", subject_image_path)
            return None

        frames = sample_frames(video_path, count)
        scores: list[float] = []
        for frame in frames:
            embedding = _largest_face_embedding(app, frame)
            if embedding is not None:
                scores.append(round(cosine_similarity(subject_embedding, embedding), 4))

        result: dict[str, Any] = {
            "scores": scores,
            "frames_sampled": len(frames),
            "frames_with_face": len(scores),
            "mean": round(sum(scores) / len(scores), 4) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
        }
        log_event(logger, "lynx.identity_qa", correlation_id, **result)
        return result
    except Exception as exc:
        # Deliberately broad: a QA metric must never take down a successful render.
        logger.warning("identity QA failed for %s: %s", video_path, exc, exc_info=True)
        return None
