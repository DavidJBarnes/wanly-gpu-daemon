"""Tests for per-segment identity scoring.

The behaviours pinned here are the ones that were learned the hard way elsewhere in this
codebase and would silently produce wrong numbers if they regressed:

  - the SUBJECT's face is measured, not the largest face. On a two-person frame the largest
    face is frequently the other person (the same trap gate.py fell into).
  - undetected frames are COUNTED, not skipped. Silently dropping them inflates the mean:
    a clip where the face vanishes half the time would otherwise score the same as one
    where it never does.
  - scoring NEVER raises. It runs at the end of a ~30-minute generation that already
    succeeded; a scoring bug must not fail that segment.
  - two means are reported, not one. Collapsing "drift from start" and "is it the character"
    into a single figure loses the distinction the whole feature exists to expose.
"""

import numpy as np
import pytest

from daemon import identity_score
from daemon.identity_score import IdentityScore, as_result_fields, score_video


class FakeFace:
    def __init__(self, emb, x0=0.0, x1=100.0, yaw=0.0):
        self.normed_embedding = np.asarray(emb, dtype=np.float32)
        self.bbox = np.array([x0, 0.0, x1, x1 - x0], dtype=np.float32)
        self.pose = np.array([0.0, yaw, 0.0], dtype=np.float32)


def unit(*v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


SUBJECT = unit(1, 0, 0)
OTHER = unit(0, 1, 0)          # orthogonal -> cosine 0 against the subject
DRIFTED = unit(1, 1, 0)        # ~0.707 against the subject


class FakeApp:
    """Returns a scripted list of faces per successive call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, _img):
        i = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[i]


@pytest.fixture
def patched(monkeypatch):
    """Drive score_video's frame loop without needing a real video or the real model."""
    def install(frame_faces, ref_faces, n_frames=None):
        script = [ref_faces] + list(frame_faces)
        app = FakeApp(script)
        monkeypatch.setattr(identity_score, "_get_app", lambda: app)

        import cv2
        monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: np.zeros((8, 8, 3), np.uint8))

        frames = list(frame_faces)
        total = n_frames if n_frames is not None else len(frames)
        state = {"i": 0}

        class FakeCap:
            def __init__(self, *a, **k): pass
            def get(self, _p): return total
            def read(self):
                if state["i"] >= len(frames):
                    return False, None
                state["i"] += 1
                return True, np.zeros((8, 8, 3), np.uint8)
            def release(self): pass

        monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
        return app
    return install


class TestPicksTheSubjectNotTheLargestFace:
    def test_two_person_frame_measures_the_subject(self, patched):
        """The other person's face is deliberately much larger. Taking the largest would
        score ~0.0 and report catastrophic identity loss on a perfectly good clip."""
        big_other = FakeFace(OTHER, x0=0, x1=400)
        small_subject = FakeFace(SUBJECT, x0=500, x1=560)
        patched(frame_faces=[[big_other, small_subject]] * 4,
                ref_faces=[FakeFace(SUBJECT, x0=0, x1=200)])

        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s is not None
        assert s.mean_cos_ref == pytest.approx(1.0, abs=1e-4)
        assert s.metrics["multi_face_frames"] == 4

    def test_single_face_frames_still_work(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 3, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.mean_cos_ref == pytest.approx(1.0, abs=1e-4)
        assert s.metrics["multi_face_frames"] == 0


class TestUndetectedFramesAreCounted:
    def test_no_face_frames_are_reported_not_dropped(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)], [], [FakeFace(SUBJECT)], []],
                ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.frames == 4
        assert s.faces_detected == 2
        assert s.no_face == 2

    def test_all_frames_missing_returns_a_score_not_none(self, patched):
        """'No face anywhere' is a real result about the clip, distinct from 'scoring
        could not run'. Returning None for both would conflate them."""
        patched(frame_faces=[[], [], []], ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s is not None
        assert s.faces_detected == 0 and s.no_face == 3
        assert s.mean_cos_ref is None


class TestTwoReferences:
    def test_start_and_reference_scored_separately(self, patched):
        """A clip can sit close to its start frame while being far from the character -
        exactly what the CTRL clip did (0.811 vs start, 0.489 vs kellydw)."""
        patched(frame_faces=[[FakeFace(DRIFTED)]] * 4,
                ref_faces=[FakeFace(DRIFTED)])   # reference-face lookup returns this for both

        s, _ = score_video(b"video", start_frame_bytes=b"start", reference_bytes=b"ref")
        assert s.mean_cos_start is not None
        assert s.mean_cos_ref is not None

    def test_reference_only_still_scores(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 3, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.mean_cos_ref is not None and s.mean_cos_start is None

    def test_no_reference_at_all_returns_none(self):
        score, reason = score_video(b"video")
        assert score is None
        assert "no reference" in reason


class TestSlope:
    def test_degrading_identity_gives_a_negative_slope(self, patched):
        """The diagnostic half of the feature: a good mean with a steep slope is temporal
        drift, which needs a different fix than a uniformly low mean."""
        decaying = [[FakeFace(unit(1, t * 0.25, 0))] for t in range(8)]
        patched(frame_faces=decaying, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.slope is not None and s.slope < 0

    def test_stable_identity_gives_a_flat_slope(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 8, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.slope == pytest.approx(0.0, abs=1e-6)


class TestNeverRaises:
    def test_empty_video_returns_none(self):
        score, reason = score_video(b"", reference_bytes=b"ref")
        assert score is None
        assert reason == "no video data"

    def test_model_load_failure_returns_none(self, monkeypatch):
        def boom(): raise RuntimeError("onnxruntime exploded")
        monkeypatch.setattr(identity_score, "_get_app", boom)
        score, reason = score_video(b"video", reference_bytes=b"ref")
        assert score is None
        assert "error:" in reason or "dependency missing" in reason

    def test_unreadable_reference_returns_none(self, monkeypatch):
        monkeypatch.setattr(identity_score, "_get_app", lambda: FakeApp([[]]))
        import cv2
        monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: None)
        score, reason = score_video(b"video", reference_bytes=b"notanimage")
        assert score is None
        assert "reference" in reason

    def test_prewarm_failure_is_survivable(self, monkeypatch):
        def boom(): raise RuntimeError("no model")
        monkeypatch.setattr(identity_score, "_get_app", boom)
        assert identity_score.prewarm() is False


class TestResultFields:
    def test_none_flattens_to_empty(self):
        assert as_result_fields(None) == {}

    def test_field_names_match_the_segment_result_schema(self):
        from daemon.schemas import SegmentResult
        score = IdentityScore(
            frames=10, faces_detected=9, no_face=1, mean_cos_start=0.81,
            mean_cos_ref=0.49, min_cos_start=0.7, slope=-0.001,
            face_px_p50=110.0, yaw_max=28.0, start_cos_ref=0.98, end_cos_ref=0.60,
            metrics={"stride": 1},
        )
        fields = as_result_fields(score)
        unknown = set(fields) - set(SegmentResult.model_fields)
        assert not unknown, f"fields not on SegmentResult would be dropped silently: {unknown}"
        assert SegmentResult(status="completed", **fields).identity_mean_cos == 0.81


class TestStride:
    def test_long_clips_are_strided(self, patched):
        """Bounds the cost so a long stitched segment cannot add minutes to every job."""
        n = identity_score.MAX_FRAMES_DENSE * 3
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 30, ref_faces=[FakeFace(SUBJECT)], n_frames=n)
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.metrics["stride"] > 1
        assert s.metrics["total_frames"] == n


class TestFailureReasonIsSpecific:
    """The bug this class exists for: a missing insightface reported as
    "not scored (no usable reference)", which sent us looking for a missing reference image
    when the real cause was a dependency that was never installed. Six different failures
    were collapsing into one message."""

    def test_missing_dependency_says_so_by_name(self, monkeypatch):
        def boom():
            raise ModuleNotFoundError("No module named 'insightface'", name="insightface")
        monkeypatch.setattr(identity_score, "_get_app", boom)
        score, reason = score_video(b"video", reference_bytes=b"ref")
        assert score is None
        assert "dependency missing" in reason and "insightface" in reason

    def test_no_reference_is_distinguishable_from_a_bad_one(self, monkeypatch):
        _, none_at_all = score_video(b"video")
        monkeypatch.setattr(identity_score, "_get_app", lambda: FakeApp([[]]))
        import cv2
        monkeypatch.setattr(cv2, "imdecode", lambda *a, **k: None)
        _, unreadable = score_video(b"video", reference_bytes=b"junk")
        assert none_at_all != unreadable, "distinct failures must not share a message"

    def test_success_has_an_empty_reason(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 3, ref_faces=[FakeFace(SUBJECT)])
        score, reason = score_video(b"video", reference_bytes=b"ref")
        assert score is not None and reason == ""


class TestTrajectory:
    """The mean blurs the shape. A segment going 0.95 -> 0.65 averages about the same as one
    sitting flat at 0.80, and only the first has lost the character. Loss across a segment is
    start - end against the job's ground truth; because a continuation begins where the
    previous ended, these chain across the whole job."""

    def test_endpoints_are_recorded_against_the_reference(self, patched):
        decaying = [[FakeFace(unit(1, t * 0.3, 0))] for t in range(6)]
        patched(frame_faces=decaying, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.start_cos_ref is not None and s.end_cos_ref is not None
        assert s.start_cos_ref > s.end_cos_ref, "a decaying clip must end lower than it started"

    def test_loss_is_start_minus_end(self, patched):
        decaying = [[FakeFace(unit(1, t * 0.3, 0))] for t in range(6)]
        patched(frame_faces=decaying, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.loss == pytest.approx(s.start_cos_ref - s.end_cos_ref)
        assert s.loss > 0, "positive loss means identity was lost"

    def test_a_stable_clip_has_near_zero_loss(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 6, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", reference_bytes=b"ref")
        assert s.loss == pytest.approx(0.0, abs=1e-6)

    def test_mean_alone_cannot_distinguish_the_two(self, patched):
        """Why endpoints exist: build a decaying clip and a flat clip with a similar mean and
        confirm only the trajectory separates them."""
        decaying = [[FakeFace(unit(1, t * 0.22, 0))] for t in range(8)]
        patched(frame_faces=decaying, ref_faces=[FakeFace(SUBJECT)])
        drift, _ = score_video(b"video", reference_bytes=b"ref")

        flat_val = drift.mean_cos_ref
        flat_emb = unit(1, (1 / flat_val ** 2 - 1) ** 0.5, 0)
        patched(frame_faces=[[FakeFace(flat_emb)]] * 8, ref_faces=[FakeFace(SUBJECT)])
        flat, _ = score_video(b"video", reference_bytes=b"ref")

        assert flat.mean_cos_ref == pytest.approx(drift.mean_cos_ref, abs=0.02)
        assert abs(flat.loss) < 1e-6 < drift.loss, "means match; only loss tells them apart"

    def test_loss_is_none_without_a_reference(self, patched):
        patched(frame_faces=[[FakeFace(SUBJECT)]] * 3, ref_faces=[FakeFace(SUBJECT)])
        s, _ = score_video(b"video", start_frame_bytes=b"start")
        assert s.loss is None


class TestGroundTruthIsSegmentZeroStartFrame:
    """"Her" is defined by where the job began — segment 0's start frame — and by nothing
    else. No external reference, no override, nothing hardcoded.

    Scoring previously piggybacked on `initial_reference_image`, which resolves to
    `job.identity_reference_image or job.starting_image`. That field exists for the
    PainterLongVideo anchor and is overridable, so setting it would have silently changed
    what the numbers were measured against, partway through a job.
    """

    def test_claim_carries_an_explicit_ground_truth_field(self):
        from daemon.schemas import SegmentClaim
        assert "identity_ground_truth" in SegmentClaim.model_fields

    def test_ground_truth_is_independent_of_the_painter_anchor(self):
        """The two fields must be separately settable, or one can shadow the other."""
        from tests.conftest import make_segment
        seg = make_segment(
            identity_ground_truth="s3://bucket/seg0_start.png",
            initial_reference_image="s3://bucket/some_other_anchor.png",
        )
        assert seg.identity_ground_truth != seg.initial_reference_image

    def test_scoring_never_reads_the_painter_anchor(self):
        """Source-level guard: the scoring helper must not reference the overridable field.
        A future edit that reintroduces the coupling fails here rather than silently
        changing what every number means."""
        import ast as _ast, inspect
        from daemon import executor
        raw = inspect.getsource(executor._score_segment_identity)
        # Strip comments and docstrings: the helper deliberately NAMES the anchor field in a
        # comment explaining why it is not used, and that must not trip the guard.
        tree = _ast.parse(raw.lstrip())
        for node in _ast.walk(tree):
            if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)) \
                    and _ast.get_docstring(node):
                node.body = node.body[1:]
        src = _ast.unparse(tree)
        assert "initial_reference_image" not in src, (
            "identity scoring must use identity_ground_truth, not the overridable "
            "PainterLongVideo anchor"
        )
        assert "identity_ground_truth" in src

    def test_no_hardcoded_identity_face_map_remains(self):
        from daemon import executor
        assert not hasattr(executor, "_HARDCODE_IDENTITY_FACE")


class TestFaceDetail:
    """Cosine is largely blur-invariant, so it cannot see a face going soft.

    On a real 2x5s chain, face detail fell 233 -> 79 (66%) while identity moved only
    0.901 -> 0.870. David could see it; the metric could not. Detail is measured as Laplacian
    variance of the face crop resized to a fixed 128x128, so it tracks sharpness rather than
    how big the face happens to be in frame."""

    def test_score_exposes_detail_fields(self):
        from daemon.identity_score import IdentityScore
        f = IdentityScore.__dataclass_fields__
        for name in ("face_sharp_mean", "face_sharp_start", "face_sharp_end"):
            assert name in f, name
            assert f[name].default is None, f"{name} must default None - older clips have none"

    def test_summary_reports_detail_when_present(self):
        from daemon.identity_score import IdentityScore
        s = IdentityScore(
            frames=10, faces_detected=10, no_face=0, mean_cos_start=0.9, mean_cos_ref=0.9,
            min_cos_start=0.8, slope=0.0, face_px_p50=100.0, yaw_max=3.0,
            start_cos_ref=0.95, end_cos_ref=0.90,
            face_sharp_mean=150.0, face_sharp_start=233.0, face_sharp_end=79.0,
        )
        assert "detail 233->79" in s.summary()

    def test_summary_omits_detail_when_unmeasured(self):
        """Faces under 40px are skipped, so a clip can legitimately have no detail reading."""
        from daemon.identity_score import IdentityScore
        s = IdentityScore(
            frames=10, faces_detected=10, no_face=0, mean_cos_start=0.9, mean_cos_ref=0.9,
            min_cos_start=0.8, slope=0.0, face_px_p50=100.0, yaw_max=3.0,
            start_cos_ref=0.95, end_cos_ref=0.90,
        )
        assert "detail" not in s.summary()
