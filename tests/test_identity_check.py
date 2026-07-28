"""Tests for Lynx identity QA.

InsightFace and its ONNX models are not available in CI, so the detector is faked.
What matters here is the arithmetic, the sampling, and — above all — that a QA failure
can never propagate out and fail an otherwise good render.
"""

import json
import logging

import cv2
import numpy as np
import pytest

from daemon import identity_check
from daemon.identity_check import (
    _largest_face_embedding,
    cosine_similarity,
    measure_identity,
    sample_frames,
)

CID = "11111111-2222-3333-4444-555555555555"


class FakeFace:
    def __init__(self, embedding, bbox=(0, 0, 10, 10)):
        self.embedding = np.array(embedding, dtype=np.float32)
        self.bbox = np.array(bbox, dtype=np.float32)


class FakeApp:
    """Stand-in for InsightFace FaceAnalysis: returns queued detections per call."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def get(self, _image):
        self.calls += 1
        if not self._results:
            return []
        return self._results.pop(0)


@pytest.fixture(autouse=True)
def _reset_app_cache():
    identity_check._face_app = None
    yield
    identity_check._face_app = None


@pytest.fixture
def video(tmp_path):
    """A real 20-frame mp4 so sample_frames exercises actual OpenCV decoding."""
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 64))
    for i in range(20):
        writer.write(np.full((64, 64, 3), i * 10 % 256, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def subject(tmp_path):
    path = tmp_path / "subject.png"
    cv2.imwrite(str(path), np.full((64, 64, 3), 128, dtype=np.uint8))
    return path


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(a, -a) == pytest.approx(-1.0)

    def test_zero_vector_does_not_divide_by_zero(self):
        a = np.zeros(3, dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(a, b) == 0.0


class TestLargestFaceEmbedding:
    def test_returns_none_when_no_face(self):
        assert _largest_face_embedding(FakeApp([[]]), np.zeros((8, 8, 3))) is None

    def test_picks_the_largest_bbox(self):
        small = FakeFace([1.0, 0.0], bbox=(0, 0, 5, 5))
        large = FakeFace([0.0, 1.0], bbox=(0, 0, 50, 50))
        app = FakeApp([[small, large]])
        assert _largest_face_embedding(app, np.zeros((8, 8, 3))).tolist() == [0.0, 1.0]


class TestSampleFrames:
    def test_samples_requested_count(self, video):
        assert len(sample_frames(str(video), 5)) == 5

    def test_zero_count_returns_nothing(self, video):
        assert sample_frames(str(video), 0) == []

    def test_unreadable_video_returns_empty(self, tmp_path):
        assert sample_frames(str(tmp_path / "nope.mp4"), 5) == []


class TestMeasureIdentity:
    def test_scores_frames_against_subject(self, monkeypatch, video, subject, caplog):
        subject_face = [FakeFace([1.0, 0.0])]
        # frame 1 identical, frame 2 orthogonal, frame 3 has no face at all
        frames = [[FakeFace([1.0, 0.0])], [FakeFace([0.0, 1.0])], []]
        app = FakeApp([subject_face, *frames])
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: app)

        with caplog.at_level(logging.INFO):
            result = measure_identity(str(video), str(subject), CID, sample_count=3)

        assert result["scores"] == [1.0, 0.0]
        assert result["frames_sampled"] == 3
        assert result["frames_with_face"] == 2
        assert result["mean"] == pytest.approx(0.5)
        assert result["min"] == 0.0 and result["max"] == 1.0

        logged = [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]
        qa = [e for e in logged if e["event"] == "lynx.identity_qa"]
        assert qa and qa[0]["correlation_id"] == CID

    def test_uses_settings_default_sample_count(self, monkeypatch, video, subject):
        from daemon.config import settings
        app = FakeApp([[FakeFace([1.0, 0.0])]] * (settings.lynx_identity_sample_frames + 1))
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: app)
        result = measure_identity(str(video), str(subject), CID)
        assert result["frames_sampled"] == settings.lynx_identity_sample_frames

    def test_no_faces_anywhere_yields_null_stats(self, monkeypatch, video, subject):
        app = FakeApp([[FakeFace([1.0, 0.0])], [], [], []])
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: app)
        result = measure_identity(str(video), str(subject), CID, sample_count=3)
        assert result["scores"] == []
        assert result["mean"] is None and result["min"] is None and result["max"] is None

    def test_returns_none_without_insightface(self, monkeypatch, video, subject):
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: None)
        assert measure_identity(str(video), str(subject), CID) is None

    def test_returns_none_for_unreadable_subject(self, monkeypatch, video, tmp_path):
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: FakeApp([]))
        assert measure_identity(str(video), str(tmp_path / "missing.png"), CID) is None

    def test_returns_none_when_subject_has_no_face(self, monkeypatch, video, subject):
        monkeypatch.setattr(identity_check, "_get_face_app", lambda: FakeApp([[]]))
        assert measure_identity(str(video), str(subject), CID) is None

    def test_never_raises(self, monkeypatch, video, subject):
        """A QA metric must not be able to fail a good render."""
        def explode():
            raise RuntimeError("detector exploded")
        monkeypatch.setattr(identity_check, "_get_face_app", explode)
        assert measure_identity(str(video), str(subject), CID) is None


class TestGetFaceApp:
    def test_missing_insightface_returns_none(self, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("insightface"):
                raise ImportError("no insightface here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert identity_check._get_face_app() is None

    def test_app_is_cached(self, monkeypatch):
        sentinel = object()
        identity_check._face_app = sentinel
        assert identity_check._get_face_app() is sentinel

    def test_builds_and_prepares_face_analysis(self, monkeypatch):
        """insightface is absent locally, so stand a fake module in to cover the happy path."""
        import sys
        import types

        prepared = {}

        class FakeFaceAnalysis:
            def prepare(self, ctx_id, det_size):
                prepared["ctx_id"] = ctx_id
                prepared["det_size"] = det_size

        app_module = types.ModuleType("insightface.app")
        app_module.FaceAnalysis = FakeFaceAnalysis
        root = types.ModuleType("insightface")
        root.app = app_module
        monkeypatch.setitem(sys.modules, "insightface", root)
        monkeypatch.setitem(sys.modules, "insightface.app", app_module)

        app = identity_check._get_face_app()
        assert isinstance(app, FakeFaceAnalysis)
        assert prepared == {"ctx_id": 0, "det_size": (640, 640)}
        # second call returns the cached instance rather than rebuilding
        assert identity_check._get_face_app() is app
