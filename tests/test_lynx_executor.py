"""Tests for the Lynx segment execution path.

ComfyUI and the queue API are faked; the concern here is the orchestration contract —
what gets submitted, what gets reported back, and which failures are surfaced how.
"""

import io
import json
import logging

import numpy as np
import pytest
from PIL import Image

from daemon import executor
from daemon.comfyui_client import ComfyUIExecutionError
from daemon.executor import _execute_lynx, _measure_lynx_identity, _resolve_lynx_subject
from daemon.workflow_builder import LynxValidationError
from tests.conftest import make_segment

CID = "11111111-2222-3333-4444-555555555555"


def png_bytes(size=(256, 256)) -> bytes:
    """A noisy PNG — _validate_image_data rejects anything under 1 KB, and a flat
    fill compresses well below that."""
    buf = io.BytesIO()
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 255, (*size, 3), dtype=np.uint8)).save(buf, format="PNG")
    return buf.getvalue()


class FakeComfyUI:
    def __init__(self, video_output=True, execution_error=None):
        self.submitted = None
        self.uploaded = []
        self._video_output = video_output
        self._execution_error = execution_error

    async def upload_image(self, data, filename):
        self.uploaded.append(filename)
        return filename

    async def submit_workflow(self, workflow):
        self.submitted = workflow
        return "promptid123456", "clientid"

    async def monitor_execution(self, prompt_id, client_id, progress=None):
        if self._execution_error:
            raise self._execution_error

    async def get_history(self, prompt_id):
        return {"outputs": {}}

    def find_video_output(self, history):
        return {"filename": "out.mp4", "subfolder": "", "type": "output"} if self._video_output else None

    async def download_output(self, filename, subfolder, output_type):
        return b"video-bytes"


class FakeQueue:
    def __init__(self, file_bytes=None):
        self.updates = []
        self.outputs = []
        self.progress_logs = []
        self._file_bytes = file_bytes or png_bytes()

    async def download_file(self, path):
        return self._file_bytes

    async def update_segment(self, segment_id, result):
        self.updates.append(result)

    async def upload_segment_output(self, segment_id, video, last_frame, result):
        self.outputs.append(result)

    async def update_segment_progress(self, *a, **kw):
        self.progress_logs.append((a, kw))


@pytest.fixture(autouse=True)
def _stub_side_effects(monkeypatch):
    """Neutralise LoRA sync, last-frame extraction and GPU polling."""
    async def noop_loras(loras, queue):
        return None

    async def fake_last_frame(data):
        return png_bytes()

    monkeypatch.setattr(executor, "ensure_loras_available", noop_loras)
    monkeypatch.setattr(executor, "_extract_last_frame", fake_last_frame)
    monkeypatch.setattr("daemon.stage_log.get_gpu_stats",
                        lambda: {"gpu_name": "x", "vram_used_mb": 18000, "vram_total_mb": 24576})


def _failure(queue):
    """The terminal failure update. ProgressLog also calls update_segment with
    'processing' rows, so index 0 is not the outcome."""
    failed = [u for u in queue.updates if u.status == "failed"]
    assert failed, f"no failure update recorded (got {[u.status for u in queue.updates]})"
    return failed[-1]


def _events(caplog):
    return [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]


class TestResolveSubject:
    async def test_s3_subject_is_downloaded_and_uploaded(self):
        seg = make_segment(lynx_subject_image="s3://bucket/face.png")
        comfy, queue = FakeComfyUI(), FakeQueue()
        filename, data = await _resolve_lynx_subject(seg, comfy, queue)
        assert filename.startswith("lynx_subject_")
        assert filename.endswith(".png")
        assert data == queue._file_bytes
        assert comfy.uploaded == [filename]

    async def test_plain_filename_passes_through(self):
        seg = make_segment(lynx_subject_image="already_there.png")
        filename, data = await _resolve_lynx_subject(seg, FakeComfyUI(), FakeQueue())
        assert filename == "already_there.png"
        assert data == b""

    async def test_missing_subject_raises_validation_error(self):
        seg = make_segment(lynx_subject_image=None)
        with pytest.raises(LynxValidationError, match="lynx_subject_image"):
            await _resolve_lynx_subject(seg, FakeComfyUI(), FakeQueue())


class TestExecuteLynx:
    async def test_happy_path_submits_and_reports_completed(self, monkeypatch, caplog):
        monkeypatch.setattr(executor, "measure_identity",
                            lambda v, s, c: {"mean": 0.61, "scores": [0.6, 0.62]})
        comfy, queue = FakeComfyUI(), FakeQueue()
        seg = make_segment(lynx_subject_image="s3://bucket/face.png")

        with caplog.at_level(logging.INFO):
            await _execute_lynx(seg, comfy, queue)

        assert comfy.submitted is not None
        assert comfy.submitted["650"]["class_type"] == "WanVideoSampler"
        assert len(queue.outputs) == 1
        result = queue.outputs[0]
        assert result.status == "completed"
        assert result.lynx_identity_scores == {"mean": 0.61, "scores": [0.6, 0.62]}

    async def test_character_lora_is_synced_before_generating(self, monkeypatch):
        from daemon.schemas import LoraItem

        synced = {}

        async def fake_sync(loras, queue):
            synced["files"] = [item.high_file for item in loras]

        monkeypatch.setattr(executor, "ensure_loras_available", fake_sync)
        monkeypatch.setattr(executor, "measure_identity", lambda v, s, c: None)
        seg = make_segment(
            lynx_subject_image="s3://b/f.png",
            loras=[LoraItem(lora_id="a", high_file="k3lly.safetensors", high_weight=0.8)],
        )
        comfy = FakeComfyUI()
        await _execute_lynx(seg, comfy, FakeQueue())
        assert synced["files"] == ["k3lly.safetensors"]
        # and the LoRA actually reached the graph, stacked after the distill LoRA
        assert comfy.submitted["701"]["inputs"]["lora"] == "k3lly.safetensors"

    async def test_stage_logging_carries_correlation_id_and_vram(self, monkeypatch, caplog):
        monkeypatch.setattr(executor, "measure_identity", lambda v, s, c: None)
        with caplog.at_level(logging.INFO):
            await _execute_lynx(make_segment(lynx_subject_image="s3://b/f.png"),
                                FakeComfyUI(), FakeQueue())
        events = _events(caplog)
        names = [e["event"] for e in events]
        for expected in ("lynx.segment_start", "lynx.build.start", "lynx.build.end",
                         "lynx.generate.start", "lynx.generate.end", "lynx.segment_complete"):
            assert expected in names
        assert {e["correlation_id"] for e in events} == {CID}
        generate_end = next(e for e in events if e["event"] == "lynx.generate.end")
        assert generate_end["vram_peak_mb"] == 18000
        assert generate_end["vram_total_mb"] == 24576

    async def test_validation_failure_reports_failed_without_submitting(self, caplog):
        comfy, queue = FakeComfyUI(), FakeQueue()
        seg = make_segment(width=512, height=512, lynx_subject_image="s3://b/f.png")
        with caplog.at_level(logging.INFO):
            await _execute_lynx(seg, comfy, queue)
        assert comfy.submitted is None
        assert _failure(queue).status == "failed"
        assert "512x512" in _failure(queue).error_message
        assert "lynx.validation_failed" in [e["event"] for e in _events(caplog)]

    async def test_faceless_subject_gets_an_explanatory_error(self):
        err = ComfyUIExecutionError("No face detected in the image")
        err.node_id, err.node_type, err.traceback = "631", "LynxInsightFaceCrop", None
        comfy = FakeComfyUI(execution_error=err)
        queue = FakeQueue()
        await _execute_lynx(make_segment(lynx_subject_image="s3://b/f.png"), comfy, queue)
        message = _failure(queue).error_message
        assert _failure(queue).status == "failed"
        assert "ArcFace-conditioned" in message

    async def test_other_comfyui_errors_keep_node_context(self):
        err = ComfyUIExecutionError("OOM")
        err.node_id, err.node_type, err.traceback = "650", "WanVideoSampler", "trace"
        queue = FakeQueue()
        await _execute_lynx(make_segment(lynx_subject_image="s3://b/f.png"),
                            FakeComfyUI(execution_error=err), queue)
        assert "node 650" in _failure(queue).error_message
        assert "WanVideoSampler" in _failure(queue).error_message

    async def test_missing_video_output_fails_the_segment(self):
        queue = FakeQueue()
        await _execute_lynx(make_segment(lynx_subject_image="s3://b/f.png"),
                            FakeComfyUI(video_output=False), queue)
        assert _failure(queue).status == "failed"
        assert "No video output" in _failure(queue).error_message

    async def test_identity_qa_failure_does_not_fail_the_segment(self, monkeypatch):
        def explode(*a, **kw):
            raise RuntimeError("qa died")
        monkeypatch.setattr(executor, "measure_identity", explode)
        queue = FakeQueue()
        await _execute_lynx(make_segment(lynx_subject_image="s3://b/f.png"),
                            FakeComfyUI(), queue)
        assert queue.outputs[0].status == "completed"
        assert queue.outputs[0].lynx_identity_scores is None


class TestMeasureLynxIdentity:
    async def test_returns_none_without_subject_bytes(self):
        assert await _measure_lynx_identity(b"video", b"", CID) is None

    async def test_delegates_to_identity_check(self, monkeypatch):
        captured = {}

        def fake_measure(video_path, subject_path, correlation_id):
            captured["video"] = open(video_path, "rb").read()
            captured["subject"] = open(subject_path, "rb").read()
            return {"mean": 0.7}

        monkeypatch.setattr(executor, "measure_identity", fake_measure)
        result = await _measure_lynx_identity(b"video-bytes", b"subject-bytes", CID)
        assert result == {"mean": 0.7}
        assert captured == {"video": b"video-bytes", "subject": b"subject-bytes"}

    async def test_swallows_errors(self, monkeypatch):
        def explode(*a, **kw):
            raise OSError("disk gone")
        monkeypatch.setattr(executor, "measure_identity", explode)
        assert await _measure_lynx_identity(b"v", b"s", CID) is None


class TestDispatch:
    async def test_generation_engine_routes_to_lynx(self, monkeypatch):
        called = {}

        async def fake_lynx(segment, comfyui, queue):
            called["hit"] = segment.id

        monkeypatch.setattr(executor, "_execute_lynx", fake_lynx)
        await executor.execute_segment(make_segment(generation_engine="lynx"),
                                       FakeComfyUI(), FakeQueue())
        assert called["hit"] == make_segment().id

    async def test_other_engines_do_not_route_to_lynx(self, monkeypatch):
        called = {}

        async def fake_lynx(segment, comfyui, queue):
            called["hit"] = True

        def fake_build(*a, **kw):
            raise RuntimeError("stop here")

        monkeypatch.setattr(executor, "_execute_lynx", fake_lynx)
        monkeypatch.setattr(executor, "build_workflow", fake_build)
        seg = make_segment(generation_engine=None)
        queue = FakeQueue()
        await executor.execute_segment(seg, FakeComfyUI(), queue)
        # routed to the traditional builder (which we made raise), never to Lynx
        assert "hit" not in called
        assert "stop here" in _failure(queue).error_message
