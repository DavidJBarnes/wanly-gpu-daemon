"""Tests for structured stage logging and VRAM sampling."""

import json
import logging

import pytest

from daemon import stage_log
from daemon.stage_log import _VramSampler, log_event, stage

CID = "11111111-2222-3333-4444-555555555555"


def _json_records(caplog):
    return [json.loads(r.message) for r in caplog.records if r.message.startswith("{")]


class TestLogEvent:
    def test_emits_json_with_correlation_id(self, caplog):
        with caplog.at_level(logging.INFO):
            payload = log_event(logging.getLogger("t"), "thing.happened", CID, frames=81)
        assert payload == {"event": "thing.happened", "correlation_id": CID, "frames": 81}
        assert _json_records(caplog) == [payload]

    def test_respects_level(self, caplog):
        with caplog.at_level(logging.DEBUG):
            log_event(logging.getLogger("t"), "oops", CID, level=logging.ERROR)
        assert caplog.records[0].levelno == logging.ERROR

    def test_non_serialisable_values_are_stringified(self, caplog):
        class Weird:
            def __str__(self) -> str:
                return "weird"

        with caplog.at_level(logging.INFO):
            log_event(logging.getLogger("t"), "e", CID, obj=Weird())
        assert _json_records(caplog)[0]["obj"] == "weird"


class TestVramSampler:
    def test_tracks_peak_across_samples(self, monkeypatch):
        readings = iter([
            {"gpu_name": "x", "vram_used_mb": 1000, "vram_total_mb": 24576},
            {"gpu_name": "x", "vram_used_mb": 21000, "vram_total_mb": 24576},
            {"gpu_name": "x", "vram_used_mb": 5000, "vram_total_mb": 24576},
        ])
        monkeypatch.setattr(stage_log, "get_gpu_stats", lambda: next(readings, None))
        sampler = _VramSampler(interval=0.01)
        sampler._sample()
        sampler._sample()
        sampler._sample()
        assert sampler.peak_mb == 21000
        assert sampler.total_mb == 24576

    def test_degrades_when_nvidia_smi_unavailable(self, monkeypatch):
        monkeypatch.setattr(stage_log, "get_gpu_stats", lambda: None)
        sampler = _VramSampler(interval=0.01)
        sampler.start()
        sampler.stop()
        assert sampler.peak_mb is None
        assert sampler.total_mb is None

    def test_start_stop_runs_background_thread(self, monkeypatch):
        monkeypatch.setattr(
            stage_log, "get_gpu_stats",
            lambda: {"gpu_name": "x", "vram_used_mb": 512, "vram_total_mb": 24576},
        )
        sampler = _VramSampler(interval=0.01)
        sampler.start()
        sampler.stop()
        assert sampler.peak_mb == 512
        assert sampler._thread is not None
        assert not sampler._thread.is_alive()


class TestStageContext:
    def test_logs_start_and_end_with_duration_and_vram(self, monkeypatch, caplog):
        monkeypatch.setattr(
            stage_log, "get_gpu_stats",
            lambda: {"gpu_name": "x", "vram_used_mb": 18000, "vram_total_mb": 24576},
        )
        with caplog.at_level(logging.INFO):
            with stage(logging.getLogger("t"), "segment.generate", CID, num_frames=81):
                pass
        events = _json_records(caplog)
        assert [e["event"] for e in events] == ["segment.generate.start", "segment.generate.end"]
        end = events[1]
        assert end["correlation_id"] == CID
        assert end["num_frames"] == 81
        assert end["vram_peak_mb"] == 18000
        assert end["vram_total_mb"] == 24576
        assert end["duration_sec"] >= 0

    def test_extra_dict_is_merged_into_end_record(self, monkeypatch, caplog):
        monkeypatch.setattr(stage_log, "get_gpu_stats", lambda: None)
        with caplog.at_level(logging.INFO):
            with stage(logging.getLogger("t"), "segment.build", CID) as extra:
                extra["nodes"] = 17
        assert _json_records(caplog)[1]["nodes"] == 17

    def test_failure_logs_and_reraises(self, monkeypatch, caplog):
        monkeypatch.setattr(stage_log, "get_gpu_stats", lambda: None)
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError, match="boom"):
                with stage(logging.getLogger("t"), "segment.generate", CID):
                    raise ValueError("boom")
        events = _json_records(caplog)
        assert [e["event"] for e in events] == ["segment.generate.start", "segment.generate.failed"]
        assert events[1]["error"] == "boom"
        assert events[1]["error_type"] == "ValueError"
        assert caplog.records[-1].levelno == logging.ERROR
