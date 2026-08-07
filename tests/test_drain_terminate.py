"""Tests for _terminate_runpod_pod's handling of missing RunPod credentials.

The regression this guards: the terminate call returned silently when RUNPOD_POD_ID or
RUNPOD_API_KEY was unset. That runs at the end of a drain — so the pod kept running, the
container respawned, the daemon re-registered and resumed claiming work, and the log said
nothing at all about why the drain appeared to do nothing.
"""

import logging

import pytest

from daemon import main


class TestMissingCredentials:
    @pytest.mark.parametrize("pod_id,api_key,expect_missing", [
        ("",        "key", "RUNPOD_POD_ID=MISSING"),
        ("pod123",  "",    "RUNPOD_API_KEY=MISSING"),
        ("",        "",    "RUNPOD_POD_ID=MISSING"),
        (None,      "key", "RUNPOD_POD_ID=MISSING"),
    ])
    async def test_warns_and_names_what_is_missing(
        self, monkeypatch, caplog, pod_id, api_key, expect_missing
    ):
        monkeypatch.setattr(main.settings, "runpod_pod_id", pod_id)
        monkeypatch.setattr(main.settings, "runpod_api_key", api_key)
        with caplog.at_level(logging.WARNING, logger=main.logger.name):
            await main._terminate_runpod_pod()
        assert caplog.records, "missing credentials must not fail silently"
        msg = caplog.records[-1].getMessage()
        assert expect_missing in msg
        assert "respawn" in msg, "the operator needs to know the worker will come back"

    async def test_does_not_raise(self, monkeypatch):
        """It runs inside the shutdown finally-block; raising would mask the real path."""
        monkeypatch.setattr(main.settings, "runpod_pod_id", "")
        monkeypatch.setattr(main.settings, "runpod_api_key", "")
        await main._terminate_runpod_pod()  # must simply return

    async def test_set_credentials_are_reported_as_set(self, monkeypatch, caplog):
        """Only the absent one is named — so the log points at the actual gap."""
        monkeypatch.setattr(main.settings, "runpod_pod_id", "pod123")
        monkeypatch.setattr(main.settings, "runpod_api_key", "")
        with caplog.at_level(logging.WARNING, logger=main.logger.name):
            await main._terminate_runpod_pod()
        msg = caplog.records[-1].getMessage()
        assert "RUNPOD_POD_ID=set" in msg
        assert "RUNPOD_API_KEY=MISSING" in msg


class TestReturnValue:
    """The caller now decides whether to deregister based on this, so it has to be honest.

    Before wanly-console#286 the function returned None either way and the shutdown path
    deregistered regardless — so a failed stop silently discarded the pending drain.
    """

    async def test_returns_false_when_credentials_missing(self, monkeypatch):
        monkeypatch.setattr(main.settings, "runpod_pod_id", "")
        monkeypatch.setattr(main.settings, "runpod_api_key", "")
        assert await main._terminate_runpod_pod() is False

    async def test_returns_false_when_the_api_rejects_the_key(self, monkeypatch):
        """A revoked key 401s. That is what actually happened, and it must not read as success."""
        import httpx

        monkeypatch.setattr(main.settings, "runpod_pod_id", "pod123")
        monkeypatch.setattr(main.settings, "runpod_api_key", "revoked")

        class _Resp:
            text = "unauthorized"
            status_code = 401

            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "401 Unauthorized", request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(401),
                )

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def delete(self, *a, **k):
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
        assert await main._terminate_runpod_pod() is False

    async def test_returns_true_on_success(self, monkeypatch):
        import httpx

        monkeypatch.setattr(main.settings, "runpod_pod_id", "pod123")
        monkeypatch.setattr(main.settings, "runpod_api_key", "goodkey")

        class _Resp:
            text = ""
            status_code = 204

            def raise_for_status(self):
                return None

        class _Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def delete(self, *a, **k):
                return _Resp()

        monkeypatch.setattr(httpx, "AsyncClient", lambda **k: _Client())
        assert await main._terminate_runpod_pod() is True


class TestDrainWaitWindow:
    """The drain must outlast a real segment.

    The old timeout was 600s. Measured segment durations on the same worker:

        480p / 3s  ->  329s
        480p / 5s  ->  566s, 575s
        720x1056/5s -> 1774s, 1787s

    So a drain issued during a 720p segment gave up at the 10 minute mark and abandoned roughly
    twenty more minutes of work that was about to finish. With terminate rather than stop, that
    stops being wasted time and becomes a destroyed pod with the segment still inside it.
    """

    def test_default_outlasts_the_longest_measured_segment(self):
        from daemon.config import Settings

        longest_measured = 1787.4
        assert Settings().drain_wait_seconds > longest_measured, (
            "the drain wait must exceed a real 720p segment, or draining discards finished work"
        )

    def test_is_configurable_per_worker(self, monkeypatch):
        """Bigger shapes will take longer than anything measured so far."""
        monkeypatch.setenv("DRAIN_WAIT_SECONDS", "7200")
        from daemon.config import Settings

        assert Settings().drain_wait_seconds == 7200
