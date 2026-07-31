"""Tests for _stop_runpod_pod's handling of missing RunPod credentials.

The regression this guards: _stop_runpod_pod returned silently when RUNPOD_POD_ID or
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
            await main._stop_runpod_pod()
        assert caplog.records, "missing credentials must not fail silently"
        msg = caplog.records[-1].getMessage()
        assert expect_missing in msg
        assert "respawn" in msg, "the operator needs to know the worker will come back"

    async def test_does_not_raise(self, monkeypatch):
        """It runs inside the shutdown finally-block; raising would mask the real path."""
        monkeypatch.setattr(main.settings, "runpod_pod_id", "")
        monkeypatch.setattr(main.settings, "runpod_api_key", "")
        await main._stop_runpod_pod()  # must simply return

    async def test_set_credentials_are_reported_as_set(self, monkeypatch, caplog):
        """Only the absent one is named — so the log points at the actual gap."""
        monkeypatch.setattr(main.settings, "runpod_pod_id", "pod123")
        monkeypatch.setattr(main.settings, "runpod_api_key", "")
        with caplog.at_level(logging.WARNING, logger=main.logger.name):
            await main._stop_runpod_pod()
        msg = caplog.records[-1].getMessage()
        assert "RUNPOD_POD_ID=set" in msg
        assert "RUNPOD_API_KEY=MISSING" in msg
