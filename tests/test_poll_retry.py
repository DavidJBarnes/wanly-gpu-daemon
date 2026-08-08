"""Tests for transient poll/heartbeat failure handling (#105).

Context: 72 of these in 24 hours on one worker — 61 with "Server disconnected without sending a
response", 11 with an **empty** message. The empty ones could not even be classified, and two
separate theories were built on that ambiguity and withdrawn.

The split by loop was the consistent clue: heartbeat and hologram poll fail, the GPU claim loop
does not. Those two are the loops still issuing requests while the GPU loop sits blocked awaiting
ComfyUI — pointing at the shared connection pool rather than at any particular network.
"""

import httpx
import pytest

from daemon.queue_client import QueueClient


class _Recorder:
    """Fails the first call with a transport error, succeeds on the second."""

    def __init__(self, error):
        self.error = error
        self.calls = 0

    async def request(self, method, url, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise self.error
        return httpx.Response(200, json={"ok": True}, request=httpx.Request(method, "http://x"))


class TestRetry:
    @pytest.mark.parametrize("error", [
        httpx.RemoteProtocolError("Server disconnected without sending a response."),
        httpx.ReadTimeout(""),          # the empty-message variant
        httpx.ConnectError("boom"),
    ])
    async def test_retries_once_on_transport_failure(self, error):
        """A reaped keepalive connection fails instantly and succeeds on a fresh socket."""
        client = QueueClient()
        rec = _Recorder(error)
        client.client = rec
        resp = await client._request_with_retry("GET", "/segments/next")
        assert rec.calls == 2
        assert resp.status_code == 200

    async def test_does_not_retry_http_errors(self):
        """An HTTP status is a real answer. Claiming is not idempotent — never repeat it."""
        calls = {"n": 0}

        class _Http:
            async def request(self, method, url, **kwargs):
                calls["n"] += 1
                return httpx.Response(409, request=httpx.Request(method, "http://x"))

        client = QueueClient()
        client.client = _Http()
        resp = await client._request_with_retry("GET", "/segments/next")
        assert calls["n"] == 1
        assert resp.status_code == 409

    async def test_a_persistent_failure_still_raises(self):
        """One retry, not infinite. A genuinely down API must surface, not spin."""
        class _Dead:
            async def request(self, method, url, **kwargs):
                raise httpx.ConnectError("down")

        client = QueueClient()
        client.client = _Dead()
        with pytest.raises(httpx.ConnectError):
            await client._request_with_retry("GET", "/segments/next")


class TestKeepaliveExpiry:
    def test_pool_retires_connections_before_the_far_end_does(self):
        """The root cause: httpx hands out a connection the server already closed."""
        client = QueueClient()
        # httpx keeps the effective value on the connection pool, not on the client.
        expiry = client.client._transport._pool._keepalive_expiry
        assert expiry is not None, "an unbounded keepalive is the bug"
        assert expiry <= 60, (
            "keepalive_expiry must sit below the far end's idle timeout, or reaped connections "
            "keep being handed out"
        )


class TestRegistrationReportsPodId:
    """The console pairs a pod with its worker by pod id (wanly-console#291 follow-up).

    Pairing on name only worked for pods the console launcher created. A pod started from the
    RunPod template has an auto-generated name while its worker registers as runpod-<pod id>,
    so the two never matched and the pod showed as "Starting" forever.
    """

    async def test_pod_id_is_sent_when_running_on_runpod(self, monkeypatch):
        from daemon.config import settings

        monkeypatch.setattr(settings, "runpod_pod_id", "pod-xyz")
        sent = {}

        class _Client:
            async def post(self, url, json=None, **kw):
                sent.update(json or {})
                return httpx.Response(
                    201, json={"id": "00000000-0000-0000-0000-000000000001",
                               "friendly_name": "w"},
                    request=httpx.Request("POST", "http://x"),
                )

        client = QueueClient()
        client.client = _Client()
        await client.register("w", "h", "127.0.0.1", True)
        assert sent.get("runpod_pod_id") == "pod-xyz"

    async def test_omitted_when_self_hosted(self, monkeypatch):
        """The 3090 is not a pod. Sending an empty string would create a worker that looks like
        it belongs to a pod called "" and never pairs with anything."""
        from daemon.config import settings

        monkeypatch.setattr(settings, "runpod_pod_id", None)
        sent = {}

        class _Client:
            async def post(self, url, json=None, **kw):
                sent.update(json or {})
                return httpx.Response(
                    201, json={"id": "00000000-0000-0000-0000-000000000001",
                               "friendly_name": "w"},
                    request=httpx.Request("POST", "http://x"),
                )

        client = QueueClient()
        client.client = _Client()
        await client.register("w", "h", "127.0.0.1", True)
        assert "runpod_pod_id" not in sent
