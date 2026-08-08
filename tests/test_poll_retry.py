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
