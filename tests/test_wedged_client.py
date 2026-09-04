"""A worker that cannot reach the API must stop pretending it can (#160).

On 2026-09-02 a worker sat registered and healthy-looking for 4.6 hours, claiming nothing,
with work queued behind it. Every request failed with RemoteProtocolError while the API was
demonstrably reachable — verified from inside that worker's own container. Only the daemon's
HTTP client was broken, and it never recovered.

Nothing errored and nothing paged. The queue just stopped, next to a worker that looked fine.
That is the failure this file guards.
"""
import httpx
import pytest

from daemon.queue_client import QueueClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("daemon.queue_client.settings.queue_url", "http://example.invalid")
    return QueueClient()


class TestFailureStreak:
    def test_a_single_blip_does_not_trip_anything(self, client):
        """A reaped keepalive connection is normal and already handled by the one retry.
        Rebuilding the pool on those would throw away a healthy one constantly."""
        client._note_failure("GET", "/x", httpx.ConnectError("blip"))
        assert client._fail_streak == 1
        assert client._fail_streak < QueueClient._REBUILD_AFTER
        assert not client.is_wedged()

    def test_any_success_resets_the_streak(self, client):
        """This counts a SUSTAINED inability to talk to the API, not a lifetime total. A
        worker that fails five times and then works is not wedged."""
        for _ in range(5):
            client._note_failure("GET", "/x", httpx.ConnectError("x"))
        client._note_success()
        assert client._fail_streak == 0
        assert not client.is_wedged()

    def test_it_gives_up_only_after_a_long_streak(self, client):
        """Between rebuild and give-up there is a lot of room on purpose: a rebuilt pool
        deserves a real chance before the worker takes itself out of the fleet."""
        for _ in range(QueueClient._GIVE_UP_AFTER - 1):
            client._note_failure("GET", "/x", httpx.ConnectError("x"))
        assert not client.is_wedged()
        client._note_failure("GET", "/x", httpx.ConnectError("x"))
        assert client.is_wedged()

    def test_rebuild_comes_well_before_give_up(self):
        """Rebuilding is cheap and often sufficient — it is what `docker restart` achieved
        by accident. Exiting is the last resort, not the first."""
        assert QueueClient._REBUILD_AFTER < QueueClient._GIVE_UP_AFTER


class TestPoolRebuild:
    @pytest.mark.asyncio
    async def test_rebuilding_replaces_the_pool(self, client):
        """The pool was the broken thing, not the process. Replacing it is the whole fix."""
        before = client.client
        await client._rebuild_client()
        assert client.client is not before

    @pytest.mark.asyncio
    async def test_rebuilding_survives_a_pool_that_fails_to_close(self, client, monkeypatch):
        """Closing an already-broken pool can itself raise. The new client is in place by
        then, so that must not propagate and undo the recovery."""
        async def boom():
            raise RuntimeError("pool already broken")
        monkeypatch.setattr(client.client, "aclose", boom)
        await client._rebuild_client()          # must not raise
        assert client.client is not None


class TestLogEscalation:
    def test_a_long_streak_logs_at_error_not_info(self, client, caplog):
        """4.6 hours of identical INFO lines is indistinguishable from ordinary polling
        unless somebody reads the timestamps — which is exactly what happened. A streak is a
        different event from a blip and must not look the same."""
        import logging
        caplog.set_level(logging.INFO, logger="daemon.queue_client")
        for _ in range(QueueClient._REBUILD_AFTER):
            client._note_failure("GET", "/segments/next", httpx.ConnectError("x"))
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "a sustained streak must escalate above INFO"
        assert "CONSECUTIVE" in errors[-1].getMessage()
        assert "claiming nothing" in errors[-1].getMessage()
