"""The daemon must say what code it is running (wanly-gpu-docker#72).

Two workers ran different code for 14 hours and nothing said so. It surfaced as a 422 that
looked random, and diagnosing it meant SSHing into both boxes and reading git.

The failure this guards is not an exception -- it is a field quietly going missing, which
puts us straight back to "SSH in and check". So these tests are about what actually reaches
the payload.
"""
from unittest.mock import AsyncMock, patch

import pytest

from daemon import build_identity
from daemon.queue_client import QueueClient


class TestReadingIt:
    def test_an_absent_image_env_is_none_not_the_string_none(self):
        """The daemon also runs outside the image in development. "None" as a value would be
        reported as if it were a real image ref."""
        with patch.dict("os.environ", {}, clear=True):
            assert build_identity._read_image() is None
        with patch.dict("os.environ", {"WANLY_IMAGE_REF": "   "}):
            assert build_identity._read_image() is None
        with patch.dict("os.environ", {"WANLY_IMAGE_REF": "sha256:abc"}):
            assert build_identity._read_image() == "sha256:abc"

    def test_a_non_git_checkout_reports_none_rather_than_guessing(self, tmp_path):
        """None means "cannot tell", which is different from a commit and must not be dressed
        up as one."""
        with patch("daemon.build_identity.os.path.abspath",
                   return_value=str(tmp_path / "x" / "y")):
            assert build_identity._read_commit() is None

    def test_describe_says_unknown_rather_than_omitting(self):
        """A missing field in the boot log reads as "nothing to report"; an explicit unknown
        reads as a question worth asking."""
        with patch.object(build_identity, "DAEMON_COMMIT", None), \
             patch.object(build_identity, "IMAGE_REF", None):
            assert "unknown" in build_identity.describe()


@pytest.mark.asyncio
class TestWhatTheHeartbeatCarries:
    async def _payload(self, **kw):
        q = QueueClient.__new__(QueueClient)
        captured = {}

        async def fake(method, url, **kwargs):
            captured.update(kwargs.get("json") or {})
            class R:
                is_success = True
                status_code = 200
                @staticmethod
                def json(): return {}
            return R()

        q._request_with_retry = AsyncMock(side_effect=fake)
        import uuid as _u
        await q.heartbeat(_u.uuid4(), True, **kw)
        return captured

    async def test_both_are_sent_when_known(self):
        p = await self._payload(daemon_commit="a044cef", image_ref="sha256:abc")
        assert p["daemon_commit"] == "a044cef"
        assert p["image_ref"] == "sha256:abc"

    async def test_unknown_values_are_omitted_not_nulled(self):
        """The API reads a missing key as "no change". Sending null would blank a value a
        previous beat established — the trap every optional worker field documents."""
        p = await self._payload(daemon_commit=None, image_ref=None)
        assert "daemon_commit" not in p
        assert "image_ref" not in p

    async def test_they_can_be_reported_independently(self):
        """`docker restart` re-clones the daemon and reuses the image, so a new commit on an
        old image is a real state — it is the one the 3090 was in for 37 hours."""
        p = await self._payload(daemon_commit="a044cef")
        assert p["daemon_commit"] == "a044cef"
        assert "image_ref" not in p


def test_the_heartbeat_call_actually_passes_them():
    """The wiring, not just the plumbing. Everything above can pass while main() never
    supplies the values — which would leave the field permanently null and look, from the
    API, exactly like a daemon too old to report it."""
    import inspect
    from daemon import main as main_mod
    src = inspect.getsource(main_mod.heartbeat_loop)
    assert "daemon_commit=build_identity.DAEMON_COMMIT" in src
    assert "image_ref=build_identity.IMAGE_REF" in src
