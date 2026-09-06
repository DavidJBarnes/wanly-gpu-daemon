"""The heartbeat says what this worker can FETCH, not only what it holds (console#422).

The API refuses to hand a worker a segment whose models it cannot load. Without this field
it would also refuse one naming a LoRA this box has never seen — work the daemon downloads
inside the claim anyway — and the worker would sit idle waiting for a restart it does not
need. A silent starve, on the one path where silence looks exactly like an empty queue.
"""
import uuid

import httpx

from daemon import main
from daemon.queue_client import QueueClient


class _Capture:
    def __init__(self):
        self.payload = None

    async def request(self, method, url, **kwargs):
        self.payload = kwargs.get("json")
        return httpx.Response(200, json={"friendly_name": "w"},
                              request=httpx.Request(method, "http://x"))


async def _beat(**kwargs) -> dict:
    client = QueueClient()
    cap = _Capture()
    client.client = cap
    await client.heartbeat(uuid.uuid4(), True, **kwargs)
    return cap.payload


class TestWhatTheHeartbeatCarries:
    async def test_fetchable_kinds_are_sent_when_given(self):
        assert (await _beat(fetchable_kinds=["lora"]))["fetchable_kinds"] == ["lora"]

    async def test_absent_rather_than_null_when_not_given(self):
        """Every optional field here works this way: the API reads a missing key as "no
        change", so sending null would overwrite what a previous beat established."""
        assert "fetchable_kinds" not in await _beat()

    def test_this_daemon_declares_loras_and_checkpoints(self):
        """Both are now fetched at claim time — LoRAs by
        lora_sync.ensure_named_loras_present, base models by
        checkpoint_sync.ensure_checkpoint_present (console#423).

        This is the seam console#422 was built around: the API holds no second opinion about
        what a daemon can do, so declaring "checkpoint" here opens the model gate on its own,
        with no API change and no coordinated deploy.

        The order matters to nothing but reading it — the API treats this as a set — but it
        is kept stable so a diff of a heartbeat payload is legible.
        """
        assert main.FETCHABLE_KINDS == ["lora", "checkpoint"]

    def test_declaring_a_kind_means_the_code_can_actually_fetch_it(self):
        """The declaration is a PROMISE the API acts on: it stops gating that kind and hands
        this worker segments needing files it does not hold. Declaring a kind nothing
        implements is how a pod claims work it will certainly fail — which is exactly what
        happened when the content-LoRA pre-fetch read a dead key (#176) while this list still
        said "lora"."""
        from daemon import checkpoint_sync, lora_sync
        assert callable(lora_sync.ensure_named_loras_present)
        assert callable(checkpoint_sync.ensure_checkpoint_present)
