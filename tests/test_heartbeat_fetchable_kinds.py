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

    def test_this_daemon_declares_loras_and_not_checkpoints(self):
        """LoRAs are fetched at claim time (lora_sync.ensure_named_loras_present); a 46 GB
        checkpoint is not, and claiming a segment that needs one would fail ten minutes in.
        console#423 is what flips the second half of this."""
        assert main.FETCHABLE_KINDS == ["lora"]
