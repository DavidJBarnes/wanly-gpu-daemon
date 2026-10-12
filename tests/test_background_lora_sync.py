"""wanly-gpu-daemon#189: register first, sync LoRAs in the background, say what it is doing.

On 2026-10-12 a mode switch into render re-ran the LoRA sync BEFORE registration, and one
650 MB LoRA at ~0.4 MB/s kept 3090a off the Workers page for 10+ minutes while a training run
waited. These pin the three halves of the fix:

  * registration does not wait for the sync,
  * a LoRA the sync has not landed yet is reported as pending (so the API leaves segments
    naming it in the queue) and released the moment it lands,
  * the phase the heartbeat carries names the file, MB done / total, rate, ETA and files left.
"""
import asyncio
import hashlib
import inspect

import pytest

from daemon import lora_sync, main as daemon_main, phase

BIG = 20 * 1024 * 1024


def _md5(payload: bytes) -> str:
    return hashlib.md5(payload + b"\0" * (BIG - len(payload))).hexdigest()


class SlowQueue:
    """A download that does not finish until the test says so."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.release = asyncio.Event()
        self.started = asyncio.Event()
        self.downloaded = []

    async def list_loras(self):
        return self.catalog

    async def stream_file(self, uri, dest, on_progress=None, total: int = 0, step: int = 0):
        self.downloaded.append(uri)
        if on_progress:
            await on_progress(total // 2, total)
        self.started.set()
        await self.release.wait()
        data = b"NEW" + b"\0" * (BIG - 3)
        with open(dest, "wb") as f:
            f.write(data)
        return hashlib.md5(data).hexdigest()


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_sync, "_loras_dir", lambda: str(tmp_path))
    return tmp_path


def _remote(name, kind="character"):
    return {"name": name, "size": BIG, "etag": _md5(b"NEW"), "multipart": False,
            "uri": f"s3://loras/{name}", "kind": kind}


async def test_a_lora_in_flight_is_pending_until_it_lands(lora_dir):
    """The claim must skip segments naming it while it downloads, and stop skipping after."""
    q = SlowQueue([_remote("Payton_v2_final.safetensors")])
    task = asyncio.create_task(lora_sync.sync_lora_catalog(q))
    await asyncio.wait_for(q.started.wait(), 5)

    assert lora_sync.pending_loras() == ["Payton_v2_final.safetensors"]

    q.release.set()
    assert await asyncio.wait_for(task, 5) is True
    assert lora_sync.pending_loras() == []
    assert (lora_dir / "Payton_v2_final.safetensors").exists()


async def test_a_current_lora_is_released_without_waiting_for_the_downloads(lora_dir):
    """Only the file really being fetched holds work back; a verified one never does."""
    with open(lora_dir / "Karoline_v3_final.safetensors", "wb") as f:
        f.write(b"NEW" + b"\0" * (BIG - 3))
    q = SlowQueue([_remote("Karoline_v3_final.safetensors"),
                   _remote("Payton_v2_final.safetensors")])
    task = asyncio.create_task(lora_sync.sync_lora_catalog(q))
    await asyncio.wait_for(q.started.wait(), 5)

    assert lora_sync.pending_loras() == ["Payton_v2_final.safetensors"]
    q.release.set()
    await asyncio.wait_for(task, 5)


async def test_a_deferred_content_lora_is_never_pending(lora_dir):
    """Content LoRAs are fetched at claim time; holding them back would starve that path."""
    q = SlowQueue([_remote("motion_v1.safetensors", kind="content")])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert lora_sync.pending_loras() == []
    assert q.downloaded == []


async def test_the_phase_names_the_file_mb_rate_eta_and_files_left(lora_dir):
    """What the console shows instead of 'offline' -- every number the ticket asked for."""
    q = SlowQueue([_remote("Payton_v2_final.safetensors")])
    task = asyncio.create_task(lora_sync.sync_lora_catalog(q))
    await asyncio.wait_for(q.started.wait(), 5)

    p = phase.snapshot()
    assert p["phase"] == "syncing_loras"
    assert p["file"] == "Payton_v2_final.safetensors"
    assert p["done_bytes"] == BIG // 2 and p["total_bytes"] == BIG
    assert p["rate_bps"] and p["rate_bps"] > 0
    assert p["eta_s"] is not None
    assert p["files_total"] == 1 and p["files_remaining"] == 1
    assert "Payton_v2_final" in p["detail"] and "MB" in p["detail"]

    q.release.set()
    await asyncio.wait_for(task, 5)


async def test_a_failed_download_is_released_not_held_forever(lora_dir):
    class Broken(SlowQueue):
        async def stream_file(self, *a, **k):
            raise RuntimeError("connection reset")

    assert await lora_sync.sync_lora_catalog(Broken([_remote("x_v1.safetensors")])) is False
    assert lora_sync.pending_loras() == []


async def test_a_claim_naming_a_pending_lora_waits_for_the_sync_instead_of_racing_it(lora_dir):
    """The fallback for an API that ignores pending_loras: never two writers on one .tmp."""
    q = SlowQueue([_remote("Payton_v2_final.safetensors")])
    task = asyncio.create_task(lora_sync.sync_lora_catalog(q))
    await asyncio.wait_for(q.started.wait(), 5)

    lines = []

    async def progress(m):
        lines.append(m)

    claim = asyncio.create_task(
        lora_sync.ensure_named_loras_present(["Payton_v2_final"], q, progress=progress))
    await asyncio.sleep(0.05)
    assert not claim.done()                       # waiting, not downloading
    assert any("Waiting for the background LoRA sync" in m for m in lines)

    q.release.set()
    await asyncio.wait_for(task, 5)
    assert await asyncio.wait_for(claim, 5) == []  # landed by the sync, nothing to fetch
    assert q.downloaded == ["s3://loras/Payton_v2_final.safetensors"]   # exactly once


def test_registration_comes_before_the_sync():
    """The whole bug: the sync used to run before register_with_retry."""
    src = inspect.getsource(daemon_main.run)
    assert "await sync_lora_catalog" not in src
    assert src.index("register_with_retry(") < src.index("_background_lora_sync(")


def test_the_claim_sends_the_pending_set():
    src = inspect.getsource(daemon_main.job_poll_loop)
    assert "pending_loras=lora_sync.pending_loras()" in src


def test_the_heartbeat_and_registration_carry_the_phase():
    assert "phase=phase.snapshot()" in inspect.getsource(daemon_main.heartbeat_loop)
    assert "phase=phase.snapshot()" in inspect.getsource(daemon_main.register_with_retry)


async def test_the_background_sync_always_ends_ready(monkeypatch):
    async def boom(_q):
        raise RuntimeError("catalogue exploded")

    monkeypatch.setattr(daemon_main, "sync_lora_catalog", boom)
    phase.set_phase("syncing_loras", "x")
    await daemon_main._background_lora_sync(object())
    assert phase.current() == phase.READY
