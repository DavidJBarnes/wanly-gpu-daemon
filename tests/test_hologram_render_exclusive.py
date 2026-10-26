"""A render and an AR hologram never run at the same time on one worker (#192).

On 2026-10-26 3090b's container hit its 26 GiB cap and the kernel OOM-killed ComfyUI: an LTX
render held 19.7 GiB, and a hologram claimed 37 s into it took the daemon to 4.4 GiB matting
482 frames. The hologram loop ran concurrently on purpose -- it never touches the GPU -- but
RAM is shared. These tests run BOTH real poll loops against a fake queue and assert the two
kinds of work never overlap, in either order.
"""
import asyncio

import pytest

from daemon import main as mod


class FakeComfy:
    async def check_health(self):
        return True

    async def check_queue_busy(self):
        return False

    async def get_system_info(self):
        return None


class FakeQueue:
    """Hands out queued work by kind, and records what was running when each claim landed."""

    def __init__(self, gpu=0, hologram=0):
        self.left = {"gpu": gpu, "hologram": hologram}
        self.running = set()
        self.overlaps = []
        self.done = []

    def is_wedged(self):
        return False

    async def claim_next(self, worker_id, name, kind, pending_loras=None):
        await asyncio.sleep(0.01)          # a real HTTP round trip: the race window
        if not self.left[kind]:
            return None
        self.left[kind] -= 1
        return type("Seg", (), {"id": f"{kind}-{self.left[kind]}", "kind": kind})()

    async def update_status(self, *a, **k):
        pass

    async def update_segment(self, *a, **k):
        pass


@pytest.fixture
def harness(monkeypatch):
    monkeypatch.setattr(mod.settings, "poll_interval", 0.02)
    monkeypatch.setattr(mod.settings, "engine", "wan")
    monkeypatch.setattr(mod.settings, "runpod_pod_id", None)
    monkeypatch.setattr(mod, "get_sd_scripts_status", lambda: {})

    async def listed(timeout=60):
        return None
    monkeypatch.setattr(mod.lora_sync, "wait_until_listed", listed)
    monkeypatch.setattr(mod.lora_sync, "pending_loras", lambda: set())

    def install(queue, work_s):
        async def execute(segment, comfyui, q):
            if q.running:
                q.overlaps.append((segment.kind, set(q.running)))
            q.running.add(segment.kind)
            await asyncio.sleep(work_s)
            q.running.discard(segment.kind)
            q.done.append(segment.kind)
        monkeypatch.setattr(mod, "execute_segment", execute)
    return install


async def _run(queue, until, timeout=5.0):
    shutdown, executing, drain = asyncio.Event(), asyncio.Event(), asyncio.Event()
    lock = asyncio.Lock()
    comfy = FakeComfy()
    tasks = [
        asyncio.create_task(mod.job_poll_loop(queue, comfy, "w", ["w"], shutdown, executing,
                                              drain, work_lock=lock)),
        asyncio.create_task(mod.hologram_poll_loop(queue, comfy, "w", ["w"], shutdown,
                                                   work_lock=lock)),
    ]
    deadline = asyncio.get_running_loop().time() + timeout
    while not until() and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.01)
    shutdown.set()
    await asyncio.gather(*tasks)


async def test_renders_and_holograms_never_overlap(harness):
    q = FakeQueue(gpu=4, hologram=3)
    harness(q, work_s=0.08)
    await _run(q, until=lambda: len(q.done) == 7)
    assert sorted(q.done) == ["gpu"] * 4 + ["hologram"] * 3   # everything still ran
    assert q.overlaps == []


async def test_a_hologram_waits_for_a_render_already_in_flight(harness):
    """The 2026-10-26 order: render first, hologram arrives mid-render."""
    q = FakeQueue(gpu=1, hologram=0)
    harness(q, work_s=0.2)
    shutdown_after = lambda: len(q.done) == 2

    async def hologram_arrives_mid_render():
        while "gpu" not in q.running:
            await asyncio.sleep(0.005)
        q.left["hologram"] = 1
    arrive = asyncio.create_task(hologram_arrives_mid_render())
    await _run(q, until=shutdown_after)
    await arrive
    assert q.done == ["gpu", "hologram"]
    assert q.overlaps == []


async def test_a_render_waits_for_a_hologram_already_in_flight(harness):
    q = FakeQueue(gpu=0, hologram=1)
    harness(q, work_s=0.2)

    async def render_arrives_mid_hologram():
        while "hologram" not in q.running:
            await asyncio.sleep(0.005)
        q.left["gpu"] = 1
    arrive = asyncio.create_task(render_arrives_mid_hologram())
    await _run(q, until=lambda: len(q.done) == 2)
    await arrive
    assert q.done == ["hologram", "gpu"]
    assert q.overlaps == []


async def test_a_failed_claim_does_not_hold_the_lock(harness):
    """A claim that raises must release, or the other loop waits forever."""
    q = FakeQueue(gpu=0, hologram=1)
    harness(q, work_s=0.01)
    calls = {"n": 0}
    real = q.claim_next

    async def flaky(worker_id, name, kind, pending_loras=None):
        if kind == "gpu" and calls["n"] < 3:
            calls["n"] += 1
            raise RuntimeError("API down")
        return await real(worker_id, name, kind, pending_loras)
    q.claim_next = flaky
    await _run(q, until=lambda: q.done == ["hologram"])
    assert q.done == ["hologram"]
