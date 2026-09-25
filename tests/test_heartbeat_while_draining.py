"""A worker that is still rendering must not report itself dead.

On 2026-09-25 a box was flipped to caption mode mid-render (wanly-gpu-docker#131). The
drain did exactly the right thing -- the segment was left to finish -- but the heartbeat
loop exited the instant SIGTERM arrived, so the row went stale for the ~10 minutes the
render still had to run. The Workers page showed:

    0 online / 1 total
    Nothing is running — 1 segment queued and no workers online — last worker seen 4m ago.
    Start a worker, or launch a RunPod pod, or this queue will sit.

beside a job correctly reading `processing`. That is worse than a cosmetic wrong label: it
tells an operator to launch a pod for work that is already being done, and invites a second
worker onto a queue that has one.

Any `docker stop` did this. The mode switch made it routine, because flipping a busy box to
captions IS the shutdown-with-work-in-flight case.
"""
import asyncio
import inspect
import pathlib

import pytest

from daemon import main as mod


def test_the_loop_takes_the_executing_flag():
    """It cannot tell "shutting down" from "shutting down but still rendering" without it."""
    assert "executing_event" in inspect.signature(mod.heartbeat_loop).parameters


def test_shutdown_alone_does_not_end_it():
    src = inspect.getsource(mod.heartbeat_loop)
    assert "shutdown_event.is_set() and not executing_event.is_set()" in src, \
        "the loop still exits on shutdown regardless of work in flight"


def test_it_does_not_spin_once_shutdown_is_set():
    """shutdown_event.wait() returns instantly once set, so the draining path has to sleep
    the interval instead or this becomes a hot loop for the length of a render."""
    src = inspect.getsource(mod.heartbeat_loop)
    drain = src.index("if shutdown_event.is_set():", src.index("while True:"))
    assert "asyncio.sleep(settings.heartbeat_interval)" in src[drain:drain + 400]


def test_the_call_site_passes_it():
    """A parameter nothing passes is a parameter that defaults to broken."""
    src = pathlib.Path(mod.__file__).read_text()
    i = src.index("heartbeat_loop(queue")
    assert "executing_event" in src[i:i + 220], \
        "the loop was given the flag but never receives it"


class _Beats:
    """Counts heartbeats. `beat` is whatever the loop calls per tick."""

    def __init__(self):
        self.n = 0


@pytest.mark.asyncio
async def test_it_keeps_beating_through_a_drain_then_stops(monkeypatch):
    """The behaviour, not the source: beats continue while executing is set after shutdown,
    and stop once it clears."""
    monkeypatch.setattr(mod.settings, "heartbeat_interval", 0.01)
    shutdown = asyncio.Event()
    executing = asyncio.Event()
    beats = _Beats()

    # Stand in for everything the body does with the outside world.
    class _Comfy:
        async def check_health(self):
            beats.n += 1
            return True

        async def check_queue_busy(self):
            return True

    class _Queue:
        async def send_heartbeat(self, *a, **k):
            return {}

        async def heartbeat(self, *a, **k):
            return {}

    monkeypatch.setattr(mod, "get_gpu_stats", lambda: {})
    monkeypatch.setattr(mod, "get_sd_scripts_status", lambda: {"sd_scripts_training": False})

    executing.set()
    task = asyncio.create_task(mod.heartbeat_loop(
        _Queue(), _Comfy(), "w-1", lambda: "3090.zero", shutdown, asyncio.Event(), executing))
    await asyncio.sleep(0.05)
    shutdown.set()                      # SIGTERM while a segment is running
    await asyncio.sleep(0.05)
    during = beats.n
    assert during > 0, "it stopped beating the moment shutdown was requested"

    executing.clear()                   # the segment finished
    await asyncio.wait_for(task, timeout=2)
    assert beats.n >= during
