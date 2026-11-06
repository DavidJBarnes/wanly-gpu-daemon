"""Hologram-only worker: claims AR holograms and smashcut concats, nothing else (wanly-gpu-docker#228).

    python -m daemon.hologram_worker

WHY IT EXISTS

    Holograms used to run only inside the render daemon, behind its work lock (#192): one
    waited for the render in flight -- up to ~30 minutes at 720p -- and a 3090 that picked one
    up held its next render for the 4-6.5 minutes the hologram took. Neither needs a 3090. This
    runs in the 2070's scene container instead, beside the captioner.

WHAT IT IS NOT

    Not a worker row of its own. The container's supervisor already registers the box and
    heartbeats it (as kind `hologram`, which is what lets this claim at all), and two writers on
    one row fight. This process only claims, against that row's id, which the supervisor writes
    to WORKER_ID_FILE -- re-read on every poll, because a re-registration can change it.

    No ComfyUI, no render loop, no LoRA sync: execute_segment's hologram and smashcut branches
    never touch them.

SHARING THE BOX: RAM IS THE ONLY CONTENDED RESOURCE

    The image's onnxruntime is CPU-only, so a hologram never touches the card the captioner and
    the prompt enhancer take turns on. RAM is shared with them, through the container's memory
    cap. The frame pass streams from disk (one frame in memory, not the clip), and on top of that
    nothing is claimed unless the container has HOLOGRAM_MIN_HEADROOM_GB of non-reclaimable
    headroom -- so a hologram waits its turn instead of pushing the captioner into the OOM
    killer.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import uuid

from daemon.config import settings
from daemon.executor import execute_segment
from daemon.queue_client import QueueClient
from daemon.schemas import SegmentResult

logger = logging.getLogger("daemon.hologram_worker")

#: Where the supervisor writes the box's worker id. The same path the render daemon uses.
WORKER_ID_FILE = os.environ.get("WORKER_ID_FILE", "/run/wanly/worker-id")
#: Non-reclaimable memory the container must have free before a claim. A streamed hologram
#: peaks well under this; the margin is for whatever the captioner does meanwhile.
MIN_HEADROOM_GB = float(os.environ.get("HOLOGRAM_MIN_HEADROOM_GB", "3"))
CGROUP = os.environ.get("HOLOGRAM_CGROUP_DIR", "/sys/fs/cgroup")


def worker_id() -> uuid.UUID | None:
    """The box's worker row, or None until the supervisor has registered it."""
    try:
        with open(WORKER_ID_FILE) as f:
            return uuid.UUID(f.read().strip())
    except (OSError, ValueError):
        return None


def headroom_gb() -> float | None:
    """The container's memory cap minus its anonymous memory, in GB; None when uncapped.

    Anonymous, not memory.current: current counts page cache, and the captioner's model files
    sit in it -- reclaimable, so counting them would read a healthy box as full.
    """
    try:
        with open(os.path.join(CGROUP, "memory.max")) as f:
            cap = f.read().strip()
        if cap == "max":
            return None
        with open(os.path.join(CGROUP, "memory.stat")) as f:
            stat = dict(line.split() for line in f if line.strip())
        return (int(cap) - int(stat["anon"])) / 1024 ** 3
    except (OSError, KeyError, ValueError):
        return None


def announce_ready() -> None:
    """Tell the supervisor this process is up (WANLY_READY_FILE). It has no port to probe."""
    path = os.environ.get("WANLY_READY_FILE")
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("ready\n")
    except OSError as e:
        logger.warning("could not write WANLY_READY_FILE (%s): %s", path, e)


def may_claim() -> tuple[bool, str]:
    room = headroom_gb()
    if room is not None and room < MIN_HEADROOM_GB:
        return False, f"{room:.1f} GB memory headroom, want {MIN_HEADROOM_GB:.1f}"
    return True, ""


async def poll_once(queue: QueueClient) -> bool:
    """Claim and run at most one segment. True when one ran."""
    wid = worker_id()
    if wid is None:
        return False
    ok, _ = may_claim()
    if not ok:
        return False
    segment = await queue.claim_next(wid, settings.friendly_name, kind="hologram")
    if segment is None:
        return False
    try:
        # No ComfyUI: the hologram and smashcut branches never use it.
        await execute_segment(segment, None, queue)
    except Exception as e:
        logger.exception("Unexpected error executing hologram segment %s", segment.id)
        try:
            await queue.update_segment(
                segment.id,
                SegmentResult(status="failed", error_message=f"{type(e).__name__}: {e}"[:2000]),
            )
        except Exception as report_err:
            logger.error("Failed to report hologram failure: %s", report_err)
    return True


async def run() -> None:
    queue = QueueClient()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    logger.info("=== Wanly hologram worker === %s | API %s | onnx threads %s | min headroom %.1f GB",
                settings.friendly_name, settings.queue_url, settings.onnx_threads or "default",
                MIN_HEADROOM_GB)
    announce_ready()
    said = ""
    try:
        while not stop.is_set():
            ran = False
            try:
                if worker_id() is None:
                    note = f"waiting for the supervisor to register this box ({WORKER_ID_FILE})"
                else:
                    note = may_claim()[1]
                if note != said:
                    logger.info(note or "claiming holograms")
                    said = note
                ran = await poll_once(queue)
            except Exception as e:
                logger.error("Hologram poll failed: %s: %s", type(e).__name__, e)
            if not ran:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=settings.poll_interval)
                except asyncio.TimeoutError:
                    pass
    finally:
        await queue.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    # As in daemon.main: otherwise every 5 s poll logs a line (~17k a day).
    logging.getLogger("httpx").setLevel(logging.WARNING)
    asyncio.run(run())


if __name__ == "__main__":
    main()
