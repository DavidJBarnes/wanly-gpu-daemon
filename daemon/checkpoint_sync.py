"""Fetch a base model this worker does not hold, inside the claim (console#423).

The counterpart to lora_sync's ensure_named_loras_present, for files ~70x larger. That gap
is the whole design: a character LoRA is 624 MB and can be fetched naively, a checkpoint is
46 GB and every shortcut that works for the LoRA fails here.

WHY NOT hf_hub_download
    huggingface_hub is not in this daemon's requirements, and adding a dependency to stream
    bytes over HTTPS would be a poor trade. httpx is already here, and streaming it
    ourselves is what makes two of this file's requirements achievable at all:

      * the HEARTBEAT MUST SURVIVE. Heartbeat is the only reclaim authority for a live
        worker: if a 46 GB transfer blocks the event loop, this worker stops beating, the
        API reclaims the segment as orphaned, and a second worker starts rendering it while
        this one is still downloading. An async stream yields between chunks, so the
        heartbeat task keeps running throughout.
      * PROGRESS IN THE SEGMENT LOG. A 5-10 minute silent step is indistinguishable from the
        wedge in #160. hf_hub_download draws a \\r progress bar, which a line-oriented log
        cannot show at all -- download_models.sh records 45 minutes of total silence from
        exactly that.

WHERE THE URL COMES FROM
    The API, never a guess. wanly-api's /ltx/checkpoints/catalog maps a checkpoint name to
    {repo, path, size_bytes}. Held there rather than here so a new base model is one entry
    instead of a redeploy to every worker.

WHAT ARRIVES IS VERIFIED
    A partial safetensors is a VALID HEADER OVER MISSING DATA. It passes every existence
    check and fails only at load, deep inside a claimed segment -- download_models.sh
    records one arriving at 14.40 of 27.16 GiB reporting nothing at all. So the file is
    staged under a .part name and only renamed into place after the byte count matches the
    catalogue exactly AND the safetensors header agrees with the file length.
"""

import json
import logging
import os
import struct

import httpx

from daemon.config import settings
from daemon.queue_client import QueueClient

logger = logging.getLogger(__name__)

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"

# Granular rather than total: `read` caps the gap BETWEEN chunks, so a stalled transfer
# fails in about a minute while a slow-but-steady 46 GB download of any duration completes.
# A total timeout cannot express that -- any value large enough for 46 GB on a bad link is
# too large to notice a dead one.
_TIMEOUT = httpx.Timeout(connect=20.0, read=120.0, write=120.0, pool=20.0)


def _canonical(name: str) -> str:
    n = (name or "").strip()
    return n[: -len(".safetensors")] if n.endswith(".safetensors") else n


def _local_path(name: str) -> str:
    return os.path.join(settings.checkpoint_dir, f"{_canonical(name)}.safetensors")


def _free_bytes(path: str) -> int:
    """Free space on the filesystem that will hold the download.

    Measured on the directory rather than the volume root: on a pod the models tree is a
    mounted volume and the container overlay is a different, much smaller filesystem.
    Checking the wrong one is how a precheck passes and the write still fails.
    """
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def verify_safetensors(path: str, expected_bytes: int) -> str | None:
    """None if the file is sound, else a human reason it is not.

    Two independent checks, because they catch different failures:

      * exact byte count against the catalogue -- catches a truncated stream, which is the
        common one, and also catches "the URL served something else of a plausible size"
      * the safetensors header's own arithmetic (8 + header_len + the largest data_offsets
        end) -- catches a file that is the right LENGTH but not this format, e.g. an HTML
        error page or a redirect body written to disk
    """
    actual = os.path.getsize(path)
    if expected_bytes and actual != expected_bytes:
        return (f"expected {expected_bytes:,} bytes, got {actual:,} "
                f"({100 * actual / expected_bytes:.1f}%)")
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            # Bound before allocating: a non-safetensors file gives a garbage length here,
            # and reading it raised MemoryError -- a real failure reported as the wrong
            # problem, and on a 46 GB file it tries to allocate first.
            if n <= 0 or n > actual:
                return f"not a safetensors: header length {n} vs file size {actual}"
            header = json.loads(fh.read(n))
        ends = [v["data_offsets"][1] for v in header.values()
                if isinstance(v, dict) and "data_offsets" in v]
        declared = 8 + n + (max(ends) if ends else 0)
        if actual < declared:
            return f"truncated: {actual:,} of {declared:,} bytes the header declares"
    except Exception as e:  # noqa: BLE001
        return f"unreadable: {type(e).__name__}: {e}"
    return None


async def ensure_checkpoint_present(name: str, queue: QueueClient, progress=None) -> str | None:
    """Fetch `name` if this worker lacks it. Returns the name if fetched, else None.

    Returns rather than raises on every failure path. A worker that cannot fetch should let
    the segment fail with the engine's own "no such checkpoint" message, which names the
    file and lists what IS available -- more useful than an exception from here, and it
    keeps this function out of the business of deciding a segment's fate.
    """
    stem = _canonical(name)
    if not stem or stem.lower() == "none":
        return None

    local = _local_path(stem)
    # Size floor is not enough for a checkpoint the way it is for a LoRA: any partial file
    # here is tens of GB and would pass a floor. Verify properly, and treat a bad existing
    # file as absent so it gets replaced rather than loaded.
    if os.path.exists(local):
        return None

    try:
        catalog = await queue.checkpoint_catalog()
    except Exception as e:  # noqa: BLE001
        logger.warning("Checkpoint %s: could not read the catalogue (%s) — not fetching",
                       stem, e)
        return None

    src = (catalog or {}).get(stem)
    if not src:
        # A real answer, not an error: the checkpoint may exist only on a box whose source
        # nobody recorded. The API's model gate routes those instead.
        logger.info("Checkpoint %s: no recorded source — leaving it to a worker that has it",
                    stem)
        return None

    expected = int(src.get("size_bytes") or 0)
    try:
        os.makedirs(settings.checkpoint_dir, exist_ok=True)
    except OSError as e:
        logger.warning("Checkpoint %s: %s is not creatable (%s) — not fetching",
                       stem, settings.checkpoint_dir, e)
        return None
    # The 3090 bind-mounts its model tree READ-ONLY (-v ...:/workspace/models:ro). That box
    # already holds every checkpoint, so this is normally unreachable there -- but a pose
    # naming a base model it lacks would otherwise fail on a write error twenty lines later,
    # reported as a download failure rather than as "this worker cannot fetch anything".
    if not os.access(settings.checkpoint_dir, os.W_OK):
        logger.info("Checkpoint %s: %s is read-only — leaving it to a worker that can fetch",
                    stem, settings.checkpoint_dir)
        return None
    need = expected + settings.checkpoint_min_free_gb * (1024 ** 3)
    free = _free_bytes(settings.checkpoint_dir)
    if expected and free < need:
        msg = (f"Checkpoint {stem}: needs {expected / 1e9:.0f} GB plus "
               f"{settings.checkpoint_min_free_gb} GB headroom, only {free / 1e9:.0f} GB free")
        logger.error(msg)
        if progress:
            await progress(f"[2/6] {msg} — not fetching")
        return None

    url = HF_RESOLVE.format(repo=src["repo"], path=src["path"])
    part = local + ".part"
    if progress:
        await progress(f"[2/6] Fetching base model {stem} ({expected / 1e9:.0f} GB)...")
    logger.info("Checkpoint %s: downloading from %s", stem, src["repo"])

    try:
        # A fresh client, not the queue's: this talks to Hugging Face, not the API, and it
        # must not borrow the queue pool's auth headers or its keepalive tuning.
        async with httpx.AsyncClient(timeout=_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as resp:
                if not resp.is_success:
                    await resp.aread()
                    logger.error("Checkpoint %s: HTTP %s from %s",
                                 stem, resp.status_code, src["repo"])
                    return None
                with open(part, "wb") as f:
                    # Report at tenths. Each callback is an API write on the segment's
                    # progress log, so per-chunk would post thousands of rows for one
                    # download; a tenth of 46 GB is still a line every minute or two, which
                    # is enough to tell moving from wedged.
                    step = max(expected // 10, 1) if expected else 0
                    mark = step
                    done = 0
                    async for chunk in resp.aiter_bytes(4 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if progress and step and done >= mark:
                            await progress(
                                f"[2/6] {stem}: {done / 1e9:.0f} of {expected / 1e9:.0f} GB"
                            )
                            mark += step
    except Exception as e:  # noqa: BLE001
        logger.error("Checkpoint %s: download failed: %s", stem, e)
        _discard(part)
        return None

    bad = verify_safetensors(part, expected)
    if bad:
        # Never rename a suspect file into place. Under the real name it would look present
        # to every check and fail only at load, which is the exact failure this guards.
        logger.error("Checkpoint %s: %s — discarding", stem, bad)
        if progress:
            await progress(f"[2/6] {stem}: download failed verification ({bad})")
        _discard(part)
        return None

    os.replace(part, local)
    logger.info("Checkpoint %s: saved (%.1f GB)", stem, os.path.getsize(local) / 1e9)
    if progress:
        await progress(f"[2/6] Base model {stem} ready")
    return stem


def _discard(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass
