"""Download LoRA files from S3 (via API proxy) to ComfyUI's local loras directory.

Only downloads files that are missing locally. Files are cached forever.
Uses atomic writes (.tmp → rename) to prevent corrupt partial files.
"""

import hashlib
import logging
import os

from daemon.config import settings
from daemon.queue_client import QueueClient
from daemon.schemas import LoraItem

logger = logging.getLogger(__name__)

MIN_LORA_SIZE = 10 * 1024 * 1024  # 10 MB — anything smaller is likely corrupt
PARTIAL_EXTENSIONS = (".aria2", ".tmp", ".part")


def _loras_dir() -> str:
    """Return the LoRA download directory path.

    Uses ``lora_cache_dir`` when set (e.g. a persistent volume on RunPod),
    otherwise falls back to ``comfyui_path/models/loras``.
    """
    if settings.lora_cache_dir:
        return settings.lora_cache_dir
    return os.path.join(settings.comfyui_path, "models", "loras")


def _cleanup_partials(lora_dir: str, filename: str) -> None:
    """Remove any partial download artifacts for a given filename."""
    for ext in PARTIAL_EXTENSIONS:
        partial = os.path.join(lora_dir, filename + ext)
        if os.path.exists(partial):
            os.remove(partial)
            logger.info("       Removed partial: %s", partial)


async def ensure_loras_available(
    loras: list[LoraItem], queue: QueueClient
) -> None:
    """Ensure all LoRA .safetensors files are present locally.

    Downloads missing files from S3 via the API proxy.
    Uses atomic writes: download to .tmp then rename to final path.
    Re-downloads files that are suspiciously small (< 10 MB).
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    for item in loras:
        for label, filename, s3_uri in [
            ("high", item.high_file, item.high_s3_uri),
            ("low", item.low_file, item.low_s3_uri),
        ]:
            if not filename or not s3_uri:
                continue

            local_path = os.path.join(lora_dir, filename)

            # Clean up any partial download artifacts
            _cleanup_partials(lora_dir, filename)

            # Check if existing file is too small (likely corrupt/partial)
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size < MIN_LORA_SIZE:
                    logger.warning(
                        "       %s: %s exists but only %.1f MB (< 10 MB) — re-downloading",
                        label, filename, file_size / (1024 * 1024),
                    )
                    os.remove(local_path)
                else:
                    mb = file_size / (1024 * 1024)
                    logger.info("       %s: %s (cached, %.1f MB)", label, filename, mb)
                    continue

            logger.info("       %s: downloading %s...", label, filename)
            data = await queue.download_file(s3_uri)

            mb = len(data) / (1024 * 1024)
            if len(data) < MIN_LORA_SIZE:
                raise RuntimeError(
                    f"LoRA {filename} download too small: {mb:.1f} MB (expected >= 10 MB)"
                )

            # Atomic write: write to .tmp then rename
            tmp_path = local_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.rename(tmp_path, local_path)
            logger.info("       %s: %s saved (%.1f MB)", label, filename, mb)


# ---------------------------------------------------------------------------
# Catalogue sync: hold every LoRA the bucket has, not just the one a claim named.
#
# ensure_loras_available() above is the WAN-era, pull-on-demand path: it fetches what a
# claimed segment names, and skips anything already on disk WITH THAT NAME. The LTX path
# never had an equivalent at all, so an LTX worker could only ever use whatever happened to
# be baked onto its volume — which is why a fresh pod renders nothing with a character.
#
# The trap this fixes, and the reason the diff is on content and not on names: LoRAs get
# RETRAINED and republished under the same name. k3lly2026_v2 today is not the file that
# name meant last week. A name-only check calls that a hit, keeps the old weights, and the
# worker renders the wrong character while the console shows the right one — wrong output
# rather than a failure, which is the expensive kind.
# ---------------------------------------------------------------------------


def _local_md5(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _needs_download(local_path: str, remote: dict) -> str | None:
    """Return a human reason to download, or None when the local copy is already right.

    The reason is returned rather than logged here so the caller can put it in one line
    per LoRA — when this goes wrong on a pod, the question is always "why did it decide
    to (not) fetch this one", and the answer should be in the log without a rerun.
    """
    if not os.path.exists(local_path):
        return "missing"

    size = os.path.getsize(local_path)
    if size < MIN_LORA_SIZE:
        return f"local copy is only {size / (1024 * 1024):.1f} MB — truncated"

    if remote.get("multipart"):
        # A multipart ETag is "<md5-of-md5s>-<partcount>", not the md5 of the content, so
        # there is nothing to compare it against. Falling back to size keeps us from
        # re-downloading the same file on every single sync, forever; it also cannot catch
        # a retrain that happened to land on the same byte count. Say so, out loud, rather
        # than let a silent weaker check look like the strong one.
        if size != remote["size"]:
            return f"size {size} != remote {remote['size']}"
        logger.warning(
            "       %s: multipart ETag — verified by SIZE ONLY, a same-size retrain "
            "would not be detected", os.path.basename(local_path),
        )
        return None

    if size != remote["size"]:
        return f"size {size} != remote {remote['size']}"

    local = _local_md5(local_path)
    if local != remote["etag"]:
        return f"content changed (local md5 {local[:12]}, remote {remote['etag'][:12]})"

    return None


async def sync_lora_catalog(queue: QueueClient) -> bool:
    """Diff the local LoRA directory against the bucket and fetch what differs.

    Returns True when the local set matches the catalogue. Deliberately NOT fatal to boot:
    a worker that exits here restart-loops, and a restart loop on a rented pod costs real
    money and is miserable to diagnose (we have already lost an hour to one). A worker with
    a missing LoRA fails the segment that needs it, loudly, with the name in the error —
    which is a better failure than no worker at all.
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    try:
        catalog = await queue.list_loras()
    except Exception as e:
        logger.error("LoRA sync: could not list the catalogue (%s) — keeping what is on disk", e)
        return False

    logger.info("LoRA sync: %d in the catalogue, checking %s", len(catalog), lora_dir)

    # The bucket files LoRAs under character/ and content/, but they land here FLAT, because
    # that is what a ComfyUI LoraLoader and ltx_characters.char_lora both expect — neither
    # knows the prefix exists. Flattening means two kinds could claim one filename, and the
    # second would silently overwrite the first: a character LoRA replaced by a motion LoRA
    # renders a stranger, successfully. Refuse the pair rather than pick one.
    seen: dict[str, str] = {}
    clashing: set[str] = set()
    for r in catalog:
        prior = seen.get(r["name"])
        if prior is not None and prior != r.get("key", r["name"]):
            clashing.add(r["name"])
            logger.error(
                "       %s: same filename under two prefixes (%s and %s) — SKIPPING BOTH, "
                "rename one in the bucket", r["name"], prior, r.get("key", r["name"]),
            )
        seen[r["name"]] = r.get("key", r["name"])

    ok = not clashing
    fetched = 0
    for remote in catalog:
        name = remote["name"]
        if name in clashing:
            continue
        local_path = os.path.join(lora_dir, name)
        _cleanup_partials(lora_dir, name)

        reason = _needs_download(local_path, remote)
        if reason is None:
            logger.info("       %s: current", name)
            continue

        logger.info("       %s: %s — downloading", name, reason)
        tmp_path = local_path + ".tmp"
        try:
            got = await queue.stream_file(remote["uri"], tmp_path)
        except Exception as e:
            logger.error("       %s: download failed: %s", name, e)
            ok = False
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            continue

        # Verify BEFORE the rename, so a bad transfer can never take the place of a good
        # local copy. The .tmp is dropped and whatever was there is left alone.
        if not remote.get("multipart") and got != remote["etag"]:
            logger.error(
                "       %s: md5 mismatch after download (got %s, expected %s) — discarding",
                name, got[:12], remote["etag"][:12],
            )
            os.remove(tmp_path)
            ok = False
            continue

        size = os.path.getsize(tmp_path)
        if size != remote["size"]:
            logger.error(
                "       %s: size mismatch after download (got %d, expected %d) — discarding",
                name, size, remote["size"],
            )
            os.remove(tmp_path)
            ok = False
            continue

        os.rename(tmp_path, local_path)
        fetched += 1
        logger.info("       %s: saved (%.1f MB)", name, size / (1024 * 1024))

    logger.info("LoRA sync: %d fetched, %d already current", fetched, len(catalog) - fetched)
    return ok


async def ensure_named_loras_present(
    names: list[str | None], queue: QueueClient
) -> list[str]:
    """Fetch any of `names` this worker does not already hold. Returns what was fetched.

    The boot sync (sync_lora_catalog) answers "is my whole set correct?" and verifies by
    CONTENT, because a retrained LoRA republished under the same name is the failure that
    renders a stranger. This answers a narrower question at claim time — "do I have a file
    by this name at all?" — and deliberately does NOT re-verify hashes.

    That split is the point. Hashing 650 MB per LoRA on every claim would cost seconds of
    GPU time per segment to re-answer a question boot already answered, while the case this
    exists for is simply a LoRA published after this worker booted. Correctness stays with
    boot; availability lives here.

    Missing LoRAs are fetched, not merely reported: a worker that could have downloaded the
    file and instead failed the segment is a worker that made the queue wait for a restart.
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    wanted: list[str] = []
    for raw in names:
        if not raw:
            continue
        n = str(raw).strip()
        if not n or n.lower() == "none":
            continue
        if not n.endswith(".safetensors"):
            n += ".safetensors"
        local = os.path.join(lora_dir, n)
        # Size floor as well as existence: a truncated file from an interrupted fetch is
        # present but useless, and would otherwise be treated as a hit forever.
        if os.path.exists(local) and os.path.getsize(local) >= MIN_LORA_SIZE:
            continue
        wanted.append(n)

    if not wanted:
        return []

    logger.info("LoRA on-demand: %s not present locally — fetching", ", ".join(wanted))
    try:
        catalog = {o["name"]: o for o in await queue.list_loras()}
    except Exception as e:
        logger.error("LoRA on-demand: could not list the catalogue (%s)", e)
        return []

    fetched: list[str] = []
    for name in wanted:
        remote = catalog.get(name)
        if remote is None:
            # Nothing to fetch. Said plainly here so the engine's later "no such lora" is
            # not the first hint, and so the cause reads as "not published" rather than
            # "download failed".
            logger.error(
                "       %s: not in the bucket at all — the pose names a LoRA that was "
                "never uploaded", name,
            )
            continue
        local = os.path.join(lora_dir, name)
        _cleanup_partials(lora_dir, name)
        tmp = local + ".tmp"
        try:
            got = await queue.stream_file(remote["uri"], tmp)
        except Exception as e:
            logger.error("       %s: download failed: %s", name, e)
            if os.path.exists(tmp):
                os.remove(tmp)
            continue
        # Verified before the rename, exactly as the boot sync does. A bad transfer must
        # never take the place of a file the next claim will trust.
        if not remote.get("multipart") and got != remote["etag"]:
            logger.error("       %s: md5 mismatch after download — discarding", name)
            os.remove(tmp)
            continue
        os.rename(tmp, local)
        fetched.append(name)
        logger.info("       %s: fetched (%.1f MB)", name, os.path.getsize(local) / (1024 * 1024))
    return fetched
