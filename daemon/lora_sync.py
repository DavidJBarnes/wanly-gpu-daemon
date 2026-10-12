"""Download LoRA files from S3 (via API proxy) to ComfyUI's local loras directory.

Only downloads files that are missing locally. Files are cached forever.
Uses atomic writes (.tmp → rename) to prevent corrupt partial files.
"""

import asyncio
import hashlib
import logging
import os
import time
from datetime import datetime, timezone

from daemon import phase

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


# ---------------------------------------------------------------------------------------
# The inventory this worker reports to the API (console#... / daemon#165).
#
# Deliberately a CACHE of what the last sync concluded, never a fresh computation. The
# honest answer to "do I hold the right k3lly2026_v2?" is an md5 of 650 MB; doing that on
# every heartbeat would hash gigabytes a minute to render a status page. sync_lora_catalog()
# already reaches that verdict — this stores it.
#
# Which means it can go stale in one specific way: ensure_named_loras_present() changes the
# disk mid-life, hours after boot. It updates this too. Without that the page would show
# "deferred" for a LoRA already downloaded, which is worse than showing nothing — it is
# confidently wrong.
# ---------------------------------------------------------------------------------------

# state          meaning                                            severity
# current        present, md5 matches the ETag                      fine
# deferred       absent BY DESIGN (content, fetched on demand)      fine
# unverifiable   present, multipart ETag, size-only check           note
# stale          present but the content differs                    the one that matters
# missing        should be here and is not                          real problem
_INVENTORY: dict = {"synced_at": None, "dir": None, "items": []}


def inventory() -> dict:
    """What the last sync concluded, for the heartbeat. Never recomputes."""
    return dict(_INVENTORY, items=list(_INVENTORY["items"]))


def _record(items: list[dict]) -> None:
    _INVENTORY["synced_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _INVENTORY["dir"] = _loras_dir()
    _INVENTORY["items"] = items


def _note_fetched(names: list[str]) -> None:
    """Mark on-demand fetches current, so the report does not contradict the disk."""
    if not names:
        return
    by_name = {i["name"]: i for i in _INVENTORY["items"]}
    for n in names:
        item = by_name.get(n)
        if item is not None:
            item["state"] = "current"
            item.pop("note", None)
        else:
            _INVENTORY["items"].append({"name": n, "kind": "unknown", "state": "current"})
    _INVENTORY["synced_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")


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


# ---------------------------------------------------------------------------------------
# LoRAs the background sync has not finished with yet (wanly-gpu-daemon#189).
#
# The sync used to run BEFORE registration, which made the claim question moot: nothing could
# claim until every eager LoRA was in place. It now runs AFTER registration, in the
# background, so a claim can arrive while Payton_v2_final is 358 MB into a 650 MB download.
# A segment naming that LoRA must not be claimed until the file lands -- not failed, not
# fetched a second time in parallel into the same .tmp, just left in the queue.
#
# So every eager LoRA the sync has not yet VERIFIED or FETCHED is "pending". The daemon
# sends the set on every claim and the API skips segments naming any of them (wanly-api,
# `pending_loras`). Same-size files count until their md5 is checked, because a retrain
# republished under the same name at the same size is exactly what the content check exists
# to catch; that pass is about a second per file, so the window is short.
#
# Each pending name carries an Event, set when the sync is done with it either way, so the
# claim-time path can WAIT for a file the sync is already fetching rather than race it --
# the fallback for an API too old to honour `pending_loras`.
# ---------------------------------------------------------------------------------------
_PENDING: dict[str, asyncio.Event] = {}
#: Set once the catalogue has been listed (or listing failed): until then nobody knows which
#: LoRAs are pending, so the claim loop waits for it. It is one API call -- about a second.
_LISTED = asyncio.Event()


def pending_loras() -> list[str]:
    """Filenames the background sync has not verified or fetched yet, for the claim."""
    return sorted(_PENDING)


async def wait_until_listed(timeout: float | None = None) -> None:
    """Block until the sync knows what is pending. Bounded: a hung listing must not stop
    claims forever, and an unlisted catalogue means nothing is pending."""
    try:
        await asyncio.wait_for(_LISTED.wait(), timeout=timeout)
    except asyncio.TimeoutError:
        pass


def _hold(names) -> None:
    for n in names:
        _PENDING.setdefault(n, asyncio.Event())


def _release(name: str) -> None:
    ev = _PENDING.pop(name, None)
    if ev is not None:
        ev.set()


def _release_all() -> None:
    for n in list(_PENDING):
        _release(n)


def _fmt_eta(s: float | None) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 60}m {s % 60:02d}s" if s >= 60 else f"{s}s"


async def sync_lora_catalog(queue: QueueClient, eager_kinds: tuple[str, ...] = ("character",)) -> bool:
    """Diff the local LoRA directory against the bucket and fetch what differs.

    RUNS IN THE BACKGROUND, AFTER REGISTRATION (wanly-gpu-daemon#189). It used to block
    registration, and every byte fetched here was a worker missing from the Workers page:
    on 2026-10-12 a mode switch into render re-ran it, Payton_v2_final (~650 MB at ~0.4 MB/s)
    took the box offline for 10+ minutes and a training run waited behind it. Now the worker
    is on the board within seconds, says what it is fetching (daemon/phase.py), and only the
    segments naming a LoRA still in flight wait for it (pending_loras()).

    TWO PASSES. First verify everything (cheap: stat, then md5 off the event loop), releasing
    each LoRA from `pending` as it is found current -- so within about a minute only the files
    really being downloaded hold anything back. Then download what needs it, one at a time,
    reporting file / MB / rate / ETA / files left.

    EAGER ONLY FOR SOME KINDS, and by default only for character LoRAs. Character LoRAs are a
    small set and effectively every job names one. Content LoRAs are left to
    ensure_named_loras_present() at claim time: the cost moves from every sync to the first
    claim that needs one. Non-eager kinds are still CHECKED, and a stale one is reported --
    only the download is deferred.

    Returns True when everything eager is in place and nothing deferred looks broken.
    Deliberately NOT fatal: a worker with a missing LoRA fails the segment that needs it,
    loudly, with the name in the error -- a better failure than no worker at all.
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    phase.set_phase("syncing_loras", "Listing the LoRA catalogue")
    try:
        catalog = await queue.list_loras()
    except Exception as e:
        logger.error("LoRA sync: could not list the catalogue (%s) — keeping what is on disk", e)
        _LISTED.set()
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

    candidates = [r for r in catalog if r["name"] not in clashing]
    # Held BEFORE the claim loop is let go: every eager LoRA is pending until proven current.
    _hold(r["name"] for r in candidates if r.get("kind", "character") in eager_kinds)
    _LISTED.set()

    ok = not clashing
    fetched = 0
    deferred: list[str] = []
    to_fetch: list[tuple[dict, str]] = []
    # One entry per catalogue object, carrying the verdict _needs_download() already reached.
    # Built here rather than in a second pass so there is exactly one implementation of
    # "is this current" — two would drift, which is the recipes.json trap in the wiki.
    report: list[dict] = []
    try:
        # ---- pass 1: verify ----
        for i, remote in enumerate(candidates, 1):
            name = remote["name"]
            phase.update(f"Checking LoRAs on disk ({i}/{len(candidates)})",
                         files_checked=i, files_in_catalogue=len(candidates))
            local_path = os.path.join(lora_dir, name)
            _cleanup_partials(lora_dir, name)

            # Off the event loop: an md5 of 650 MB is about a second, and forty of them on the
            # loop would stall the heartbeat and the claim this whole change exists to keep
            # moving.
            reason = await asyncio.to_thread(_needs_download, local_path, remote)
            if reason is None:
                logger.info("       %s: current", name)
                report.append({
                    "name": name, "kind": remote.get("kind", "unknown"),
                    # A multipart ETag is not an md5, so "current" here rests on size alone.
                    "state": "unverifiable" if remote.get("multipart") else "current",
                    **({"note": "multipart ETag — size only"} if remote.get("multipart") else {}),
                })
                _release(name)
                continue

            kind = remote.get("kind", "character")
            if kind not in eager_kinds:
                # Checked, reported, and deliberately not fetched here. The claim that names it
                # will fetch it; nothing waits for a LoRA nothing has asked for yet.
                deferred.append(name)
                logger.info("       %s: %s — deferred (%s, fetched on demand)", name, reason, kind)
                # "missing" is by design here and must not read as a fault; but a file that is
                # PRESENT AND WRONG is a fault whether or not we chose to defer the download.
                report.append({
                    "name": name, "kind": kind,
                    "state": "deferred" if reason == "missing" else "stale",
                    "note": reason,
                })
                continue
            logger.info("       %s: %s — will download", name, reason)
            to_fetch.append((remote, reason))

        # ---- pass 2: download ----
        total_all = sum(int(r.get("size") or 0) for r, _ in to_fetch)
        done_before = 0
        for idx, (remote, reason) in enumerate(to_fetch):
            name = remote["name"]
            size = int(remote.get("size") or 0)
            left = len(to_fetch) - idx
            local_path = os.path.join(lora_dir, name)
            tmp_path = local_path + ".tmp"
            started = time.monotonic()
            stem = name[: -len(".safetensors")] if name.endswith(".safetensors") else name
            phase.set_phase(
                "syncing_loras", f"Downloading {stem} (0 / {phase.mb(size)} MB), "
                f"{left} file{'s' if left != 1 else ''} left",
                file=name, done_bytes=0, total_bytes=size, rate_bps=None, eta_s=None,
                files_done=idx, files_total=len(to_fetch), files_remaining=left,
                all_done_bytes=done_before, all_total_bytes=total_all,
            )
            logger.info("       %s: %s — downloading", name, reason)

            async def _tick(done, total, _name=stem, _started=started, _before=done_before,
                            _left=left, _size=size):
                elapsed = max(time.monotonic() - _started, 0.001)
                rate = done / elapsed
                remaining = (total_all - _before - done)
                eta = remaining / rate if rate > 0 else None
                phase.update(
                    f"Downloading {_name} ({phase.mb(done)} / {phase.mb(_size or total)} MB, "
                    f"{rate / (1024 * 1024):.1f} MB/s, ETA {_fmt_eta(eta)}), "
                    f"{_left} file{'s' if _left != 1 else ''} left",
                    done_bytes=done, rate_bps=round(rate), eta_s=round(eta) if eta else None,
                    all_done_bytes=_before + done,
                )

            try:
                got = await queue.stream_file(remote["uri"], tmp_path, on_progress=_tick,
                                              total=size, step=4 * 1024 * 1024)
            except Exception as e:
                logger.error("       %s: download failed: %s", name, e)
                ok = False
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                _release(name)
                done_before += size
                continue
            done_before += size

            # Verify BEFORE the rename, so a bad transfer can never take the place of a good
            # local copy. The .tmp is dropped and whatever was there is left alone.
            if not remote.get("multipart") and got != remote["etag"]:
                logger.error(
                    "       %s: md5 mismatch after download (got %s, expected %s) — discarding",
                    name, got[:12], remote["etag"][:12],
                )
                os.remove(tmp_path)
                ok = False
                _release(name)
                continue

            got_size = os.path.getsize(tmp_path)
            if got_size != remote["size"]:
                logger.error(
                    "       %s: size mismatch after download (got %d, expected %d) — discarding",
                    name, got_size, remote["size"],
                )
                os.remove(tmp_path)
                ok = False
                _release(name)
                continue

            os.rename(tmp_path, local_path)
            fetched += 1
            report.append({"name": name, "kind": remote.get("kind", "unknown"),
                           "state": "current"})
            _release(name)
            logger.info("       %s: saved (%.1f MB in %.0fs)", name,
                        got_size / (1024 * 1024), time.monotonic() - started)
    finally:
        # Whatever happens -- including a cancel on shutdown -- nothing may stay pending, or a
        # claim-time waiter would hang on a sync that is no longer running.
        _release_all()

    # Whatever never made it into the report failed to download — present in the catalogue,
    # absent from disk, and not deferred. That is the "real problem" state.
    seen_names = {r["name"] for r in report}
    for remote in candidates:
        if remote["name"] not in seen_names:
            report.append({"name": remote["name"], "kind": remote.get("kind", "unknown"),
                           "state": "missing", "note": "download failed"})
    for name in sorted(clashing):
        report.append({"name": name, "kind": "conflict", "state": "missing",
                       "note": "same filename on two shelves — refused"})
    _record(report)

    logger.info(
        "LoRA sync: %d fetched, %d already current%s",
        fetched, len(candidates) - len(to_fetch) - len(deferred),
        f", {len(deferred)} deferred to first use ({', '.join(deferred)})" if deferred else "",
    )
    return ok


async def ensure_named_loras_present(
    names: list[str | None], queue: QueueClient, progress=None
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

    `progress` is the SEGMENT's progress log, not the daemon's. This download happens inside
    a claimed segment and can take minutes on a 400 MB LoRA — without it the job page sits at
    "[1/6] Start image ready" for five minutes and looks wedged, which is indistinguishable
    from the real wedge in #160. Saying "downloading a LoRA, 40% of 408 MB" is the difference
    between a slow step and a broken worker.
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    # A LoRA the background sync is still verifying or downloading (#189). The API skips such
    # segments when it honours `pending_loras`; this is the fallback for one that does not.
    # WAIT for the sync rather than fetch alongside it: two writers on one .tmp is a corrupt
    # file, and a same-size retrain still being md5-checked must not be trusted early.
    for raw in names:
        n = str(raw or "").strip()
        if n and not n.endswith(".safetensors"):
            n += ".safetensors"
        ev = _PENDING.get(n)
        if ev is None:
            continue
        logger.info("LoRA on-demand: %s is still being synced — waiting for it", n)
        if progress:
            await progress(f"[2/6] Waiting for the background LoRA sync to finish {n}")
        await ev.wait()

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
    if progress:
        await progress(
            f"[2/6] {len(wanted)} LoRA(s) not on this worker — fetching: {', '.join(wanted)}"
        )
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
            if progress:
                await progress(f"[2/6] {name} is not in the bucket — never uploaded")
            continue
        local = os.path.join(lora_dir, name)
        _cleanup_partials(lora_dir, name)
        tmp = local + ".tmp"
        mb = remote.get("size", 0) / (1024 * 1024)
        if progress:
            await progress(f"[2/6] Downloading {name} ({mb:.0f} MB)...")

        async def _tick(done, total, _name=name):
            if progress and total:
                await progress(f"[2/6] {_name}: {done * 100 // total}% of {total / (1024 * 1024):.0f} MB")

        try:
            got = await queue.stream_file(remote["uri"], tmp,
                                          on_progress=_tick, total=remote.get("size", 0))
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
    # Keep the reported inventory honest: this ran hours after boot and changed the disk.
    _note_fetched(fetched)
    return fetched
