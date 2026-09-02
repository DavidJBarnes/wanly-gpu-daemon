"""The LoRA catalogue diff — what makes a worker fetch, skip, or refuse a file.

The behaviour worth protecting is the retrain case. LoRAs are republished under the SAME
name after retraining, so a name-only check keeps the old weights and the worker renders
the wrong character while the console shows the right one. That is wrong output rather
than a failure, so it is the one this file guards hardest.
"""
import hashlib
import logging
import os

import pytest

from daemon import lora_sync

BIG = 20 * 1024 * 1024  # comfortably over MIN_LORA_SIZE


def _write(path, payload: bytes, size=BIG):
    with open(path, "wb") as f:
        f.write(payload + b"\0" * (size - len(payload)))
    return hashlib.md5(open(path, "rb").read()).hexdigest()


class FakeQueue:
    """Records what was asked for, and writes whatever content it was configured with."""

    def __init__(self, catalog, content=b"NEW"):
        self.catalog = catalog
        self.content = content
        self.downloaded = []

    async def list_loras(self):
        return self.catalog

    async def stream_file(self, uri, dest):
        self.downloaded.append(uri)
        return _write(dest, self.content)


@pytest.fixture
def lora_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(lora_sync, "_loras_dir", lambda: str(tmp_path))
    return tmp_path


async def test_a_retrain_under_the_same_name_is_detected_and_replaced(lora_dir):
    """Same filename, same size, different bytes — the case a name check misses.

    If this ever regresses the worker keeps rendering the previous character silently.
    """
    path = lora_dir / "k3lly2026_v2.safetensors"
    _write(str(path), b"OLD-WEIGHTS")
    new_md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()

    q = FakeQueue([{
        "name": "k3lly2026_v2.safetensors", "size": BIG,
        "etag": new_md5, "multipart": False,
        "uri": "s3://ltx-loras/k3lly2026_v2.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert q.downloaded == ["s3://ltx-loras/k3lly2026_v2.safetensors"]
    assert open(path, "rb").read(3) == b"NEW"


async def test_an_unchanged_lora_is_not_downloaded_again(lora_dir):
    """Otherwise every boot re-pulls gigabytes over the wire for nothing."""
    path = lora_dir / "p@y.safetensors"
    md5 = _write(str(path), b"SAME")

    q = FakeQueue([{
        "name": "p@y.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/p@y.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert q.downloaded == []


async def test_a_corrupt_download_never_replaces_a_good_local_copy(lora_dir):
    """Verification happens on the .tmp, before the rename.

    A transfer that truncates or flips bytes must leave the working file untouched — losing
    a good LoRA to a bad network is not an acceptable outcome of a routine sync.
    """
    path = lora_dir / "k3llydw_v2.safetensors"
    _write(str(path), b"GOOD-LOCAL")

    q = FakeQueue([{
        "name": "k3llydw_v2.safetensors", "size": BIG,
        "etag": "0" * 32,  # will not match whatever the fake writes
        "multipart": False, "uri": "s3://ltx-loras/k3llydw_v2.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is False   # reported, not swallowed
    assert open(path, "rb").read(10) == b"GOOD-LOCAL"      # untouched
    assert not os.path.exists(str(path) + ".tmp")          # and cleaned up


async def test_a_multipart_etag_falls_back_to_size_instead_of_looping(lora_dir):
    """A multipart ETag is not an md5, so comparing it as one never matches.

    Treated as a hash it would re-download the same file on every sync forever.
    """
    path = lora_dir / "big.safetensors"
    _write(str(path), b"WHATEVER")

    q = FakeQueue([{
        "name": "big.safetensors", "size": BIG, "etag": "abc-7",
        "multipart": True, "uri": "s3://ltx-loras/big.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert q.downloaded == []


async def test_an_unreachable_catalogue_leaves_the_disk_alone(lora_dir):
    """The API being down must not delete or disturb LoRAs the worker already has."""
    path = lora_dir / "k3lly2026_v2.safetensors"
    _write(str(path), b"KEEP-ME")

    class Broken(FakeQueue):
        async def list_loras(self):
            raise RuntimeError("connection refused")

    assert await lora_sync.sync_lora_catalog(Broken([])) is False
    assert open(path, "rb").read(7) == b"KEEP-ME"


async def test_two_prefixes_claiming_one_filename_are_both_refused(lora_dir):
    """The bucket files under character/ and content/, but files land here FLAT.

    ComfyUI's LoraLoader and ltx_characters.char_lora both use the bare filename and neither
    knows a prefix exists, so flattening is required. The cost is that two kinds can claim
    one name — and the loser is silently overwritten, which renders a stranger successfully.
    Refuse the pair; a skipped LoRA fails loudly, a swapped one does not.
    """
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([
        {"name": "clash.safetensors", "kind": "character", "key": "character/clash.safetensors",
         "size": BIG, "etag": md5, "multipart": False, "uri": "s3://ltx-loras/character/clash.safetensors"},
        {"name": "clash.safetensors", "kind": "content", "key": "content/clash.safetensors",
         "size": BIG, "etag": md5, "multipart": False, "uri": "s3://ltx-loras/content/clash.safetensors"},
    ])
    assert await lora_sync.sync_lora_catalog(q) is False   # reported, not silently resolved
    assert q.downloaded == []                              # neither one wins
    assert not os.path.exists(lora_dir / "clash.safetensors")


async def test_prefixed_keys_land_flat_where_comfyui_looks(lora_dir):
    """A character/ key must become <lora_dir>/<basename>, not <lora_dir>/character/...

    ComfyUI resolves lora_name relative to its loras directory, and the DB stores the bare
    name. A file written into a subdirectory is a file ComfyUI cannot find.
    """
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "k3lly2026_v2.safetensors", "kind": "character",
        "key": "character/k3lly2026_v2.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/character/k3lly2026_v2.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert (lora_dir / "k3lly2026_v2.safetensors").exists()          # flat
    assert not (lora_dir / "character").exists()                     # no subdirectory
    assert q.downloaded == ["s3://ltx-loras/character/k3lly2026_v2.safetensors"]


# ---------------------------------------------------------------------------------------
# On-demand fetch at claim time.
#
# The boot sync is a snapshot; the console offers a LoRA the moment it reaches the bucket.
# Without this, a pose naming a newly published LoRA fails every segment until somebody
# restarts the pod — the queue waiting on an operator for something the worker could just
# download.
# ---------------------------------------------------------------------------------------


async def test_a_lora_published_after_boot_is_fetched_at_claim(lora_dir):
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "sfbehind_LTX2_3_v0_1.safetensors", "kind": "content",
        "key": "content/sfbehind_LTX2_3_v0_1.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/content/sfbehind_LTX2_3_v0_1.safetensors",
    }])
    got = await lora_sync.ensure_named_loras_present(["sfbehind_LTX2_3_v0_1"], q)
    assert got == ["sfbehind_LTX2_3_v0_1.safetensors"]
    assert (lora_dir / "sfbehind_LTX2_3_v0_1.safetensors").exists()


async def test_the_common_case_costs_no_network_call(lora_dir):
    """Both LoRAs already present — the overwhelmingly normal claim.

    It must not list the catalogue, and must not re-hash: 650 MB per LoRA per segment to
    re-answer what boot already answered would be seconds of GPU time thrown away.
    """
    _write(str(lora_dir / "k3lly2026_v2.safetensors"), b"X")
    _write(str(lora_dir / "sfbehind_LTX2_3_v0_1.safetensors"), b"Y")

    class Loud(FakeQueue):
        async def list_loras(self):
            raise AssertionError("must not hit the API when everything is present")

    got = await lora_sync.ensure_named_loras_present(
        ["k3lly2026_v2", "sfbehind_LTX2_3_v0_1"], Loud([]))
    assert got == []


async def test_none_and_empty_are_not_treated_as_filenames(lora_dir):
    """"none" is how a pose says "no content LoRA". Looking it up would fetch nothing and
    log a scary "not in the bucket" for a perfectly normal pose."""
    class Loud(FakeQueue):
        async def list_loras(self):
            raise AssertionError("must not hit the API for 'none'")

    assert await lora_sync.ensure_named_loras_present(["none", "", None], Loud([])) == []


async def test_a_truncated_local_file_is_replaced_not_trusted(lora_dir):
    """Present-but-tiny is what an interrupted fetch leaves behind. Treated as a hit, it
    would be trusted forever and every render with it would fail in ComfyUI instead."""
    p = lora_dir / "k3lly2026_v2.safetensors"
    p.write_bytes(b"truncated")
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "k3lly2026_v2.safetensors", "kind": "character",
        "key": "character/k3lly2026_v2.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/character/k3lly2026_v2.safetensors",
    }])
    assert await lora_sync.ensure_named_loras_present(["k3lly2026_v2"], q) == [
        "k3lly2026_v2.safetensors"]
    assert p.stat().st_size == BIG


async def test_a_lora_that_was_never_uploaded_is_named_plainly(lora_dir, caplog):
    """The pose references something not in the bucket at all. Nothing to fetch — but the
    cause is "never published", not "download failed", and the log should say which."""
    got = await lora_sync.ensure_named_loras_present(["ghost_lora"], FakeQueue([]))
    assert got == []
    assert "never uploaded" in caplog.text


async def test_a_corrupt_on_demand_download_does_not_land(lora_dir):
    """Same rule as the boot sync: verify before the rename, so the next claim never
    inherits a bad file that now looks present."""
    q = FakeQueue([{
        "name": "k3lly2026_v2.safetensors", "kind": "character",
        "key": "character/k3lly2026_v2.safetensors", "size": BIG, "etag": "0" * 32,
        "multipart": False, "uri": "s3://ltx-loras/character/k3lly2026_v2.safetensors",
    }])
    assert await lora_sync.ensure_named_loras_present(["k3lly2026_v2"], q) == []
    assert not (lora_dir / "k3lly2026_v2.safetensors").exists()
    assert not (lora_dir / "k3lly2026_v2.safetensors.tmp").exists()


# ---------------------------------------------------------------------------------------
# Boot sync is eager for characters only.
#
# Boot sync blocks registration, so every byte fetched there is a worker not yet taking
# work. Measured on the 3090: one 408 MB content LoRA published while the pod was down
# added five minutes to boot, with no worker on the queue for that time. The content
# library is expected to grow; the character set is bounded by how many characters exist.
# ---------------------------------------------------------------------------------------


async def test_a_missing_character_lora_is_still_fetched_at_boot(lora_dir):
    """Effectively every job names one, so this download was going to happen on the first
    claim anyway. Doing it at boot costs nothing extra and keeps the first claim fast."""
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "k3lly2026_v2.safetensors", "kind": "character",
        "key": "character/k3lly2026_v2.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/character/k3lly2026_v2.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert q.downloaded == ["s3://ltx-loras/character/k3lly2026_v2.safetensors"]


async def test_a_missing_content_lora_is_not_downloaded_at_boot(lora_dir):
    """The five minutes this saves is five minutes of no worker on the queue, for a LoRA
    that may never be asked for on this box."""
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "sfbehind_LTX2_3_v0_1.safetensors", "kind": "content",
        "key": "content/sfbehind_LTX2_3_v0_1.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/content/sfbehind_LTX2_3_v0_1.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q) is True
    assert q.downloaded == []                                     # deferred
    assert not (lora_dir / "sfbehind_LTX2_3_v0_1.safetensors").exists()


async def test_a_deferred_lora_is_still_checked_and_named(lora_dir, caplog):
    """Deferring the DOWNLOAD must not mean skipping the CHECK.

    A content LoRA that is present but stale is exactly what the content comparison exists
    to catch, and silence about it would be the failure that renders the wrong motion.
    """
    # caplog captures WARNING and above by default; the deferral line is INFO, which is the
    # right level for it — this is normal operation, not a problem.
    caplog.set_level(logging.INFO, logger="daemon.lora_sync")
    _write(str(lora_dir / "sfbehind_LTX2_3_v0_1.safetensors"), b"OLD")
    q = FakeQueue([{
        "name": "sfbehind_LTX2_3_v0_1.safetensors", "kind": "content",
        "key": "content/sfbehind_LTX2_3_v0_1.safetensors", "size": BIG,
        "etag": hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest(),
        "multipart": False, "uri": "s3://ltx-loras/content/sfbehind_LTX2_3_v0_1.safetensors",
    }])
    await lora_sync.sync_lora_catalog(q)
    assert "sfbehind_LTX2_3_v0_1.safetensors" in caplog.text
    assert "deferred" in caplog.text
    assert "content changed" in caplog.text     # the reason, not just the fact


async def test_eager_kinds_is_overridable(lora_dir):
    """A box that wants everything up front can still say so, without editing this file."""
    md5 = hashlib.md5(b"NEW" + b"\0" * (BIG - 3)).hexdigest()
    q = FakeQueue([{
        "name": "sfbehind_LTX2_3_v0_1.safetensors", "kind": "content",
        "key": "content/sfbehind_LTX2_3_v0_1.safetensors", "size": BIG, "etag": md5,
        "multipart": False, "uri": "s3://ltx-loras/content/sfbehind_LTX2_3_v0_1.safetensors",
    }])
    assert await lora_sync.sync_lora_catalog(q, eager_kinds=("character", "content")) is True
    assert q.downloaded == ["s3://ltx-loras/content/sfbehind_LTX2_3_v0_1.safetensors"]
