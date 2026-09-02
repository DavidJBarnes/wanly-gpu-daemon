"""The LoRA catalogue diff — what makes a worker fetch, skip, or refuse a file.

The behaviour worth protecting is the retrain case. LoRAs are republished under the SAME
name after retraining, so a name-only check keeps the old weights and the worker renders
the wrong character while the console shows the right one. That is wrong output rather
than a failure, so it is the one this file guards hardest.
"""
import hashlib
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
