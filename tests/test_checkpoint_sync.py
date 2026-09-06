"""Fetching a base model this worker lacks (console#423).

Everything here guards a failure that costs twenty minutes and a claimed segment. A LoRA
that fails to download costs 40 seconds; a checkpoint is ~70x larger, so each check exists
because the naive version of it is expensive to get wrong.

The single most important behaviour is the one at the end: a file that fails verification is
DISCARDED, never renamed into place. A partial safetensors is a valid header over missing
data -- present to every existence check, fatal only at load, inside a claim.
"""
import json
import os
import struct

import pytest

from daemon import checkpoint_sync as cs
from daemon.config import settings


def _safetensors(path, payload_len=64, declared_len=None, header_extra=None):
    """Write a syntactically real safetensors file."""
    hdr = {"w": {"dtype": "F32", "shape": [1],
                 "data_offsets": [0, declared_len if declared_len is not None else payload_len]}}
    if header_extra:
        hdr.update(header_extra)
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.write(b"\0" * payload_len)
    return os.path.getsize(path)


class TestVerification:
    def test_a_sound_file_passes(self, tmp_path):
        p = tmp_path / "ok.safetensors"
        size = _safetensors(p)
        assert cs.verify_safetensors(str(p), size) is None

    def test_a_short_download_is_caught_by_byte_count(self, tmp_path):
        """The common failure: the stream ended early. The catalogue's size makes this
        checkable without parsing anything."""
        p = tmp_path / "short.safetensors"
        size = _safetensors(p)
        reason = cs.verify_safetensors(str(p), size + 1000)
        assert reason and "expected" in reason and "%" in reason

    def test_a_truncated_body_is_caught_by_the_header(self, tmp_path):
        """The nastier failure: the header declares more data than the file holds. Catches a
        file whose length happens to match but whose contents do not."""
        p = tmp_path / "trunc.safetensors"
        _safetensors(p, payload_len=10, declared_len=9999)
        reason = cs.verify_safetensors(str(p), 0)   # 0 = skip the size check
        assert reason and "truncated" in reason

    def test_something_that_is_not_a_safetensors_at_all(self, tmp_path):
        """An HTML error page or a redirect body written to disk. Bounded before allocating:
        a garbage header length used to raise MemoryError, which reported the wrong problem
        and, on a 46 GB file, tried to allocate first."""
        p = tmp_path / "junk.safetensors"
        p.write_bytes(b"<html>404 not found</html>" * 4)
        reason = cs.verify_safetensors(str(p), 0)
        assert reason and ("not a safetensors" in reason or "unreadable" in reason)


class _Q:
    """A queue client that answers the catalogue and nothing else."""
    def __init__(self, catalog=None, boom=False):
        self._c = catalog if catalog is not None else {}
        self._boom = boom

    async def checkpoint_catalog(self):
        if self._boom:
            raise RuntimeError("api unreachable")
        return self._c


CATALOG = {"10Eros_v1.5_bf16": {"repo": "TenStrip/LTX2.3-10Eros",
                                "path": "10Eros_v1.5_bf16.safetensors",
                                "size_bytes": 46_139_886_366}}


@pytest.mark.asyncio
class TestWhenItDeclinesToFetch:
    async def test_a_file_already_here_is_left_alone(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        (tmp_path / "10Eros_v1.5_bf16.safetensors").write_bytes(b"x")
        assert await cs.ensure_checkpoint_present("10Eros_v1.5_bf16", _Q(CATALOG)) is None

    async def test_none_and_empty_are_not_filenames(self, tmp_path, monkeypatch):
        """"none" is how a recipe says "no override". Fetching it would be a 404 for a file
        called none.safetensors."""
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        for spelling in (None, "", "  ", "none", "NONE"):
            assert await cs.ensure_checkpoint_present(spelling, _Q(CATALOG)) is None

    async def test_an_unknown_checkpoint_is_left_to_a_worker_that_has_it(self, tmp_path, monkeypatch):
        """No recorded source is a real answer, not an error. Guessing a URL from the name
        would produce a confident 404 after a long wait."""
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        assert await cs.ensure_checkpoint_present("mystery_model", _Q(CATALOG)) is None

    async def test_an_unreachable_api_does_not_raise(self, tmp_path, monkeypatch):
        """The segment should fail with the engine's own "no such checkpoint", which names
        the file and lists what IS present — more useful than an exception from here."""
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        assert await cs.ensure_checkpoint_present("10Eros_v1.5_bf16", _Q(boom=True)) is None

    async def test_it_refuses_rather_than_filling_the_volume(self, tmp_path, monkeypatch):
        """statvfs up front, because the alternative is dying at 92% of 46 GB inside a
        claimed segment — having also evicted the page cache on the way."""
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        monkeypatch.setattr(cs, "_free_bytes", lambda p: 5 * 1024 ** 3)
        logged = []

        async def prog(msg):
            logged.append(msg)

        assert await cs.ensure_checkpoint_present(
            "10Eros_v1.5_bf16", _Q(CATALOG), progress=prog) is None
        assert any("free" in m for m in logged), logged
        assert not list(tmp_path.iterdir()), "nothing should have been written"


@pytest.mark.asyncio
class TestTheDownloadItself:
    async def test_a_bad_download_is_discarded_not_renamed(self, tmp_path, monkeypatch):
        """THE one that matters. Renamed into place, a truncated 46 GB file looks present to
        every check and fails only at load, deep inside a claimed segment."""
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        monkeypatch.setattr(cs, "_free_bytes", lambda p: 500 * 1024 ** 3)
        _fake_stream(monkeypatch, b"\x00" * 32)      # far short of size_bytes

        assert await cs.ensure_checkpoint_present("10Eros_v1.5_bf16", _Q(CATALOG)) is None
        assert not (tmp_path / "10Eros_v1.5_bf16.safetensors").exists(), \
            "a file that failed verification was renamed into place"
        assert not list(tmp_path.glob("*.part")), "the staged file was left behind"

    async def test_a_good_download_lands_under_the_real_name(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        monkeypatch.setattr(cs, "_free_bytes", lambda p: 500 * 1024 ** 3)
        src = tmp_path / "src.safetensors"
        size = _safetensors(src, payload_len=128)
        catalog = {"tiny": {"repo": "r", "path": "tiny.safetensors", "size_bytes": size}}
        _fake_stream(monkeypatch, src.read_bytes())

        assert await cs.ensure_checkpoint_present("tiny", _Q(catalog)) == "tiny"
        assert (tmp_path / "tiny.safetensors").exists()
        assert not list(tmp_path.glob("*.part"))

    async def test_an_http_error_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
        monkeypatch.setattr(cs, "_free_bytes", lambda p: 500 * 1024 ** 3)
        _fake_stream(monkeypatch, b"", status=404)
        assert await cs.ensure_checkpoint_present("10Eros_v1.5_bf16", _Q(CATALOG)) is None
        assert not list(tmp_path.iterdir())


def _fake_stream(monkeypatch, body: bytes, status: int = 200):
    """Replace httpx.AsyncClient with one that yields `body` in chunks.

    Chunked and async on purpose: the real requirement is that a 46 GB transfer never blocks
    the event loop, because the heartbeat is the only reclaim authority for a live worker. A
    blocking read here would let the API declare this worker dead and hand the segment to
    someone else mid-download.
    """
    class _Resp:
        status_code = status
        is_success = 200 <= status < 300

        async def aread(self):
            return b""

        async def aiter_bytes(self, n):
            for i in range(0, len(body), n or 1):
                yield body[i:i + (n or 1)]

    class _Stream:
        async def __aenter__(self): return _Resp()
        async def __aexit__(self, *a): return False

    class _Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        def stream(self, *a, **k): return _Stream()

    monkeypatch.setattr(cs.httpx, "AsyncClient", _Client)


@pytest.mark.asyncio
async def test_a_read_only_model_tree_declines_clearly(tmp_path, monkeypatch):
    """The 3090 bind-mounts its models read-only. That box holds every checkpoint so this is
    normally unreachable, but without the check a pose naming a missing base model would
    fail on a write error reported as a download failure."""
    monkeypatch.setattr(settings, "checkpoint_dir", str(tmp_path))
    monkeypatch.setattr(cs, "_free_bytes", lambda p: 500 * 1024 ** 3)
    monkeypatch.setattr(cs.os, "access", lambda p, mode: False)
    assert await cs.ensure_checkpoint_present("10Eros_v1.5_bf16", _Q(CATALOG)) is None
    assert not list(tmp_path.iterdir())
