"""The character's identity reference: fetched at claim, cached by content hash (#187).

Part of the Character identity epic (wanly-console#581). A character can carry a 1536x1024
character sheet or a face close-up; when the segment's character has one and the job has not
turned it off, the API puts a presigned URL for it in the claim, and ltx-engine conditions both
stages on it through LTXIdentityOverlapConditioning (wanly-gpu-docker#156). Phase 0 showed
LoRA + sheet holding identity best in wanly's own recipe graph (wanly-gpu-docker#155).

WHY A CACHE, when the image is a few MB against a ten-minute render:

  * The file on disk is named by its CONTENT HASH, so the same sheet is one file however many
    segments use it, and two different images can never be confused because they shared a
    name or an S3 key that was overwritten.
  * A small index remembers which content hash an S3 URI last resolved to. When the download
    fails -- S3 blip, presigned URL expired in a queue backlog -- a reference this worker has
    already fetched is used from disk, and the progress log says so. A sheet character failing
    its render over a transient fetch error would be the worse outcome.

The URL itself is never logged: it carries a signature.
"""

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from daemon.config import settings

logger = logging.getLogger(__name__)

#: The two modes the engine accepts (recipe.IDENTITY_LORAS in wanly-gpu-docker).
MODES = ("sheet", "face")
#: The layout the CharacterSheet LoRA was trained on. Anything else still renders; it is noted.
SHEET_SIZE = (1536, 1024)
#: Files kept on disk. A worker sees a handful of characters; this only bounds a long life.
MAX_CACHED = 64


class IdentityRefError(RuntimeError):
    """The reference could not be fetched (and was not cached), or is not an image."""


@dataclass
class FetchedRef:
    data: bytes
    sha256: str
    mode: str
    size: tuple[int, int]
    #: "downloaded" or "cache (download failed: ...)"
    source: str

    def data_uri(self) -> str:
        return "data:image/png;base64," + base64.b64encode(self.data).decode()

    def describe(self) -> str:
        return (f"identity ref ({self.mode}) {self.size[0]}x{self.size[1]} "
                f"sha256 {self.sha256[:12]} from {self.source}")


def cache_dir() -> Path:
    d = Path(settings.identity_ref_cache_dir or
             os.path.join(tempfile.gettempdir(), "wanly-identity-refs"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(d: Path) -> Path:
    return d / "index.json"


def _read_index(d: Path) -> dict[str, str]:
    try:
        raw = json.loads(_index_path(d).read_text())
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_index(d: Path, index: dict[str, str]) -> None:
    tmp = _index_path(d).with_suffix(".tmp")
    tmp.write_text(json.dumps(index, sort_keys=True))
    tmp.replace(_index_path(d))


def _image_size(data: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            return im.size
    except Exception as e:
        raise IdentityRefError(f"identity reference is not a readable image "
                               f"({type(e).__name__}: {e})") from e


def _store(d: Path, data: bytes) -> str:
    sha = hashlib.sha256(data).hexdigest()
    path = d / f"{sha}.img"
    if not path.exists():
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
    else:
        path.touch()
    _prune(d)
    return sha


def _prune(d: Path) -> None:
    files = sorted(d.glob("*.img"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[MAX_CACHED:]:
        try:
            stale.unlink()
        except OSError:
            pass


async def _download(url: str) -> bytes:
    timeout = httpx.Timeout(connect=15.0, read=60.0, write=60.0, pool=15.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
        r = await c.get(url)
        if not r.is_success:
            # Status only: the URL is a credential.
            raise IdentityRefError(f"identity reference download failed: HTTP {r.status_code}")
        return r.content


async def fetch_identity_ref(url: str, mode: str, uri: str | None = None) -> FetchedRef:
    """Download the reference, cache it by content hash, and return its bytes.

    `uri` is the S3 object the presigned `url` points at. It is only the cache's index key --
    the cached bytes are always addressed by their own hash -- so a missing uri just means no
    fallback.
    """
    if mode not in MODES:
        raise IdentityRefError(f"identity mode {mode!r}; expected one of {MODES}")
    d = cache_dir()
    index = _read_index(d)
    try:
        data = await _download(url)
        size = _image_size(data)
        source = "download"
    except (httpx.HTTPError, IdentityRefError) as e:
        sha = index.get(uri or "")
        cached = d / f"{sha}.img" if sha else None
        if not cached or not cached.exists():
            if isinstance(e, IdentityRefError):
                raise
            raise IdentityRefError(f"identity reference download failed: "
                                   f"{type(e).__name__}") from e
        logger.warning("Identity reference download failed (%s); using the cached copy %s",
                       e if isinstance(e, IdentityRefError) else type(e).__name__, sha[:12])
        data = cached.read_bytes()
        size = _image_size(data)
        source = f"cache (download failed: {e if isinstance(e, IdentityRefError) else type(e).__name__})"
    sha = _store(d, data)
    if uri and index.get(uri) != sha:
        index[uri] = sha
        _write_index(d, index)
    return FetchedRef(data=data, sha256=sha, mode=mode, size=size, source=source)
