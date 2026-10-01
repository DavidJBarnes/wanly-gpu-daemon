"""The character's identity reference, from claim to engine request (#187).

The API puts a presigned URL for the segment character's sheet or face reference in the claim
(wanly-api#379); the daemon fetches it, caches it by content hash, and hands it to ltx-engine as
`identity_ref` + `identity_mode` (wanly-gpu-docker#156). What these tests hold:

  * no reference -> the engine request is byte-for-byte what it was, and nothing is fetched or
    asked of the engine;
  * a reference -> the engine gets the bytes and the mode, the segment log says what was used,
    and the presigned URL (a credential) is never written anywhere;
  * an engine that predates the feature is refused rather than silently ignoring the field;
  * the cache is content-addressed and covers a failed download of a reference seen before.
"""
import asyncio
import base64
import hashlib
import io
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from daemon import identity_ref as idref
from daemon import ltx_executor
from daemon.ltx_client import build_submit_payload
from daemon.schemas import IdentityRef
from tests.conftest import make_segment

URL = "https://wanly-images.s3.amazonaws.com/chars/k.png?X-Amz-Signature=SECRET"
URI = "s3://wanly-images/chars/k.png"


def _png(w=1536, h=1024, colour=(200, 180, 170)):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def cache(monkeypatch, tmp_path):
    monkeypatch.setattr(idref.settings, "identity_ref_cache_dir", str(tmp_path / "idrefs"))
    return tmp_path / "idrefs"


# ---------------------------------------------------------------- the claim

def test_a_claim_without_a_reference_parses_as_before():
    assert make_segment().identity_ref is None


def test_a_claim_with_a_reference_parses():
    seg = make_segment(identity_ref={"url": URL, "mode": "sheet", "uri": URI,
                                     "character": "Kelly"})
    assert isinstance(seg.identity_ref, IdentityRef)
    assert seg.identity_ref.mode == "sheet" and seg.identity_ref.uri == URI


# ---------------------------------------------------------------- the engine request

BASE = dict(image_bytes=b"img", prompt="p", negative_prompt=None, width=832, height=1216,
            num_frames=121, frame_rate=24, seed=1,
            recipe={"recipe": "Turn", "char_lora": "k3lly2026_v2", "char_s1": 0.8,
                    "char_s2": 1.5})


def test_no_reference_leaves_the_request_exactly_as_it_was():
    assert build_submit_payload(**BASE) == build_submit_payload(
        **BASE, identity_ref=None, identity_mode=None)
    assert "identity_ref" not in build_submit_payload(**BASE)
    assert "identity_mode" not in build_submit_payload(**BASE)


def test_a_reference_is_sent_with_its_mode():
    p = build_submit_payload(**BASE, identity_ref="data:image/png;base64,AA==",
                             identity_mode="sheet")
    assert p["identity_ref"] == "data:image/png;base64,AA=="
    assert p["identity_mode"] == "sheet"


@pytest.mark.parametrize("kw", [{"identity_ref": "data:,"}, {"identity_mode": "face"}])
def test_half_a_reference_is_never_sent(kw):
    p = build_submit_payload(**BASE, **kw)
    assert "identity_ref" not in p and "identity_mode" not in p


def test_a_sheet_only_character_sends_no_character_lora():
    """A character can be a sheet and nothing else: no LoRA in the blob, so no `loras`."""
    p = build_submit_payload(**{**BASE, "recipe": {
        "recipe": "Turn", "characters": [{"name": "Kelly", "trigger": None, "char_lora": None,
                                          "s1": None, "s2": None}]}},
        identity_ref="data:,", identity_mode="sheet")
    assert "loras" not in p
    assert p["identity_mode"] == "sheet"


# ---------------------------------------------------------------- fetch + cache

def _serve(monkeypatch, data=None, exc=None):
    calls = []

    async def fake_download(url):
        calls.append(url)
        if exc is not None:
            raise exc
        return data
    monkeypatch.setattr(idref, "_download", fake_download)
    return calls


def test_a_reference_is_downloaded_and_stored_by_content_hash(monkeypatch, cache):
    sheet = _png()
    _serve(monkeypatch, sheet)
    ref = asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    sha = hashlib.sha256(sheet).hexdigest()
    assert ref.data == sheet and ref.sha256 == sha and ref.size == (1536, 1024)
    assert (cache / f"{sha}.img").read_bytes() == sheet
    assert ref.source == "download"
    assert ref.data_uri() == "data:image/png;base64," + base64.b64encode(sheet).decode()


def test_the_same_image_is_one_file(monkeypatch, cache):
    sheet = _png()
    _serve(monkeypatch, sheet)
    asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    asyncio.run(idref.fetch_identity_ref(URL, "sheet", "s3://wanly-images/elsewhere.png"))
    assert len(list(cache.glob("*.img"))) == 1


def test_a_failed_download_falls_back_to_the_cached_copy(monkeypatch):
    sheet = _png()
    _serve(monkeypatch, sheet)
    asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    _serve(monkeypatch, exc=httpx.ConnectError("boom"))
    ref = asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    assert ref.data == sheet and ref.source.startswith("cache")


def test_a_failed_download_with_nothing_cached_fails(monkeypatch):
    _serve(monkeypatch, exc=httpx.ConnectError("boom"))
    with pytest.raises(idref.IdentityRefError, match="download failed"):
        asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))


def test_a_replaced_object_is_not_served_stale(monkeypatch, cache):
    """The bytes are addressed by their own hash: a new image at the same key is a new
    file, and the index moves to it."""
    old, new = _png(colour=(1, 1, 1)), _png(colour=(9, 9, 9))
    _serve(monkeypatch, old)
    asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    _serve(monkeypatch, new)
    assert asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI)).data == new


def test_something_that_is_not_an_image_is_refused(monkeypatch):
    _serve(monkeypatch, b"<Error>AccessDenied</Error>")
    with pytest.raises(idref.IdentityRefError, match="not a readable image"):
        asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))


def test_an_unknown_mode_is_refused(monkeypatch):
    _serve(monkeypatch, _png())
    with pytest.raises(idref.IdentityRefError, match="identity mode"):
        asyncio.run(idref.fetch_identity_ref(URL, "both", URI))


def test_an_http_error_names_the_status_not_the_url(monkeypatch):
    class Resp:
        is_success, status_code, content = False, 403, b""

    class Client:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass
        async def get(self, url): return Resp()
    monkeypatch.setattr(idref.httpx, "AsyncClient", Client)
    with pytest.raises(idref.IdentityRefError) as e:
        asyncio.run(idref.fetch_identity_ref(URL, "sheet", URI))
    assert "403" in str(e.value) and "SECRET" not in str(e.value)


# ---------------------------------------------------------------- the executor

class _Stop(Exception):
    pass


def _run(monkeypatch, identity, features=("identity_ref",), data=None):
    """Drive execute_ltx_segment up to the engine submit and capture what it was sent."""
    seen = {"features_asked": 0, "logs": [], "failed": None}

    async def no_fetch(*a, **k):
        return []

    async def image(segment, queue):
        return b"x"

    class Client:
        def __init__(self, *a, **k): pass

        async def features(self):
            seen["features_asked"] += 1
            return set(features)

        async def submit(self, **kw):
            seen["submit"] = kw
            raise _Stop()

        async def close(self): pass

    class Progress:
        def __init__(self, *a, **k): self.text = ""
        async def log(self, line): seen["logs"].append(line)

    class Queue:
        async def update_segment(self, seg_id, result):
            seen["failed"] = result.error_message

    monkeypatch.setattr(ltx_executor, "ensure_named_loras_present", no_fetch)
    monkeypatch.setattr(ltx_executor, "_start_image_bytes", image)
    monkeypatch.setattr(ltx_executor, "LtxClient", Client)
    monkeypatch.setattr(ltx_executor, "ProgressLog", Progress)
    import daemon.checkpoint_sync as cs

    async def no_ckpt(*a, **k):
        return False
    monkeypatch.setattr(cs, "ensure_checkpoint_present", no_ckpt)
    _serve(monkeypatch, data if data is not None else _png())
    seg = SimpleNamespace(
        id="seg-1", job_id="job-1", index=0, prompt="p", negative_prompt=None,
        width=832, height=1216, fps=24, duration_seconds=5, seed=1,
        ltx_recipe={"recipe": "Turn", "char_lora": "k3lly2026_v2"}, reprocess_type=None,
        identity_ref=IdentityRef(**identity) if identity else None,
    )
    asyncio.run(ltx_executor.execute_ltx_segment(seg, queue=Queue()))
    return seen


def test_no_reference_sends_none_and_asks_the_engine_nothing(monkeypatch):
    seen = _run(monkeypatch, None)
    assert seen["submit"]["identity_ref"] is None and seen["submit"]["identity_mode"] is None
    assert seen["features_asked"] == 0
    assert not any("identity" in line for line in seen["logs"])


def test_a_reference_reaches_the_engine_and_the_log(monkeypatch):
    sheet = _png()
    seen = _run(monkeypatch, {"url": URL, "mode": "sheet", "uri": URI, "character": "Kelly"},
                data=sheet)
    assert seen["submit"]["identity_mode"] == "sheet"
    assert seen["submit"]["identity_ref"] == ("data:image/png;base64,"
                                              + base64.b64encode(sheet).decode())
    line = next(x for x in seen["logs"] if "identity ref" in x)
    assert "1536x1024" in line and "Kelly" in line
    assert not any("SECRET" in x for x in seen["logs"]), "logged a presigned URL"


def test_an_off_size_sheet_is_warned_about(monkeypatch):
    seen = _run(monkeypatch, {"url": URL, "mode": "sheet"}, data=_png(1024, 1024))
    assert any("not 1536x1024" in x for x in seen["logs"])


def test_an_engine_without_the_feature_is_refused_not_ignored(monkeypatch):
    """An older engine would drop the field and render somebody else."""
    seen = _run(monkeypatch, {"url": URL, "mode": "sheet"}, features=())
    assert "submit" not in seen
    assert "does not support identity references" in seen["failed"]
    assert "re-pin" in seen["failed"]


# ---------------------------------------------------------------- what the worker checks

def test_bfsnodes_is_an_ltx_node_pinned_to_phase_0s_commit():
    from daemon.node_checker import required_packages
    ltx = required_packages("ltx")
    assert "LTXIdentityOverlapConditioning" in ltx["ComfyUI-BFSNodes"]["nodes"]
    assert ltx["ComfyUI-BFSNodes"]["commit"].startswith("bd23236")
    # A WAN worker keeps exactly the packages it always had.
    assert "ComfyUI-BFSNodes" not in required_packages("wan22")
    assert "PainterLongVideo" in str(required_packages("wan22"))


def test_an_ltx_worker_checks_the_identity_loras():
    from daemon.config import settings
    from daemon.model_validator import MODEL_CHECKS, get_model_checks
    ltx = get_model_checks("ltx")
    assert [settings_name for settings_name, *_ in ltx] == ["identity_face_lora",
                                                          "identity_sheet_lora"]
    assert settings.identity_face_lora == "Best_FaceID_v1.0_LoRA.safetensors"
    assert settings.identity_sheet_lora == "Best_FaceID_CharacterSheet_v1.0_LoRA.safetensors"
    assert get_model_checks("wan22") == MODEL_CHECKS
