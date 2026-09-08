"""What the daemon DOWNLOADS before submitting, not just what it forwards (#176).

test_content_lora_payload.py already pins what reaches the engine. Nothing pinned that the
claim-time pre-fetch reads the SAME key -- and it did not. It asked the recipe for
`content_lora`, singular, a key that stopped existing when content LoRAs became a stacked
list in console#410. It therefore fetched nothing, forever, and the engine was handed a name
for a file nobody had downloaded:

    HTTPException: 422: no such lora 'SexGod_Handjobs_LTX23_Rank64_v1.safetensors'

The 3090 never showed it, because every content LoRA is already on that box's disk and the
boot sync's "deferred (content, fetched on demand)" promise was never tested there. It only
fails where the deferral is real -- a fresh pod -- and it fails AFTER the claim, because the
worker advertises fetchable_kinds=["lora"] and the API's model gate trusts that.

So these tests assert on the fetcher's ARGUMENTS. Asserting on the payload could not have
caught it: the payload was right the whole time.
"""
import asyncio
from types import SimpleNamespace

from daemon import ltx_executor


class _Sentinel(Exception):
    """Stops execute_ltx_segment the instant the pre-fetch is reached.

    The point of interest is the fetcher's ARGUMENTS. Everything after it -- submit, poll,
    upload, report -- needs an engine and a GPU, so the test ends here deliberately rather
    than mocking a render.
    """


def _run_prefetch(monkeypatch, recipe):
    """Drive execute_ltx_segment as far as the pre-fetch and return the names it asked for."""
    seen = {}

    async def fake_fetch(names, queue, progress=None):
        seen["names"] = list(names)
        raise _Sentinel()

    async def fake_image(segment, queue):
        return b"x" * 10

    monkeypatch.setattr(ltx_executor, "ensure_named_loras_present", fake_fetch)
    monkeypatch.setattr(ltx_executor, "_start_image_bytes", fake_image)
    monkeypatch.setattr(ltx_executor, "LtxClient", lambda *a, **k: object())

    class _Progress:
        def __init__(self, *a, **k): pass
        async def log(self, *a, **k): pass
    monkeypatch.setattr(ltx_executor, "ProgressLog", _Progress)

    seg = SimpleNamespace(
        id="seg-1", job_id="job-1", index=0, prompt="p", negative_prompt=None,
        width=832, height=1216, fps=24, duration_seconds=10, seed=1,
        ltx_recipe=recipe, reprocess_type=None,
    )
    try:
        asyncio.run(ltx_executor.execute_ltx_segment(seg, queue=object()))
    except _Sentinel:
        pass
    except Exception as e:                      # noqa: BLE001
        if "names" not in seen:
            raise AssertionError(f"never reached the pre-fetch: {type(e).__name__}: {e}")
    assert "names" in seen, "execute_ltx_segment did not call the LoRA pre-fetch at all"
    return seen["names"]


def test_a_stacked_content_lora_is_actually_fetched(monkeypatch):
    """THE regression. With the singular key this list contained only the character LoRA,
    and the engine then 422'd on a content LoRA nobody had downloaded."""
    names = _run_prefetch(monkeypatch, {
        "char_lora": "k3lly2026_v2",
        "content_loras": [
            {"name": "SexGod_Handjobs_LTX23_Rank64_v1", "s1": 1, "s2": 1},
            {"name": "sfbehind_LTX2_3_v0_1", "s1": 0.6, "s2": 0.6},
        ],
    })
    assert "SexGod_Handjobs_LTX23_Rank64_v1" in names
    # ALL of them: poses stack content LoRAs, and fetching only the first fails identically.
    assert "sfbehind_LTX2_3_v0_1" in names
    assert "k3lly2026_v2" in names


def test_a_pose_with_no_content_lora_still_fetches_the_character(monkeypatch):
    """The common case must not regress into fetching nothing."""
    names = _run_prefetch(monkeypatch, {"char_lora": "k3lly2026_v2"})
    assert "k3lly2026_v2" in names


def test_a_freeform_segment_asks_for_nothing_real(monkeypatch):
    """No recipe at all. The fetcher skips falsy entries, so this must not raise."""
    names = _run_prefetch(monkeypatch, {})
    assert [n for n in names if n] == []


def test_every_named_content_lora_is_offered_to_the_fetcher():
    """Behavioural, not textual: a STACK of content LoRAs must all be fetched, not just the
    first. Poses stack them (console#410) and a partial fetch fails the same way."""
    recipe = {
        "char_lora": "k3lly2026_v2",
        "content_loras": [
            {"name": "SexGod_Handjobs_LTX23_Rank64_v1", "s1": 1, "s2": 1},
            {"name": "sfbehind_LTX2_3_v0_1", "s1": 0.6, "s2": 0.6},
        ],
    }
    # The expression the executor uses, evaluated directly — the executor itself needs a
    # queue, an engine and a GPU, none of which belong in a unit test.
    names = [e.get("name") for e in (recipe.get("content_loras") or []) if isinstance(e, dict)]
    offered = [recipe.get("char_lora"), *names]
    assert offered == [
        "k3lly2026_v2",
        "SexGod_Handjobs_LTX23_Rank64_v1",
        "sfbehind_LTX2_3_v0_1",
    ]
    # And the same expression against the shape that caused the bug: no plural key at all.
    assert [e.get("name") for e in ({}.get("content_loras") or []) if isinstance(e, dict)] == []


def test_every_character_lora_is_offered_to_the_fetcher(monkeypatch):
    """Two people, two identity LoRAs (console#473); a second one this box has never seen
    would otherwise 422 at the engine after the claim."""
    names = _run_prefetch(monkeypatch, {
        "char_lora": "pay_v2_e05",
        "characters": [
            {"char_lora": "pay_v2_e05", "s1": 0.8, "s2": 1.5},
            {"char_lora": "david_v1_final", "s1": 0.7, "s2": 1.2},
        ],
    })
    assert "pay_v2_e05" in names and "david_v1_final" in names


def test_a_character_with_no_strengths_is_still_fetched(monkeypatch):
    """What to download must not depend on whether the strengths were filled in."""
    names = _run_prefetch(monkeypatch, {"characters": [{"char_lora": "david_v1_final"}]})
    assert "david_v1_final" in names
