"""HTTP client for ltx-engine, the LTX 2.3 render service.

The engine runs in the same container and owns graph assembly and recipe resolution. This
client does not build graphs, patch workflows or know what a node is — it submits a job,
waits, and collects an mp4.

That division is deliberate. Every structural bug on the storyboard project came from
rewriting a downloaded graph's topology in a caller to cover a job shape it was not built
for; none came from the reference workflows themselves. So the engine keeps the graph and
the daemon keeps the queue.
"""

import asyncio
import base64
import logging
import time
from typing import Any

import httpx

from .config import settings

logger = logging.getLogger(__name__)

# The engine's own vocabulary. "None" means queued — its word, not ours.
TERMINAL = {"Done", "Failed"}


class LtxEngineError(RuntimeError):
    """The engine reported a failed render, or could not be reached."""


def build_submit_payload(
    *,
    image_bytes: bytes | None,
    prompt: str,
    negative_prompt: str | None,
    width: int,
    height: int,
    num_frames: int,
    frame_rate: int,
    seed: int,
    recipe: dict[str, Any] | None,
) -> dict[str, Any]:
    """The engine's /job body. Pure, so it can be asserted without an engine.

    `recipe` is the RESOLVED configuration handed down in the claim. It is read, never
    looked up — see the module docstring.
    """
    payload: dict[str, Any] = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        # The engine takes a signed 32-bit seed; the queue derives a 63-bit one. Narrow it
        # here rather than letting the engine reject it or silently wrap it.
        "seed": seed % (2**31 - 1),
        "keyframes": [],
    }
    if image_bytes is not None:
        b64 = base64.b64encode(image_bytes).decode()
        payload["keyframes"] = [{"image": f"data:image/png;base64,{b64}"}]

    if recipe:
        # ONLY the fields the engine's request model carries. Everything else in the blob is
        # a record of what ran, not an input — graph_sha256 above all. Sending that back
        # would offer the engine a hash to honour instead of one to produce, which inverts
        # the entire point of hashing the resolved graph.
        for key in ("recipe", "character"):
            if recipe.get(key):
                payload[key] = recipe[key]
        # The pose's base model. Forwarded as-is; the engine appends .safetensors when
        # missing and moves every loader that names the file, because 2.3 checkpoints are
        # monoliths rather than a transformer plus separate parts.
        #
        # Worth knowing when reading a render that came back wrong: character LoRAs were
        # trained against sulphur, and against another base a LoRA whose keys do not line up
        # fuses NOTHING, silently. The engine logs its fusion count per render — that number
        # is the thing to check first, not the prompt.
        if recipe.get("checkpoint"):
            payload["checkpoint"] = str(recipe["checkpoint"]).strip()
        # `is not None`, not truthiness: img_compression 0 is a real setting — it bypasses
        # the conditioning-frame encode — and a falsy check would drop it, leaving the engine
        # on its workflow default while the pose said otherwise.
        if recipe.get("img_compression") is not None:
            payload["img_compression"] = int(recipe["img_compression"])
        lora = recipe.get("char_lora")
        # "none" is how a render says "no character" — useful for judging what the LoRA is
        # actually contributing, and for a shot whose start frame already carries the
        # identity (console#412).
        #
        # It has to be filtered HERE, not left to the engine. `if lora` alone passes,
        # because the STRING "none" is truthy, and the entry would then be normalised to
        # "none.safetensors" — which is no longer the literal "none" the engine's own
        # want_char check looks for, so it would sail past that and 422 on a file that does
        # not exist, ten minutes into a claimed segment.
        if lora and str(lora).strip().lower() == "none":
            lora = None
        s1, s2 = recipe.get("char_s1"), recipe.get("char_s2")
        if lora and s1 is not None and s2 is not None:
            # The engine matches LoRAs by EXACT filename. Its own recipe path appends the
            # extension; the explicit-lora path this uses does not. Names arrive bare because
            # that is how the recipe sheet wrote them and how the console displays them, so a
            # real render died on `no such lora 'pay_v2_e05'` while 'pay_v2_e05.safetensors'
            # sat in the very list the error printed.
            #
            # Normalised here, at the boundary that talks to the engine, rather than in the
            # database: what a LoRA is CALLED is a display concern and what file it IS is the
            # engine's, and everything in between should not have to agree on the extension.
            if not lora.endswith(".safetensors"):
                lora = f"{lora}.safetensors"
            # Per-stage, never flat. Stage 1 generates at half size from noise and stage 2
            # refines the 2x-upscaled latent; the validated recipe runs 0.8 then 1.5, and
            # collapsing them to one number is a different configuration.
            payload["loras"] = [{
                "name": lora,
                "strength": float(s1),
                "strength_stage_1": float(s1),
                "strength_stage_2": float(s2),
            }]

        # The POSE's content LoRAs — motion and act — chained ahead of the character LoRA on
        # both stage branches, IN ORDER. They stack: motion, act and framing are separable
        # and a pose may want several (console#410). Order is part of the configuration, so
        # it is forwarded exactly as stored.
        #
        # Sent as their own field, not appended to `loras`. On this path the engine reads
        # loras[0] as the CHARACTER LoRA, so a content LoRA added to that list would be
        # loaded as a character — both load, the render succeeds, and it is the wrong person.
        contents = []
        for entry in (recipe.get("content_loras") or []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name or name.lower() == "none":
                # "none" is how a pose says off. Forwarded, the engine would look for
                # 'none.safetensors' and the segment would die ten minutes into a claim.
                continue
            # Same extension normalisation as the character LoRA, for the same reason: names
            # are stored bare because that is how they are displayed, and the engine matches
            # files exactly. A real render died on `no such lora 'pay_v2_e05'` while the
            # .safetensors sat in the very list the error printed.
            item = {"name": name if name.endswith(".safetensors") else f"{name}.safetensors"}
            # `is not None`, not truthiness: a strength of 0 is a REAL setting — it loads the
            # LoRA and gives it no weight, which is how you measure what it contributes. A
            # falsy check would drop it and the engine would apply its own 0.6 default, so
            # the measurement would silently be of a different configuration.
            for k in ("s1", "s2"):
                if entry.get(k) is not None:
                    item[k] = float(entry[k])
            contents.append(item)
        if contents:
            payload["content_loras"] = contents

    return payload


class LtxClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or settings.ltx_engine_url).rstrip("/")
        # Submitting returns immediately; only the video GET streams anything large.
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> dict[str, Any]:
        r = await self._client.get("/health")
        r.raise_for_status()
        result: dict[str, Any] = r.json()
        return result

    async def purge(self, job_id: str) -> dict:
        """Drop the engine's local media for a finished job (console#380).

        Only ever called AFTER a successful upload: the local file is the only copy until
        that upload lands, and reclaiming disk is not worth risking a render.

        The engine keeps graph.json and prompt.txt — ~7 MB across 500 jobs against 2.7 GB of
        media, and the only local record of what a render actually did.
        """
        r = await self._client.post(f"/job/{job_id}/purge", timeout=30.0)
        r.raise_for_status()
        return r.json()

    async def checkpoints(self) -> list[str]:
        """Base models this worker can actually load.

        Asked of the engine rather than globbed off disk, because the engine asks ComfyUI,
        and ComfyUI answers from the folder mapping in extra_model_paths.yaml. A file the
        mapping does not cover is invisible to a render however present it is on the
        filesystem — so this lists what will LOAD, not what exists.

        The daemon is the only thing that can ask: the engine binds to 127.0.0.1 inside the
        container, so nothing upstream can reach it. That is why this is reported through
        the heartbeat rather than fetched by the API (console#404).
        """
        r = await self._client.get("/checkpoints", timeout=30.0)
        r.raise_for_status()
        names = r.json().get("checkpoints") or []
        # Bare names, matching how a recipe stores one and how the console shows it. The
        # engine re-appends the extension when it resolves the file.
        return sorted({n[: -len(".safetensors")] if n.endswith(".safetensors") else n
                       for n in names})

    async def submit(
        self,
        *,
        image_bytes: bytes | None,
        prompt: str,
        negative_prompt: str | None,
        width: int,
        height: int,
        num_frames: int,
        frame_rate: int,
        seed: int,
        recipe: dict[str, Any] | None,
    ) -> str:
        """Queue a render. Returns the engine's job id."""
        payload = build_submit_payload(
            image_bytes=image_bytes, prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames, frame_rate=frame_rate,
            seed=seed, recipe=recipe,
        )
        r = await self._client.post("/job", json=payload)
        if r.status_code >= 400:
            raise LtxEngineError(f"submit failed: {r.status_code} {r.text[:500]}")
        job_id: str = r.json()["job_id"]
        logger.info("ltx-engine accepted job %s", job_id)
        return job_id

    async def wait(self, job_id: str, progress: Any = None) -> dict[str, Any]:
        """Poll until the render finishes. Returns the final job record.

        A transient failure to reach the engine is NOT treated as a failed render: the engine
        holds jobs in memory, so a restart loses the record while the mp4 survives on disk.
        Declaring failure on a dropped poll would abandon work that is still running or
        already finished.
        """
        deadline = time.monotonic() + settings.ltx_timeout_seconds
        last_status = None
        consecutive_errors = 0

        while True:
            if time.monotonic() > deadline:
                raise LtxEngineError(
                    f"job {job_id} still {last_status or 'unknown'} after "
                    f"{settings.ltx_timeout_seconds}s"
                )
            try:
                r = await self._client.get(f"/job/{job_id}")
                r.raise_for_status()
                job: dict[str, Any] = r.json()
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                # Loud after a while, but never fatal — see the docstring.
                if consecutive_errors in (1, 12) or consecutive_errors % 60 == 0:
                    logger.warning(
                        "ltx-engine poll failed (%d in a row, still waiting): %s",
                        consecutive_errors, e,
                    )
                await asyncio.sleep(settings.ltx_poll_interval)
                continue

            status = job.get("status")
            if status != last_status:
                logger.info("ltx-engine job %s: %s", job_id, status)
                if progress is not None:
                    await progress.log(f"[4/6] ltx-engine: {status}")
                last_status = status

            if status in TERMINAL:
                if status == "Failed":
                    raise LtxEngineError(job.get("error") or "engine reported Failed")
                return job

            await asyncio.sleep(settings.ltx_poll_interval)

    async def fetch_video(self, job_id: str) -> bytes:
        """Download the rendered mp4."""
        async with httpx.AsyncClient(base_url=self.base_url, timeout=900.0) as c:
            r = await c.get(f"/job/{job_id}/video")
            if r.status_code >= 400:
                raise LtxEngineError(f"video fetch failed: {r.status_code} {r.text[:200]}")
            return r.content
