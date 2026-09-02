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
        # `is not None`, not truthiness: img_compression 0 is a real setting — it bypasses
        # the conditioning-frame encode — and a falsy check would drop it, leaving the engine
        # on its workflow default while the pose said otherwise.
        if recipe.get("img_compression") is not None:
            payload["img_compression"] = int(recipe["img_compression"])
        lora = recipe.get("char_lora")
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

    # The POSE's content LoRA — motion and act — chained ahead of the character LoRA on both
    # stage branches. Sent as its own field, not appended to `loras`: that list is the
    # CHARACTER LoRA on this path and the engine reads loras[0] as such, so a second entry
    # would be silently taken for a character.
    content = recipe.get("content_lora")
    if content and str(content).strip().lower() != "none":
        content = str(content).strip()
        # Same extension normalisation, and for the same reason: names are stored bare
        # because that is how they are displayed, and the engine matches files exactly.
        if not content.endswith(".safetensors"):
            content = f"{content}.safetensors"
        payload["content_lora"] = content
        # `is not None`, exactly as img_compression above. A content strength of 0 is a REAL
        # setting: it loads the LoRA and gives it no weight, which is how you measure what it
        # contributes. A falsy check would drop it and the engine would apply its 0.6
        # default, so the measurement would silently be of a different configuration.
        for key, field in (("content_s1", "content_s1"), ("content_s2", "content_s2")):
            if recipe.get(key) is not None:
                payload[field] = float(recipe[key])
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
