# BUILD: wanly Lynx identity-preserving engine

Status: **POC, RunPod-only.** Graph builder, model staging, API plumbing and identity QA
are implemented and unit-tested. Nothing has been generated yet — every VRAM and
similarity number below is marked either MEASURED or UNMEASURED, and the calibration
matrix is not yet run.

---

## 1. What Lynx is, and why we want it

Lynx (ByteDance, Apache 2.0) runs on **Wan 2.1 T2V-14B** and adds two adapters:

| Adapter | Mechanism | What it controls |
|---|---|---|
| **ID-adapter** (`ip`) | ArcFace embedding of a 112×112 aligned face crop → Perceiver resampler → identity tokens | *Who the face is.* Too high tends to freeze expression. |
| **Ref-adapter** (`ref`) | Dense VAE features of a wider 256px crop, cross-attended in the DiT blocks | *Fine appearance* — skin, hair, lighting. Too high drags the reference's pose and lighting in. |

The important difference from a character LoRA: **identity is re-asserted at every denoising
step**, rather than baked into the weights once. That is why this is worth trying after the
identity-drift audit — the failure mode there was that identity conditioning got out-weighted
as the denoise progressed.

**This is a T2V model conditioned on a subject image, not first-frame i2v.** The subject
never appears as frame 0. Parameters are named `subject_image`, never `start_frame`.

---

## 2. Graph topology

Node IDs use the `6xx` block (VACE owns `5xx`, LoRA chains `7xx`), built by
`build_lynx_workflow` in `daemon/workflow_builder.py`.

```
                    630 LoadImage (subject)
                             │
                    631 LynxInsightFaceCrop
                     ┌───────┴────────┐
              out 0 (112px)      out 1 (256px)
              ArcFace crop        ref crop
                     │                │
632 LoadLynxResampler│                │
        └──► 633 LynxEncodeFaceIP     │
                     │ (LYNXIP)       │
601 ExtraModelSelect (ip layers)      │
        └──► 602 ExtraModelSelect (ref layers, prev_model=601)
                     │                │
600 WanVideoBlockSwap│                │        640 WanVideoEmptyEmbeds (w, h, frames)
700+ LoRA chain      │                │                │
        └──► 610 WanVideoModelLoader  │                │
                     │                └──► 641 WanVideoAddLynxEmbeds ◄── 620 VAE
                     │                          ▲               ◄── 622 ref text embed
                     │                          │
                     └──────► 650 WanVideoSampler ◄── 621 main text embed
                                      │
                              660 WanVideoDecode
                                      │
                          [670 RIFE VFI]  (only when target fps > 15)
                                      │
                              680 VHS_VideoCombine
```

Three things here are easy to get wrong:

1. **Both adapters reach the model through one chained `extra_model` list.** `602` chains
   onto `601` via `prev_model`, and only the tail (`602`) is wired to the loader. Wiring
   both separately loads only one.
2. **The ref path needs its own text embed and the VAE.** `WanVideoAddLynxEmbeds` raises
   `ValueError` if `ref_image` is supplied without `ref_text_embed`. ByteDance hardcode the
   prompt as `"image of a face"`; it is `lynx_ref_prompt` in settings. This is a *second*
   `WanVideoTextEncodeCached` (node 622), separate from the main prompt (621).
3. **`ref_blocks_to_use` is omitted when empty**, not passed as `""`. It is declared
   `forceInput`, and the node already treats absent as "all blocks".

Both text encodes use `WanVideoTextEncodeCached` with `device: gpu` and
`use_disk_cache: False` — the same offload handling the VACE path uses, because umt5-xxl
residency has OOM'd this box before.

---

## 3. Models

All Lynx files live in `models/diffusion_models/` — both `LoadLynxResampler` and
`WanVideoExtraModelSelect` read from that folder, regardless of what the file is.

| Setting | File | Size | Purpose |
|---|---|---|---|
| `lynx_t2v_model` | `Wan2_1-T2V-14B_fp8_e4m3fn_scaled_KJ.safetensors` | 14.5 GB | Base |
| `lynx_ref_layers` | `Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors` | 4.2 GB | Ref-adapter |
| `lynx_ip_layers` | `Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors` | 0.84 GB | ID-adapter (**default**) |
| `lynx_resampler` | `lynx_lite_resampler_fp32.safetensors` | 0.33 GB | Resampler (**default**) |
| — | `Wan2_1-T2V-14B-Lynx_full_ip_layers_fp16.safetensors` | 4.2 GB | ID-adapter (A/B arm) |
| — | `lynx_full_resampler_fp32.safetensors` | 0.34 GB | Resampler (A/B arm) |
| `lynx_distill_lora` | `lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors` | 0.63 GB | cfg-step distill |

Total **25 GB**, plus the shared `models_t5_umt5-xxl-enc-bf16.pth` (11 GB) and
`wan_2.1_vae.safetensors`.

### Why Kijai's repacks, not `Wan-AI` / `ByteDance/lynx`

The original task named the upstream repos. Those ship sharded diffusers-format weights;
WanVideoWrapper's loaders want a single file. Kijai publishes exactly that, already in our
layout, and his `_scaled_KJ` fp8 quant is the same family as our `Wan2_2-Animate-14B` model.
Using upstream would mean a conversion step for no benefit.

### The `lite` vs `full` ip trap

The task said to skip `lynx_lite`. **Kijai's own workflow note says the opposite:**

> "Original implementation uses full ip layers with full ref layers, I don't know if
> there's some mistake in my implementation as the full ip adapter seems very weak, and
> using lite ip instead seems better"

His shipped reference workflow runs **lite ip + full ref**, which is why that is our
default. Both arms are staged so this is settled by measurement rather than by either
model card.

**The ip layers and the resampler are a matched pair** — the resampler's `proj_out`
dimension must match the ip layers it feeds. A mismatched pair *loads without raising* and
produces garbage identity, so `_validate_lynx` rejects it up front. Swap
`lynx_ip_layers` and `lynx_resampler` together, always.

### Face models fetched at runtime

Two models are downloaded lazily on first use, both outside `/workspace`:

- **facexlib ArcFace IR-SE50** (`recognition_arcface_ir_se50.pth`, from GitHub releases) →
  `custom_nodes/ComfyUI-WanVideoWrapper/lynx/face/facexlib/weights/`. This is what Lynx
  actually conditions identity on.
- **insightface `buffalo_l`** → `~/.insightface/models/`. Used only for the 5-point
  landmarks that drive the crop — and separately by our identity QA.

`download_models.sh` pre-stages both, so a pod boot does not depend on GitHub being
reachable mid-job.

---

## 4. Parameters

Precedence everywhere: **per-job override → settings default**. `None` means "not set";
every other value including `0` and `""` is an intentional override and wins.

| Param | Default | Semantics |
|---|---|---|
| `ip_scale` | **0.7** | ID-adapter strength. Raise for likeness, at the cost of expression range. |
| `ref_scale` | **0.6** | Ref-adapter strength. Raise for appearance fidelity, at the cost of importing the reference's pose/lighting. |
| `lynx_cfg_scale` | 2.0 | **Inert on the distilled path.** Only triggers an extra pass when the *main* cfg is also > 1.0, and the distill LoRA pins cfg to 1.0. Matters only once de-distilled. |
| `start_percent` | 0.0 | Fraction of the denoise at which the ref adapter *starts* applying. |
| `end_percent` | 1.0 | Fraction at which it *stops*. Narrowing this (e.g. 0.0–0.6) frees the late steps from the reference — the lever to try if identity holds but motion looks stiff. |
| `ref_blocks_to_use` | `""` (all) | Which DiT blocks receive the ref feature, e.g. `"0-20, 25, 35-39"`. Restricting to early blocks applies identity to structure while leaving late blocks free for detail. Untested by us. |
| `steps` / `cfg` / `shift` / `scheduler` | 6 / 1.0 / 8.0 / `lcm` | From Kijai's reference workflow. |
| `distill_strength` | 1.0 | **0 drops the LoRA node entirely** rather than applying strength 0 — the de-distilled path, where you would also raise cfg and steps. |

Note the defaults are Kijai's 0.7/0.6, *not* the 1.0/1.0 in the original task. Both are
in the A/B matrix.

### Constraints — validated, never clamped

`LynxValidationError` is raised with the offending value named. Silent clamping was
rejected because an off-bucket resolution or off-grid frame count degrades identity in
ways that are very hard to attribute afterwards.

- **Resolution:** `832×480` or `1280×720` only.
- **Frames:** `4n+1`, minimum 5 (81 is the smoke-test shape).
- **Adapter arms:** ip layers and resampler must both be `lite` or both `full`.
- **Subject image:** required.

---

## 5. Placement

**All Lynx jobs go to RunPod.** This is not a tuning decision — the local 3090 physically
cannot run Lynx today:

| | WanVideoWrapper | Lynx nodes |
|---|---|---|
| 3090 (`3090.zero`) | **1.3.3** | ✗ (Lynx landed in 1.3.5) |
| RunPod image | **1.3.9** (pinned `e926f7a0`) | ✓ |

The RunPod wrapper's `lynx/nodes.py` is **byte-identical to current upstream `main`**, so
the graph builder is version-stable across 1.3.9 → 1.4.7.

Upgrading the 3090's wrapper is deferred deliberately: its ComfyUI core is pinned to an
Apr-28 rollback for the activation-memory fix, and a 6-month wrapper jump risks disturbing
that. Revisit only if the POC's identity numbers justify it.

`_lynx_preflight` fails fast with a diagnosis rather than a bare "node not found", because
two very different causes present identically:

1. the wrapper predates 1.3.5, or
2. the wrapper is new enough but its Lynx import raised — **it logs a warning and
   registers nothing**, and a missing `insightface` does exactly this.

There is no silent fallback to the 2.2 i2v path: Lynx is a different base model family, so
a worker that cannot run it must say so.

---

## 6. VRAM

**UNMEASURED.** No Lynx generation has run yet. To be filled from
`lynx.generate.end` → `vram_peak_mb` on the first pod run.

| Config | Peak VRAM | Status |
|---|---|---|
| 832×480, 81f, fp8 + block swap 35 | — | pending |
| 1280×720, 81f | — | pending |

Budget sketch, for expectations only: 14.5 GB fp8 base (block-swapped, so resident share
is well under that) + ~5 GB adapters + VAE + latents. `lynx_blocks_to_swap` is 35, from
Kijai's reference workflow — higher than our VACE path's 25, because the adapters add
resident weight on top of the base.

**Measurement caveat:** the daemon is a *separate process* from ComfyUI, so
`torch.cuda.max_memory_allocated()` would report the daemon's own empty allocator. We poll
device-level usage via `nvidia-smi` on a background thread (0.5 s) and report the peak.
That figure includes anything else resident on the card — which is the number that
actually matters for "does this fit in 24 GB", but is not directly comparable to a
PyTorch allocator figure.

---

## 7. Identity QA

After each Lynx render the daemon samples 5 frames (interior — the first and last frames
are least representative), embeds the largest face in each with InsightFace, and
cosine-compares against the subject crop. Results are logged as `lynx.identity_qa` with
the correlation ID and persisted to `segments.lynx_identity_scores`.

**Measurement only.** No gating, no retries — that is a separate task. The QA path is
wrapped so it can never fail an otherwise good render; if insightface is missing or the
video is unreadable, scores are simply `null`.

**Embedding-space caveat:** Lynx *conditions* on facexlib ArcFace IR-SE50; we *measure*
with InsightFace buffalo_l. Different spaces, so absolute values are not comparable to
published Lynx numbers. They are only meaningful **relative to each other across A/B
arms**, which is exactly how the matrix below uses them.

---

## 8. Calibration matrix — NOT YET RUN

Same prompt, subject and seed across all cells; 832×480, 81 frames.

| # | ip arm | ip_scale | ref_scale | mean cos | notes |
|---|---|---|---|---|---|
| 1 | lite | 0.7 | 0.6 | — | Kijai default |
| 2 | lite | 1.0 | 1.0 | — | task-spec default |
| 3 | full | 0.7 | 0.6 | — | tests Kijai's "full ip is weak" claim |
| 4 | full | 1.0 | 1.0 | — | |

Reference points from the original task, folded in: `(0.5, 0.5)` and `(1.5, 1.0)` are
worth adding once the four above show which arm is worth exploring.

---

## 9. Open questions

- Does lite ip actually beat full ip on measured similarity, or is Kijai's note specific
  to his test subject?
- Does `end_percent` < 1.0 buy motion freedom without losing likeness?
- Does a character LoRA stacked on Lynx help or fight it? Both mechanisms target identity;
  they may not compose.
- Wan 2.1 is a generation behind our 2.2 i2v path. Even if identity improves, is the
  motion/quality regression acceptable? This is the real go/no-go, and similarity scores
  alone will not answer it.

---

## 10. Files

| Repo | Path | What |
|---|---|---|
| daemon | `daemon/workflow_builder.py` | `build_lynx_workflow`, `_validate_lynx`, `lynx_num_frames` |
| daemon | `daemon/executor.py` | `_execute_lynx`, `_lynx_preflight`, `_resolve_lynx_subject` |
| daemon | `daemon/stage_log.py` | structured JSON stage events + VRAM peaks |
| daemon | `daemon/identity_check.py` | cosine-similarity QA |
| daemon | `daemon/config.py` | `lynx_*` settings |
| daemon | `tests/golden/lynx_832x480_81f.json` | golden graph for a fixed param set |
| api | `alembic/versions/050_add_lynx_engine.py` | job tunables + segment scores |
| runpod | `download_models.sh` | Lynx staging, `MODEL_PROFILE=lynx` |
