# wanly-gpu-daemon -- Codebase Audit

**Audit Date:** 2026-03-12T15:59:35Z
**Branch:** feature/daemon-safeguards
**Commit:** 3768d97f9d2d644f1a1d0e9aa670f1bd448d47c6 Add daemon safeguards: stale process killer, queue clear, progress timeout, GPU stats
**Auditor:** Claude Code (Automated)
**Purpose:** Zero-context reference for AI-assisted development
**Stack:** Python / asyncio / httpx / ComfyUI integration
**Audit File:** wanly-gpu-daemon-Audit.md
**Scorecard:** wanly-gpu-daemon-Scorecard.md

> This audit is the source of truth for the wanly-gpu-daemon codebase structure, job execution pipeline, ComfyUI integration, and configuration.
> An AI reading this audit should be able to generate accurate code changes, new features, tests, and fixes without filesystem access.

---

## Table of Contents

1. [Project Identity](#1-project-identity)
2. [Directory Structure](#2-directory-structure)
3. [Build & Dependency Manifest](#3-build--dependency-manifest)
4. [Configuration & Infrastructure Summary](#4-configuration--infrastructure-summary)
5. [Startup & Runtime Behavior](#5-startup--runtime-behavior)
6. [Data Models](#6-data-models)
7. [Enum Inventory](#7-enum-inventory)
8. [HTTP Client Layer](#8-http-client-layer)
9. [Service/Business Logic Layer](#9-servicebusiness-logic-layer)
10. [ComfyUI Integration](#10-comfyui-integration)
11. [S3/Storage Integration](#11-s3storage-integration)
12. [Utility Modules](#12-utility-modules)
13. [Environment Variable Inventory](#13-environment-variable-inventory)
14. [Service Dependency Map](#14-service-dependency-map)
15. [Known Technical Debt](#15-known-technical-debt)

---

## 1. Project Identity

- **Name:** wanly-gpu-daemon
- **Type:** Long-running async Python daemon (NOT a web server)
- **Role:** GPU worker -- polls for video generation jobs, executes ComfyUI workflows, uploads results
- **Entry point:** `python -m daemon.main` (calls `daemon.main:main()` which runs `asyncio.run(run())`)
- **Total source:** 2,235 lines across 14 Python files
- **Test files:** 0 (no test suite exists)

---

## 2. Directory Structure

```
wanly-gpu-daemon/
  .env                        # Active config (points to gpu-registry.wanly22.com)
  .env.example                # Template with all settings
  requirements.txt            # 6 pinned-free dependencies
  README.md                   # Setup docs
  daemon/
    __init__.py               # Empty
    config.py                 # Pydantic Settings (33 lines)
    main.py                   # Entry point + main loop (420 lines)
    schemas.py                # Pydantic models (52 lines)
    comfyui_client.py         # ComfyUI HTTP + WebSocket client (247 lines)
    executor.py               # Segment execution pipeline (300 lines)
    workflow_builder.py       # ComfyUI workflow JSON construction (475 lines)
    queue_client.py           # wanly-api HTTP client (80 lines)
    registry_client.py        # wanly-gpu-registry HTTP client (62 lines)
    progress.py               # Progress log accumulator (32 lines)
    lora_sync.py              # LoRA file download + cache (93 lines)
    model_validator.py        # Pre-flight model checks (182 lines)
    node_checker.py           # Custom node installer (177 lines)
    resource_sync.py          # Model weight downloader (82 lines)
```

---

## 3. Build & Dependency Manifest

**File:** `/home/david/projects/wanly/wanly-gpu-daemon/requirements.txt`

| Package | Purpose |
|---------|---------|
| httpx | Async HTTP client for API + ComfyUI communication |
| Pillow | Image validation (PIL.Image.open/verify) |
| pydantic-settings | Config from env vars |
| python-dotenv | .env file loading |
| pyyaml | Parsing ComfyUI extra_model_paths.yaml |
| websockets | ComfyUI execution monitoring via WebSocket |

**No version pins.** All dependencies are unpinned.

**No Dockerfile.** The daemon is deployed via git pull (start.sh on RunPod clones the repo), not via container image.

**No pyproject.toml, setup.py, or setup.cfg.**

---

## 4. Configuration & Infrastructure Summary

**File:** `/home/david/projects/wanly/wanly-gpu-daemon/daemon/config.py`

```python
class Settings(BaseSettings):
    registry_url: str = "http://localhost:8000"
    friendly_name: str = "gpu-worker-1"
    heartbeat_interval: int = 30
    comfyui_url: str = "http://localhost:8188"
    comfyui_api_key: str = ""
    comfyui_path: str = ""
    lora_cache_dir: str = ""
    queue_url: str = "http://localhost:8001"
    poll_interval: int = 5
    clip_model: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_model: str = "wan_2.1_vae.safetensors"
    unet_high_model: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_model: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    lightx2v_lora_high: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    lightx2v_lora_low: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    lightx2v_strength_high: float = 2.0
    lightx2v_strength_low: float = 1.0
    clip_vision_model: str = "clip_vision_h.safetensors"
    runpod_pod_id: str | None = None
    runpod_api_key: str | None = None
    model_config = {"env_file": ".env"}
```

Singleton: `settings = Settings()` at module level.

**No env_prefix** -- env vars map directly (e.g., `REGISTRY_URL`, not `WANLY_DAEMON_REGISTRY_URL`). This differs from the CLAUDE.md convention.

---

## 5. Startup & Runtime Behavior

**File:** `/home/david/projects/wanly/wanly-gpu-daemon/daemon/main.py`

### Startup Sequence (`async def run()`)

1. **`kill_stale_daemons()`** -- Scans `/proc/` for other `daemon.main` Python processes and SIGKILL's them
2. Create client instances: `RegistryClient`, `ComfyUIClient`, `QueueClient`
3. Set up `shutdown_event`, `executing_event`, `drain_event` (all `asyncio.Event`)
4. Register SIGINT/SIGTERM handlers to set `shutdown_event`
5. **ComfyUI health check** -- `comfyui.check_health()` via GET `/system_stats`
6. **Clear ComfyUI queue** -- `comfyui.clear_queue()` via POST `/queue` with `{"clear": true}`
7. **`check_and_install_nodes(comfyui)`** -- Verify/install custom ComfyUI nodes (exits on failure)
8. **`sync_resources(queue)`** -- Download RIFE model weights from S3 if missing (exits on failure)
9. **`cleanup_partial_downloads(comfyui_path)`** -- Remove .aria2/.tmp/.part files
10. **`validate_models(comfyui)`** -- Check all model files exist and meet size thresholds (exits on failure)
11. **Log GPU/VRAM info** from ComfyUI system stats
12. **`register_with_retry()`** -- Register with GPU registry, retry every 10s until success or shutdown
13. **Launch two concurrent tasks:**
    - `heartbeat_loop()` -- every `heartbeat_interval` (default 30s)
    - `job_poll_loop()` -- every `poll_interval` (default 5s)

### Heartbeat Loop (`async def heartbeat_loop()`)

- Sends heartbeat to registry with `comfyui_running` status and GPU stats (VRAM used/total, GPU name, torch VRAM)
- Picks up friendly name renames from registry response
- Detects `"draining"` status from registry and sets `drain_event`
- Logs ComfyUI state changes (offline/busy/idle)

### Job Poll Loop (`async def job_poll_loop()`)

- Skips polling if `drain_event` is set (triggers shutdown)
- Skips polling if ComfyUI is offline or queue busy
- Calls `queue.claim_next(worker_id, friendly_name)` via GET `/segments/next`
- On claim: sets `executing_event`, updates registry to `online-busy`, calls `execute_segment()`
- After execution: logs VRAM usage, clears `executing_event`, updates registry to `online-idle`
- If draining after segment completes, triggers shutdown

### Shutdown Sequence

- Waits up to 10 minutes for current segment to finish (if executing)
- Deregisters from registry
- If draining on RunPod: calls RunPod GraphQL API to stop the pod
- Closes all HTTP clients

### Key Functions

| Function | Signature | Purpose |
|----------|-----------|---------|
| `get_ip_address()` | `() -> str` | UDP socket trick to find local IP |
| `_log_system_info(system_info)` | `(dict \| None) -> None` | Log GPU/VRAM/RAM from ComfyUI |
| `register_with_retry(client, ...)` | `async (client, *, friendly_name, hostname, ip_address, comfyui_running, shutdown_event)` | Register with retry every 10s |
| `heartbeat_loop(...)` | `async (registry, comfyui, worker_id, friendly_name_ref, shutdown_event, drain_event)` | Heartbeat coroutine |
| `job_poll_loop(...)` | `async (registry, comfyui, queue, worker_id, friendly_name_ref, shutdown_event, executing_event, drain_event)` | Job polling coroutine |
| `kill_stale_daemons()` | `() -> None` | Kill leftover daemon.main processes |
| `_stop_runpod_pod()` | `async () -> None` | RunPod GraphQL mutation to stop pod |
| `run()` | `async () -> None` | Main orchestrator |
| `main()` | `() -> None` | Entry point, calls `asyncio.run(run())` |

---

## 6. Data Models

**File:** `/home/david/projects/wanly/wanly-gpu-daemon/daemon/schemas.py`

### LoraItem
```python
class LoraItem(BaseModel):
    lora_id: Optional[str] = None
    high_file: Optional[str] = None
    high_s3_uri: Optional[str] = None
    high_weight: float = 1.0
    low_file: Optional[str] = None
    low_s3_uri: Optional[str] = None
    low_weight: float = 1.0
```

### SegmentClaim
```python
class SegmentClaim(BaseModel):
    id: UUID
    job_id: UUID
    index: int
    prompt: str
    duration_seconds: float
    speed: float = 1.0
    start_image: Optional[str] = None
    loras: Optional[list[LoraItem]] = None
    faceswap_enabled: bool
    faceswap_method: Optional[str] = None        # "reactor" or "facefusion" (default)
    faceswap_source_type: Optional[str] = None
    faceswap_image: Optional[str] = None
    faceswap_faces_order: Optional[str] = None
    faceswap_faces_index: Optional[str] = None
    initial_reference_image: Optional[str] = None
    lightx2v_strength_high: Optional[float] = None
    lightx2v_strength_low: Optional[float] = None
    width: int
    height: int
    fps: int
    seed: int
```

### SegmentResult
```python
class SegmentResult(BaseModel):
    status: str                          # "completed", "failed", or "processing"
    output_path: Optional[str] = None
    last_frame_path: Optional[str] = None
    error_message: Optional[str] = None
    progress_log: Optional[str] = None
```

---

## 7. Enum Inventory

**No enums defined.** Status values are string literals:
- Worker status: `"online-idle"`, `"online-busy"` (sent to registry)
- Segment status: `"processing"`, `"completed"`, `"failed"` (sent to API)
- Faceswap method: `"reactor"` or default (FaceFusion)

---

## 8. HTTP Client Layer

### QueueClient (`daemon/queue_client.py`)

Talks to **wanly-api** (settings.queue_url, default `http://localhost:8001`).

| Method | HTTP | Endpoint | Purpose |
|--------|------|----------|---------|
| `claim_next(worker_id, worker_name)` | GET | `/segments/next?worker_id=&worker_name=` | Claim next pending segment |
| `update_segment(segment_id, result)` | PATCH | `/segments/{id}` | Report status/progress/failure |
| `upload_segment_output(segment_id, video_data, last_frame_data)` | POST | `/segments/{id}/upload` | Multipart upload of video + last frame |
| `download_file(s3_path)` | GET | `/files?path=s3://...` | Download file from S3 via API proxy |

- Timeout: 10s default, 300s for uploads, 600s for .safetensors/.pth downloads
- Error handling: `_raise_with_details()` logs HTTP body before re-raising

### RegistryClient (`daemon/registry_client.py`)

Talks to **wanly-gpu-registry** (settings.registry_url, default `http://localhost:8000`).

| Method | HTTP | Endpoint | Purpose |
|--------|------|----------|---------|
| `register(friendly_name, hostname, ip_address, comfyui_running)` | POST | `/workers` | Register worker, returns (worker_id, friendly_name) |
| `heartbeat(worker_id, comfyui_running, gpu_stats)` | POST | `/workers/{id}/heartbeat` | Send heartbeat with GPU stats |
| `update_status(worker_id, status)` | PATCH | `/workers/{id}/status` | Update worker status |
| `deregister(worker_id)` | DELETE | `/workers/{id}` | Remove worker |

- Timeout: 10s for all requests
- Returns raw dict from heartbeat (includes `friendly_name`, `status` for drain detection)

### ComfyUIClient (`daemon/comfyui_client.py`)

Talks to **local ComfyUI** (settings.comfyui_url, default `http://localhost:8188`).

| Method | HTTP | Endpoint | Purpose |
|--------|------|----------|---------|
| `check_health()` | GET | `/system_stats` | Health check |
| `check_queue_busy()` | GET | `/queue` | Check if prompt is running |
| `clear_queue()` | POST | `/queue` | Clear pending/running items |
| `get_system_info()` | GET | `/system_stats` | GPU/VRAM/RAM stats |
| `upload_image(data, filename)` | POST | `/upload/image` | Upload start/faceswap image |
| `submit_workflow(workflow)` | POST | `/prompt` | Submit workflow, returns (prompt_id, client_id) |
| `get_history(prompt_id)` | GET | `/history/{prompt_id}` | Get execution outputs (5 retries, 1s delay) |
| `download_output(filename, subfolder, output_type)` | GET | `/view` | Download output video |
| `monitor_execution(prompt_id, client_id)` | WS | `/ws?clientId=` | WebSocket monitoring with timeout |

- Auth: Bearer token via `Authorization` header + `?token=` on WebSocket (from `comfyui_api_key`)
- Timeout: 30s for HTTP, 1800s (30min) overall execution, 300s (5min) progress timeout

---

## 9. Service/Business Logic Layer

### Segment Execution Pipeline (`daemon/executor.py`)

**Function:** `async def execute_segment(segment: SegmentClaim, comfyui: ComfyUIClient, queue: QueueClient) -> None`

7-step pipeline with timing for each step:

| Step | Description | Key Operations |
|------|-------------|----------------|
| 1/7 | Download start image | S3 -> API proxy -> ComfyUI upload; validates with PIL |
| 1b | Download initial reference image | For PainterLongVideo identity anchoring (segment > 0) |
| 1c | Download faceswap image | If faceswap enabled |
| 2/7 | Sync LoRA files | `ensure_loras_available()` downloads missing LoRAs from S3 |
| 3/7 | Build workflow | `build_workflow()` constructs ComfyUI API JSON |
| 4/7 | Submit to ComfyUI | POST `/prompt` |
| 5/7 | Wait for execution | WebSocket monitoring |
| 6/7 | Download output | Get history, find video, download bytes |
| 7/7 | Extract last frame + upload | ffmpeg `-sseof -0.1`, multipart upload to API |

**Error handling:** Catches `ComfyUIExecutionError` and general exceptions, reports failure with truncated error message (max 2000 chars) and progress log to the API.

**Helper functions:**
- `_validate_image_data(data, label)` -- PIL verify + size check
- `_download_with_retry(coro_factory, label, attempts=3, delay=2.0)` -- Retries on httpx transient errors
- `_resolve_start_image(segment, comfyui, queue)` -- Downloads S3 image via API, uploads to ComfyUI
- `_resolve_faceswap_image(segment, comfyui, queue)` -- Same for faceswap source
- `_extract_last_frame(video_data)` -- ffmpeg subprocess: `-sseof -0.1 -frames:v 1`

### LoRA Sync (`daemon/lora_sync.py`)

**Function:** `async def ensure_loras_available(loras: list[LoraItem], queue: QueueClient) -> None`

- Downloads LoRA .safetensors from S3 via API proxy
- Cache dir: `lora_cache_dir` setting or `{comfyui_path}/models/loras`
- Skips files already cached and >= 10 MB
- Re-downloads files < 10 MB (likely corrupt)
- Atomic writes: `.tmp` -> `os.rename()`
- Cleans up `.aria2`, `.tmp`, `.part` artifacts before each download

### Progress Log (`daemon/progress.py`)

**Class:** `ProgressLog(segment_id: UUID, queue)`

- Accumulates timestamped log lines (`[HH:MM:SS] message`)
- PATCHes accumulated text to API via `queue.update_segment()` after each line
- Non-fatal on PATCH failure (just logs debug)
- `.text` property returns newline-joined log

---

## 10. ComfyUI Integration

### Workflow Builder (`daemon/workflow_builder.py`)

**Function:** `def build_workflow(segment: SegmentClaim, start_image_filename: str | None = None, initial_reference_image_filename: str | None = None) -> dict`

**Base workflow:** `WAN_I2V_API_WORKFLOW` -- hardcoded dict of ComfyUI API-format nodes, defined at module level.

**Core nodes (always present):**

| Node ID | class_type | Purpose |
|---------|-----------|---------|
| 84 | CLIPLoader | Load text encoder |
| 85 | KSamplerAdvanced | Low noise sampler (steps 2-4) |
| 86 | KSamplerAdvanced | High noise sampler (steps 0-2) |
| 87 | VAEDecode | Decode latent to images |
| 89 | CLIPTextEncode | Negative prompt (Chinese text, hardcoded) |
| 90 | VAELoader | Load VAE |
| 93 | CLIPTextEncode | Positive prompt (user's prompt) |
| 95 | UNETLoader | High noise UNET |
| 96 | UNETLoader | Low noise UNET |
| 97 | LoadImage | Start image (removed for text-to-video) |
| 98 | WanImageToVideo | Main generation node |
| 101 | LoraLoaderModelOnly | LightX2V high noise LoRA |
| 102 | LoraLoaderModelOnly | LightX2V low noise LoRA |
| 103 | ModelSamplingSD3 | Shift=5.0 for low noise path |
| 104 | ModelSamplingSD3 | Shift=5.0 for high noise path |

**Dynamic nodes (added at runtime):**

| Node ID(s) | class_type | Condition |
|------------|-----------|-----------|
| 118/119, 120/121, 122/123 | LoraLoaderModelOnly | User LoRAs (up to 3 pairs high/low) |
| 183 | ReActorFaceSwapOpt or AdvancedSwapFaceImage | Faceswap enabled |
| 186 | VHS_VideoCombine | Always added |
| 188 | LoadImage | Faceswap source image |
| 189 | ReActorOptions | ReActor method only |
| 200 | RIFE VFI | Always added |
| 300 | LoadImage | PainterLongVideo reference image |
| 301 | CLIPVisionLoader | PainterLongVideo |
| 302 | CLIPVisionEncode | PainterLongVideo |

**Processing pipeline order:** VAEDecode (87) -> FaceSwap (183, if enabled) -> RIFE (200) -> VideoCombine (186)

**Generation parameters:** `_calculate_generation_params(target_fps, duration_sec, speed)`
- Generation always at 15fps
- `wan_frames = ceil(duration * 15 * speed)`, minimum 5
- `rife_multiplier = target_fps // 15` (2x for 30fps, 4x for 60fps)
- `output_fps = round(total_frames / duration)`

**PainterLongVideo:** When `initial_reference_image_filename` is provided and there is a start image, node 98 is replaced from `WanImageToVideo` to `PainterLongVideo` with dual-reference inputs (previous frame + identity anchor). Adds CLIP vision nodes (301, 302) for reference encoding.

### WebSocket Monitoring (`daemon/comfyui_client.py`)

- Connects to `ws://{host}/ws?clientId={uuid}`
- Handles message types: `execution_start`, `executing`, `progress`, `executed`, `execution_success`, `execution_error`
- Progress logging at 0%, 25%, 50%, 75%, 100% thresholds
- **Progress timeout:** 300s without any activity raises `ComfyUIExecutionError`
- **Overall timeout:** 1800s (30 minutes)
- Skips binary preview frames
- Filters messages by `prompt_id`

### Custom Node Checker (`daemon/node_checker.py`)

**Function:** `async def check_and_install_nodes(comfyui_client) -> bool`

Required custom node packages:

| Package | Nodes | Repo |
|---------|-------|------|
| ComfyUI-Frame-Interpolation | RIFE VFI | Fannovel16/ComfyUI-Frame-Interpolation |
| ComfyUI-VideoHelperSuite | VHS_VideoCombine | Kosinkadink/ComfyUI-VideoHelperSuite |
| comfyui-reactor-node | ReActorFaceSwapOpt, ReActorOptions | Gourieff/ComfyUI-ReActor |
| ComfyUI-PainterLongVideo | PainterLongVideo | princepainter/ComfyUI-PainterLongVideo |

- Phase 1: Check if package directories exist (including alt_dirs), git clone if missing, pip install requirements
- Phase 2: Verify via `/object_info` that node types are loaded
- If new nodes were installed, returns False (ComfyUI restart required)
- Missing nodes after install do NOT block startup (returns True with warning)

### Model Validator (`daemon/model_validator.py`)

**Function:** `async def validate_models(comfyui_client) -> bool`

Validates 6 model files at startup:

| Setting | Loader Node | Search Dirs | Min Size |
|---------|-------------|-------------|----------|
| clip_model | CLIPLoader | clip, text_encoders | 100 MB |
| vae_model | VAELoader | vae | 100 MB |
| unet_high_model | UNETLoader | diffusion_models, unet | 10 GB |
| unet_low_model | UNETLoader | diffusion_models, unet | 10 GB |
| lightx2v_lora_high | LoraLoaderModelOnly | loras | 100 MB |
| lightx2v_lora_low | LoraLoaderModelOnly | loras | 100 MB |

- Queries ComfyUI `/object_info` for known model lists
- Searches filesystem including `extra_model_paths.yaml` entries
- Trusts ComfyUI's model list if file can't be located on disk

**Function:** `def cleanup_partial_downloads(comfyui_path: str) -> int`

Walks `{comfyui_path}/models/` and removes `.aria2`, `.tmp`, `.part` files.

### Resource Sync (`daemon/resource_sync.py`)

**Function:** `async def sync_resources(queue: QueueClient) -> bool`

Hardcoded manifest of one resource:

| S3 URI | Local Path | Min Size |
|--------|-----------|----------|
| `s3://wanly-resources/comfyui/models/rife/rife49.pth` | `custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth` | 10 MB |

Downloads via API proxy (`queue.download_file()`), atomic writes, re-downloads undersized files.

---

## 11. S3/Storage Integration

The daemon has **no direct S3 access**. All file transfers go through wanly-api as a proxy:

- **Downloads:** `GET /files?path=s3://...` via `QueueClient.download_file()`
- **Uploads:** `POST /segments/{id}/upload` via `QueueClient.upload_segment_output()` (multipart: video + last_frame)
- **LoRA downloads:** Same `download_file()` path, 600s timeout for large .safetensors
- **Resource downloads:** Same path for RIFE weights

This design eliminates the need for AWS credentials on GPU workers (critical for RunPod ephemeral instances).

---

## 12. Utility Modules

### Progress Log (`daemon/progress.py`)

| Item | Detail |
|------|--------|
| Class | `ProgressLog(segment_id: UUID, queue)` |
| `.log(message)` | Appends `[HH:MM:SS] message`, PATCHes to API |
| `.text` | Property returning newline-joined lines |
| Error policy | PATCH failures are non-fatal (debug log) |

### ComfyUIExecutionError (`daemon/comfyui_client.py`)

```python
class ComfyUIExecutionError(Exception):
    def __init__(self, message: str, node_id: str = "", node_type: str = "", traceback: str = ""):
```

Custom exception carrying ComfyUI execution error details.

---

## 13. Environment Variable Inventory

All env vars are loaded by `pydantic_settings` from `.env` file (no prefix).

| Variable | Type | Default | Required | Purpose |
|----------|------|---------|----------|---------|
| REGISTRY_URL | str | `http://localhost:8000` | No | GPU registry URL |
| FRIENDLY_NAME | str | `gpu-worker-1` | No | Worker display name |
| HEARTBEAT_INTERVAL | int | 30 | No | Seconds between heartbeats |
| COMFYUI_URL | str | `http://localhost:8188` | No | Local ComfyUI URL |
| COMFYUI_API_KEY | str | `""` | No | Bearer token for ComfyUI auth |
| COMFYUI_PATH | str | `""` | Recommended | Path to ComfyUI installation |
| LORA_CACHE_DIR | str | `""` | No | Override LoRA cache directory |
| QUEUE_URL | str | `http://localhost:8001` | No | wanly-api URL |
| POLL_INTERVAL | int | 5 | No | Seconds between job polls |
| CLIP_MODEL | str | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | No | CLIP model filename |
| VAE_MODEL | str | `wan_2.1_vae.safetensors` | No | VAE model filename |
| UNET_HIGH_MODEL | str | `wan2.2_i2v_high_noise_14B_fp16.safetensors` | No | High noise UNET |
| UNET_LOW_MODEL | str | `wan2.2_i2v_low_noise_14B_fp16.safetensors` | No | Low noise UNET |
| LIGHTX2V_LORA_HIGH | str | (see config) | No | LightX2V high LoRA filename |
| LIGHTX2V_LORA_LOW | str | (see config) | No | LightX2V low LoRA filename |
| LIGHTX2V_STRENGTH_HIGH | float | 2.0 | No | High noise LoRA strength |
| LIGHTX2V_STRENGTH_LOW | float | 1.0 | No | Low noise LoRA strength |
| CLIP_VISION_MODEL | str | `clip_vision_h.safetensors` | No | CLIP vision model for PainterLongVideo |
| RUNPOD_POD_ID | str\|None | None | No | RunPod pod ID for auto-stop |
| RUNPOD_API_KEY | str\|None | None | No | RunPod API key for auto-stop |

---

## 14. Service Dependency Map

```
wanly-gpu-daemon
  |
  |--> wanly-api (QUEUE_URL)
  |      GET  /segments/next         (claim work)
  |      PATCH /segments/{id}        (status updates)
  |      POST  /segments/{id}/upload (video + last frame)
  |      GET   /files?path=s3://...  (S3 proxy downloads)
  |
  |--> wanly-gpu-registry (REGISTRY_URL)
  |      POST   /workers             (register)
  |      POST   /workers/{id}/heartbeat
  |      PATCH  /workers/{id}/status
  |      DELETE /workers/{id}        (deregister)
  |
  |--> ComfyUI (COMFYUI_URL, local)
  |      GET   /system_stats
  |      GET   /queue
  |      POST  /queue (clear)
  |      POST  /upload/image
  |      POST  /prompt
  |      GET   /history/{prompt_id}
  |      GET   /view (download output)
  |      GET   /object_info
  |      WS    /ws?clientId=
  |
  |--> RunPod API (optional, on drain)
  |      POST https://api.runpod.io/graphql (podStop mutation)
  |
  |--> ffmpeg (subprocess, local)
         Extract last frame from video
```

---

## 15. Known Technical Debt

### CRITICAL: No Test Suite
Zero test files exist. No unit tests, integration tests, or mocking of any kind.

### CRITICAL: No Dependency Pinning
`requirements.txt` has no version pins. Any `pip install -r requirements.txt` could pull breaking changes.

### CRITICAL: No Packaging Metadata
No `pyproject.toml`, `setup.py`, or `setup.cfg`. Cannot be installed as a package.

### HIGH: No TODO/FIXME Markers in Code
While the codebase has no TODO comments, the project MEMORY.md documents untested features:
- Faceswap end-to-end (ReActor and FaceFusion paths untested)
- RIFE verification (needs visual confirmation)
- LoRAs end-to-end (built but untested with real files)

### MEDIUM: Missing Type Annotations
18 out of 64 functions lack return type annotations, including key functions:
- `main.py`: `register_with_retry()`, `heartbeat_loop()`, `job_poll_loop()`, `run()`, `kill_stale_daemons()`
- `registry_client.py`: `register()`, `update_status()`, `deregister()`, `close()`
- `queue_client.py`: `__init__()`, `close()`

### MEDIUM: Missing Docstrings
18 out of 64 functions lack docstrings (largely overlapping with missing type annotations).

### MEDIUM: Hardcoded Node IDs
Workflow node IDs (84, 85, 86, 87, etc.) are magic numbers spread across `workflow_builder.py` and `comfyui_client.py` (node 186 lookup in `find_video_output`).

### MEDIUM: Hardcoded Negative Prompt
Node 89's negative prompt is hardcoded Chinese text in the base workflow template. Not configurable per-segment.

### LOW: env_prefix Mismatch
Settings class uses no `env_prefix`, while CLAUDE.md documents the convention `WANLY_DAEMON_` prefix. Current env vars are bare (e.g., `REGISTRY_URL` not `WANLY_DAEMON_REGISTRY_URL`).

### LOW: LoRA Download via Full In-Memory Bytes
`QueueClient.download_file()` loads entire files into memory. LoRA files can be hundreds of MB. No streaming support.

### LOW: Single Unicode Emoji in Source
`node_checker.py` line 125 uses a Unicode cross mark character in a log message.
