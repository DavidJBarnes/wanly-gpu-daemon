from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    friendly_name: str = "gpu-worker-1"
    heartbeat_interval: int = 30
    comfyui_url: str = "http://localhost:8188"
    comfyui_api_key: str = ""  # Bearer token for ComfyUI auth (RunPod sets this)
    comfyui_path: str = ""  # Path to ComfyUI installation (for custom node management)
    lora_cache_dir: str = ""  # Override LoRA download dir (e.g. /workspace/models/loras for persistence)
    queue_url: str = "http://localhost:8001"
    queue_api_key: str = ""
    poll_interval: int = 5

    # Which generation engine this worker drives.
    #
    # "wan22" -> ComfyUI directly, via workflow_builder (the original path)
    # "ltx"   -> ltx-engine over HTTP, which owns graph assembly itself
    #
    # LTX 2.3 replaces WAN 2.2 (wanly-gpu-docker#41), but the default stays "wan22" so that
    # merging LTX support cannot change what an existing worker does. start.sh git-pulls this
    # daemon on every boot, so a default flip here would retarget any running worker the moment
    # it restarted. The LTX image sets ENGINE=ltx explicitly instead.
    engine: str = "wan22"

    # ltx-engine's job API, in the same container. NOT ComfyUI on 8191: the engine owns the
    # graph — it uploads keyframes, resolves the recipe, patches the workflow and submits.
    # Driving ComfyUI directly for LTX would put graph assembly back in a caller, which is
    # where every structural bug on that project came from.
    ltx_engine_url: str = "http://localhost:8190"
    # A render is 8-12 minutes and can queue behind another. Sized from the slowest realistic
    # case and then doubled: a timeout that gives up early costs a finished render, and that
    # has happened before (4 of 9 lost to a 90-minute wait loop).
    ltx_timeout_seconds: int = 5400
    ltx_poll_interval: int = 5

    # Model filenames (vary per GPU worker — override in .env)
    clip_model: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_model: str = "wan_2.1_vae.safetensors"
    unet_high_model: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_model: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    lightx2v_lora_high: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    lightx2v_lora_low: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    lightx2v_strength_high: float = 2.0  # Strength for high noise lightx2v LoRA (community range: 1.0–5.6)
    lightx2v_strength_low: float = 1.0  # Strength for low noise lightx2v LoRA (community range: 1.0–2.0)
    cfg_high: float = 1.0  # CFG for high noise KSampler (node 86)
    cfg_low: float = 1.0  # CFG for low noise KSampler (node 85)
    # Sampler schedule (distilled default = 4/2). Per-job override lets you de-distill:
    # pass lightx2v strength 0 (builder drops the Lightning LoRA) + raise cfg + raise steps.
    steps_total: int = 4  # total KSamplerAdvanced schedule length (both passes share it)
    high_noise_steps: int = 2  # boundary: high runs [0, high_noise_steps], low runs [high_noise_steps, steps_total]
    flow_shift: float = 5.0  # ModelSamplingSD3 schedule shift; higher = more high-noise steps = more motion
    clip_vision_model: str = "clip_vision_h.safetensors"

    # AR hologram: Robust Video Matting ONNX model (auto-used when a clip is NOT green-screen).
    # ~15MB; auto-downloaded on first use if missing. Path is relative to the daemon workdir.
    rvm_model_path: str = "models/rvm_mobilenetv3_fp32.onnx"

    # AR hologram "2.5d_depth" flavor: monocular depth model (Depth Anything V2 small, ONNX).
    # ~100MB; auto-downloaded on first use. Runs on GPU when onnxruntime-gpu is present, else CPU.
    depth_model_path: str = "models/depth_anything_v2_vits.onnx"
    depth_model_url: str = (
        "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
    )
    # Relief depth in meters: how far the nearest subject pixels are pushed toward the viewer
    # in the displaced mesh. Fallback only — the per-request value from the claim
    # (hologram_depth_scale_m, set in the console dialog) wins when present.
    depth_scale_m: float = 0.30


    # Identity scoring device. Defaults to CPU, which is what every recorded identity score to
    # date was produced on.
    #
    # Moving it to GPU is issue #106 and is NOT yet justified: the measurement that motivated it
    # (scoring = 23-29% of job time) did not reproduce — a later run scored 289 frames in 9s, and
    # motion analysis, not scoring, dominated that phase. Worse, nobody has yet checked whether
    # CUDA and CPU produce the SAME embeddings. If they differ, every score after the switch sits
    # on a different scale from every score before it, and the recorded history stops being
    # comparable — the same hazard the DET_SIZE comment warns about.
    #
    # So: opt in per worker, never on by default, until #106 measures both the speedup and the
    # embedding agreement.
    identity_scoring_gpu: bool = False

    # How long a drain waits for the in-flight segment before giving up.
    #
    # Was 600s, which is shorter than the work: measured segments run 329s at 480p/3s, ~570s at
    # 480p/5s and ~1780s at 720x1056/5s. A drain during a 720p segment hit the old timeout at
    # the 10 minute mark and abandoned ~20 minutes of finished-but-unreported work. Now generous
    # by default, because the cost of waiting too long is a slightly later shutdown, while the
    # cost of not waiting long enough is a discarded segment.
    drain_wait_seconds: int = 3600

    # The PainterLongVideo and motion-amplitude settings were removed with the swap that read
    # them (see #124). extra="ignore" below means a .env still carrying them is harmless.

    # sd-scripts LoRA training monitor
    sd_scripts_path: str = "~/projects/sd-scripts"

    # RunPod auto-stop (set by RunPod environment + user config)
    runpod_pod_id: str | None = None
    runpod_api_key: str | None = None

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
