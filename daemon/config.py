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

    # Model filenames (vary per GPU worker — override in .env)
    clip_model: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_model: str = "wan_2.1_vae.safetensors"
    unet_high_model: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_model: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    # UNETLoader weight_dtype (nodes 95 high / 96 low). Clarity vs. speed/VRAM lever.
    #   "default"          — load at native precision (sharpest, most VRAM)
    #   "fp8_e4m3fn"       — fp8 storage, standard matmul (clean; recommended for baked-fp8 DaSiWa)
    #   "fp8_e4m3fn_fast"  — fp8 + reduced-precision fast GEMM (fuzziest; Ada/Hopper only — wasted on the 3090)
    unet_weight_dtype: str = "fp8_e4m3fn"
    lightx2v_lora_high: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    lightx2v_lora_low: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
    lightx2v_strength_high: float = 2.0  # Strength for high noise lightx2v LoRA (community range: 1.0–5.6)
    lightx2v_strength_low: float = 1.0  # Strength for low noise lightx2v LoRA (community range: 1.0–2.0)
    cfg_high: float = 1.0  # CFG for high noise KSampler (node 86)
    cfg_low: float = 1.0  # CFG for low noise KSampler (node 85)
    clip_vision_model: str = "clip_vision_h.safetensors"

    # Sampler schedule (previously hardcoded in the workflow template)
    generation_fps: int = 16  # Wan 2.2 14B is trained at 16fps; was hardcoded 15. CORRECTNESS FIX.
    steps_total: int = 4  # total KSamplerAdvanced schedule length (shared by both passes)
    high_noise_steps: int = 2  # boundary: high-noise expert runs steps [0, high_noise_steps); low-noise runs [high_noise_steps, steps_total)
    shift_high: float = 5.0  # ModelSamplingSD3 shift for the high-noise expert (node 104)
    shift_low: float = 5.0  # ModelSamplingSD3 shift for the low-noise expert (node 103)

    # --- Realism profile (de-distilled high-noise) — single A/B toggle against the distilled baseline ---
    # When True, the high-noise expert runs de-distilled: its lightx2v LoRA is dropped (the builder
    # rewires the graph) and it runs real steps at real CFG, for stronger motion and more natural
    # facial expression. Low-noise pass stays distilled. The bundle below moves together because
    # steps_total is shared across both passes, so a single switch is the clean way to flip it.
    # Distilled baseline: strength_high 2.0, cfg_high 1.0, steps_total 4, high_noise_steps 2, shift_high 5.0
    # Realism bundle:     strength_high 0.0, cfg_high 3.5, steps_total 8, high_noise_steps 4, shift_high 7.0
    # Per-segment overrides (from the job) still win over the profile.
    high_noise_realism: bool = False

    # Faceswap / output quality knobs
    # ReActor CodeFormer blend: lower lets more of the raw, expressive generated face show
    # through instead of the smoothed CodeFormer prior (was hardcoded 1.0 → generic faces).
    faceswap_restore_visibility: float = 0.5
    # RIFE VFI fast mode: False = cleaner interpolated frames (was hardcoded True).
    rife_fast_mode: bool = False

    # PainterLongVideo motion parameters (identity anchoring)
    painter_motion_amplitude: float = 1.3  # Range: 1.0-2.0, higher = more motion
    painter_motion_frames: int = 5  # Range: 1-20, controls motion cycle length

    # Motion matching (optical flow based)
    motion_matching_enabled: bool = True  # Enable automatic motion amplitude matching
    motion_amplitude_default: float = 1.3  # Default motion_amplitude for segment 0
    motion_amplitude_min: float = 1.0  # Minimum motion_amplitude (no motion boost)
    motion_amplitude_max: float = 2.0  # Maximum motion_amplitude (extreme motion)

    # sd-scripts LoRA training monitor
    sd_scripts_path: str = "~/projects/sd-scripts"

    # RunPod auto-stop (set by RunPod environment + user config)
    runpod_pod_id: str | None = None
    runpod_api_key: str | None = None

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()


# Generation mode presets — each is a FULL profile (model + sampler knobs) selected
# per job via GenerationMode (wanly-api) and resolved in workflow_builder by segment.mode.
# identity/expression are the proven walking-5/walking-7 recipes. dasiwa values are
# best-known and want one confirming test before they're locked.
_BASE_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
_BASE_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"

MODE_PRESETS: dict[str, dict] = {
    # Wan22 Base (Character Identity) — fully resident, ~16m, 10/10 identity
    "identity": {
        "unet_high_model": _BASE_HIGH, "unet_low_model": _BASE_LOW,
        "unet_weight_dtype": "fp8_e4m3fn",
        "lightx2v_strength_high": 0.5, "lightx2v_strength_low": 0.5,
        "cfg_high": 1.0, "cfg_low": 1.0,
        "steps_total": 10, "high_noise_steps": 5,
        "shift_high": 5.0, "shift_low": 5.0,
    },
    # Wan22 Base (Identity + Expression) — de-distilled high (lightx2v 0 + real CFG),
    # model offloads, ~21m, natural motion + expression. Needs the cfg-aware estimate.
    "expression": {
        "unet_high_model": _BASE_HIGH, "unet_low_model": _BASE_LOW,
        "unet_weight_dtype": "fp8_e4m3fn",
        "lightx2v_strength_high": 0.0, "lightx2v_strength_low": 0.5,
        "cfg_high": 3.5, "cfg_low": 1.0,
        "steps_total": 20, "high_noise_steps": 12,
        "shift_high": 5.0, "shift_low": 5.0,
    },
    # DaSiWa (Fast) — baked-distilled remix, ~13m, natural motion / weaker identity.
    # Confirmed 2026-07-01 (first preset guess ran clean + looked great).
    "dasiwa": {
        "unet_high_model": "DasiwaWAN22I2V14BLightspeed_snatchkissHighV11.safetensors",
        "unet_low_model": "DasiwaWAN22I2V14BLightspeed_snatchkissLowV11.safetensors",
        "unet_weight_dtype": "fp8_e4m3fn",
        "lightx2v_strength_high": 0.0, "lightx2v_strength_low": 0.0,
        "cfg_high": 1.0, "cfg_low": 1.0,
        "steps_total": 8, "high_noise_steps": 4,
        "shift_high": 5.0, "shift_low": 5.0,
    },
}


def get_mode_preset(mode: str | None) -> dict:
    """Resolve a job's generation mode to its full model+sampler preset.
    Unknown/legacy modes fall back to the safe resident 'identity' profile."""
    return MODE_PRESETS.get(mode or "identity", MODE_PRESETS["identity"])
