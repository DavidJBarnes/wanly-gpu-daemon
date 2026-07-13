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

    # VACE continuation (Fun-VACE modules on the Wan2.2 T2V-A14B base, via WanVideoWrapper).
    # Validated defaults from the 3090 bench (Lightning 4-step / cfg 1). The T2V base differs
    # from the I2V unet_* models above; WanVideoTextEncodeCached needs the bf16 .pth T5.
    vace_t2v_high_model: str = "wan2.2_t2v_high_noise_14B_fp16.safetensors"
    vace_t2v_low_model: str = "wan2.2_t2v_low_noise_14B_fp16.safetensors"
    vace_module_high: str = "Wan2_2_Fun_VACE_module_A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors"
    vace_module_low: str = "Wan2_2_Fun_VACE_module_A14B_LOW_fp8_e4m3fn_scaled_KJ.safetensors"
    vace_lightning_high: str = "Wan2.2-Lightning_T2V-A14B-4steps-lora_HIGH_fp16.safetensors"
    vace_lightning_low: str = "Wan2.2-Lightning_T2V-A14B-4steps-lora_LOW_fp16.safetensors"
    vace_t5_model: str = "models_t5_umt5-xxl-enc-bf16.pth"
    vace_steps: int = 6
    vace_cfg: float = 1.0
    vace_boundary: int = 3  # high expert runs [0, boundary], low runs [boundary, end]
    vace_shift: float = 8.0
    vace_blocks_to_swap: int = 25
    vace_lightning: bool = True  # use the 4-step distill LoRAs (fast path)

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
    # in the displaced mesh. Aesthetic knob — tune by eye on the Quest 3.
    depth_scale_m: float = 0.12

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
