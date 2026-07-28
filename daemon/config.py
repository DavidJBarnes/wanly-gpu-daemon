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

    # Which model families this worker was provisioned with. Mirrors MODEL_PROFILE in
    # wanly-runpod's download_models.sh, and drives startup model validation:
    #   "full" — Wan 2.2 i2v + VACE + Lynx (the default)
    #   "lynx" — Wan2.1 T2V + Lynx adapters only (a lean pod; no 2.2 i2v weights)
    # A mismatch here fails startup on a perfectly healthy worker, so the two must agree.
    model_profile: str = "full"

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

    # === Lynx identity-preserving generation (ByteDance Lynx on Wan2.1 T2V-14B) ===
    # Lynx re-asserts identity at EVERY denoising step via two adapters, unlike a
    # character LoRA which bakes it into the weights:
    #   ip  (ID-adapter)  ArcFace embedding -> Perceiver resampler -> identity tokens
    #   ref (Ref-adapter) dense VAE features of the reference face, cross-attended in DiT blocks
    # This is a T2V base conditioned on a subject image — NOT first-frame i2v, so the
    # subject image never appears as frame 0.
    lynx_t2v_model: str = "wan2.1_t2v_14B_fp16.safetensors"
    # ip layers and resampler are a MATCHED PAIR (the resampler's proj_out dim must match
    # the ip layers). Default is Kijai's shipped combination — lite ip + full ref — because
    # his own workflow note reports the full ip adapter as "very weak". full ip + full ref
    # is the A/B arm; swap BOTH lynx_ip_layers and lynx_resampler together.
    lynx_ip_layers: str = "Wan2_1-T2V-14B-Lynx_lite_ip_layers_fp16.safetensors"
    lynx_resampler: str = "lynx_lite_resampler_fp32.safetensors"
    lynx_ref_layers: str = "Wan2_1-T2V-14B-Lynx_full_ref_layers_fp16.safetensors"
    lynx_resampler_precision: str = "fp16"
    # Loader precision. Everything stays fp16: no fp8 anywhere in the graph.
    #
    # An fp8 base does NOT work with Lynx on this stack. The adapters are plain
    # nn.Linear (lynx/modules.py: to_k_ip, to_v_ip, to_k_ref, to_v_ref), so they are not
    # wrapped by the wrapper's fp8-aware linear. With an fp8 base their weights get cast
    # to fp8 while activations stay fp16, and WanVideoSampler dies with
    # "self and mat2 must have the same dtype, but got Half and Float8_e4m3fn".
    # Both fp8 settings fail identically (quantization=fp8_e4m3fn_scaled AND =disabled),
    # so quantization is not the lever — the base checkpoint's dtype is.
    #
    # Kijai's reference workflow gets away with an fp8 base only via base_precision
    # "fp16_fast", which needs torch >= 2.7.0.dev2025-02-26 (our image is older).
    #
    # Cost of fp16: a 28GB checkpoint instead of 14.5GB. Block swap carries it, exactly
    # as the VACE path already runs 28GB fp16 models on a 24GB card.
    # Valid base_precision: fp32 | bf16 | fp16 | fp16_fast.
    lynx_base_precision: str = "fp16"
    lynx_quantization: str = "disabled"
    # Wan2.1 T2V cfg-step-distill LoRA (the i2v lightx2v_* above are a different family and
    # do NOT apply to the 2.1 T2V base). Strength 0 drops it -> de-distilled path.
    lynx_distill_lora: str = "lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors"
    lynx_distill_strength: float = 1.0
    lynx_t5_model: str = "models_t5_umt5-xxl-enc-bf16.pth"
    # Adapter strengths. Kijai's reference workflow ships 0.7/0.6 rather than the 1.0/1.0
    # implied by the model card; treated as the calibration starting point, not gospel.
    lynx_ip_scale: float = 0.7
    lynx_ref_scale: float = 0.6
    # Only takes effect when the main cfg is also > 1.0 (it triggers an extra pass). With the
    # distill LoRA on, cfg is 1.0, so this is inert until you de-distill.
    lynx_lynx_cfg_scale: float = 2.0
    # Denoise window over which the ref adapter is applied (fraction of total steps).
    lynx_start_percent: float = 0.0
    lynx_end_percent: float = 1.0
    # Comma-separated DiT block indices/ranges for the ref feature, e.g. "0-20, 25, 35-39".
    # Empty = all blocks.
    lynx_ref_blocks_to_use: str = ""
    # Reference-extraction pass prompt. Hardcoded in ByteDance's original implementation;
    # Kijai's node requires it whenever ref_image is supplied.
    lynx_ref_prompt: str = "image of a face"
    lynx_steps: int = 6
    lynx_cfg: float = 1.0
    lynx_shift: float = 8.0
    lynx_scheduler: str = "lcm"
    lynx_blocks_to_swap: int = 35
    # Identity QA: sample N frames from the render and cosine-compare their InsightFace
    # embeddings against the subject crop. Measurement only — nothing gates on the score.
    lynx_identity_sample_frames: int = 5

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
