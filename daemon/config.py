from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    registry_url: str = "http://localhost:8000"
    friendly_name: str = "gpu-worker-1"
    heartbeat_interval: int = 30
    comfyui_url: str = "http://localhost:8188"
    comfyui_api_key: str = ""  # Bearer token for ComfyUI auth (RunPod sets this)
    comfyui_path: str = ""  # Path to ComfyUI installation (for custom node management)
    lora_cache_dir: str = ""  # Override LoRA download dir (e.g. /workspace/models/loras for persistence)
    queue_url: str = "http://localhost:8001"
    poll_interval: int = 5

    # Model filenames (vary per GPU worker — override in .env)
    clip_model: str = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
    vae_model: str = "wan_2.1_vae.safetensors"
    unet_high_model: str = "wan2.2_i2v_high_noise_14B_fp16.safetensors"
    unet_low_model: str = "wan2.2_i2v_low_noise_14B_fp16.safetensors"
    lightx2v_lora_high: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
    lightx2v_lora_low: str = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

    model_config = {"env_file": ".env"}


settings = Settings()
