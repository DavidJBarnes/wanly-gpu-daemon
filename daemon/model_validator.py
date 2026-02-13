"""Pre-flight model validation — runs at startup before registration.

Checks that every model referenced in settings actually exists in ComfyUI
and is large enough to be a complete download. Blocks startup on failure.
"""

import logging
import os

import yaml

from daemon.config import settings

logger = logging.getLogger(__name__)

# Minimum file sizes to catch partial/corrupt downloads
MIN_SIZES = {
    "clip": 100 * 1024 * 1024,       # 100 MB
    "vae": 100 * 1024 * 1024,        # 100 MB
    "unet": 10 * 1024 * 1024 * 1024, # 10 GB
    "loras": 100 * 1024 * 1024,      # 100 MB
}

# Map config setting names → (ComfyUI loader node, model subfolder, min size key)
MODEL_CHECKS = [
    ("clip_model",          "CLIPLoader",           "clip",  "clip",  MIN_SIZES["clip"]),
    ("vae_model",           "VAELoader",            "vae",   "vae",   MIN_SIZES["vae"]),
    ("unet_high_model",     "UNETLoader",           "diffusion_models", "unet", MIN_SIZES["unet"]),
    ("unet_low_model",      "UNETLoader",           "diffusion_models", "unet", MIN_SIZES["unet"]),
    ("lightx2v_lora_high",  "LoraLoaderModelOnly",  "loras", "loras", MIN_SIZES["loras"]),
    ("lightx2v_lora_low",   "LoraLoaderModelOnly",  "loras", "loras", MIN_SIZES["loras"]),
]

PARTIAL_EXTENSIONS = {".aria2", ".tmp", ".part"}


def _human_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _get_model_search_paths(comfyui_path: str, subfolder: str) -> list[str]:
    """Return all directories where ComfyUI might store models of the given type."""
    paths = [os.path.join(comfyui_path, "models", subfolder)]

    # Parse extra_model_paths.yaml for additional search paths (used by RunPod)
    extra_paths_file = os.path.join(comfyui_path, "extra_model_paths.yaml")
    if os.path.isfile(extra_paths_file):
        try:
            with open(extra_paths_file) as f:
                data = yaml.safe_load(f) or {}
            for _section_name, section in data.items():
                if not isinstance(section, dict):
                    continue
                # Resolve base_path for relative paths in the section
                base_path = section.get("base_path", "")
                model_dir = section.get(subfolder)
                if model_dir:
                    if base_path and not os.path.isabs(model_dir):
                        model_dir = os.path.join(base_path, model_dir)
                    paths.append(model_dir)
        except Exception as e:
            logger.warning("Failed to parse %s: %s", extra_paths_file, e)

    return paths


def _find_model_file(comfyui_path: str, subfolder: str, filename: str) -> str | None:
    """Search all model paths for a file, return full path or None."""
    for search_dir in _get_model_search_paths(comfyui_path, subfolder):
        candidate = os.path.join(search_dir, filename)
        if os.path.isfile(candidate):
            return candidate
    return None


async def validate_models(comfyui_client) -> bool:
    """Validate that all required models exist and meet minimum size thresholds.

    Returns True if all checks pass, False if any critical model is missing or undersized.
    """
    comfyui_path = settings.comfyui_path
    if not comfyui_path:
        logger.warning("COMFYUI_PATH not set — skipping model validation")
        return True

    # Try to get available models from ComfyUI's /object_info
    available_models: dict[str, set[str]] = {}
    try:
        resp = await comfyui_client.http.get("/object_info", timeout=30)
        if resp.status_code == 200:
            object_info = resp.json()
            for _setting, loader_node, _subfolder, _size_key, _min_size in MODEL_CHECKS:
                node_info = object_info.get(loader_node, {})
                inputs = node_info.get("input", {}).get("required", {})
                # Model filename is typically the first input parameter
                for _param_name, param_info in inputs.items():
                    if isinstance(param_info, list) and isinstance(param_info[0], list):
                        available_models[loader_node] = set(param_info[0])
                        break
    except Exception as e:
        logger.warning("Could not query /object_info for model lists: %s", e)

    all_ok = True
    logger.info("--- Model validation ---")

    for setting_name, loader_node, subfolder, size_key, min_size in MODEL_CHECKS:
        filename = getattr(settings, setting_name)
        if not filename:
            continue

        # Check if ComfyUI knows about it
        known_models = available_models.get(loader_node)
        if known_models and filename not in known_models:
            logger.error("  MISSING %s=%s — not found in ComfyUI %s model list", setting_name, filename, loader_node)
            all_ok = False
            continue

        # Find on disk and check size
        file_path = _find_model_file(comfyui_path, subfolder, filename)
        if not file_path:
            logger.error("  MISSING %s=%s — file not found on disk", setting_name, filename)
            all_ok = False
            continue

        file_size = os.path.getsize(file_path)
        if file_size < min_size:
            logger.error(
                "  UNDERSIZED %s=%s — %s (minimum %s) at %s",
                setting_name, filename, _human_size(file_size), _human_size(min_size), file_path,
            )
            all_ok = False
            continue

        logger.info("  OK %s=%s (%s) %s", setting_name, filename, _human_size(file_size), file_path)

    if all_ok:
        logger.info("--- All models validated ---")
    else:
        logger.error("--- Model validation FAILED — cannot start ---")

    return all_ok


def cleanup_partial_downloads(comfyui_path: str) -> int:
    """Remove .aria2, .tmp, .part files from ComfyUI model directories.

    Returns count of files deleted.
    """
    if not comfyui_path:
        return 0

    models_dir = os.path.join(comfyui_path, "models")
    if not os.path.isdir(models_dir):
        return 0

    count = 0
    for root, _dirs, files in os.walk(models_dir):
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in PARTIAL_EXTENSIONS:
                full_path = os.path.join(root, fname)
                try:
                    os.remove(full_path)
                    logger.info("Cleaned up partial download: %s", full_path)
                    count += 1
                except OSError as e:
                    logger.warning("Failed to remove %s: %s", full_path, e)

    if count:
        logger.info("Cleaned up %d partial download file(s)", count)
    return count
