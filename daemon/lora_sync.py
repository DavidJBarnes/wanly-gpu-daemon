"""Download LoRA files from S3 (via API proxy) to ComfyUI's local loras directory.

Only downloads files that are missing locally. Files are cached forever.
Uses atomic writes (.tmp → rename) to prevent corrupt partial files.
"""

import logging
import os

from daemon.config import settings
from daemon.queue_client import QueueClient
from daemon.schemas import LoraItem

logger = logging.getLogger(__name__)

MIN_LORA_SIZE = 10 * 1024 * 1024  # 10 MB — anything smaller is likely corrupt
PARTIAL_EXTENSIONS = (".aria2", ".tmp", ".part")


def _loras_dir() -> str:
    """Return the LoRA download directory path.

    Uses ``lora_cache_dir`` when set (e.g. a persistent volume on RunPod),
    otherwise falls back to ``comfyui_path/models/loras``.
    """
    if settings.lora_cache_dir:
        return settings.lora_cache_dir
    return os.path.join(settings.comfyui_path, "models", "loras")


def _cleanup_partials(lora_dir: str, filename: str) -> None:
    """Remove any partial download artifacts for a given filename."""
    for ext in PARTIAL_EXTENSIONS:
        partial = os.path.join(lora_dir, filename + ext)
        if os.path.exists(partial):
            os.remove(partial)
            logger.info("       Removed partial: %s", partial)


async def ensure_loras_available(
    loras: list[LoraItem], queue: QueueClient
) -> None:
    """Ensure all LoRA .safetensors files are present locally.

    Downloads missing files from S3 via the API proxy.
    Uses atomic writes: download to .tmp then rename to final path.
    Re-downloads files that are suspiciously small (< 10 MB).
    """
    lora_dir = _loras_dir()
    os.makedirs(lora_dir, exist_ok=True)

    for item in loras:
        for label, filename, s3_uri in [
            ("high", item.high_file, item.high_s3_uri),
            ("low", item.low_file, item.low_s3_uri),
        ]:
            if not filename or not s3_uri:
                continue

            local_path = os.path.join(lora_dir, filename)

            # Clean up any partial download artifacts
            _cleanup_partials(lora_dir, filename)

            # Check if existing file is too small (likely corrupt/partial)
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path)
                if file_size < MIN_LORA_SIZE:
                    logger.warning(
                        "       %s: %s exists but only %.1f MB (< 10 MB) — re-downloading",
                        label, filename, file_size / (1024 * 1024),
                    )
                    os.remove(local_path)
                else:
                    mb = file_size / (1024 * 1024)
                    logger.info("       %s: %s (cached, %.1f MB)", label, filename, mb)
                    continue

            logger.info("       %s: downloading %s...", label, filename)
            data = await queue.download_file(s3_uri)

            mb = len(data) / (1024 * 1024)
            if len(data) < MIN_LORA_SIZE:
                raise RuntimeError(
                    f"LoRA {filename} download too small: {mb:.1f} MB (expected >= 10 MB)"
                )

            # Atomic write: write to .tmp then rename
            tmp_path = local_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(data)
            os.rename(tmp_path, local_path)
            logger.info("       %s: %s saved (%.1f MB)", label, filename, mb)
