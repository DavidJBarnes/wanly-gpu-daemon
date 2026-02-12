"""Download LoRA files from S3 (via API proxy) to ComfyUI's local loras directory.

Only downloads files that are missing locally. Files are cached forever.
"""

import logging
import os

from daemon.config import settings
from daemon.queue_client import QueueClient
from daemon.schemas import LoraItem

logger = logging.getLogger(__name__)


def _loras_dir() -> str:
    """Return the ComfyUI loras directory path (wan2.2 subdirectory)."""
    return os.path.join(settings.comfyui_path, "models", "loras", "wan2.2")


async def ensure_loras_available(
    loras: list[LoraItem], queue: QueueClient
) -> None:
    """Ensure all LoRA .safetensors files are present locally.

    Downloads missing files from S3 via the API proxy.
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
            if os.path.exists(local_path):
                logger.debug("LoRA %s already cached: %s", label, filename)
                continue

            logger.info("Downloading LoRA %s: %s", label, filename)
            data = await queue.download_file(s3_uri)
            with open(local_path, "wb") as f:
                f.write(data)
            logger.info(
                "Saved LoRA %s: %s (%d bytes)", label, filename, len(data)
            )
