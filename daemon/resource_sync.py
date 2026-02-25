"""Download required model resources from S3 on daemon startup.

Checks a hardcoded manifest of resources that ComfyUI custom nodes need
(e.g. RIFE model weights). Skips files already present and large enough.
Uses atomic writes (.tmp -> rename) to prevent corrupt partial files.
"""

import logging
import os

from daemon.config import settings
from daemon.queue_client import QueueClient

logger = logging.getLogger(__name__)

RESOURCES = [
    {
        "s3_uri": "s3://wanly-resources/comfyui/models/rife/rife49.pth",
        "local_path": "custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth",
        "min_size_mb": 10,
    },
]


async def sync_resources(queue: QueueClient) -> bool:
    """Download any missing resources from S3 via the API proxy.

    Returns True if all resources are available, False on any failure.
    """
    if not settings.comfyui_path:
        logger.warning("comfyui_path not set — skipping resource sync")
        return True

    all_ok = True
    for res in RESOURCES:
        local_path = os.path.join(settings.comfyui_path, res["local_path"])
        min_bytes = res["min_size_mb"] * 1024 * 1024
        filename = os.path.basename(local_path)

        # Check if existing file is large enough
        if os.path.exists(local_path):
            file_size = os.path.getsize(local_path)
            if file_size >= min_bytes:
                mb = file_size / (1024 * 1024)
                logger.info("Resource %s: cached (%.1f MB)", filename, mb)
                continue
            logger.warning(
                "Resource %s: exists but only %.1f MB (< %d MB) — re-downloading",
                filename,
                file_size / (1024 * 1024),
                res["min_size_mb"],
            )
            os.remove(local_path)

        # Download
        logger.info("Resource %s: downloading from %s ...", filename, res["s3_uri"])
        try:
            data = await queue.download_file(res["s3_uri"])
        except Exception as e:
            logger.error("Resource %s: download failed: %s", filename, e)
            all_ok = False
            continue

        if len(data) < min_bytes:
            logger.error(
                "Resource %s: download too small (%.1f MB, expected >= %d MB)",
                filename,
                len(data) / (1024 * 1024),
                res["min_size_mb"],
            )
            all_ok = False
            continue

        # Atomic write
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = local_path + ".tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
        os.rename(tmp_path, local_path)
        logger.info("Resource %s: saved (%.1f MB)", filename, len(data) / (1024 * 1024))

    return all_ok
