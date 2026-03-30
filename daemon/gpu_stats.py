"""Collect GPU stats via nvidia-smi, independent of ComfyUI."""

import logging
import subprocess

logger = logging.getLogger(__name__)


def get_gpu_stats() -> dict | None:
    """Query nvidia-smi for GPU stats. Returns None on failure."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return None

        # Parse first GPU line: "NVIDIA GeForce RTX 3090, 18432, 24576"
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            return None

        gpu_name = parts[0]
        vram_used_mb = int(parts[1])
        vram_total_mb = int(parts[2])

        return {
            "gpu_name": gpu_name,
            "vram_used_mb": vram_used_mb,
            "vram_total_mb": vram_total_mb,
        }
    except Exception as e:
        logger.debug("nvidia-smi failed: %s", e)
        return None
