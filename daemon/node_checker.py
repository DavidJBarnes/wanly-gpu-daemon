"""Check and install required ComfyUI custom nodes on daemon startup.

Queries ComfyUI's /object_info endpoint to discover available node types,
compares against the nodes our workflows require, and git-clones any missing
custom node packages into ComfyUI's custom_nodes directory.
"""

import asyncio
import logging
import os
from pathlib import Path

from daemon.config import settings

logger = logging.getLogger(__name__)

# Map of custom node package → (git repo URL, list of node class_types it provides).
# Native ComfyUI nodes (CLIPLoader, KSamplerAdvanced, VAEDecode, etc.) are NOT listed
# because they ship with ComfyUI itself.
CUSTOM_NODE_PACKAGES: dict[str, dict] = {
    "ComfyUI-Frame-Interpolation": {
        "repo": "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation",
        "nodes": ["RIFE VFI"],
    },
    "ComfyUI-VideoHelperSuite": {
        "repo": "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        "nodes": ["VHS_VideoCombine"],
    },
    "comfyui-reactor-node": {
        "repo": "https://github.com/Gourieff/comfyui-reactor-node",
        "nodes": ["ReActorFaceSwapOpt", "ReActorOptions"],
        "alt_dirs": ["comfyui-reactor", "ComfyUI-ReActor-NSFW"],
    },
}


async def _run(cmd: list[str], cwd: str | None = None) -> tuple[int, str]:
    """Run a subprocess and return (returncode, combined output)."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=cwd,
    )
    stdout, _ = await proc.communicate()
    return proc.returncode, stdout.decode(errors="replace").strip()


async def _git_clone(repo_url: str, target_dir: Path) -> bool:
    """Clone a git repo into target_dir. Returns True on success."""
    logger.info("Cloning %s → %s", repo_url, target_dir)
    rc, output = await _run(["git", "clone", "--depth", "1", repo_url, str(target_dir)])
    if rc != 0:
        logger.error("git clone failed (rc=%d): %s", rc, output)
        return False
    logger.info("Cloned %s successfully", repo_url)
    return True


async def _pip_install_requirements(node_dir: Path) -> None:
    """Install pip requirements if requirements.txt exists in the node directory."""
    req_file = node_dir / "requirements.txt"
    if not req_file.exists():
        return
    logger.info("Installing pip requirements for %s", node_dir.name)
    rc, output = await _run(["pip", "install", "-r", str(req_file)], cwd=str(node_dir))
    if rc != 0:
        logger.warning("pip install failed for %s (rc=%d): %s", node_dir.name, rc, output)
    else:
        logger.info("pip install completed for %s", node_dir.name)


async def check_and_install_nodes(comfyui_client) -> bool:
    """Check for required custom nodes and install any that are missing.

    Args:
        comfyui_client: ComfyUIClient instance (used to query /object_info if ComfyUI is running).

    Returns:
        True if all required nodes are available (or were just installed and need a restart).
        False if installation failed or ComfyUI path is not configured.
    """
    comfyui_path = settings.comfyui_path
    if not comfyui_path:
        logger.warning("COMFYUI_PATH not set — skipping custom node check")
        return True

    custom_nodes_dir = Path(comfyui_path) / "custom_nodes"
    if not custom_nodes_dir.is_dir():
        logger.error("custom_nodes directory not found at %s", custom_nodes_dir)
        return False

    # Phase 1: Check which node directories are missing and install them
    installed_any = False
    for pkg_name, pkg_info in CUSTOM_NODE_PACKAGES.items():
        pkg_dir = custom_nodes_dir / pkg_name
        if pkg_dir.is_dir():
            logger.info("✓ %s — already installed", pkg_name)
            continue

        # Check alternate directory names (e.g. comfyui-reactor vs comfyui-reactor-node)
        alt_found = False
        for alt in pkg_info.get("alt_dirs", []):
            if (custom_nodes_dir / alt).is_dir():
                logger.info("✓ %s — found as %s", pkg_name, alt)
                alt_found = True
                break
        if alt_found:
            continue

        logger.warning("✗ %s — not found, installing...", pkg_name)
        success = await _git_clone(pkg_info["repo"], pkg_dir)
        if not success:
            logger.error("Failed to install %s — workflows using %s will fail", pkg_name, pkg_info["nodes"])
            continue

        await _pip_install_requirements(pkg_dir)
        installed_any = True

    # Phase 2: If ComfyUI is running, verify nodes are actually loaded via /object_info
    comfyui_running = await comfyui_client.check_health()
    if not comfyui_running:
        if installed_any:
            logger.info("Custom nodes installed. ComfyUI is not running — nodes will load on next start.")
        return True

    if installed_any:
        logger.warning(
            "New custom nodes were installed. ComfyUI must be restarted to load them. "
            "Please restart ComfyUI and re-run the daemon."
        )
        return False

    # Verify via /object_info that all required node types are available
    try:
        available_nodes = await _get_available_nodes(comfyui_client)
    except Exception as e:
        logger.warning("Could not query /object_info: %s — skipping node verification", e)
        return True

    all_required = []
    for pkg_info in CUSTOM_NODE_PACKAGES.values():
        all_required.extend(pkg_info["nodes"])

    missing = [n for n in all_required if n not in available_nodes]
    if missing:
        logger.warning(
            "The following node types are not available in ComfyUI despite packages being installed: %s. "
            "Try restarting ComfyUI.",
            missing,
        )
        return True  # Don't block startup — nodes might work after a manual restart

    logger.info("All required custom nodes verified via /object_info")
    return True


async def _get_available_nodes(comfyui_client) -> set[str]:
    """Query ComfyUI /object_info and return set of available node class_types."""
    resp = await comfyui_client.http.get("/object_info")
    resp.raise_for_status()
    data = resp.json()
    return set(data.keys())
