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
        "repo": "https://github.com/Gourieff/ComfyUI-ReActor",
        "nodes": ["ReActorFaceSwapOpt", "ReActorOptions"],
        "alt_dirs": ["comfyui-reactor", "ComfyUI-ReActor", "ComfyUI-ReActor-NSFW"],
    },
    "ComfyUI-PainterLongVideo": {
        "repo": "https://github.com/princepainter/ComfyUI-PainterLongVideo",
        "nodes": ["PainterLongVideo"],
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
    # GIT_TERMINAL_PROMPT=0 prevents git from hanging on credential prompts
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    proc = await asyncio.create_subprocess_exec(
        "git", "clone", "--depth", "1", repo_url, str(target_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    stdout, _ = await proc.communicate()
    rc = proc.returncode
    output = stdout.decode(errors="replace").strip()
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


def _package_dir_present(custom_nodes_dir: Path, pkg_name: str, pkg_info: dict) -> bool:
    """Return True if the package directory exists under its primary or any alternate name."""
    if (custom_nodes_dir / pkg_name).is_dir():
        return True
    return any((custom_nodes_dir / alt).is_dir() for alt in pkg_info.get("alt_dirs", []))


async def check_and_install_nodes(comfyui_client) -> bool:
    """Check for required custom nodes and install any that are missing.

    Args:
        comfyui_client: ComfyUIClient instance (used to query /object_info if ComfyUI is running).

    Returns:
        True if all required node types are verified available (or COMFYUI_PATH is unset, or
        ComfyUI is not yet running so verification is deferred to its next start).
        False — caller should abort startup — if a package failed to clone/install, if nodes
        were just installed and ComfyUI must restart to load them, or if a package is present
        on disk but failed to import (its node types are missing from /object_info).
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
    install_failed: list[str] = []
    for pkg_name, pkg_info in CUSTOM_NODE_PACKAGES.items():
        if _package_dir_present(custom_nodes_dir, pkg_name, pkg_info):
            logger.debug("Node %s: installed", pkg_name)
            continue

        logger.warning("✗ %s — not found, installing...", pkg_name)
        success = await _git_clone(pkg_info["repo"], custom_nodes_dir / pkg_name)
        if not success:
            logger.error("Failed to install %s — workflows using %s will fail", pkg_name, pkg_info["nodes"])
            install_failed.append(pkg_name)
            continue

        await _pip_install_requirements(custom_nodes_dir / pkg_name)
        installed_any = True

    # Phase 2: If ComfyUI is running, verify nodes are actually loaded via /object_info
    comfyui_running = await comfyui_client.check_health()
    if not comfyui_running:
        if install_failed:
            logger.error(
                "Could not install required custom node package(s): %s. These must be present "
                "before the daemon can serve jobs.", install_failed,
            )
            return False
        if installed_any:
            logger.info("Custom nodes installed. ComfyUI is not running — nodes will load on next start.")
        return True

    if installed_any:
        logger.warning(
            "New custom nodes were installed. ComfyUI must be restarted to load them. "
            "Exiting so the daemon relaunches against a freshly-started ComfyUI."
        )
        return False

    if install_failed:
        logger.error(
            "Could not install required custom node package(s): %s. Refusing to start.",
            install_failed,
        )
        return False

    # Verify via /object_info that every on-disk package actually registered its node types.
    # A package whose directory exists but whose nodes are absent from /object_info failed to
    # import at ComfyUI startup (bad dependency, version skew, syntax error, etc.). If we let
    # the daemon proceed it would register as a healthy worker and then fail EVERY segment with
    # a 400 "missing_node_type" — a silent, money-burning failure. Treat it as fatal instead.
    try:
        available_nodes = await _get_available_nodes(comfyui_client)
    except Exception as e:
        logger.warning("Could not query /object_info: %s — skipping node verification", e)
        return True

    failed_imports: dict[str, list[str]] = {}
    for pkg_name, pkg_info in CUSTOM_NODE_PACKAGES.items():
        missing = [n for n in pkg_info["nodes"] if n not in available_nodes]
        if missing:
            failed_imports[pkg_name] = missing

    if failed_imports:
        for pkg_name, missing in failed_imports.items():
            logger.error(
                "Custom node package '%s' is installed on disk but its node type(s) %s are NOT "
                "registered in ComfyUI — the package failed to import at ComfyUI startup. Check the "
                "ComfyUI log (e.g. /workspace/logs/comfyui.log) for an 'IMPORT FAILED: %s' traceback.",
                pkg_name, missing, pkg_name,
            )
        total = sum(len(m) for m in failed_imports.values())
        logger.error(
            "Refusing to start: %d required node type(s) across %d package(s) failed to register. "
            "The daemon would otherwise accept jobs and fail every segment with a 400 "
            "missing_node_type error.",
            total, len(failed_imports),
        )
        return False

    logger.info("All required custom nodes verified via /object_info")
    return True


async def _get_available_nodes(comfyui_client) -> set[str]:
    """Query ComfyUI /object_info and return set of available node class_types."""
    resp = await comfyui_client.http.get("/object_info")
    resp.raise_for_status()
    data = resp.json()
    return set(data.keys())
