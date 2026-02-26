import asyncio
import logging
import signal
import socket
import sys

from daemon.comfyui_client import ComfyUIClient
from daemon.config import settings
from daemon.executor import execute_segment
from daemon.model_validator import cleanup_partial_downloads, validate_models
from daemon.node_checker import check_and_install_nodes
from daemon.queue_client import QueueClient
from daemon.registry_client import RegistryClient
from daemon.resource_sync import sync_resources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress noisy httpx request-level logging (every /system_stats ping)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def get_ip_address() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except Exception:
            return "127.0.0.1"


def _log_system_info(system_info: dict | None) -> None:
    """Log GPU/VRAM/RAM info from ComfyUI system stats."""
    if not system_info:
        logger.warning("Could not retrieve ComfyUI system info")
        return

    devices = system_info.get("devices", [])
    for dev in devices:
        name = dev.get("name", "unknown")
        vram_total = dev.get("vram_total", 0)
        vram_free = dev.get("vram_free", 0)
        vram_total_gb = vram_total / (1024**3) if vram_total else 0
        vram_free_gb = vram_free / (1024**3) if vram_free else 0
        logger.info("GPU: %s — VRAM: %.1f GB total, %.1f GB free", name, vram_total_gb, vram_free_gb)

    system = system_info.get("system", {})
    ram = system.get("ram", {})
    ram_total = ram.get("total", 0)
    ram_free = ram.get("free", 0)
    if ram_total:
        logger.info("RAM: %.1f GB total, %.1f GB free", ram_total / (1024**3), ram_free / (1024**3))


async def register_with_retry(client, *, friendly_name, hostname, ip_address, comfyui_running, shutdown_event):
    """Attempt to register with the registry, retrying every 10s until success or shutdown."""
    attempt = 0
    while not shutdown_event.is_set():
        attempt += 1
        try:
            worker_id = await client.register(
                friendly_name=friendly_name,
                hostname=hostname,
                ip_address=ip_address,
                comfyui_running=comfyui_running,
            )
            logger.info("Registered as %s (id=%s)", friendly_name, worker_id)
            return worker_id
        except Exception as e:
            logger.error("Failed to register with registry at %s (attempt %d): %s", settings.registry_url, attempt, e)
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
    return None


async def heartbeat_loop(registry, comfyui, worker_id, shutdown_event, drain_event):
    """Send heartbeats to the registry every heartbeat_interval seconds."""
    beat_count = 0
    last_comfyui_state = None
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=settings.heartbeat_interval
            )
        except asyncio.TimeoutError:
            pass

        if shutdown_event.is_set():
            break

        comfyui_running = await comfyui.check_health()
        comfyui_busy = await comfyui.check_queue_busy() if comfyui_running else False
        try:
            data = await registry.heartbeat(worker_id, comfyui_running)
            beat_count += 1

            # Check if registry signals drain
            if data.get("status") == "draining" and not drain_event.is_set():
                logger.info("Drain requested by registry — will stop after current work")
                drain_event.set()

            # Log state changes immediately, otherwise only every 5th beat
            state = (comfyui_running, comfyui_busy)
            if state != last_comfyui_state:
                status_str = "offline" if not comfyui_running else ("busy" if comfyui_busy else "idle")
                logger.info("ComfyUI: %s", status_str)
                last_comfyui_state = state
            elif beat_count % 5 == 0:
                logger.debug("Heartbeat OK (beat #%d)", beat_count)
        except Exception as e:
            logger.error("Heartbeat failed: %s", e)


async def job_poll_loop(registry, comfyui, queue, worker_id, shutdown_event, executing_event, drain_event):
    """Poll the queue for segments and execute them one at a time."""
    poll_count = 0
    while not shutdown_event.is_set():
        try:
            await asyncio.wait_for(
                shutdown_event.wait(), timeout=settings.poll_interval
            )
        except asyncio.TimeoutError:
            pass

        if shutdown_event.is_set():
            break

        # If draining and not executing, trigger shutdown
        if drain_event.is_set():
            logger.info("Drain active and no work in progress — shutting down")
            shutdown_event.set()
            break

        # Don't claim work if ComfyUI isn't running
        if not await comfyui.check_health():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.warning("ComfyUI offline — skipping poll")
            poll_count += 1
            continue

        # Don't claim work if ComfyUI is already processing (e.g. leftover from previous daemon run)
        if await comfyui.check_queue_busy():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.info("ComfyUI queue busy — waiting for current job to finish")
            poll_count += 1
            continue

        try:
            segment = await queue.claim_next(worker_id)
        except Exception as e:
            logger.error("Poll failed: %s", e)
            continue

        poll_count += 1
        if segment is None:
            if poll_count == 1 or poll_count % 60 == 0:
                logger.info("Waiting for work... (polled %d times)", poll_count)
            continue

        poll_count = 0

        executing_event.set()
        try:
            await registry.update_status(worker_id, "online-busy")
        except Exception as e:
            logger.error("Failed to update status to busy: %s", e)

        try:
            await execute_segment(segment, comfyui, queue)
        except Exception as e:
            logger.exception("Unexpected error executing segment %s", segment.id)
        finally:
            executing_event.clear()
            # If draining, don't go back to idle — shut down
            if drain_event.is_set():
                logger.info("Drain active — segment finished, shutting down")
                shutdown_event.set()
            else:
                try:
                    await registry.update_status(worker_id, "online-idle")
                except Exception as e:
                    logger.error("Failed to update status to idle: %s", e)


async def _stop_runpod_pod():
    """Call RunPod API to stop this pod (if running on RunPod)."""
    import httpx as _httpx

    pod_id = settings.runpod_pod_id
    api_key = settings.runpod_api_key
    if not pod_id or not api_key:
        return

    logger.info("Stopping RunPod pod %s ...", pod_id)
    query = f'mutation {{ podStop(input: {{podId: "{pod_id}"}}) {{ id desiredStatus }} }}'
    try:
        async with _httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"https://api.runpod.io/graphql?api_key={api_key}",
                json={"query": query},
            )
            resp.raise_for_status()
            logger.info("RunPod stop response: %s", resp.text)
    except Exception as e:
        logger.error("Failed to stop RunPod pod: %s", e)


async def run():
    registry = RegistryClient()
    comfyui = ComfyUIClient()
    queue = QueueClient()
    shutdown_event = asyncio.Event()
    executing_event = asyncio.Event()
    drain_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    hostname = socket.gethostname()
    ip_address = get_ip_address()
    comfyui_running = await comfyui.check_health()

    logger.info("=== Wanly GPU Daemon ===")
    logger.info("Python %s | Worker: %s | ComfyUI: %s | API: %s",
        sys.version.split()[0],
        settings.friendly_name,
        "running" if comfyui_running else "NOT RUNNING",
        settings.queue_url,
    )
    logger.info("Models: clip=%s vae=%s unet_high=%s unet_low=%s",
        settings.clip_model, settings.vae_model,
        settings.unet_high_model, settings.unet_low_model,
    )
    logger.info("LightX2V strengths: high=%.1f low=%.1f",
        settings.lightx2v_strength_high, settings.lightx2v_strength_low,
    )

    # Check and install required ComfyUI custom nodes
    nodes_ok = await check_and_install_nodes(comfyui)
    if not nodes_ok:
        logger.error("Required custom nodes are missing or could not be installed. Exiting.")
        await comfyui.close()
        await registry.close()
        await queue.close()
        return

    # Pre-flight: sync required resources (model weights for custom nodes)
    resources_ok = await sync_resources(queue)
    if not resources_ok:
        logger.error("Resource sync failed. Exiting.")
        await comfyui.close()
        await registry.close()
        await queue.close()
        return

    # Pre-flight: clean up partial downloads
    cleaned = cleanup_partial_downloads(settings.comfyui_path)
    if cleaned:
        logger.info("Cleaned %d partial download(s)", cleaned)

    # Pre-flight: validate all required models
    models_ok = await validate_models(comfyui)
    if not models_ok:
        logger.error("Model validation failed. Exiting.")
        await comfyui.close()
        await registry.close()
        await queue.close()
        return

    # Log GPU/VRAM info
    system_info = await comfyui.get_system_info()
    _log_system_info(system_info)

    worker_id = await register_with_retry(
        registry,
        friendly_name=settings.friendly_name,
        hostname=hostname,
        ip_address=ip_address,
        comfyui_running=comfyui_running,
        shutdown_event=shutdown_event,
    )

    if worker_id is None:
        logger.info("Shutdown requested before registration completed")
        await registry.close()
        await queue.close()
        await comfyui.close()
        return

    try:
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(registry, comfyui, worker_id, shutdown_event, drain_event)
        )
        job_task = asyncio.create_task(
            job_poll_loop(registry, comfyui, queue, worker_id, shutdown_event, executing_event, drain_event)
        )

        await asyncio.gather(heartbeat_task, job_task)
    finally:
        # Graceful shutdown: wait for current segment if executing
        if executing_event.is_set():
            logger.info("Waiting for current segment to finish (up to 10 minutes)...")
            try:
                await asyncio.wait_for(
                    asyncio.create_task(_wait_for_clear(executing_event)),
                    timeout=600,
                )
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for segment, forcing shutdown")

        logger.info("Shutting down, deregistering...")
        try:
            await registry.deregister(worker_id)
            logger.info("Deregistered successfully")
        except Exception as e:
            logger.error("Failed to deregister: %s", e)

        # If draining on RunPod, stop the pod
        if drain_event.is_set():
            await _stop_runpod_pod()

        await registry.close()
        await queue.close()
        await comfyui.close()


async def _wait_for_clear(event: asyncio.Event):
    """Wait until the event is cleared."""
    while event.is_set():
        await asyncio.sleep(1)


def main():
    asyncio.run(run())


if __name__ == "__main__":
    main()
