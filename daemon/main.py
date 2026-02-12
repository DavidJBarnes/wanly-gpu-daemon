import asyncio
import logging
import signal
import socket

from daemon.comfyui_client import ComfyUIClient
from daemon.config import settings
from daemon.executor import execute_segment
from daemon.node_checker import check_and_install_nodes
from daemon.queue_client import QueueClient
from daemon.registry_client import RegistryClient

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


async def heartbeat_loop(registry, comfyui, worker_id, shutdown_event):
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
            await registry.heartbeat(worker_id, comfyui_running)
            beat_count += 1
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


async def job_poll_loop(registry, comfyui, queue, worker_id, shutdown_event, executing_event):
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

        # Don't claim work if ComfyUI isn't running
        if not await comfyui.check_health():
            if poll_count == 0 or poll_count % 60 == 0:
                logger.warning("ComfyUI offline — skipping poll")
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
            try:
                await registry.update_status(worker_id, "online-idle")
            except Exception as e:
                logger.error("Failed to update status to idle: %s", e)


async def run():
    registry = RegistryClient()
    comfyui = ComfyUIClient()
    queue = QueueClient()
    shutdown_event = asyncio.Event()
    executing_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    hostname = socket.gethostname()
    ip_address = get_ip_address()
    comfyui_running = await comfyui.check_health()

    logger.info("=== Wanly GPU Daemon ===")
    logger.info("Worker: %s | ComfyUI: %s | API: %s",
        settings.friendly_name,
        "running" if comfyui_running else "NOT RUNNING",
        settings.queue_url,
    )

    # Check and install required ComfyUI custom nodes
    nodes_ok = await check_and_install_nodes(comfyui)
    if not nodes_ok:
        logger.error("Required custom nodes are missing or could not be installed. Exiting.")
        await comfyui.close()
        await registry.close()
        await queue.close()
        return

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
            heartbeat_loop(registry, comfyui, worker_id, shutdown_event)
        )
        job_task = asyncio.create_task(
            job_poll_loop(registry, comfyui, queue, worker_id, shutdown_event, executing_event)
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
