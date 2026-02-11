# wanly-gpu-daemon

Background daemon that runs on each GPU worker machine. Registers with the GPU registry, sends periodic heartbeats, and monitors local ComfyUI availability.

## Setup

### Prerequisites

- Python 3.11+
- ComfyUI running locally (optional, status is reported to registry)

### Environment

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description | Default |
|----------|-------------|---------|
| `REGISTRY_URL` | URL of the wanly-gpu-registry service | `http://localhost:8000` |
| `FRIENDLY_NAME` | Display name for this worker | `gpu-worker-1` |
| `HEARTBEAT_INTERVAL` | Seconds between heartbeats | `30` |
| `COMFYUI_URL` | Local ComfyUI instance URL | `http://localhost:8188` |

### Install and run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m daemon.main
```

## Behavior

1. On startup: detects hostname, IP, and ComfyUI status, then registers with the GPU registry
2. Sends heartbeats to the registry every `HEARTBEAT_INTERVAL` seconds
3. On shutdown (SIGINT/SIGTERM): deregisters from the registry
