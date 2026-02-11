from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    registry_url: str = "http://localhost:8000"
    friendly_name: str = "gpu-worker-1"
    heartbeat_interval: int = 30
    comfyui_url: str = "http://localhost:8188"
    queue_url: str = "http://localhost:8001"
    poll_interval: int = 5

    model_config = {"env_file": ".env"}


settings = Settings()
