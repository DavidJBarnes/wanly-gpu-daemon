"""Tests for startup model validation.

validate_models is a startup gate: returning False exits the daemon before it registers.
So the regressions worth guarding are the two directions of that gate — it must fail when
a required weight is genuinely missing or truncated, and it must NOT fail for reasons that
say nothing about the weights (no comfyui_path configured, a flaky /object_info).
"""

import pytest

from daemon import model_validator
from daemon.model_validator import MODEL_CHECKS, get_model_checks, validate_models


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


class FakeHTTP:
    def __init__(self, object_info=None):
        self._object_info = object_info if object_info is not None else {}

    async def get(self, path, timeout=None):
        return FakeResponse(self._object_info)


class FakeComfyUI:
    def __init__(self, object_info=None):
        self.http = FakeHTTP(object_info)


class TestModelChecks:
    def test_checks_cover_the_native_i2v_path(self):
        names = [c[0] for c in get_model_checks()]
        assert names == [
            "vae_model", "clip_model", "unet_high_model", "unet_low_model",
            "lightx2v_lora_high", "lightx2v_lora_low",
        ]

    def test_no_retired_engine_models_are_validated(self):
        """VACE and Lynx weights are no longer staged, so requiring them would fail
        startup on a correctly-provisioned worker."""
        names = [c[0] for c in get_model_checks()]
        assert not [n for n in names if n.startswith(("vace_", "lynx_"))]

    def test_every_check_resolves_to_a_real_setting(self):
        """A typo'd setting name silently skips the check (getattr -> None -> continue)."""
        from daemon.config import settings
        for name, _loader, _subs, _min in MODEL_CHECKS:
            assert getattr(settings, name, None), f"{name} is not a Settings field"

    def test_every_check_has_a_comfyui_loader(self):
        assert all(loader for _n, loader, _s, _m in MODEL_CHECKS)


class TestValidateModels:
    @pytest.fixture(autouse=True)
    def _comfy_path(self, monkeypatch, tmp_path):
        monkeypatch.setattr(model_validator.settings, "comfyui_path", str(tmp_path))
        return tmp_path

    @staticmethod
    def _small(checks):
        """Same checks with tiny minimums — the real ones are GB-scale."""
        return [(n, loader, subs, 16) for n, loader, subs, _m in checks]

    def _write(self, tmp_path, subfolder, name, size):
        d = tmp_path / "models" / subfolder
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_bytes(b"\0" * size)

    def _write_all(self, tmp_path, checks):
        for name, _loader, subfolders, _min in checks:
            self._write(tmp_path, subfolders[0],
                        getattr(model_validator.settings, name), 64)

    async def test_skips_entirely_without_comfyui_path(self, monkeypatch):
        monkeypatch.setattr(model_validator.settings, "comfyui_path", "")
        assert await validate_models(FakeComfyUI()) is True

    async def test_passes_when_every_model_is_present(self, monkeypatch, _comfy_path):
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(MODEL_CHECKS))
        self._write_all(_comfy_path, MODEL_CHECKS)
        assert await validate_models(FakeComfyUI()) is True

    async def test_fails_when_a_model_is_missing(self, monkeypatch, _comfy_path):
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(MODEL_CHECKS))
        self._write_all(_comfy_path, MODEL_CHECKS)
        # Remove one unet — the gate must catch it.
        (_comfy_path / "models" / "diffusion_models"
         / model_validator.settings.unet_low_model).unlink()
        assert await validate_models(FakeComfyUI()) is False

    async def test_undersized_file_is_rejected(self, _comfy_path):
        self._write(_comfy_path, "vae", model_validator.settings.vae_model, 8)
        assert await validate_models(FakeComfyUI()) is False

    async def test_object_info_failure_is_non_fatal(self, monkeypatch, _comfy_path):
        """A /object_info hiccup must not block startup on its own."""
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(MODEL_CHECKS))
        self._write_all(_comfy_path, MODEL_CHECKS)

        class Boom:
            http = type("H", (), {"get": staticmethod(
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no object_info")))})()

        assert await validate_models(Boom()) is True

    async def test_model_unknown_to_comfyui_is_rejected(self, monkeypatch, _comfy_path):
        """Present on disk but absent from the loader's dropdown => ComfyUI can't load it."""
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(MODEL_CHECKS))
        self._write_all(_comfy_path, MODEL_CHECKS)
        object_info = {"UNETLoader": {"input": {"required": {
            "unet_name": [["some_other_model.safetensors"]]}}}}
        assert await validate_models(FakeComfyUI(object_info)) is False
