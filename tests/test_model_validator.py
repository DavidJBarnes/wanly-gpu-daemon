"""Tests for profile-aware startup model validation.

The regression these guard: a lynx-profile pod has no Wan 2.2 i2v weights by design.
Validating them anyway fails startup on a perfectly healthy worker — which is exactly
what happened on the first Lynx pod boot.
"""

import pytest

from daemon import model_validator
from daemon.model_validator import (
    _LYNX_CHECKS,
    _WAN22_CHECKS,
    MODEL_CHECKS,
    get_model_checks,
    validate_models,
)


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


class TestProfileSelection:
    def test_full_profile_checks_wan22(self, monkeypatch):
        monkeypatch.setattr(model_validator.settings, "model_profile", "full")
        names = [c[0] for c in get_model_checks()]
        assert "unet_high_model" in names
        assert "lightx2v_lora_high" in names
        assert "lynx_t2v_model" not in names

    def test_lynx_profile_checks_lynx_only(self, monkeypatch):
        monkeypatch.setattr(model_validator.settings, "model_profile", "lynx")
        names = [c[0] for c in get_model_checks()]
        assert "lynx_t2v_model" in names
        assert "lynx_ip_layers" in names
        assert "lynx_ref_layers" in names
        assert "lynx_resampler" in names
        # The 2.2 i2v weights are absent from a lynx pod by design.
        assert "unet_high_model" not in names
        assert "lightx2v_lora_high" not in names

    @pytest.mark.parametrize("value", ["lynx", "LYNX", " Lynx ", "lYnX"])
    def test_profile_match_is_case_and_space_insensitive(self, monkeypatch, value):
        monkeypatch.setattr(model_validator.settings, "model_profile", value)
        assert [c[0] for c in get_model_checks()] == [c[0] for c in
                                                      (model_validator._COMMON_CHECKS + _LYNX_CHECKS)]

    @pytest.mark.parametrize("value", ["full", "", "anything-else"])
    def test_unknown_profile_falls_back_to_full(self, monkeypatch, value):
        monkeypatch.setattr(model_validator.settings, "model_profile", value)
        assert "unet_high_model" in [c[0] for c in get_model_checks()]

    def test_vae_is_checked_in_both_profiles(self, monkeypatch):
        for profile in ("full", "lynx"):
            monkeypatch.setattr(model_validator.settings, "model_profile", profile)
            assert "vae_model" in [c[0] for c in get_model_checks()]

    def test_backcompat_flat_list_is_the_full_profile(self):
        assert MODEL_CHECKS == model_validator._COMMON_CHECKS + _WAN22_CHECKS

    def test_lynx_checks_are_disk_only(self):
        """Lynx models load via WanVideoWrapper's loaders, not a ComfyUI dropdown."""
        assert all(loader is None for _n, loader, _s, _m in _LYNX_CHECKS)


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

    async def test_skips_entirely_without_comfyui_path(self, monkeypatch):
        monkeypatch.setattr(model_validator.settings, "comfyui_path", "")
        assert await validate_models(FakeComfyUI()) is True

    async def test_lynx_profile_passes_without_wan22_models(self, monkeypatch, _comfy_path):
        """The exact scenario that failed the first pod boot."""
        monkeypatch.setattr(model_validator.settings, "model_profile", "lynx")
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(model_validator._COMMON_CHECKS + _LYNX_CHECKS))
        # Only Lynx + VAE present — no 2.2 i2v weights anywhere.
        self._write(_comfy_path, "vae", model_validator.settings.vae_model, 64)
        for setting in ("lynx_t2v_model", "lynx_ip_layers", "lynx_ref_layers", "lynx_resampler"):
            self._write(_comfy_path, "diffusion_models",
                        getattr(model_validator.settings, setting), 64)
        assert await validate_models(FakeComfyUI()) is True

    async def test_lynx_profile_fails_when_an_adapter_is_missing(self, monkeypatch, _comfy_path):
        monkeypatch.setattr(model_validator.settings, "model_profile", "lynx")
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(model_validator._COMMON_CHECKS + _LYNX_CHECKS))
        self._write(_comfy_path, "vae", model_validator.settings.vae_model, 64)
        self._write(_comfy_path, "diffusion_models",
                    model_validator.settings.lynx_t2v_model, 64)
        # ref layers + resampler absent
        assert await validate_models(FakeComfyUI()) is False

    async def test_undersized_file_is_rejected(self, monkeypatch, _comfy_path):
        monkeypatch.setattr(model_validator.settings, "model_profile", "lynx")
        self._write(_comfy_path, "vae", model_validator.settings.vae_model, 8)
        assert await validate_models(FakeComfyUI()) is False

    async def test_object_info_failure_is_non_fatal(self, monkeypatch, _comfy_path):
        """A /object_info hiccup must not block startup on its own."""
        monkeypatch.setattr(model_validator.settings, "model_profile", "lynx")
        monkeypatch.setattr(model_validator, "get_model_checks",
                            lambda: self._small(model_validator._COMMON_CHECKS + _LYNX_CHECKS))
        self._write(_comfy_path, "vae", model_validator.settings.vae_model, 64)
        for setting in ("lynx_t2v_model", "lynx_ip_layers", "lynx_ref_layers", "lynx_resampler"):
            self._write(_comfy_path, "diffusion_models",
                        getattr(model_validator.settings, setting), 64)

        class Boom:
            http = type("H", (), {"get": staticmethod(
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no object_info")))})()

        assert await validate_models(Boom()) is True
