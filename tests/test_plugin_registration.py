# Copyright (c) 2026 BAAI. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""Tests for plugin registration mechanism."""

import os
from contextlib import contextmanager
from unittest import mock

import pytest


@contextmanager
def _fresh_registries():
    """Reset platform manager singleton for isolated tests."""
    import verl.plugin.platform.platform_manager as pm

    old_platform = pm._current_platform
    pm._current_platform = None
    try:
        yield
    finally:
        pm._current_platform = old_platform


class TestPlatformRegistration:
    """Verify that all hardware platforms register correctly."""

    def test_flagos_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.flagos import PlatformFlagOS  # noqa: F401

        assert "flagos" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("flagos")
        assert cls is PlatformFlagOS

    def test_xpu_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU  # noqa: F401

        assert "xpu" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("xpu")
        assert cls is PlatformXPU

    def test_mlu_registered(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry
        from verl_hardware_plugin.platforms.platform_mlu import PlatformMLU  # noqa: F401

        assert "mlu" in PlatformRegistry.registered_names()
        cls = PlatformRegistry.get("mlu")
        assert cls is PlatformMLU

    def test_flagos_detection_with_env(self):
        """VERL_PLATFORM=flagos should select FlagOS platform."""
        from verl.plugin.platform.platform_manager import PlatformRegistry, _detect_platform_name
        from verl_hardware_plugin.platforms.flagos import PlatformFlagOS  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "flagos"}):
                assert _detect_platform_name() == "flagos"

    def test_xpu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry, _detect_platform_name
        from verl_hardware_plugin.platforms.platform_xpu import PlatformXPU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "xpu"}):
                assert _detect_platform_name() == "xpu"

    def test_mlu_detection_with_env(self):
        from verl.plugin.platform.platform_manager import PlatformRegistry, _detect_platform_name
        from verl_hardware_plugin.platforms.platform_mlu import PlatformMLU  # noqa: F401

        with _fresh_registries():
            with mock.patch.dict(os.environ, {"VERL_PLATFORM": "mlu"}):
                assert _detect_platform_name() == "mlu"


class TestEngineRegistration:
    """Verify that engine classes register correctly."""

    def test_fsdp_flagos_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_flagos import (
            FSDPFlagOSEngineWithLMHead,
            FSDPFlagOSEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"]["flagos"] is FSDPFlagOSEngineWithLMHead
        assert EngineRegistry._engines["language_model"]["fsdp2"]["flagos"] is FSDPFlagOSEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"]["flagos"] is FSDPFlagOSEngineWithValueHead

    def test_fsdp_xpu_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_xpu import (
            FSDPXPUEngineWithLMHead,
            FSDPXPUEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"]["xpu"] is FSDPXPUEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"]["xpu"] is FSDPXPUEngineWithValueHead

    def test_fsdp_mlu_engines_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.fsdp_mlu import (
            FSDPMLUEngineWithLMHead,
            FSDPMLUEngineWithValueHead,
        )

        assert EngineRegistry._engines["language_model"]["fsdp"]["mlu"] is FSDPMLUEngineWithLMHead
        assert EngineRegistry._engines["value_model"]["fsdp"]["mlu"] is FSDPMLUEngineWithValueHead

    def test_megatron_flagos_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_flagos import MegatronFlagOSEngineWithLMHead

        assert EngineRegistry._engines["language_model"]["megatron"]["flagos"] is MegatronFlagOSEngineWithLMHead

    def test_megatron_xpu_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_xpu import MegatronXPUEngineWithLMHead

        assert EngineRegistry._engines["language_model"]["megatron"]["xpu"] is MegatronXPUEngineWithLMHead

    def test_megatron_mlu_engine_registered(self):
        from verl.workers.engine.base import EngineRegistry
        from verl_hardware_plugin.engines.megatron_mlu import MegatronMLUEngineWithLMHead

        assert EngineRegistry._engines["language_model"]["megatron"]["mlu"] is MegatronMLUEngineWithLMHead


class TestFLEnvManager:
    """Test FLEnvManager utility."""

    def test_flaggems_disabled_by_default(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {}, clear=True):
            assert not FLEnvManager.is_flaggems_enabled()

    def test_flaggems_enabled(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"USE_FLAGGEMS": "true"}):
            assert FLEnvManager.is_flaggems_enabled()

    def test_whitelist_parsing(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"TRAINING_FL_FLAGOS_WHITELIST": "rmsnorm,layernorm,softmax"}):
            wl = FLEnvManager.get_training_whitelist()
            assert wl == ["rmsnorm", "layernorm", "softmax"]

    def test_summary(self):
        from verl_hardware_plugin.utils import FLEnvManager

        with mock.patch.dict(os.environ, {"USE_FLAGGEMS": "1", "USE_FLAGCX": "0"}):
            summary = FLEnvManager.get_summary()
            assert "FlagGems=ON" in summary
            assert "FlagCX=OFF" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
