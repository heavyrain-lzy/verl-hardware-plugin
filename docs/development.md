# Development Guide: Adding a New Hardware Platform

This document explains how to add support for a new hardware accelerator to verl-hardware-plugin.

## Architecture Overview

```
verl-FL (main framework)
    │
    └── import verl_hardware_plugin (via VERL_USE_EXTERNAL_MODULES)
            │
            ├── platforms/  → @PlatformRegistry.register()
            └── engines/    → @EngineRegistry.register()
```

The plugin integrates with verl through two registries:
1. **PlatformRegistry** — registers hardware platform abstractions (device management, communication, memory, etc.)
2. **EngineRegistry** — registers training engines (hardware-specific variants of FSDP/Megatron)

## Steps to Add a New Platform

### Step 1: Create the Platform Class

Create a new file under `verl_hardware_plugin/platforms/`, e.g. `my_device.py`:

```python
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Optional

import torch

from verl.plugin.platform.platform_base import PlatformBase
from verl.plugin.platform.platform_manager import PlatformRegistry


@PlatformRegistry.register(platform="my_device")
class PlatformMyDevice(PlatformBase):

    @property
    def device_name(self) -> str:
        return "my_device"

    @property
    def device_module(self) -> ModuleType:
        return torch.my_device  # torch device namespace

    def is_available(self, use_smi_check: bool = False) -> bool:
        return hasattr(torch, "my_device") and torch.my_device.is_available()

    def current_device(self) -> int:
        return torch.my_device.current_device()

    def device_count(self) -> int:
        return torch.my_device.device_count()

    def set_device(self, device_index: int) -> None:
        torch.my_device.set_device(device_index)

    def synchronize(self, device_index: Optional[int] = None) -> None:
        torch.my_device.synchronize()

    def manual_seed(self, seed: int) -> None:
        torch.my_device.manual_seed(seed)

    def manual_seed_all(self, seed: int) -> None:
        torch.my_device.manual_seed_all(seed)

    def set_allocator_settings(self, settings: str) -> None:
        pass  # leave empty if unsupported

    def empty_cache(self) -> None:
        torch.my_device.empty_cache()

    def get_device_capability(self, device_index: int = 0) -> tuple[Optional[int], Optional[int]]:
        return (None, None)

    def communication_backend_name(self) -> str:
        return "my_ccl"  # collective communication backend name

    def visible_devices_envvar(self) -> str:
        return "MY_DEVICE_VISIBLE_DEVICES"

    def ray_resource_name(self) -> str:
        return "MY_DEVICE"  # Ray resource name

    def ray_noset_envvars(self) -> list[str]:
        return ["RAY_EXPERIMENTAL_NOSET_MY_DEVICE_VISIBLE_DEVICES"]

    def is_ipc_supported(self) -> bool:
        return False

    @contextmanager
    def nvtx_range(self, msg: str):
        yield  # no-op if profiling is not supported

    def profiler_start(self) -> None:
        pass

    def profiler_stop(self) -> None:
        pass

    def cudart(self) -> Any:
        return None
```

### Step 2: Register in platforms/__init__.py

Add the following to `register_all_platforms()`:

```python
try:
    from verl_hardware_plugin.platforms import my_device  # noqa: F401
    logger.info("Registered platform: my_device")
except Exception as e:
    logger.debug("my_device platform not registered: %s", e)
```

### Step 3: Create Engine Extensions

Create `verl_hardware_plugin/engines/fsdp_my_device.py`:

```python
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.fsdp import FSDPEngineWithLMHead
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithValueHead


@EngineRegistry.register(model_type="language_model", backend=["fsdp", "fsdp2"], device="my_device")
class FSDPMyDeviceEngineWithLMHead(FSDPEngineWithLMHead):
    def initialize(self):
        super().initialize()
        # Add device-specific initialization logic


@EngineRegistry.register(model_type="value_model", backend=["fsdp", "fsdp2"], device="my_device")
class FSDPMyDeviceEngineWithValueHead(FSDPEngineWithValueHead):
    def initialize(self):
        super().initialize()
```

### Step 4: Register in engines/__init__.py

Add the corresponding import to `register_all_engines()`.

### Step 5: Add Documentation

Create `docs/my_device.md` containing:
- Overview and hardware description
- Prerequisites and installation
- Environment variable configuration
- Usage examples
- FAQ

### Step 6: Add Tests

Add registration verification tests in `tests/test_plugin_registration.py`.

## Key Design Principles

1. **Conditional imports**: Platform modules are imported inside `try/except` blocks — a missing hardware SDK does not affect other platforms.
2. **Last writer wins**: A platform registered later with the same name overrides the earlier one, allowing plugins to override built-in implementations.
3. **Auto-detection**: The first platform whose `is_available(use_smi_check=True)` returns True is selected, or it can be forced via `VERL_PLATFORM`.
4. **Minimal intrusion**: Engine extensions inject logic through inheritance + `initialize()` without modifying base class behavior.
