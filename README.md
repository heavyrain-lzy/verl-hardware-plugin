# verl-hardware-plugin

Multi-chip hardware platform and engine plugins for [verl](https://github.com/verl-project/verl).

This package provides platform abstraction and training engine extensions for non-CUDA accelerators, enabling verl to run on diverse hardware through a unified plugin interface.

## Supported Hardware

| Platform | Device | Communication | Status |
|----------|--------|---------------|--------|
| FlagOS | 天数智芯/摩尔线程等国产GPU | FlagCX | ✅ |
| Intel XPU | Data Center GPU Max / Arc | xccl (oneCCL) | ✅ |
| Cambricon MLU | MLU370/MLU590 | CNCL | ✅ |
| Huawei NPU | Ascend 910B | HCCL | Built-in (verl core) |

## Installation

```bash
pip install -e .
```

## Usage

### Method 1: Environment Variable (recommended for explicit control)

```bash
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
python train.py
```

### Method 2: Auto-discovery via entry_points

After `pip install`, the plugin is automatically discoverable by verl through the `verl.plugins` entry point group. No additional configuration needed if verl supports entry_point scanning.

### Platform Selection

The platform is auto-detected based on hardware availability. To force a specific platform:

```bash
export VERL_PLATFORM=flagos   # Force FlagOS
export VERL_PLATFORM=xpu      # Force Intel XPU
export VERL_PLATFORM=mlu      # Force Cambricon MLU
```

### FlagOS Configuration

```bash
# Enable FlagGems operator library
export USE_FLAGGEMS=true

# Enable FlagCX communication library
export USE_FLAGCX=1

# Optional: operator whitelist/blacklist
export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax
# OR
export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention
```

## Architecture

```
verl-FL (main framework)
    └── VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
            │
            ├── PlatformRegistry.register("flagos") → PlatformFlagOS
            ├── PlatformRegistry.register("xpu")    → PlatformXPU
            ├── PlatformRegistry.register("mlu")    → PlatformMLU
            │
            ├── EngineRegistry.register(device="flagos") → FSDP/Megatron engines
            ├── EngineRegistry.register(device="xpu")    → FSDP/Megatron engines
            └── EngineRegistry.register(device="mlu")    → FSDP/Megatron engines
```

The plugin uses verl's decorator-based registration:
- `@PlatformRegistry.register(platform="xxx")` for platform classes
- `@EngineRegistry.register(model_type=..., backend=..., device=...)` for engine classes

Registration happens at import time. The "last writer wins" semantics allow this plugin to override built-in defaults.

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

Apache License 2.0
