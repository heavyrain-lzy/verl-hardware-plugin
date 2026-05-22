# Cambricon MLU Platform

## Overview

Supports Cambricon MLU (Machine Learning Unit) series accelerators, including MLU370 and MLU590, running via the `torch_mlu` extension and CNCL (Cambricon NCCL) communication backend.

## Prerequisites

- PyTorch 2.1+
- Cambricon PyTorch (torch_mlu): refer to the official Cambricon documentation for installation
- CNCL (Cambricon NCCL) communication library

### Installing torch_mlu

```bash
# Refer to the official Cambricon release page
pip install torch_mlu
```

## Configuration

```bash
# Load the plugin
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

# Force MLU platform (optional — auto-detected when MLU hardware is present)
export VERL_PLATFORM=mlu
```

### Device Visibility

```bash
# Control visible MLU devices
export MLU_VISIBLE_DEVICES=0,1,2,3
```

### Distributed Communication

MLU uses `cncl` as the collective communication backend:

```python
import torch.distributed as dist
dist.init_process_group(backend="cncl")
```

## Training Engines

| Engine | model_type | backend |
|--------|-----------|---------|
| `FSDPMLUEngineWithLMHead` | language_model | fsdp, fsdp2 |
| `FSDPMLUEngineWithValueHead` | value_model | fsdp, fsdp2 |
| `MegatronMLUEngineWithLMHead` | language_model | megatron |

## Usage Example

```bash
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_PLATFORM=mlu
export MLU_VISIBLE_DEVICES=0,1,2,3

python -m verl.trainer.main --config your_config.yaml
```

## Ray Integration

MLU maps to the `MLU` resource type in Ray. You need to declare MLU resources in the Ray cluster configuration:

```yaml
# ray cluster config
resources:
  MLU: 8
```

## Platform Interface

```python
from verl.plugin.platform import get_platform

platform = get_platform()  # auto-detected as MLU
print(platform.device_name)                    # "mlu"
print(platform.communication_backend_name())   # "cncl"
print(platform.visible_devices_envvar())       # "MLU_VISIBLE_DEVICES"
print(platform.device_count())                 # number of MLU devices
```

## FAQ

**Q: torch.mlu is reported as unavailable?**

Verify that `torch_mlu` is installed correctly:
```python
import torch_mlu
print(torch.mlu.is_available())
print(torch.mlu.device_count())
```

**Q: Is IPC supported?**

Currently `is_ipc_supported()` returns `False` for the MLU platform. If a future version of torch_mlu adds IPC support, the plugin can be updated accordingly.

**Q: Memory allocator settings not taking effect?**

Some versions of torch_mlu do not support `_set_allocator_settings`. The plugin will gracefully degrade and emit a warning. Upgrading torch_mlu to the latest version should resolve this.

**Q: get_device_capability returns (None, None)?**

MLU does not use the CUDA compute capability concept. Returning `(None, None)` is expected behavior and does not affect training.
