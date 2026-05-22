# Intel XPU Platform

## Overview

Supports Intel Data Center GPU Max Series (Ponte Vecchio), Arc series, and other XPU devices, running via `torch.xpu` and the oneCCL (xccl) communication backend.

## Prerequisites

- PyTorch 2.1+
- Intel Extension for PyTorch (IPEX): `pip install intel-extension-for-pytorch`
- oneAPI Base Toolkit (includes oneCCL)

### Docker Environment (Recommended)

```bash
docker pull intel/deep-learning-essentials:2025.3.2
```

## Configuration

```bash
# Load the plugin
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

# Force XPU platform (optional — auto-detected when XPU hardware is present)
export VERL_PLATFORM=xpu
```

### Device Visibility

```bash
# Control visible devices (similar to CUDA_VISIBLE_DEVICES)
export ZE_AFFINITY_MASK=0,1,2,3
```

### Distributed Communication

XPU uses `xccl` (oneCCL) as the communication backend. The composite backend is configured as `cpu:gloo,xpu:xccl`.

Known limitations:
- xccl does not support `ReduceOp.AVG` — the plugin automatically uses SUM + division as a workaround
- FSDP2 requires `set_force_sum_reduction_for_comms(True)` (handled automatically by the plugin)

## Training Engines

| Engine | model_type | backend |
|--------|-----------|---------|
| `FSDPXPUEngineWithLMHead` | language_model | fsdp, fsdp2 |
| `FSDPXPUEngineWithValueHead` | value_model | fsdp, fsdp2 |
| `MegatronXPUEngineWithLMHead` | language_model | megatron |

## Usage Example

```bash
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_PLATFORM=xpu
export ZE_AFFINITY_MASK=0,1,2,3

python -m verl.trainer.main --config your_config.yaml
```

## Ray Integration

XPU maps to the `GPU` resource type in Ray (via `IntelGPUAccelerator`), so scheduling behavior is consistent with CUDA.

```python
# Ray workers automatically sync ONEAPI_DEVICE_SELECTOR
# No manual configuration needed
```

## FAQ

**Q: torch.xpu is reported as unavailable?**

Ensure `intel-extension-for-pytorch` is installed and the oneAPI environment is initialized:
```bash
source /opt/intel/oneapi/setvars.sh
```

**Q: Communication error "AVG not supported"?**

This is a known xccl limitation. The plugin's FSDP engine automatically enables `force_sum_reduction_for_comms`. If you encounter this in custom code, manually use SUM + division:
```python
torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
tensor /= world_size
```

**Q: Notes on vLLM rollout with XPU?**

- sleep_mode must be disabled
- TP=1 forces the "uni" executor
- `ONEAPI_DEVICE_SELECTOR` must be unset before launch
