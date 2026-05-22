# FlagOS Multi-Chip Platform

## Overview

FlagOS is a multi-chip software stack that runs on various domestic GPUs (Tianshu Zhixin, Moore Threads, etc.) through FlagGems (operator library) and FlagCX (communication library), providing a CUDA-compatible interface.

## Prerequisites

- PyTorch (CUDA version)
- FlagGems: `pip install flag-gems`
- FlagCX (optional, for multi-GPU communication)

## Configuration

### Basic Setup

```bash
# Enable FlagGems operator replacement
export USE_FLAGGEMS=true

# Enable FlagCX communication library (multi-GPU scenarios)
export USE_FLAGCX=1

# Load the plugin
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin

# Force FlagOS platform (optional — auto-detected by default)
export VERL_PLATFORM=flagos
```

### Operator Control

FlagGems supports per-operator granularity control:

```bash
# Option 1: Whitelist — only enable specified operators
export TRAINING_FL_FLAGOS_WHITELIST=rmsnorm,layernorm,softmax,gelu

# Option 2: Blacklist — disable specified operators (mutually exclusive with whitelist)
export TRAINING_FL_FLAGOS_BLACKLIST=flash_attention

# Separate control for the rollout phase
export VLLM_FL_FLAGOS_WHITELIST=rmsnorm,layernorm
```

### TE-FL Configuration (Transformer Engine with FlagOS)

```bash
# Backend priority: flagos > vendor > reference
export TE_FL_PREFER=flagos

# Strict mode: do not fall back to other backends
export TE_FL_STRICT=1

# Allowed vendor list
export TE_FL_ALLOW_VENDORS=nvidia,amd

# Log level
export TEFL_LOG_LEVEL=INFO
```

## Training Engines

The plugin registers the following engines for FlagOS:

| Engine | model_type | backend |
|--------|-----------|---------|
| `FSDPFlagOSEngineWithLMHead` | language_model | fsdp, fsdp2 |
| `FSDPFlagOSEngineWithValueHead` | value_model | fsdp, fsdp2 |
| `MegatronFlagOSEngineWithLMHead` | language_model | megatron |

These engines automatically call `may_enable_flag_gems(phase="training")` during `initialize()` — no manual activation is needed.

## Usage Example

```bash
export USE_FLAGGEMS=true
export USE_FLAGCX=1
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export VERL_PLATFORM=flagos

python -m verl.trainer.main --config your_config.yaml
```

## FAQ

**Q: FlagGems is not taking effect?**

Check that `USE_FLAGGEMS` is set to `true` or `1`, and confirm the `flag_gems` package is installed.

**Q: Can whitelist and blacklist be set at the same time?**

No. Only one can be set per phase (training/rollout). Setting both will raise a `ValueError`.

**Q: How to check the current FL status?**

```python
from verl_hardware_plugin.utils import FLEnvManager
print(FLEnvManager.get_summary())
print(FLEnvManager.get_env_snapshot("training"))
```
