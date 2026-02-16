# Configs

Training configurations for Tiny Alignment Studio.

## Structure

- `base.yaml` — default config with sensible defaults for consumer hardware
- `experiments/` — per-experiment overrides that inherit from `base.yaml`
- `schemas/` — JSON Schema files for static validation

## Usage

```python
from src.utils.config import load_config

config = load_config("configs/base.yaml")
config = load_config("configs/experiments/dpo_tinyllama.yaml")
```

## Conventions

- All parameters go through config files, never hardcoded in source.
- Experiment configs only override what differs from `base.yaml`.
- Schema files allow validation without running Python.
