"""Configuration loader with YAML parsing and Pydantic validation."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.contracts.training import TrainConfig


def load_config(config_path: str | Path) -> TrainConfig:
    """Load and validate a YAML training configuration.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Validated TrainConfig instance.

    Raises:
        FileNotFoundError: If config file does not exist.
        ValidationError: If config fails Pydantic validation.
    """
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        raw = yaml.safe_load(f)

    return TrainConfig(**_flatten_config(raw))


def _flatten_config(raw: dict) -> dict:
    """Flatten nested YAML config into TrainConfig-compatible kwargs.

    Args:
        raw: Raw parsed YAML dict.

    Returns:
        Flat dict matching TrainConfig fields.
    """
    model_cfg = raw.get("model", {})
    training_cfg = raw.get("training", {})
    adapter_cfg = raw.get("adapter", {})

    return {
        "model_name": model_cfg.get("name", ""),
        "algorithm": training_cfg.get("algorithm", "dpo"),
        "adapter_type": adapter_cfg.get("type", "lora"),
        "quantization_bits": model_cfg.get("quantization", {}).get("bits", 4),
        "batch_size": training_cfg.get("batch_size", 4),
        "gradient_accumulation_steps": training_cfg.get(
            "gradient_accumulation_steps", 4
        ),
        "learning_rate": training_cfg.get("learning_rate", 5e-5),
        "num_epochs": training_cfg.get("num_epochs", 1),
        "max_length": model_cfg.get("max_length", 512),
        "seed": training_cfg.get("seed", 42),
        "output_dir": training_cfg.get("output_dir", "outputs"),
        "bf16": training_cfg.get("bf16", True),
    }
