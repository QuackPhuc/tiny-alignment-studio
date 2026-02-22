"""Configuration loader with YAML parsing and Pydantic validation."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.contracts.training import TrainConfig

logger = logging.getLogger(__name__)


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


def merge_configs(
    base_path: str | Path,
    override_path: str | Path,
) -> dict[str, Any]:
    """Deep-merge an override config on top of a base config.

    Override values replace base values at the leaf level.
    Nested dicts are merged recursively.

    Args:
        base_path: Path to the base YAML config.
        override_path: Path to the override YAML config.

    Returns:
        Merged config dict.
    """
    base_path, override_path = Path(base_path), Path(override_path)

    with base_path.open() as f:
        base = yaml.safe_load(f) or {}
    with override_path.open() as f:
        override = yaml.safe_load(f) or {}

    return _deep_merge(base, override)


def validate_config_schema(
    config: dict[str, Any],
    schema_path: str | Path | None = None,
) -> list[str]:
    """Validate a config dict against the JSON Schema.

    Args:
        config: Parsed YAML config dict.
        schema_path: Path to JSON Schema file. Defaults to
            configs/schemas/training_config.schema.json.

    Returns:
        List of validation error messages (empty if valid).
    """
    try:
        import jsonschema
    except ImportError:
        logger.warning("jsonschema not installed, skipping schema validation")
        return []

    if schema_path is None:
        schema_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs"
            / "schemas"
            / "training_config.schema.json"
        )
    schema_path = Path(schema_path)
    if not schema_path.exists():
        logger.warning("Schema file not found: %s", schema_path)
        return []

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    return [err.message for err in validator.iter_errors(config)]


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


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base.

    Args:
        base: Base config dict (not mutated).
        override: Override values to apply.

    Returns:
        New merged dict.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result
