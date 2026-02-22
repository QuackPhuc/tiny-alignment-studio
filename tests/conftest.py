"""Shared pytest fixtures for all test levels."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from src.contracts.data import PreferenceRecord


@pytest.fixture
def sample_preference_record() -> PreferenceRecord:
    """A minimal valid preference record for testing."""
    return PreferenceRecord(
        id="test-001",
        prompt="What is the capital of France?",
        chosen="The capital of France is Paris.",
        rejected="I don't know.",
        source="test",
    )


@pytest.fixture
def sample_records() -> list[PreferenceRecord]:
    """Multiple valid preference records for pipeline testing."""
    return [
        PreferenceRecord(
            id=f"rec-{i}",
            prompt=f"Question {i}?",
            chosen=f"Good answer {i}.",
            rejected=f"Bad answer {i}.",
            source="test",
        )
        for i in range(10)
    ]


@pytest.fixture
def sample_config_dict() -> dict:
    """Raw config dict matching base.yaml structure."""
    return {
        "model": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "quantization": {"enabled": True, "bits": 4},
            "max_length": 512,
        },
        "adapter": {"type": "lora", "r": 16, "alpha": 32},
        "data": {"source": "Anthropic/hh-rlhf"},
        "training": {
            "algorithm": "dpo",
            "batch_size": 4,
            "learning_rate": 5e-5,
            "num_epochs": 1,
            "seed": 42,
            "output_dir": "outputs",
        },
        "dpo": {"beta": 0.1, "loss_type": "sigmoid"},
        "telemetry": {"enabled": True, "log_dir": "logs"},
    }


@pytest.fixture
def tmp_config_file(sample_config_dict, tmp_path) -> Path:
    """Write sample config to a temporary YAML file."""
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
    return config_path


@pytest.fixture
def tmp_data_jsonl(sample_records, tmp_path) -> Path:
    """Write sample records to a temporary JSONL file."""
    data_path = tmp_path / "test_data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for record in sample_records:
            row = {
                "prompt": record.prompt,
                "chosen": record.chosen,
                "rejected": record.rejected,
            }
            f.write(json.dumps(row) + "\n")
    return data_path


@pytest.fixture
def tmp_data_jsonl_with_config(sample_records, sample_config_dict, tmp_path) -> Path:
    """Write a config YAML pointing to a local JSONL data file."""
    data_path = tmp_path / "data.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for record in sample_records:
            row = {
                "prompt": record.prompt,
                "chosen": record.chosen,
                "rejected": record.rejected,
            }
            f.write(json.dumps(row) + "\n")

    sample_config_dict["data"]["source"] = str(data_path)
    sample_config_dict["data"]["train_split"] = "train"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
    return config_path
