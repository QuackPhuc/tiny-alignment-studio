"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

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
        },
    }
