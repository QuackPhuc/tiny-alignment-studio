"""Tests for config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import load_config


class TestLoadConfig:
    def test_load_base_config(self) -> None:
        config = load_config(Path("configs/base.yaml"))
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.algorithm == "dpo"
        assert config.quantization_bits == 4

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
