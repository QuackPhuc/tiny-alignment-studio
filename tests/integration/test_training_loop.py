"""Integration tests for the training pipeline components.

Tests component interactions without GPU dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.contracts.training import TrainConfig
from src.core.algorithms import AlgorithmRegistry
from src.core.data.pipeline import DataPipeline
from src.core.trainer import AlignmentTrainer
from src.telemetry.callbacks import create_telemetry_callback
from src.telemetry.events import EventReader, EventWriter
from src.utils.config import load_config, merge_configs


class TestConfigPipeline:
    """Test config loading, validation, and merge."""

    def test_load_valid_config(self, tmp_config_file):
        config = load_config(tmp_config_file)
        assert isinstance(config, TrainConfig)
        assert config.algorithm == "dpo"
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_load_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_invalid_algorithm_uses_default(self, sample_config_dict, tmp_path):
        """Config with unknown algorithm falls back to pydantic defaults."""
        sample_config_dict["training"]["algorithm"] = "unknown"
        config_path = tmp_path / "bad.yaml"
        config_path.write_text(yaml.dump(sample_config_dict), encoding="utf-8")
        # Should still load (pydantic doesn't restrict enum on TrainConfig)
        config = load_config(config_path)
        assert config.algorithm == "unknown"

    def test_merge_base_with_override(self, tmp_path):
        base = {
            "model": {"name": "base-model", "max_length": 512},
            "training": {"algorithm": "dpo", "batch_size": 4},
        }
        override = {
            "model": {"name": "override-model"},
            "training": {"batch_size": 8},
        }
        base_path = tmp_path / "base.yaml"
        override_path = tmp_path / "override.yaml"
        base_path.write_text(yaml.dump(base), encoding="utf-8")
        override_path.write_text(yaml.dump(override), encoding="utf-8")

        merged = merge_configs(base_path, override_path)
        assert merged["model"]["name"] == "override-model"
        assert merged["model"]["max_length"] == 512
        assert merged["training"]["batch_size"] == 8
        assert merged["training"]["algorithm"] == "dpo"

    def test_merge_preserves_base_when_no_overlap(self, tmp_path):
        base = {"model": {"name": "m"}, "training": {"algorithm": "dpo"}}
        override = {"dpo": {"beta": 0.2}}
        base_path = tmp_path / "base.yaml"
        override_path = tmp_path / "override.yaml"
        base_path.write_text(yaml.dump(base), encoding="utf-8")
        override_path.write_text(yaml.dump(override), encoding="utf-8")

        merged = merge_configs(base_path, override_path)
        assert merged["model"]["name"] == "m"
        assert merged["dpo"]["beta"] == 0.2


class TestDataPipelineIntegration:
    """Test the full data pipeline chain: load → validate → format."""

    def test_load_validate_format_chain(self, tmp_data_jsonl):
        pipeline = DataPipeline(max_samples=5)
        dataset = pipeline.load(str(tmp_data_jsonl), split="train")
        assert len(dataset) == 5

        records = pipeline.validate(dataset)
        assert len(records) == 5

        dpo_dataset = pipeline.format_for_dpo(records)
        assert "prompt" in dpo_dataset.column_names
        assert "chosen" in dpo_dataset.column_names
        assert "rejected" in dpo_dataset.column_names
        assert len(dpo_dataset) == 5

    def test_manifest_roundtrip(self, sample_records, tmp_path):
        pipeline = DataPipeline()
        manifest = pipeline.create_manifest("test-ds", sample_records)
        saved_path = pipeline.save_manifest(manifest, tmp_path)

        assert saved_path.exists()
        content = json.loads(saved_path.read_text(encoding="utf-8"))
        assert content["name"] == "test-ds"
        assert content["num_records"] == 10
        assert len(content["checksum"]) == 16

    def test_empty_dataset_raises(self, tmp_path):
        data_path = tmp_path / "empty.jsonl"
        data_path.write_text(
            json.dumps({"prompt": "", "chosen": "", "rejected": ""}) + "\n",
            encoding="utf-8",
        )
        pipeline = DataPipeline()
        dataset = pipeline.load(str(data_path), split="train")
        with pytest.raises(ValueError, match="No valid records"):
            pipeline.validate(dataset)


class TestAlgorithmRegistryIntegration:
    """Test algorithm registration and retrieval."""

    def test_dpo_registered(self):
        algo = AlgorithmRegistry.get("dpo")
        assert algo.required_data_format == "preference_pairs"

    def test_ppo_registered(self):
        algo = AlgorithmRegistry.get("ppo")
        assert algo.required_data_format == "prompt_completion"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            AlgorithmRegistry.get("nonexistent")

    def test_list_algorithms(self):
        algos = AlgorithmRegistry.available()
        assert "dpo" in algos
        assert "ppo" in algos


class TestTelemetryIntegration:
    """Test telemetry callback → writer → reader chain."""

    def test_callback_writes_events_readable_by_reader(self, tmp_path):
        from types import SimpleNamespace

        run_id = "integration-test"
        callback = create_telemetry_callback(str(tmp_path), run_id)

        state = SimpleNamespace(global_step=5)
        logs = {"loss": 0.5, "learning_rate": 1e-4}
        callback.on_log(args=None, state=state, control=None, logs=logs)

        reader = EventReader(str(tmp_path), run_id)
        events = reader.read_all()
        assert len(events) == 1
        assert events[0].loss == 0.5
        assert events[0].step == 5

    def test_multiple_steps_produce_ordered_events(self, tmp_path):
        from types import SimpleNamespace

        run_id = "ordered-test"
        callback = create_telemetry_callback(str(tmp_path), run_id)

        for step in range(10):
            state = SimpleNamespace(global_step=step)
            logs = {"loss": 1.0 - step * 0.05, "learning_rate": 1e-4}
            callback.on_log(args=None, state=state, control=None, logs=logs)

        reader = EventReader(str(tmp_path), run_id)
        events = reader.read_all()
        assert len(events) == 10
        assert events[0].step == 0
        assert events[-1].step == 9
        # Loss should decrease
        assert events[-1].loss < events[0].loss


class TestTrainerInit:
    """Test AlignmentTrainer initialization (no GPU needed)."""

    def test_init_from_config(self, tmp_config_file):
        trainer = AlignmentTrainer(tmp_config_file)
        assert trainer.config.algorithm == "dpo"
        assert trainer.config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_init_missing_config_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AlignmentTrainer(tmp_path / "missing.yaml")
