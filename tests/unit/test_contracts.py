"""Tests for contract models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.contracts.data import DatasetManifest, PreferenceRecord
from src.contracts.events import RunEvent
from src.contracts.training import StepMetrics, TrainConfig


class TestPreferenceRecord:
    def test_valid_record(self, sample_preference_record: PreferenceRecord) -> None:
        assert sample_preference_record.prompt == "What is the capital of France?"
        assert sample_preference_record.source == "test"

    def test_missing_chosen_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PreferenceRecord(id="bad", prompt="test")

    def test_default_source(self) -> None:
        record = PreferenceRecord(id="r1", prompt="p", chosen="c", rejected="r")
        assert record.source == "unknown"


class TestTrainConfig:
    def test_defaults(self) -> None:
        config = TrainConfig(model_name="test/model")
        assert config.algorithm == "dpo"
        assert config.batch_size == 4
        assert config.bf16 is True

    def test_invalid_batch_size(self) -> None:
        with pytest.raises(ValidationError):
            TrainConfig(model_name="test/model", batch_size=0)


class TestStepMetrics:
    def test_valid_metrics(self) -> None:
        metrics = StepMetrics(step=10, loss=0.5, learning_rate=1e-4)
        assert metrics.step == 10
        assert metrics.reward_margin is None


class TestRunEvent:
    def test_auto_timestamp(self) -> None:
        event = RunEvent(run_id="run-001", step=0, loss=2.5, learning_rate=5e-5)
        assert event.timestamp is not None


class TestDatasetManifest:
    def test_valid_manifest(self) -> None:
        manifest = DatasetManifest(
            name="test", version="1.0", num_records=100, checksum="abc123"
        )
        assert manifest.schema_version == "1.0"
