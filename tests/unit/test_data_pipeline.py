"""Tests for data pipeline and formatters."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from datasets import Dataset

from src.contracts.data import PreferenceRecord
from src.core.data.formatters import (
    AnthropicHHFormatter,
    StandardFormatter,
    get_formatter,
)
from src.core.data.pipeline import (
    DataPipeline,
    _extract_prompt,
    _extract_response,
    _get_last_human_turn,
)


# --- Fixtures ---


@pytest.fixture
def anthropic_hh_row() -> dict:
    """Simulates a single Anthropic HH-RLHF record."""
    return {
        "chosen": (
            "\n\nHuman: What is the capital of France?"
            "\n\nAssistant: The capital of France is Paris."
        ),
        "rejected": (
            "\n\nHuman: What is the capital of France?"
            "\n\nAssistant: I'm not sure about that."
        ),
    }


@pytest.fixture
def standard_row() -> dict:
    """A record with explicit prompt/chosen/rejected fields."""
    return {
        "prompt": "Explain gravity.",
        "chosen": "Gravity is a fundamental force.",
        "rejected": "Gravity doesn't exist.",
    }


@pytest.fixture
def sample_records() -> list[PreferenceRecord]:
    return [
        PreferenceRecord(
            id="0", prompt="Q1", chosen="A1", rejected="B1", source="test"
        ),
        PreferenceRecord(
            id="1", prompt="Q2", chosen="A2", rejected="B2", source="test"
        ),
    ]


@pytest.fixture
def mock_dataset(anthropic_hh_row: dict) -> Dataset:
    """A small HF Dataset mimicking Anthropic HH format."""
    return Dataset.from_list([anthropic_hh_row] * 5)


# --- Formatter tests ---


class TestAnthropicHHFormatter:
    def test_basic_format(self, anthropic_hh_row: dict) -> None:
        record = AnthropicHHFormatter.format(anthropic_hh_row, record_id="test-1")
        assert record.prompt == "What is the capital of France?"
        assert record.chosen == "The capital of France is Paris."
        assert record.rejected == "I'm not sure about that."
        assert record.source == "anthropic_hh"

    def test_multi_turn_conversation(self) -> None:
        row = {
            "chosen": (
                "\n\nHuman: Hi"
                "\n\nAssistant: Hello!"
                "\n\nHuman: What is Python?"
                "\n\nAssistant: Python is a programming language."
            ),
            "rejected": (
                "\n\nHuman: Hi"
                "\n\nAssistant: Hello!"
                "\n\nHuman: What is Python?"
                "\n\nAssistant: I don't know."
            ),
        }
        record = AnthropicHHFormatter.format(row)
        assert record.prompt == "What is Python?"
        assert record.chosen == "Python is a programming language."
        assert record.rejected == "I don't know."

    def test_empty_response_raises(self) -> None:
        row = {"chosen": "", "rejected": ""}
        with pytest.raises(ValueError, match="could not extract responses"):
            AnthropicHHFormatter.format(row)


class TestStandardFormatter:
    def test_basic_format(self, standard_row: dict) -> None:
        record = StandardFormatter.format(standard_row, record_id="s-1")
        assert record.prompt == "Explain gravity."
        assert record.chosen == "Gravity is a fundamental force."
        assert record.source == "standard"

    def test_missing_field_raises(self) -> None:
        with pytest.raises(KeyError):
            StandardFormatter.format({"prompt": "test"})


class TestFormatterRegistry:
    def test_get_anthropic(self) -> None:
        assert get_formatter("anthropic_hh") is AnthropicHHFormatter

    def test_get_standard(self) -> None:
        assert get_formatter("standard") is StandardFormatter

    def test_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown formatter"):
            get_formatter("nonexistent")


# --- Pipeline helper tests ---


class TestHelperFunctions:
    def test_extract_prompt_explicit(self) -> None:
        row = {"prompt": "Hello", "chosen": "World"}
        assert _extract_prompt(row) == "Hello"

    def test_extract_prompt_from_conversation(self) -> None:
        row = {
            "chosen": "\n\nHuman: What is AI?\n\nAssistant: It's intelligence.",
        }
        assert _extract_prompt(row) == "What is AI?"

    def test_get_last_human_turn_single(self) -> None:
        conv = "\n\nHuman: Hello\n\nAssistant: Hi there"
        assert _get_last_human_turn(conv) == "Hello"

    def test_get_last_human_turn_multi(self) -> None:
        conv = (
            "\n\nHuman: Hi"
            "\n\nAssistant: Hello"
            "\n\nHuman: How are you?"
            "\n\nAssistant: I'm fine"
        )
        assert _get_last_human_turn(conv) == "How are you?"

    def test_extract_response_with_assistant_prefix(self) -> None:
        text = "\n\nHuman: Q\n\nAssistant: The answer is 42."
        assert _extract_response(text) == "The answer is 42."

    def test_extract_response_plain(self) -> None:
        assert _extract_response("Just a response") == "Just a response"


# --- Pipeline integration tests ---


class TestDataPipeline:
    def test_validate_valid_dataset(self, mock_dataset: Dataset) -> None:
        pipeline = DataPipeline()
        records = pipeline.validate(mock_dataset)
        assert len(records) == 5
        assert all(isinstance(r, PreferenceRecord) for r in records)

    def test_validate_empty_raises(self) -> None:
        pipeline = DataPipeline()
        empty = Dataset.from_list([{"other": "data"}])
        with pytest.raises(ValueError, match="No valid records"):
            pipeline.validate(empty)

    def test_format_for_dpo(self, sample_records: list[PreferenceRecord]) -> None:
        pipeline = DataPipeline()
        dataset = pipeline.format_for_dpo(sample_records)
        assert len(dataset) == 2
        assert set(dataset.column_names) == {"prompt", "chosen", "rejected"}
        assert dataset[0]["prompt"] == "Q1"

    def test_create_manifest(self, sample_records: list[PreferenceRecord]) -> None:
        pipeline = DataPipeline()
        manifest = pipeline.create_manifest("test_ds", sample_records)
        assert manifest.name == "test_ds"
        assert manifest.num_records == 2
        assert len(manifest.checksum) == 16

    def test_manifest_deterministic(
        self, sample_records: list[PreferenceRecord]
    ) -> None:
        pipeline = DataPipeline()
        m1 = pipeline.create_manifest("ds", sample_records)
        m2 = pipeline.create_manifest("ds", sample_records)
        assert m1.checksum == m2.checksum

    def test_save_manifest(self, sample_records: list[PreferenceRecord]) -> None:
        pipeline = DataPipeline()
        manifest = pipeline.create_manifest("test_ds", sample_records)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pipeline.save_manifest(manifest, tmpdir)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["name"] == "test_ds"
            assert data["num_records"] == 2

    def test_max_samples(self) -> None:
        pipeline = DataPipeline(max_samples=3)
        assert pipeline.max_samples == 3
