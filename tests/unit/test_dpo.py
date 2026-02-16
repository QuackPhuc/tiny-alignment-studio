"""Tests for DPO algorithm."""

from __future__ import annotations

from src.core.algorithms.registry import AlgorithmRegistry


class TestDPORegistration:
    def test_dpo_registered(self) -> None:
        assert "dpo" in AlgorithmRegistry.available()

    def test_ppo_registered(self) -> None:
        assert "ppo" in AlgorithmRegistry.available()

    def test_dpo_data_format(self) -> None:
        algo = AlgorithmRegistry.get("dpo")
        assert algo.required_data_format == "preference_pairs"

    def test_unknown_algorithm_raises(self) -> None:
        import pytest

        with pytest.raises(KeyError, match="Unknown algorithm"):
            AlgorithmRegistry.get("nonexistent")
