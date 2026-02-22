"""Unit tests for PPO algorithm registration and config creation."""

from __future__ import annotations

import pytest

from src.core.algorithms import AlgorithmRegistry


class TestPPORegistration:
    """Test PPO algorithm plugin registration."""

    def test_ppo_registered(self):
        algo = AlgorithmRegistry.get("ppo")
        assert algo is not None

    def test_ppo_data_format(self):
        algo = AlgorithmRegistry.get("ppo")
        assert algo.required_data_format == "prompt_completion"

    def test_ppo_trainer_class(self):
        """Trainer class import should succeed if trl is installed."""
        algo = AlgorithmRegistry.get("ppo")
        try:
            cls = algo.get_trainer_class()
            assert cls.__name__ == "PPOTrainer"
        except ImportError:
            pytest.skip("trl not available")

    def test_ppo_training_args(self):
        """PPOConfig creation from raw config dict."""
        algo = AlgorithmRegistry.get("ppo")
        config = {
            "training": {
                "learning_rate": 1.41e-5,
                "batch_size": 4,
                "gradient_accumulation_steps": 4,
                "seed": 42,
            },
            "ppo": {
                "mini_batch_size": 1,
                "ppo_epochs": 4,
                "init_kl_coeff": 0.2,
                "target_kl": 6.0,
            },
        }
        try:
            args = algo.create_training_args(config)
            assert args.learning_rate == 1.41e-5
            assert args.batch_size == 4
        except ImportError:
            pytest.skip("trl not available")

    def test_ppo_compute_loss_raises(self):
        algo = AlgorithmRegistry.get("ppo")
        with pytest.raises(NotImplementedError):
            algo.compute_loss({})

    def test_ppo_evaluate_raises(self):
        algo = AlgorithmRegistry.get("ppo")
        with pytest.raises(NotImplementedError):
            algo.evaluate(None)
