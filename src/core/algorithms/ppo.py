"""PPO (Proximal Policy Optimization) algorithm implementation.

Placeholder for future PPO implementation using TRL's PPOTrainer.
"""

from __future__ import annotations

from typing import Any

from src.contracts.algorithms import AlgorithmPlugin
from src.core.algorithms.registry import register


@register("ppo")
class PPOAlgorithm(AlgorithmPlugin):
    """PPO alignment using TRL's PPOTrainer (not yet implemented).

    Requires a separate reward model for scoring responses.
    """

    def get_trainer_class(self) -> type:
        """Return TRL's PPOTrainer class."""
        from trl import PPOTrainer

        return PPOTrainer

    def create_training_args(self, config: dict[str, Any]) -> Any:
        """Build PPO-specific training arguments."""
        raise NotImplementedError("PPO training args not yet implemented")

    def compute_loss(self, batch: dict[str, Any]) -> Any:
        raise NotImplementedError

    def evaluate(self, eval_dataloader: Any) -> Any:
        raise NotImplementedError

    @property
    def required_data_format(self) -> str:
        return "prompt_completion"
