"""PPO (Proximal Policy Optimization) algorithm implementation.

Uses TRL's PPOConfig and PPOTrainer for RLHF-style training
with a reward model scoring generated responses.
"""

from __future__ import annotations

from typing import Any

from src.contracts.algorithms import AlgorithmPlugin
from src.core.algorithms.registry import register


@register("ppo")
class PPOAlgorithm(AlgorithmPlugin):
    """PPO alignment using TRL's PPOTrainer.

    Unlike DPO, PPO requires:
    1. A policy model that generates responses
    2. A reward model that scores those responses
    3. A KL penalty to prevent the policy from diverging too far
    """

    def get_trainer_class(self) -> type:
        """Return TRL's PPOTrainer class."""
        from trl import PPOTrainer

        return PPOTrainer

    def create_training_args(self, config: dict[str, Any]) -> Any:
        """Build PPO-specific training arguments.

        Args:
            config: Raw YAML config dict with ppo section.

        Returns:
            PPOConfig instance.
        """
        from trl import PPOConfig

        training = config.get("training", {})
        ppo = config.get("ppo", {})

        return PPOConfig(
            learning_rate=training.get("learning_rate", 1.41e-5),
            batch_size=training.get("batch_size", 4),
            mini_batch_size=ppo.get("mini_batch_size", 1),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            ppo_epochs=ppo.get("ppo_epochs", 4),
            init_kl_coeff=ppo.get("init_kl_coeff", 0.2),
            target_kl=ppo.get("target_kl", 6.0),
            seed=training.get("seed", 42),
            log_with=None,
        )

    def compute_loss(self, batch: dict[str, Any]) -> Any:
        """PPO loss is computed internally by PPOTrainer.step()."""
        raise NotImplementedError(
            "PPO loss is computed by PPOTrainer.step(), not directly"
        )

    def evaluate(self, eval_dataloader: Any) -> Any:
        """Evaluation for PPO uses reward model scoring."""
        raise NotImplementedError("PPO evaluation requires reward model scoring")

    @property
    def required_data_format(self) -> str:
        return "prompt_completion"
