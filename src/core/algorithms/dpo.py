"""DPO (Direct Preference Optimization) algorithm implementation.

Wraps TRL's DPOTrainer with configuration from TrainConfig
and integrates with the AlgorithmPlugin protocol.
"""

from __future__ import annotations

import logging
from typing import Any

from src.contracts.algorithms import AlgorithmPlugin
from src.core.algorithms.registry import register

logger = logging.getLogger(__name__)


@register("dpo")
class DPOAlgorithm(AlgorithmPlugin):
    """DPO alignment using TRL's DPOTrainer.

    Optimizes directly on preference pairs without a separate reward model.
    """

    def get_trainer_class(self) -> type:
        """Return TRL's DPOTrainer class."""
        from trl import DPOTrainer

        return DPOTrainer

    def create_training_args(self, config: dict[str, Any]) -> Any:
        """Build DPO-specific training arguments from config.

        Args:
            config: Flat config dict with training parameters.

        Returns:
            Configured DPOConfig for the trainer.
        """
        from trl import DPOConfig

        training = config.get("training", {})
        dpo = config.get("dpo", {})

        return DPOConfig(
            output_dir=training.get("output_dir", "outputs"),
            num_train_epochs=training.get("num_epochs", 1),
            per_device_train_batch_size=training.get("batch_size", 4),
            gradient_accumulation_steps=training.get("gradient_accumulation_steps", 4),
            learning_rate=training.get("learning_rate", 5e-5),
            warmup_ratio=training.get("warmup_ratio", 0.1),
            weight_decay=training.get("weight_decay", 0.01),
            max_grad_norm=training.get("max_grad_norm", 1.0),
            bf16=training.get("bf16", True),
            fp16=training.get("fp16", False),
            logging_steps=training.get("logging_steps", 10),
            eval_steps=training.get("eval_steps", 100),
            save_steps=training.get("save_steps", 200),
            seed=training.get("seed", 42),
            beta=dpo.get("beta", 0.1),
            loss_type=dpo.get("loss_type", "sigmoid"),
            max_length=config.get("model", {}).get("max_length", 512),
            max_prompt_length=config.get("data", {})
            .get("preprocessing", {})
            .get("max_prompt_length", 256),
            remove_unused_columns=False,
        )

    def compute_loss(self, batch: dict[str, Any]) -> Any:
        """Not used directly; DPOTrainer handles loss computation."""
        raise NotImplementedError("DPOTrainer handles loss internally")

    def evaluate(self, eval_dataloader: Any) -> Any:
        """Not used directly; DPOTrainer handles evaluation."""
        raise NotImplementedError("DPOTrainer handles eval internally")

    @property
    def required_data_format(self) -> str:
        return "preference_pairs"
