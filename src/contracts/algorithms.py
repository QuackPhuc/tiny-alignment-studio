"""Algorithm plugin protocol for swappable alignment methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.contracts.training import EvalMetrics, StepMetrics


class AlgorithmPlugin(ABC):
    """Protocol for alignment algorithm implementations.

    Each algorithm (DPO, PPO, etc.) implements this interface.
    The AlignmentTrainer uses it to remain algorithm-agnostic.
    """

    @abstractmethod
    def get_trainer_class(self) -> type:
        """Return the TRL trainer class for this algorithm."""
        ...

    @abstractmethod
    def compute_loss(self, batch: dict[str, Any]) -> StepMetrics:
        """Compute alignment loss for a single batch."""
        ...

    @abstractmethod
    def evaluate(self, eval_dataloader: Any) -> EvalMetrics:
        """Run evaluation and return aggregated metrics."""
        ...

    @property
    @abstractmethod
    def required_data_format(self) -> str:
        """Data format this algorithm expects (e.g., 'preference_pairs')."""
        ...
