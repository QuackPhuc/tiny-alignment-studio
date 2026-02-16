"""Algorithm implementations for alignment training."""

from src.core.algorithms import dpo, ppo  # noqa: F401 — trigger @register
from src.core.algorithms.registry import AlgorithmRegistry

__all__ = ["AlgorithmRegistry"]
