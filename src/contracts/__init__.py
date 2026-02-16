"""Contracts: type definitions and protocols for the framework.

This package is the single source of truth for all data structures,
configuration models, and protocol definitions. All other packages
import types from here.

Zero internal dependencies.
"""

from src.contracts.algorithms import AlgorithmPlugin
from src.contracts.data import DatasetManifest, PreferenceRecord
from src.contracts.events import RunEvent
from src.contracts.training import EvalMetrics, StepMetrics, TrainConfig

__all__ = [
    "AlgorithmPlugin",
    "DatasetManifest",
    "EvalMetrics",
    "PreferenceRecord",
    "RunEvent",
    "StepMetrics",
    "TrainConfig",
]
