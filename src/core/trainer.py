"""AlignmentTrainer: main orchestrator for the training pipeline.

Receives a config, creates all components (algorithm, data pipeline,
model loader), and runs the training loop with telemetry callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from src.core.algorithms import AlgorithmRegistry
from src.core.data.pipeline import DataPipeline
from src.core.models.loader import ModelLoader
from src.utils.config import load_config

logger = logging.getLogger(__name__)


class AlignmentTrainer:
    """Orchestrates the full alignment training pipeline.

    Args:
        config_path: Path to YAML config file.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        self.config = load_config(self.config_path)
        self._raw_config = self._load_raw_config()

    def train(self, callbacks: list | None = None) -> dict[str, Any]:
        """Run the full training loop.

        Args:
            callbacks: Optional list of Trainer callbacks.

        Returns:
            Training results including metrics and output path.
        """
        logger.info("Starting training: algorithm=%s", self.config.algorithm)

        algorithm = AlgorithmRegistry.get(self.config.algorithm)
        logger.info("Algorithm: %s", self.config.algorithm)

        logger.info("Loading data from: %s", self._data_source)
        pipeline = DataPipeline(max_samples=self._max_samples)
        raw_dataset = pipeline.load(
            source=self._data_source,
            split=self._raw_config.get("data", {}).get("train_split", "train"),
        )
        records = pipeline.validate(raw_dataset)
        train_dataset = pipeline.format_for_dpo(records)

        manifest = pipeline.create_manifest(
            name=self._data_source.replace("/", "_"),
            records=records,
        )
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_manifest(manifest, output_dir)

        logger.info("Loading model: %s", self.config.model_name)
        loaded = ModelLoader.load(self.config)

        training_args = algorithm.create_training_args(self._raw_config)
        trainer_cls = algorithm.get_trainer_class()

        trainer_kwargs: dict[str, Any] = {
            "model": loaded.model,
            "args": training_args,
            "train_dataset": train_dataset,
            "processing_class": loaded.tokenizer,
        }
        if callbacks:
            trainer_kwargs["callbacks"] = callbacks

        trainer = trainer_cls(**trainer_kwargs)

        logger.info("Training started")
        result = trainer.train()
        logger.info("Training complete: loss=%.4f", result.training_loss)

        loaded.model.save_pretrained(output_dir / "adapter")
        loaded.tokenizer.save_pretrained(output_dir / "adapter")
        logger.info("Adapter saved to: %s", output_dir / "adapter")

        return {
            "training_loss": result.training_loss,
            "output_dir": str(output_dir),
            "adapter_dir": str(output_dir / "adapter"),
            "num_records": manifest.num_records,
            "checksum": manifest.checksum,
        }

    @property
    def _data_source(self) -> str:
        return self._raw_config.get("data", {}).get("source", "")

    @property
    def _max_samples(self) -> int | None:
        return self._raw_config.get("data", {}).get("max_samples")

    def _load_raw_config(self) -> dict:
        """Load raw YAML for algorithm-specific config access."""
        with self.config_path.open() as f:
            return yaml.safe_load(f)
