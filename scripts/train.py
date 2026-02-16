"""CLI entry point for training.

Usage:
    python scripts/train.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trainer import AlignmentTrainer
from src.telemetry.callbacks import create_telemetry_callback
from src.utils.logging import setup_logger

logger = setup_logger("train")


def main() -> None:
    """Parse arguments and launch training."""
    parser = argparse.ArgumentParser(description="Train an alignment model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    trainer = AlignmentTrainer(config_path)
    logger.info(
        "Config loaded: algorithm=%s, model=%s",
        trainer.config.algorithm,
        trainer.config.model_name,
    )

    # Set up telemetry callback for live monitoring
    telemetry_config = trainer._raw_config.get("telemetry", {})
    log_dir = telemetry_config.get("log_dir", "logs")
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    callbacks = []

    if telemetry_config.get("enabled", True):
        telemetry_cb = create_telemetry_callback(log_dir, run_id)
        callbacks.append(telemetry_cb)
        logger.info("Telemetry enabled: log_dir=%s, run_id=%s", log_dir, run_id)

    result = trainer.train(callbacks=callbacks)
    logger.info("Training complete!")
    logger.info("  Loss: %.4f", result["training_loss"])
    logger.info("  Output: %s", result["output_dir"])
    logger.info("  Adapter: %s", result["adapter_dir"])
    logger.info("  Records: %d", result["num_records"])


if __name__ == "__main__":
    main()
