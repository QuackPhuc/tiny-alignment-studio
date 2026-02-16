"""CLI entry point for training.

Usage:
    python scripts/train.py --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.trainer import AlignmentTrainer
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

    result = trainer.train()
    logger.info("Training complete!")
    logger.info("  Loss: %.4f", result["training_loss"])
    logger.info("  Output: %s", result["output_dir"])
    logger.info("  Adapter: %s", result["adapter_dir"])
    logger.info("  Records: %d", result["num_records"])


if __name__ == "__main__":
    main()
