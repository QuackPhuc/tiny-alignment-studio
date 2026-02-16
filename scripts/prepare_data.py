"""CLI entry point for data preparation.

Usage:
    python scripts/prepare_data.py --source Anthropic/hh-rlhf --max-samples 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.data.pipeline import DataPipeline
from src.utils.logging import setup_logger

logger = setup_logger("prepare_data")


def main() -> None:
    """Parse arguments and run data preparation pipeline."""
    parser = argparse.ArgumentParser(description="Prepare preference data")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="HuggingFace dataset ID or local path",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit records")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/data", help="Output dir"
    )
    args = parser.parse_args()

    pipeline = DataPipeline(max_samples=args.max_samples)

    logger.info("Loading: %s (split=%s)", args.source, args.split)
    dataset = pipeline.load(args.source, split=args.split)

    logger.info("Validating %d records...", len(dataset))
    records = pipeline.validate(dataset)

    logger.info("Formatting for DPO...")
    dpo_dataset = pipeline.format_for_dpo(records)
    logger.info(
        "Formatted dataset: %d records, columns=%s",
        len(dpo_dataset),
        dpo_dataset.column_names,
    )

    manifest = pipeline.create_manifest(
        name=args.source.replace("/", "_"), records=records
    )
    pipeline.save_manifest(manifest, args.output_dir)

    # Save actual data records to JSONL
    data_path = Path(args.output_dir) / f"{manifest.name}_data.jsonl"
    import json

    with open(data_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump()) + "\n")

    logger.info("Saved data to: %s", data_path)
    logger.info("Done. %d records ready in %s", manifest.num_records, args.output_dir)


if __name__ == "__main__":
    main()
