"""Tiny Alignment Studio CLI entry point.

Provides the `tas` command with subcommands for training,
evaluation, and data preparation.

Usage:
    tas train --config configs/base.yaml
    tas eval --adapter outputs/adapter --prompt "What is AI?"
    tas data prepare --source Anthropic/hh-rlhf --max-samples 1000
"""

from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("train", help="Train an alignment model")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.set_defaults(func=_run_train)


def _add_eval_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluate a trained model")
    parser.add_argument(
        "--adapter", type=str, required=True, help="Path to adapter dir"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model name",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Evaluation prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Max tokens to generate"
    )
    parser.set_defaults(func=_run_eval)


def _add_data_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("data", help="Data pipeline operations")
    data_sub = parser.add_subparsers(dest="data_cmd")

    prepare = data_sub.add_parser("prepare", help="Prepare training data")
    prepare.add_argument(
        "--source", type=str, required=True, help="HuggingFace dataset ID"
    )
    prepare.add_argument(
        "--output-dir", type=str, default="outputs/data", help="Output directory"
    )
    prepare.add_argument(
        "--max-samples", type=int, default=1000, help="Max samples to prepare"
    )
    prepare.add_argument("--split", type=str, default="train", help="Dataset split")
    prepare.set_defaults(func=_run_data_prepare)


def _run_train(args: argparse.Namespace) -> None:
    from src.core.trainer import AlignmentTrainer
    from src.telemetry.callbacks import create_telemetry_callback
    from src.utils.logging import setup_logger

    logger = setup_logger("tas.train")

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

    telemetry_config = trainer._raw_config.get("telemetry", {})
    log_dir = telemetry_config.get("log_dir", "logs")
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    callbacks = []

    if telemetry_config.get("enabled", True):
        callbacks.append(create_telemetry_callback(log_dir, run_id))
        logger.info("Telemetry: log_dir=%s, run_id=%s", log_dir, run_id)

    result = trainer.train(callbacks=callbacks)
    logger.info("Training complete! Loss: %.4f", result["training_loss"])
    logger.info("Adapter saved to: %s", result["adapter_dir"])


def _run_eval(args: argparse.Namespace) -> None:
    from src.core.inference import (
        GenerationConfig,
        generate_response,
        load_model_for_inference,
    )
    from src.utils.logging import setup_logger

    logger = setup_logger("tas.eval")

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error("Adapter not found: %s", adapter_path)
        sys.exit(1)

    logger.info("Loading model: %s + adapter: %s", args.base_model, args.adapter)
    model, tokenizer = load_model_for_inference(args.base_model, str(adapter_path))

    config = GenerationConfig(max_new_tokens=args.max_tokens)
    response = generate_response(model, tokenizer, args.prompt, config)

    logger.info("Prompt: %s", args.prompt)
    logger.info("Response: %s", response)


def _run_data_prepare(args: argparse.Namespace) -> None:
    import json

    from src.core.data.pipeline import DataPipeline
    from src.utils.logging import setup_logger

    logger = setup_logger("tas.data")

    pipeline = DataPipeline(max_samples=args.max_samples)
    logger.info(
        "Loading: %s (split=%s, max=%d)", args.source, args.split, args.max_samples
    )

    dataset = pipeline.load(args.source, split=args.split)
    logger.info("Loaded %d raw examples", len(dataset))

    records = pipeline.validate(dataset)
    logger.info("Validated %d records", len(records))

    manifest = pipeline.create_manifest(
        name=args.source.replace("/", "_"), records=records
    )
    pipeline.save_manifest(manifest, args.output_dir)

    data_path = Path(args.output_dir) / f"{manifest.name}_data.jsonl"
    with open(data_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.model_dump()) + "\n")

    logger.info("Saved %d records to: %s", manifest.num_records, data_path)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="tas",
        description="Tiny Alignment Studio — RLHF training toolkit",
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_train_parser(subparsers)
    _add_eval_parser(subparsers)
    _add_data_parser(subparsers)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
