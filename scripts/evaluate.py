"""CLI entry point for model evaluation.

Usage:
    python scripts/evaluate.py --adapter outputs/adapter --prompt "What is AI?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging import setup_logger

logger = setup_logger("evaluate")


def main() -> None:
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate an aligned model")
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
    args = parser.parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        logger.error("Adapter not found: %s", adapter_path)
        sys.exit(1)

    logger.info("Loading model: %s + adapter: %s", args.base_model, args.adapter)

    # Lazy import to avoid loading torch at CLI parse time
    from src.core.models.adapters import AdapterManager

    model = AdapterManager.load_with_adapter(args.base_model, adapter_path)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    logger.info("Prompt: %s", args.prompt)
    logger.info("Response: %s", response)


if __name__ == "__main__":
    main()
