"""Inference utilities for generating text from loaded models.

Shared by the Arena UI and evaluation scripts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Parameters for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


def load_model_for_inference(
    model_name: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load a model and tokenizer for inference.

    Args:
        model_name: HuggingFace model identifier.
        adapter_path: Optional path to LoRA adapter directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_path and Path(adapter_path).exists():
        from peft import PeftModel

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model.eval()
        logger.info("Loaded model with adapter: %s", adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model.eval()
        logger.info("Loaded base model: %s", model_name)

    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    prompt: str,
    config: GenerationConfig | None = None,
) -> str:
    """Generate a text response from a model.

    Args:
        model: Loaded model (base or with adapter).
        tokenizer: Corresponding tokenizer.
        prompt: User prompt to respond to.
        config: Generation parameters.

    Returns:
        Generated response text.
    """
    import torch

    if config is None:
        config = GenerationConfig()

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=config.do_sample,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the new tokens (exclude prompt)
    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    return response.strip()
