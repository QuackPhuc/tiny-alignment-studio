"""Unified model loader with QLoRA support.

Handles loading base models with optional 4-bit quantization and
attaching LoRA adapters for parameter-efficient fine-tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

if TYPE_CHECKING:
    from src.contracts.training import TrainConfig

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model and its tokenizer."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    is_quantized: bool
    has_adapter: bool


class ModelLoader:
    """Load and configure models for alignment training.

    Supports QLoRA (4-bit quantization + LoRA) for consumer GPU training.
    """

    @staticmethod
    def load(config: TrainConfig) -> LoadedModel:
        """Load a model and tokenizer based on training config.

        Args:
            config: Training configuration with model and adapter settings.

        Returns:
            LoadedModel with configured model and tokenizer.
        """
        tokenizer = _load_tokenizer(config.model_name)
        quantize = config.quantization_bits in (4, 8)

        if quantize:
            model = _load_quantized_model(config.model_name, config.quantization_bits)
        else:
            model = _load_base_model(config.model_name)

        has_adapter = False
        if config.adapter_type == "lora":
            model = _attach_lora(model, quantize)
            has_adapter = True

        _log_model_info(model, quantize, has_adapter)
        return LoadedModel(
            model=model,
            tokenizer=tokenizer,
            is_quantized=quantize,
            has_adapter=has_adapter,
        )


def _load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and configure tokenizer with padding token.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Configured tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _load_quantized_model(model_name: str, bits: int) -> PreTrainedModel:
    """Load a model with BitsAndBytes quantization.

    Args:
        model_name: HuggingFace model identifier.
        bits: Quantization level (4 or 8).

    Returns:
        Quantized model ready for adapter attachment.
    """
    compute_dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        compute_dtype = torch.float16
        logger.info("bfloat16 not supported, falling back to float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=(bits == 4),
        load_in_8bit=(bits == 8),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)
    return model


def _load_base_model(model_name: str) -> PreTrainedModel:
    """Load a model without quantization (full precision or bf16).

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Full-precision model.
    """
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )


def _attach_lora(model: PreTrainedModel, is_quantized: bool) -> PreTrainedModel:
    """Attach a LoRA adapter to the model.

    Args:
        model: Base or quantized model.
        is_quantized: Whether the model is quantized (affects config).

    Returns:
        Model with LoRA adapter attached.
    """
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        "LoRA attached: %d/%d trainable params (%.2f%%)",
        trainable,
        total,
        100 * trainable / total,
    )
    return model


def _log_model_info(model: PreTrainedModel, quantized: bool, adapter: bool) -> None:
    """Log model configuration summary.

    Args:
        model: Loaded model.
        quantized: Whether quantization was applied.
        adapter: Whether LoRA adapter is attached.
    """
    logger.info(
        "Model loaded: quantized=%s, adapter=%s, device=%s",
        quantized,
        adapter,
        next(model.parameters()).device,
    )
