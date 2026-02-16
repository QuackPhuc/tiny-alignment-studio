"""LoRA adapter management: save, load, merge, and compare.

Provides utilities for managing trained LoRA adapters independently
of the training loop.
"""

from __future__ import annotations

import logging
from pathlib import Path

from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)


class AdapterManager:
    """Manage trained LoRA adapters.

    Supports saving, loading, merging adapters back into base models,
    and listing available adapters in an output directory.
    """

    @staticmethod
    def load_with_adapter(
        base_model_name: str,
        adapter_path: str | Path,
    ) -> PreTrainedModel:
        """Load a base model and apply a saved LoRA adapter.

        Args:
            base_model_name: HuggingFace model identifier.
            adapter_path: Path to saved LoRA adapter weights.

        Returns:
            Model with adapter applied, ready for inference.
        """
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto"
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_path))
        logger.info("Loaded adapter from: %s", adapter_path)
        return model

    @staticmethod
    def merge_and_save(
        model: PreTrainedModel,
        output_path: str | Path,
    ) -> Path:
        """Merge LoRA adapter into base model and save.

        Creates a standalone model without requiring PEFT at inference.

        Args:
            model: Model with LoRA adapter attached (PeftModel).
            output_path: Directory to save the merged model.

        Returns:
            Path to the saved merged model.
        """
        merged = model.merge_and_unload()
        save_path = Path(output_path)
        save_path.mkdir(parents=True, exist_ok=True)
        merged.save_pretrained(save_path)
        logger.info("Merged model saved to: %s", save_path)
        return save_path

    @staticmethod
    def list_adapters(output_dir: str | Path) -> list[Path]:
        """List all saved adapter directories under an output dir.

        Args:
            output_dir: Root directory to search for adapters.

        Returns:
            List of paths to adapter directories.
        """
        root = Path(output_dir)
        if not root.exists():
            return []
        return sorted(p.parent for p in root.rglob("adapter_config.json"))
