"""Data pipeline for loading, validating, and formatting preference data."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from src.contracts.data import DatasetManifest, PreferenceRecord

logger = logging.getLogger(__name__)


class DataPipeline:
    """Handles the full data lifecycle: ingest, validate, format.

    Loads preference data from HuggingFace Hub or local files,
    validates against the PreferenceRecord schema, and formats
    for the target algorithm.

    Args:
        max_samples: Optional limit on number of records to load.
    """

    def __init__(self, max_samples: int | None = None) -> None:
        self.max_samples = max_samples

    def load(
        self,
        source: str,
        split: str = "train",
        subset: str | None = None,
    ) -> Dataset:
        """Load a preference dataset from HuggingFace Hub or local path.

        Args:
            source: HuggingFace dataset identifier or local directory.
            split: Dataset split to load.
            subset: Optional dataset subset/configuration name.

        Returns:
            HuggingFace Dataset object.
        """
        logger.info("Loading dataset: source=%s, split=%s", source, split)

        location = Path(source)
        if (
            location.exists()
            and location.is_file()
            and location.suffix in (".json", ".jsonl")
        ):
            logger.info("Loading local file: %s", source)
            # For JSON/JSONL, we load it as a dataset
            data_files = {"train": str(location)}
            if split != "train":
                data_files[split] = str(location)

            dataset = load_dataset("json", data_files=data_files, split=split)
        else:
            load_kwargs: dict[str, Any] = {"path": source, "split": split}
            if subset:
                load_kwargs["name"] = subset
            dataset = load_dataset(**load_kwargs)

        if self.max_samples and len(dataset) > self.max_samples:
            dataset = dataset.select(range(self.max_samples))
            logger.info("Truncated to %d samples", self.max_samples)

        logger.info("Loaded %d records", len(dataset))
        return dataset

    def validate(self, dataset: Dataset) -> list[PreferenceRecord]:
        """Validate dataset records against PreferenceRecord schema.

        Skips records that fail validation and logs warnings.

        Args:
            dataset: Raw loaded dataset with 'chosen' and 'rejected' fields.

        Returns:
            List of validated PreferenceRecord objects.

        Raises:
            ValueError: If no valid records remain after filtering.
        """
        valid_records: list[PreferenceRecord] = []
        errors = 0

        for idx, row in enumerate(dataset):
            try:
                prompt = _extract_prompt(row)
                chosen = _extract_response(row.get("chosen", ""))
                rejected = _extract_response(row.get("rejected", ""))

                if not prompt or not chosen or not rejected:
                    raise ValueError("Empty prompt, chosen, or rejected")

                record = PreferenceRecord(
                    id=str(idx),
                    prompt=prompt,
                    chosen=chosen,
                    rejected=rejected,
                    source="pipeline",
                )
                valid_records.append(record)
            except Exception:
                errors += 1
                if errors <= 5:
                    logger.warning("Skipping invalid record at index %d", idx)

        if not valid_records:
            msg = f"No valid records found. {errors} records failed validation."
            raise ValueError(msg)

        if errors:
            logger.warning(
                "Validation complete: %d valid, %d skipped",
                len(valid_records),
                errors,
            )
        else:
            logger.info("All %d records valid", len(valid_records))

        return valid_records

    def format_for_dpo(self, records: list[PreferenceRecord]) -> Dataset:
        """Format validated records for DPO training.

        DPO expects a dataset with 'prompt', 'chosen', 'rejected' columns.

        Args:
            records: Validated preference records.

        Returns:
            HuggingFace Dataset formatted for DPOTrainer.
        """
        formatted = [
            {
                "prompt": r.prompt,
                "chosen": r.chosen,
                "rejected": r.rejected,
            }
            for r in records
        ]
        return Dataset.from_list(formatted)

    def create_manifest(
        self,
        name: str,
        records: list[PreferenceRecord],
        version: str = "1.0",
    ) -> DatasetManifest:
        """Create a manifest for a processed dataset.

        Args:
            name: Dataset name.
            records: Processed records to compute checksum from.
            version: Version string.

        Returns:
            DatasetManifest with checksum for reproducibility.
        """
        content = json.dumps(
            [r.model_dump() for r in records],
            sort_keys=True,
            default=str,
        )
        checksum = hashlib.sha256(content.encode()).hexdigest()[:16]

        return DatasetManifest(
            name=name,
            version=version,
            num_records=len(records),
            checksum=checksum,
        )

    def save_manifest(self, manifest: DatasetManifest, output_dir: str | Path) -> Path:
        """Save a dataset manifest to disk.

        Args:
            manifest: Manifest to persist.
            output_dir: Directory to write the manifest file.

        Returns:
            Path to the saved manifest JSON file.
        """
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / f"{manifest.name}_manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Saved manifest: %s", manifest_path)
        return manifest_path


def _extract_prompt(row: dict[str, Any]) -> str:
    """Extract the prompt from a dataset row.

    Handles both Anthropic HH format (conversation in 'chosen')
    and standard format (explicit 'prompt' field).

    Args:
        row: Single dataset row.

    Returns:
        Extracted prompt string.
    """
    if "prompt" in row and row["prompt"]:
        return row["prompt"]

    # Anthropic HH: prompt is embedded in the 'chosen' conversation
    chosen = row.get("chosen", "")
    return _get_last_human_turn(chosen)


def _get_last_human_turn(conversation: str) -> str:
    """Extract the last human turn from an Anthropic HH conversation.

    Anthropic HH format uses "\\n\\nHuman: " and "\\n\\nAssistant: "
    as turn delimiters.

    Args:
        conversation: Full conversation string.

    Returns:
        The last human message in the conversation.
    """
    parts = conversation.split("\n\nHuman: ")
    if len(parts) < 2:
        return conversation.strip()

    last_human = parts[-1]
    # Remove assistant response that follows
    assistant_idx = last_human.find("\n\nAssistant: ")
    if assistant_idx != -1:
        last_human = last_human[:assistant_idx]

    return last_human.strip()


def _extract_response(text: str) -> str:
    """Extract the assistant response from a conversation string.

    Args:
        text: Full conversation or standalone response.

    Returns:
        The last assistant response, stripped of formatting.
    """
    parts = text.split("\n\nAssistant: ")
    if len(parts) < 2:
        return text.strip()
    return parts[-1].strip()
