"""Formatters for converting raw data into preference pairs.

Each formatter handles a specific source format and normalizes it
into the PreferenceRecord structure used by the pipeline.
"""

from __future__ import annotations

from typing import Any

from src.contracts.data import PreferenceRecord


class AnthropicHHFormatter:
    """Format Anthropic HH-RLHF data into preference pairs.

    Anthropic HH conversations use "\\n\\nHuman: " and "\\n\\nAssistant: "
    as turn delimiters. Each record has 'chosen' and 'rejected'
    conversation transcripts sharing the same prompt.
    """

    @staticmethod
    def format(raw_record: dict[str, Any], record_id: str = "0") -> PreferenceRecord:
        """Convert a raw Anthropic HH record to a PreferenceRecord.

        Args:
            raw_record: Single record with 'chosen' and 'rejected' keys,
                each containing a full conversation transcript.
            record_id: Unique identifier for this record.

        Returns:
            Validated PreferenceRecord.

        Raises:
            ValueError: If neither chosen nor rejected can be parsed.
        """
        chosen_text = raw_record.get("chosen", "")
        rejected_text = raw_record.get("rejected", "")

        prompt = _extract_shared_prompt(chosen_text, rejected_text)
        chosen_response = _extract_last_assistant(chosen_text)
        rejected_response = _extract_last_assistant(rejected_text)

        if not chosen_response or not rejected_response:
            msg = f"Record {record_id}: could not extract responses"
            raise ValueError(msg)

        return PreferenceRecord(
            id=record_id,
            prompt=prompt,
            chosen=chosen_response,
            rejected=rejected_response,
            source="anthropic_hh",
        )


class StandardFormatter:
    """Format datasets that already have prompt/chosen/rejected columns."""

    @staticmethod
    def format(raw_record: dict[str, Any], record_id: str = "0") -> PreferenceRecord:
        """Convert a record with explicit prompt/chosen/rejected fields.

        Args:
            raw_record: Record with 'prompt', 'chosen', 'rejected' keys.
            record_id: Unique identifier for this record.

        Returns:
            Validated PreferenceRecord.
        """
        return PreferenceRecord(
            id=record_id,
            prompt=raw_record["prompt"],
            chosen=raw_record["chosen"],
            rejected=raw_record["rejected"],
            source=raw_record.get("source", "standard"),
        )


FORMATTERS: dict[str, type] = {
    "anthropic_hh": AnthropicHHFormatter,
    "standard": StandardFormatter,
}


def get_formatter(name: str) -> type:
    """Look up a formatter by name.

    Args:
        name: Formatter identifier.

    Raises:
        KeyError: If no formatter exists with that name.
    """
    if name not in FORMATTERS:
        available = ", ".join(sorted(FORMATTERS))
        msg = f"Unknown formatter '{name}'. Available: {available}"
        raise KeyError(msg)
    return FORMATTERS[name]


def _extract_shared_prompt(chosen: str, rejected: str) -> str:
    """Extract the shared human prompt from two conversation branches.

    Both chosen and rejected conversations share the same prompt
    prefix. This function finds the common human turn.

    Args:
        chosen: Full chosen conversation transcript.
        rejected: Full rejected conversation transcript.

    Returns:
        The shared human prompt.
    """
    chosen_parts = chosen.split("\n\nHuman: ")
    rejected_parts = rejected.split("\n\nHuman: ")

    if len(chosen_parts) < 2:
        return chosen.strip()

    # Find last shared human turn
    last_human = chosen_parts[-1]
    assistant_idx = last_human.find("\n\nAssistant: ")
    if assistant_idx != -1:
        last_human = last_human[:assistant_idx]

    return last_human.strip()


def _extract_last_assistant(conversation: str) -> str:
    """Extract the last assistant response from a conversation.

    Args:
        conversation: Full conversation with Human/Assistant turns.

    Returns:
        The final assistant response text.
    """
    parts = conversation.split("\n\nAssistant: ")
    if len(parts) < 2:
        return conversation.strip()
    return parts[-1].strip()
