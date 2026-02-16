"""Data contracts for preference datasets."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PreferenceRecord(BaseModel):
    """A single preference pair for alignment training.

    Represents one human preference judgment: given a prompt, the `chosen`
    response is preferred over the `rejected` response.
    """

    id: str = Field(description="Unique record identifier")
    prompt: str = Field(description="Input prompt shown to annotators")
    chosen: str = Field(description="Preferred response")
    rejected: str = Field(description="Dispreferred response")
    source: str = Field(default="unknown", description="Dataset origin")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    """Metadata for a processed preference dataset.

    Created after data pipeline processing to track provenance
    and enable reproducibility.
    """

    name: str
    version: str
    num_records: int = Field(ge=0)
    schema_version: str = "1.0"
    checksum: str = Field(description="SHA-256 of the processed dataset")
