"""Telemetry event contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class RunEvent(BaseModel):
    """Event emitted during training for real-time monitoring.

    The UI and telemetry systems consume these events to render
    dashboards and persist training history.
    """

    timestamp: datetime = Field(default_factory=datetime.now)
    run_id: str
    step: int = Field(ge=0)
    loss: float
    reward_margin: float | None = None
    learning_rate: float = Field(ge=0)
    gpu_memory_mb: float | None = None
    tokens_per_second: float | None = None
    extras: dict[str, Any] = Field(default_factory=dict)
