"""Training callbacks for emitting telemetry events.

Integrates with the TRL/Transformers Trainer callback system to
capture metrics at each training step without coupling to the UI.
"""

from __future__ import annotations

import logging
from typing import Any

from src.contracts.events import RunEvent
from src.telemetry.events import EventWriter

logger = logging.getLogger(__name__)


class TelemetryCallback:
    """Trainer callback that emits RunEvents to an EventWriter.

    Inherits from TrainerCallback at runtime when transformers is
    available. Works as standalone for testing otherwise.

    Args:
        event_writer: EventWriter for persisting telemetry events.
        run_id: Unique run identifier.
    """

    def __init__(self, event_writer: EventWriter, run_id: str) -> None:
        self.event_writer = event_writer
        self.run_id = run_id

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Called by Trainer after logging metrics.

        Args:
            args: Training arguments.
            state: Current trainer state with step info.
            control: Trainer control object.
            logs: Dict of logged metrics for this step.
        """
        if not logs:
            return

        step = getattr(state, "global_step", 0) if state else 0

        event = RunEvent(
            run_id=self.run_id,
            step=step,
            loss=logs.get("loss", 0.0),
            learning_rate=logs.get("learning_rate", 0.0),
            reward_margin=logs.get("rewards/margins", None),
            extras={
                k: v
                for k, v in logs.items()
                if k not in {"loss", "learning_rate", "rewards/margins"}
            },
        )
        self.event_writer.write(event)


def create_telemetry_callback(log_dir: str, run_id: str) -> TelemetryCallback:
    """Factory for creating a TelemetryCallback with its EventWriter.

    Args:
        log_dir: Directory for telemetry logs.
        run_id: Unique run identifier.

    Returns:
        Configured TelemetryCallback ready for Trainer.
    """
    writer = EventWriter(log_dir, run_id)
    return TelemetryCallback(writer, run_id)
