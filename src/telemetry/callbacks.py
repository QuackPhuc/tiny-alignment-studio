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

try:
    from transformers import TrainerCallback

    _BASE_CLASS = TrainerCallback
except ImportError:
    _BASE_CLASS = object


class TelemetryCallback(_BASE_CLASS):
    """Trainer callback that emits RunEvents to an EventWriter.

    Inherits from TrainerCallback when transformers is available,
    enabling automatic integration with the Trainer loop.

    Args:
        event_writer: EventWriter for persisting telemetry events.
        run_id: Unique run identifier.
    """

    def __init__(self, event_writer: EventWriter, run_id: str) -> None:
        if _BASE_CLASS is not object:
            super().__init__()
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


class WandbCallback(_BASE_CLASS):
    """Optional callback that logs metrics to Weights & Biases.

    Only active when wandb is installed and configured.

    Args:
        project: W&B project name.
        run_name: Display name for the run.
        config: Training configuration dict to log.
    """

    def __init__(
        self,
        project: str = "tiny-alignment-studio",
        run_name: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if _BASE_CLASS is not object:
            super().__init__()
        self._project = project
        self._run_name = run_name
        self._config = config
        self._run = None

    def on_train_begin(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize W&B run at training start."""
        try:
            import wandb

            self._run = wandb.init(
                project=self._project,
                name=self._run_name,
                config=self._config,
                reinit=True,
            )
            logger.info("Wandb run started: %s/%s", self._project, self._run_name)
        except ImportError:
            logger.warning("wandb not installed, skipping W&B logging")
        except Exception:
            logger.warning("wandb init failed, skipping W&B logging", exc_info=True)

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log metrics to W&B."""
        if not logs or self._run is None:
            return

        step = getattr(state, "global_step", 0) if state else 0
        self._run.log(logs, step=step)

    def on_train_end(
        self,
        args: Any,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> None:
        """Finish W&B run at training end."""
        if self._run is not None:
            self._run.finish()
            logger.info("Wandb run finished")


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


def create_wandb_callback(
    config: dict[str, Any],
    run_name: str | None = None,
) -> WandbCallback | None:
    """Factory for creating a WandbCallback if enabled in config.

    Args:
        config: Raw YAML config dict.
        run_name: Optional display name for the run.

    Returns:
        WandbCallback if wandb is configured, None otherwise.
    """
    wandb_config = config.get("telemetry", {}).get("wandb", {})
    if not wandb_config.get("enabled", False):
        return None

    project = wandb_config.get("project", "tiny-alignment-studio")
    return WandbCallback(project=project, run_name=run_name, config=config)
