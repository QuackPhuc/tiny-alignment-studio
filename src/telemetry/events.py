"""Event writer and reader for training telemetry.

Persists RunEvent objects to disk as JSONL for consumption by
the Streamlit dashboard and post-training analysis.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

from src.contracts.events import RunEvent

logger = logging.getLogger(__name__)


class EventWriter:
    """Append RunEvents to a JSONL log file.

    Args:
        log_dir: Directory for telemetry logs.
        run_id: Unique identifier for the training run.
    """

    def __init__(self, log_dir: str | Path, run_id: str) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self._log_file = self.log_dir / f"{run_id}.jsonl"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event: RunEvent) -> None:
        """Persist a single event to the JSONL log file.

        Args:
            event: Training step event to persist.
        """
        with self._log_file.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

    @property
    def log_path(self) -> Path:
        return self._log_file


class EventReader:
    """Read RunEvents from a JSONL log file.

    Args:
        log_dir: Directory containing telemetry logs.
        run_id: Training run identifier to read.
    """

    def __init__(self, log_dir: str | Path, run_id: str) -> None:
        self.log_dir = Path(log_dir)
        self.run_id = run_id
        self._log_file = self.log_dir / f"{run_id}.jsonl"

    def read_all(self) -> list[RunEvent]:
        """Read all events for a given run.

        Returns:
            List of RunEvent objects in chronological order.
        """
        if not self._log_file.exists():
            return []

        events = []
        with self._log_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(RunEvent(**json.loads(line)))
        return events

    def tail(self, n: int = 10) -> list[RunEvent]:
        """Read the last N events for live monitoring.

        Args:
            n: Number of most recent events to return.

        Returns:
            List of the N most recent RunEvent objects.
        """
        all_events = self.read_all()
        return all_events[-n:]

    def count(self) -> int:
        """Count total events in the log.

        Returns:
            Number of events in the log file.
        """
        if not self._log_file.exists():
            return 0
        with self._log_file.open(encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    @staticmethod
    def list_runs(log_dir: str | Path) -> list[str]:
        """List all available run IDs in a log directory.

        Args:
            log_dir: Directory to search for telemetry logs.

        Returns:
            List of run ID strings.
        """
        path = Path(log_dir)
        if not path.exists():
            return []
        return sorted(p.stem for p in path.glob("*.jsonl"))
