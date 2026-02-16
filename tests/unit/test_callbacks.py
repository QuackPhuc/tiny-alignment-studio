"""Tests for telemetry callback."""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

from src.telemetry.callbacks import TelemetryCallback, create_telemetry_callback
from src.telemetry.events import EventReader, EventWriter


class TestTelemetryCallback:
    def test_on_log_writes_event(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            writer = EventWriter(d, "run-cb")
            cb = TelemetryCallback(writer, "run-cb")

            state = SimpleNamespace(global_step=5)
            logs = {"loss": 0.3, "learning_rate": 1e-4}
            cb.on_log(args=None, state=state, control=None, logs=logs)

            reader = EventReader(d, "run-cb")
            events = reader.read_all()
            assert len(events) == 1
            assert events[0].step == 5
            assert events[0].loss == 0.3

    def test_on_log_skips_empty(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            writer = EventWriter(d, "run-skip")
            cb = TelemetryCallback(writer, "run-skip")
            cb.on_log(args=None, state=None, control=None, logs=None)

            reader = EventReader(d, "run-skip")
            assert reader.count() == 0

    def test_extras_captured(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            writer = EventWriter(d, "run-extra")
            cb = TelemetryCallback(writer, "run-extra")

            state = SimpleNamespace(global_step=1)
            logs = {"loss": 0.5, "learning_rate": 1e-5, "epoch": 0.1}
            cb.on_log(args=None, state=state, control=None, logs=logs)

            reader = EventReader(d, "run-extra")
            event = reader.read_all()[0]
            assert "epoch" in event.extras


class TestCreateTelemetryCallback:
    def test_factory(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            cb = create_telemetry_callback(d, "factory-run")
            assert isinstance(cb, TelemetryCallback)
            assert cb.run_id == "factory-run"
