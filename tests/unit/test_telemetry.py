"""Tests for telemetry event writer and reader."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.contracts.events import RunEvent
from src.telemetry.events import EventReader, EventWriter


@pytest.fixture
def tmp_log_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_event() -> RunEvent:
    return RunEvent(run_id="test-run", step=10, loss=0.5, learning_rate=5e-5)


class TestEventWriter:
    def test_write_creates_file(
        self, tmp_log_dir: Path, sample_event: RunEvent
    ) -> None:
        writer = EventWriter(tmp_log_dir, "test-run")
        writer.write(sample_event)
        assert writer.log_path.exists()

    def test_write_appends(self, tmp_log_dir: Path, sample_event: RunEvent) -> None:
        writer = EventWriter(tmp_log_dir, "test-run")
        writer.write(sample_event)
        writer.write(sample_event)
        lines = writer.log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_creates_directory(self, tmp_log_dir: Path, sample_event: RunEvent) -> None:
        nested = tmp_log_dir / "a" / "b"
        writer = EventWriter(nested, "run")
        writer.write(sample_event)
        assert writer.log_path.exists()


class TestEventReader:
    def test_read_all(self, tmp_log_dir: Path, sample_event: RunEvent) -> None:
        writer = EventWriter(tmp_log_dir, "run-1")
        for i in range(5):
            event = RunEvent(
                run_id="run-1", step=i, loss=1.0 - i * 0.1, learning_rate=5e-5
            )
            writer.write(event)

        reader = EventReader(tmp_log_dir, "run-1")
        events = reader.read_all()
        assert len(events) == 5
        assert events[0].step == 0
        assert events[4].step == 4

    def test_read_empty(self, tmp_log_dir: Path) -> None:
        reader = EventReader(tmp_log_dir, "nonexistent")
        assert reader.read_all() == []

    def test_tail(self, tmp_log_dir: Path) -> None:
        writer = EventWriter(tmp_log_dir, "run-2")
        for i in range(20):
            event = RunEvent(run_id="run-2", step=i, loss=1.0, learning_rate=5e-5)
            writer.write(event)

        reader = EventReader(tmp_log_dir, "run-2")
        tail = reader.tail(5)
        assert len(tail) == 5
        assert tail[0].step == 15
        assert tail[-1].step == 19

    def test_count(self, tmp_log_dir: Path, sample_event: RunEvent) -> None:
        writer = EventWriter(tmp_log_dir, "run-3")
        writer.write(sample_event)
        writer.write(sample_event)
        writer.write(sample_event)

        reader = EventReader(tmp_log_dir, "run-3")
        assert reader.count() == 3

    def test_list_runs(self, tmp_log_dir: Path, sample_event: RunEvent) -> None:
        EventWriter(tmp_log_dir, "alpha").write(sample_event)
        EventWriter(tmp_log_dir, "beta").write(sample_event)
        EventWriter(tmp_log_dir, "gamma").write(sample_event)

        runs = EventReader.list_runs(tmp_log_dir)
        assert runs == ["alpha", "beta", "gamma"]

    def test_list_runs_empty(self, tmp_log_dir: Path) -> None:
        empty = tmp_log_dir / "empty"
        assert EventReader.list_runs(empty) == []
