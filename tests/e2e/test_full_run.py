"""End-to-end tests: CLI scripts, config schema, and full pipeline validation."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class TestCLIScripts:
    """Verify CLI scripts are importable and respond to --help."""

    @pytest.mark.parametrize(
        "module",
        ["scripts.train", "scripts.evaluate", "scripts.prepare_data"],
    )
    def test_script_importable(self, module):
        mod = importlib.import_module(module)
        assert hasattr(mod, "main")

    @pytest.mark.parametrize(
        "script",
        ["scripts/train.py", "scripts/evaluate.py", "scripts/prepare_data.py"],
    )
    def test_script_help_exits_cleanly(self, script):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                script.replace("/", ".").replace(".py", ""),
                "--help",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=30,
        )
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower() or "--help" in result.stdout.lower()


class TestConfigSchema:
    """Validate YAML configs against JSON Schema."""

    @pytest.fixture
    def schema(self):
        schema_path = (
            PROJECT_ROOT / "configs" / "schemas" / "training_config.schema.json"
        )
        return json.loads(schema_path.read_text(encoding="utf-8"))

    def test_schema_is_valid_json_schema(self, schema):
        jsonschema.Draft7Validator.check_schema(schema)

    def test_base_config_valid(self, schema):
        config_path = PROJECT_ROOT / "configs" / "base.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        jsonschema.validate(config, schema)

    def test_colab_config_valid(self, schema):
        config_path = PROJECT_ROOT / "configs" / "colab.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        jsonschema.validate(config, schema)

    def test_experiment_config_partial_valid(self, schema):
        """Experiment configs may be partial (used with merge)."""
        config_path = PROJECT_ROOT / "configs" / "experiments" / "dpo_tinyllama.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        # Experiment config is partial, so does not need to pass full validation
        assert "model" in config or "training" in config


class TestContractsConsistency:
    """Verify contract models are consistent and instantiable."""

    def test_preference_record_roundtrip(self):
        from src.contracts.data import PreferenceRecord

        record = PreferenceRecord(
            id="e2e-1",
            prompt="Hello",
            chosen="Hi there!",
            rejected="Go away",
            source="test",
        )
        dumped = record.model_dump()
        restored = PreferenceRecord(**dumped)
        assert restored == record

    def test_train_config_roundtrip(self):
        from src.contracts.training import TrainConfig

        config = TrainConfig(
            model_name="test/model",
            algorithm="dpo",
        )
        dumped = config.model_dump()
        restored = TrainConfig(**dumped)
        assert restored.model_name == "test/model"

    def test_run_event_roundtrip(self):
        from src.contracts.events import RunEvent

        event = RunEvent(run_id="r1", step=10, loss=0.5, learning_rate=1e-4)
        json_str = event.model_dump_json()
        restored = RunEvent.model_validate_json(json_str)
        assert restored.step == 10
        assert restored.loss == 0.5
