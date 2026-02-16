# Tiny Alignment Studio

An educational RLHF framework for learning alignment techniques on consumer hardware.

## Overview

Train small language models using DPO (Direct Preference Optimization) with QLoRA on a single GPU. Features a Streamlit dashboard for monitoring training and comparing base vs aligned models.

## Features

- **DPO Training** with QLoRA 4-bit quantization via TRL
- **Data Pipeline** for Anthropic HH-RLHF and custom preference datasets
- **Live Dashboard** with loss curves, reward margins, and learning rate charts
- **Arena Mode** for side-by-side model comparison
- **Telemetry** with JSONL event streaming for real-time monitoring
- **Plugin Architecture** for adding new alignment algorithms

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Prepare data
python scripts/prepare_data.py --source Anthropic/hh-rlhf --max-samples 1000

# Train
python scripts/train.py --config configs/base.yaml

# Launch dashboard
streamlit run src/ui/app.py

# Evaluate
python scripts/evaluate.py --adapter outputs/adapter --prompt "What is AI?"
```

## Project Structure

```
tiny-alignment-studio/
  .context/         AI navigation, architecture docs, ADRs
  configs/          YAML configs + JSON schema validation
  src/
    contracts/      Pydantic data models (single source of truth)
    core/           Training engine, algorithms, models, data pipeline
    telemetry/      Event streaming and trainer callbacks
    ui/             Streamlit dashboard and arena
    utils/          Config loader, logging, hardware detection
  tests/            Unit, integration, and e2e tests
  scripts/          CLI entry points (train, evaluate, prepare_data)
  docs/             Documentation
```

## Requirements

- Python 3.10+
- GPU with 6+ GB VRAM (RTX 3050 or better) for training
- CPU-only for data preparation and testing

## Running Tests

```bash
pytest tests/ -v
```

All unit tests run on CPU without GPU dependencies.
