# Quickstart

## Prerequisites

- Python 3.10+
- GPU with 6+ GB VRAM for training (RTX 3050 or better)
- CPU-only for data prep and testing

## Installation

```bash
git clone <repo-url>
cd tiny-alignment-studio
pip install -e ".[dev]"
```

## 1. Prepare Data

Download and validate preference data:

```bash
python scripts/prepare_data.py \
    --source Anthropic/hh-rlhf \
    --max-samples 1000 \
    --output-dir outputs/data
```

## 2. Train a DPO Model

```bash
python scripts/train.py --config configs/base.yaml
```

Training uses QLoRA (4-bit quantization + LoRA) to fit within 6GB VRAM.
The adapter weights are saved to `outputs/adapter/`.

## 3. Evaluate

```bash
python scripts/evaluate.py \
    --adapter outputs/adapter \
    --prompt "What is the meaning of life?"
```

## 4. Launch Dashboard

Monitor training runs and compare models:

```bash
streamlit run src/ui/app.py
```

## 5. Run Tests

```bash
pytest tests/ -v
```

All unit tests run on CPU without GPU dependencies.

## Configuration

Training is configured via YAML files in `configs/`:

- `configs/base.yaml` — default config for TinyLlama + DPO
- `configs/experiments/dpo_tinyllama.yaml` — example experiment override

See `configs/README.md` for the full configuration reference.
