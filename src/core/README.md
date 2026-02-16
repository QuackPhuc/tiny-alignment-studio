# Core

Training logic for Tiny Alignment Studio. This package is UI-independent.

## Modules

| Module | Purpose |
|---|---|
| `trainer.py` | `AlignmentTrainer` orchestrator: loads config, creates components, runs training |
| `algorithms/` | Algorithm implementations. Each implements `AlgorithmPlugin` from contracts. |
| `models/` | Model loading (QLoRA-aware) and adapter management. |
| `data/` | Data pipeline: ingest, validate, format preference pairs. |

## Dependencies

- Imports from `src.contracts` and `src.utils` only.
- Never imports from `src.ui` or `src.telemetry`.
