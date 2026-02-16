# System Map: Tiny Alignment Studio

## Component Overview

```
configs/*.yaml
    |
    v
ConfigLoader (src/utils/config.py)
    |
    v
AlignmentTrainer (src/core/trainer.py)
    |
    +---> AlgorithmPlugin (src/core/algorithms/)
    |         |---> DPOAlgorithm (dpo.py)
    |         +---> PPOAlgorithm (ppo.py, stub)
    |
    +---> DataPipeline (src/core/data/)
    |         |---> Ingest & Validate
    |         +---> Formatters -> Preference Pairs
    |
    +---> ModelLoader (src/core/models/)
    |         |---> QLoRA 4-bit loading
    |         +---> Adapter management
    |
    +---> Callbacks -> Telemetry (src/telemetry/)
                           |
                           v
                   Streamlit Dashboard (src/ui/)
                       |---> Training Monitor
                       +---> Arena (Base vs Aligned)
```

## Data Flow

1. Raw preference data (JSONL) -> `DataPipeline.ingest()` -> validate schema
2. Validated records -> `DataPipeline.format()` -> algorithm-specific format
3. Formatted dataset + loaded model -> `AlignmentTrainer.train()`
4. Training loop emits `RunEvent` -> telemetry writer -> UI reads events

## Key Contracts

All interfaces live in `src/contracts/`:

- `PreferenceRecord` — canonical preference data shape
- `TrainConfig` — full training configuration (Pydantic model)
- `AlgorithmPlugin` — protocol for swappable alignment algorithms
- `RunEvent` — telemetry event emitted each training step

## Design Principles

1. **Trainer orchestrates** — single entry point, receives config, creates components
2. **Algorithms are plugins** — implement the protocol, register, swap via config
3. **Data pipeline is algorithm-agnostic** — same data, different algorithms
4. **UI only observes** — connects via telemetry callbacks, never mutates training state
