# src

Core source code for Tiny Alignment Studio.

## Package Layout

| Package | Purpose |
|---|---|
| `contracts/` | Type definitions and protocols. Single source of truth for interfaces. |
| `core/` | Training logic, algorithms, model loading, data pipeline. UI-independent. |
| `telemetry/` | Metrics collection, event streaming, training callbacks. |
| `ui/` | Streamlit dashboard: training monitor and arena. |
| `utils/` | Config loading, structured logging, hardware detection. |

## Dependency Rules

- `contracts` has zero internal dependencies (leaf package).
- `core` depends on `contracts` and `utils` only.
- `telemetry` depends on `contracts` only.
- `ui` depends on `contracts`, `telemetry`, and `utils`. Never imports `core` directly.
- `utils` depends on `contracts` only.
