# UI

Streamlit dashboard for Tiny Alignment Studio.

## Pages

| Page | Purpose |
|---|---|
| `1_Training.py` | Real-time training monitor with loss curves and metrics |
| `2_Arena.py` | Chat arena: compare base model vs aligned model side-by-side |

## Components

Reusable widgets in `components/`:

- `chat_widget.py` — Chat interface for arena mode
- `metric_cards.py` — Summary metric cards
- `training_charts.py` — Loss and reward margin charts

## Running

```bash
streamlit run src/ui/app.py
```
