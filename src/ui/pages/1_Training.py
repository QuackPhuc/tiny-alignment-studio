"""Training monitor page: real-time loss curves and metrics.

Reads telemetry events from JSONL logs and renders live charts.
Supports auto-refresh during active training.
"""

from __future__ import annotations

import time
from pathlib import Path

import streamlit as st

from src.telemetry.events import EventReader
from src.ui.components.metric_cards import render_metric_row
from src.ui.components.training_charts import (
    render_loss_chart,
    render_lr_chart,
    render_reward_margin_chart,
)
from src.ui.state import get, init_state, set_value

init_state()

st.set_page_config(page_title="Training Monitor", layout="wide")
st.title("Training Monitor")

# --- Sidebar: run selection ---
log_dir = st.sidebar.text_input("Log directory", value="outputs/telemetry")
available_runs = EventReader.list_runs(log_dir)

if not available_runs:
    st.info(
        "No training runs found. Start training with:\n\n"
        "```\npython scripts/train.py --config configs/base.yaml\n```"
    )
    st.stop()

selected_run = st.sidebar.selectbox("Select run", available_runs)
auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

# --- Load events ---
reader = EventReader(log_dir, selected_run)
events = reader.read_all()

if not events:
    st.warning(f"Run '{selected_run}' has no events yet.")
    st.stop()

# --- Metric summary cards ---
latest = events[-1]
first = events[0]

render_metric_row(
    {
        "Current Loss": (
            latest.loss,
            latest.loss - first.loss if len(events) > 1 else None,
        ),
        "Learning Rate": (latest.learning_rate, None),
        "Step": (float(latest.step), None),
        "Total Steps": (float(len(events)), None),
    }
)

st.markdown("---")

# --- Charts ---
tab_loss, tab_reward, tab_lr = st.tabs(["Loss", "Reward Margin", "Learning Rate"])

with tab_loss:
    render_loss_chart(events)

with tab_reward:
    has_margins = any(e.reward_margin is not None for e in events)
    if has_margins:
        render_reward_margin_chart(events)
    else:
        st.info("No reward margin data available for this run.")

with tab_lr:
    render_lr_chart(events)

# --- Event log ---
with st.expander("Raw event log (last 20)"):
    for event in reversed(events[-20:]):
        st.json(event.model_dump(), expanded=False)

# --- Auto-refresh ---
if auto_refresh:
    time.sleep(5)
    st.rerun()
