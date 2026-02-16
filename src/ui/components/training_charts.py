"""Training chart components for loss, reward margin, and LR visualization.

Uses Streamlit's native charting backed by Altair for clean,
interactive charts without additional dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
    from src.contracts.events import RunEvent


def render_loss_chart(events: list[RunEvent]) -> None:
    """Render a loss curve chart from training events.

    Args:
        events: List of RunEvent objects to plot.
    """
    data = [{"step": e.step, "loss": e.loss} for e in events]
    st.line_chart(data, x="step", y="loss", use_container_width=True)


def render_reward_margin_chart(events: list[RunEvent]) -> None:
    """Render a reward margin chart from training events.

    Only plots events that have reward margin data.

    Args:
        events: List of RunEvent objects to plot.
    """
    data = [
        {"step": e.step, "reward_margin": e.reward_margin}
        for e in events
        if e.reward_margin is not None
    ]
    if not data:
        st.info("No reward margin data available.")
        return
    st.line_chart(data, x="step", y="reward_margin", use_container_width=True)


def render_lr_chart(events: list[RunEvent]) -> None:
    """Render a learning rate schedule chart.

    Args:
        events: List of RunEvent objects to plot.
    """
    data = [{"step": e.step, "learning_rate": e.learning_rate} for e in events]
    st.line_chart(data, x="step", y="learning_rate", use_container_width=True)
