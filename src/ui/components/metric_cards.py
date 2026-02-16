"""Metric card components for dashboard summaries.

Renders a row of st.metric cards for key training stats.
"""

from __future__ import annotations

import streamlit as st


def render_metric_card(
    label: str,
    value: float,
    delta: float | None = None,
    fmt: str = ".4f",
) -> None:
    """Render a single metric card.

    Args:
        label: Metric display name.
        value: Current metric value.
        delta: Optional change from baseline.
        fmt: Format string for the value.
    """
    formatted_value = f"{value:{fmt}}"
    formatted_delta = f"{delta:{fmt}}" if delta is not None else None
    st.metric(label=label, value=formatted_value, delta=formatted_delta)


def render_metric_row(
    metrics: dict[str, tuple[float, float | None]],
) -> None:
    """Render a row of metric cards.

    Args:
        metrics: Dict mapping label to (value, delta) tuples.
    """
    cols = st.columns(len(metrics))
    for col, (label, (value, delta)) in zip(cols, metrics.items()):
        with col:
            # Use appropriate formatting based on value magnitude
            if abs(value) < 0.01 and value != 0:
                fmt = ".2e"
            elif value == int(value):
                fmt = ".0f"
            else:
                fmt = ".4f"
            render_metric_card(label, value, delta, fmt=fmt)
