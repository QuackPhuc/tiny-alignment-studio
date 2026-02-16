"""Streamlit session state management.

Centralizes all session state access to avoid scattered st.session_state
usage across pages and components.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

_DEFAULTS: dict[str, Any] = {
    "current_run_id": None,
    "training_active": False,
    "selected_model": None,
    "arena_messages": [],
}


def init_state() -> None:
    """Initialize session state with defaults if not already set."""
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def get(key: str) -> Any:
    """Get a value from session state.

    Args:
        key: State key to retrieve.

    Returns:
        The stored value, or None if key doesn't exist.
    """
    return st.session_state.get(key)


def set_value(key: str, value: Any) -> None:
    """Set a value in session state.

    Args:
        key: State key to set.
        value: Value to store.
    """
    st.session_state[key] = value
