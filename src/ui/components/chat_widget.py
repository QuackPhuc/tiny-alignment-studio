"""Chat widget components for the arena page.

Renders chat messages in Streamlit's native chat_message format
for side-by-side model comparison.
"""

from __future__ import annotations

import streamlit as st


def render_chat_column(
    messages: list[dict],
    model_key: str = "base",
) -> None:
    """Render a chat conversation for one model in the arena.

    Handles both user messages (rendered identically in both columns)
    and assistant messages (rendered from the model_key sub-dict).

    Args:
        messages: List of message dicts from session state.
        model_key: Which model's responses to display ('base' or 'aligned').
    """
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.write(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                if isinstance(content, dict):
                    st.write(content.get(model_key, "[No response]"))
                else:
                    st.write(content)


def render_chat(messages: list[dict]) -> None:
    """Render a simple chat conversation without model splitting.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
    """
    for msg in messages:
        with st.chat_message(msg.get("role", "user")):
            st.write(msg.get("content", ""))
