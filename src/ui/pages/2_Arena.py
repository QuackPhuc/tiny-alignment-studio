"""Arena page: compare base model vs aligned model side-by-side.

Provides an interactive chat interface for comparing responses
from a base model and a fine-tuned (aligned) model.
"""

from __future__ import annotations

import streamlit as st

from src.ui.components.chat_widget import render_chat_column
from src.ui.state import get, init_state, set_value

init_state()

st.set_page_config(page_title="Arena", layout="wide")
st.title("Arena: Base vs Aligned")
st.markdown("Compare responses from the base model and aligned model side-by-side.")

# --- Sidebar: model config ---
st.sidebar.subheader("Model Configuration")

base_model = st.sidebar.text_input(
    "Base model", value="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
adapter_path = st.sidebar.text_input("Adapter path", value="outputs/adapter")
max_tokens = st.sidebar.slider("Max new tokens", 32, 512, 256)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)

st.sidebar.markdown("---")
if st.sidebar.button("Clear chat"):
    set_value("arena_messages", [])
    st.rerun()

# --- Chat input ---
prompt = st.chat_input("Enter a prompt to compare responses...")

if prompt:
    messages = get("arena_messages")
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": "user", "content": prompt})

    # Generate responses (placeholder until model loading is connected)
    base_response = _generate_placeholder("base", prompt)
    aligned_response = _generate_placeholder("aligned", prompt)

    messages.append(
        {
            "role": "assistant",
            "content": {"base": base_response, "aligned": aligned_response},
        }
    )
    set_value("arena_messages", messages)

# --- Render chat columns ---
messages = get("arena_messages")
if not isinstance(messages, list):
    messages = []

if not messages:
    st.info("Enter a prompt above to start comparing models.")
else:
    col_base, col_aligned = st.columns(2)

    with col_base:
        st.subheader("Base Model")
        render_chat_column(messages, model_key="base")

    with col_aligned:
        st.subheader("Aligned Model")
        render_chat_column(messages, model_key="aligned")


def _generate_placeholder(model_type: str, prompt: str) -> str:
    """Placeholder response generator until real model inference.

    Args:
        model_type: Either 'base' or 'aligned'.
        prompt: User prompt.

    Returns:
        Placeholder response string.
    """
    return (
        f"[{model_type.upper()} MODEL] This is a placeholder response "
        f"for: '{prompt[:50]}...'. Connect a real model to see actual outputs."
    )
