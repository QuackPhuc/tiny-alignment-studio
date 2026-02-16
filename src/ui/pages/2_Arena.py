"""Arena page: compare base model vs aligned model side-by-side.

Provides an interactive chat interface for comparing responses
from a base model and a fine-tuned (aligned) model.
"""

from __future__ import annotations

from pathlib import Path

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


@st.cache_resource(show_spinner="Loading base model...")
def _load_base(model_name: str):
    """Cache the base model to avoid reloading on each interaction."""
    from src.core.inference import load_model_for_inference

    return load_model_for_inference(model_name, adapter_path=None)


@st.cache_resource(show_spinner="Loading aligned model...")
def _load_aligned(model_name: str, _adapter_path: str):
    """Cache the aligned model (base + adapter)."""
    from src.core.inference import load_model_for_inference

    return load_model_for_inference(model_name, adapter_path=_adapter_path)


def _generate(model, tokenizer, prompt: str) -> str:
    """Generate a response using the shared inference utility."""
    from src.core.inference import GenerationConfig, generate_response

    config = GenerationConfig(
        max_new_tokens=max_tokens,
        temperature=max(temperature, 0.01),
    )
    return generate_response(model, tokenizer, prompt, config)


# --- Check adapter availability ---
adapter_exists = Path(adapter_path).exists()
if not adapter_exists:
    st.warning(
        f"Adapter not found at `{adapter_path}`. "
        "Train a model first, or update the adapter path in the sidebar."
    )

# --- Chat input ---
prompt = st.chat_input("Enter a prompt to compare responses...")

if prompt:
    messages = get("arena_messages")
    if not isinstance(messages, list):
        messages = []
    messages.append({"role": "user", "content": prompt})

    with st.spinner("Generating responses..."):
        # Base model response
        base_model_obj, base_tokenizer = _load_base(base_model)
        base_response = _generate(base_model_obj, base_tokenizer, prompt)

        # Aligned model response (falls back to base if no adapter)
        if adapter_exists:
            aligned_model_obj, aligned_tokenizer = _load_aligned(
                base_model, adapter_path
            )
            aligned_response = _generate(aligned_model_obj, aligned_tokenizer, prompt)
        else:
            aligned_response = "[No adapter loaded] — train a model first."

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
