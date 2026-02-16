"""Tiny Alignment Studio Dashboard.

Entry point for the Streamlit application.
Run with: streamlit run src/ui/app.py
"""

from __future__ import annotations

import streamlit as st

from src.ui.state import init_state


def main() -> None:
    """Configure and launch the main dashboard."""
    st.set_page_config(
        page_title="Tiny Alignment Studio",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    init_state()

    st.sidebar.title("Tiny Alignment Studio")
    st.sidebar.markdown("---")
    st.sidebar.caption("An educational RLHF framework")

    st.title("Tiny Alignment Studio")
    st.markdown(
        "An educational RLHF framework for learning alignment techniques "
        "on consumer hardware."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Training Monitor")
        st.markdown(
            "Watch training in real-time with live loss curves, "
            "reward margins, and learning rate schedules."
        )
        st.page_link("pages/1_Training.py", label="Open Training Monitor")

    with col2:
        st.markdown("### Arena")
        st.markdown(
            "Compare base model vs aligned model side-by-side "
            "in an interactive chat arena."
        )
        st.page_link("pages/2_Arena.py", label="Open Arena")

    with col3:
        st.markdown("### Getting Started")
        st.markdown(
            "1. Configure training in `configs/base.yaml`\n"
            "2. Run `python scripts/train.py --config configs/base.yaml`\n"
            "3. Monitor progress in the Training page"
        )


if __name__ == "__main__":
    main()
