# Product Manifest: Tiny Alignment Studio

## 1. Vision

To democratize Reinforcement Learning from Human Feedback (RLHF) by building a **"Tiny Alignment Studio"**. This framework serves as a bridge between simple RL concepts and modern LLM alignment techniques. It is designed to be:

- **Educational**: "Học mà chơi" (Learn by playing) - making complex concepts intuitive.
- **Practical**: Produces real, working aligned models (not just simulations).
- **Visual**: Everything from training dynamics to model comparisons must be interactive and visible.

## 2. Core Philosophy

- **Vibe Coding First**: The codebase structure must support seamless AI collaboration. Context is king.
- **Modularity**: Users should be able to swap "batteries" (Models, Datasets, Algorithms) without rewriting core logic.
- **Accessibility**: Optimized for consumer hardware (Single GPU/Colab) using efficient techniques like QLoRA.

## 3. High-Level Requirements

- **Framework Architecture**: clearly separated configuration, core logic, and UI.
- **Algorithm Support**: Must support modern alignment methods (e.g., DPO) with extensibility for others (PPO).
- **Data Flexibility**: clear pipelines for ingesting and formatting preference data.
- **Interactive Dashboard**: A UI (e.g., Streamlit) for:
  - Real-time training monitoring.
  - "Arena" mode: Chat with Pre-trained vs. Aligned models side-by-side.
