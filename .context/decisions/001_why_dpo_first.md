# ADR-001: Why DPO First

## Status

Accepted

## Context

The project needs to support alignment algorithms. The two main candidates
for v1 are Direct Preference Optimization (DPO) and Proximal Policy
Optimization (PPO).

## Decision

Implement DPO as the first and primary algorithm. PPO will be a plugin stub
with only the interface contract, not a full implementation, in v1.

## Rationale

1. **Simplicity**: DPO requires no separate reward model. It optimizes
   directly on preference pairs, reducing the training pipeline complexity
   from 3 models (policy + reference + reward) to 2 (policy + reference).

2. **Hardware friendliness**: With QLoRA, DPO can run on a single consumer
   GPU (8 GB VRAM). PPO's reward model adds significant memory overhead.

3. **TRL support**: The `trl` library provides a mature `DPOTrainer` that
   handles the training loop, loss computation, and logging out of the box.

4. **Educational value**: DPO's math is more accessible. The loss function
   maps directly to the preference data, making it easier to teach.

## Consequences

- PPO support is deferred but the `AlgorithmPlugin` protocol ensures it
  can be added without modifying core training logic.
- Users who specifically need PPO will need to wait for Phase 6 or
  contribute the implementation themselves.
