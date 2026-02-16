"""Training configuration and metrics contracts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TrainConfig(BaseModel):
    """Full training configuration loaded from YAML."""

    model_name: str
    algorithm: str = "dpo"
    adapter_type: str = "lora"
    quantization_bits: int = Field(default=4, ge=4, le=8)
    batch_size: int = Field(default=4, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=5e-5, gt=0)
    num_epochs: int = Field(default=1, ge=1)
    max_length: int = Field(default=512, ge=64)
    seed: int = 42
    output_dir: str = "outputs"
    bf16: bool = True


class StepMetrics(BaseModel):
    """Metrics emitted after each training step."""

    step: int = Field(ge=0)
    loss: float
    learning_rate: float = Field(ge=0)
    reward_margin: float | None = None
    gpu_memory_mb: float | None = None
    tokens_per_second: float | None = None


class EvalMetrics(BaseModel):
    """Metrics from an evaluation pass."""

    step: int = Field(ge=0)
    eval_loss: float
    eval_reward_margin: float | None = None
    num_samples: int = Field(ge=1)
