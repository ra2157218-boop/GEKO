"""
GEKO: Gradient-Efficient Knowledge Optimization

A plug-and-play training framework that works with ANY LLM.
Like LoRA for efficient fine-tuning, GEKO is for efficient training.

Usage:
    from geko import GEKOTrainer, GEKOConfig

    trainer = GEKOTrainer(
        model=your_model,
        train_dataset=your_dataset,
    )
    trainer.train()

Paper: "GEKO: Gradient-Efficient Knowledge Optimization via
        Confidence-Gated Sample Partitioning"
Author: Syed Abdur Rehman
"""

__version__ = "0.3.0"
__author__ = "Syed Abdur Rehman"

from .core import (
    Bucket,
    SampleState,
    GEKOConfig,
)
from .trainer import GEKOTrainer, GEKOTrainingArgs, GEKODataset
from .curriculum import MountainCurriculum
from .partitioner import SamplePartitioner
from .peft_utils import apply_lora, is_peft_available

__all__ = [
    "Bucket",
    "SampleState",
    "GEKOConfig",
    "GEKOTrainer",
    "GEKOTrainingArgs",
    "GEKODataset",
    "MountainCurriculum",
    "SamplePartitioner",
    "apply_lora",
    "is_peft_available",
]
