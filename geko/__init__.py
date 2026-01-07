"""
GEKO: Gradient-Efficient Knowledge Optimization

A plug-and-play training framework that works with ANY LLM.
Like LoRA for efficient fine-tuning, GEKO is for efficient training.

Usage:
    from geko import GEKOTrainer, GEKOConfig

    trainer = GEKOTrainer(
        model=your_model,
        tokenizer=your_tokenizer,
        config=GEKOConfig()
    )
    trainer.train(dataset)

Paper: "GEKO: Gradient-Efficient Knowledge Optimization via
        Confidence-Gated Sample Partitioning"
Author: Syed Abdur Rehman
"""

__version__ = "0.1.0"
__author__ = "Syed Abdur Rehman"

from .core import (
    Bucket,
    SampleState,
    GEKOConfig,
)
from .trainer import GEKOTrainer
from .curriculum import MountainCurriculum
from .partitioner import SamplePartitioner

__all__ = [
    "Bucket",
    "SampleState",
    "GEKOConfig",
    "GEKOTrainer",
    "MountainCurriculum",
    "SamplePartitioner",
]
