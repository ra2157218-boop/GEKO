"""
GEKO Core Components

Defines the fundamental building blocks:
- Bucket: The 4 GEKO sample categories
- SampleState: Tracks per-sample learning state
- GEKOConfig: Configuration for GEKO training
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import math


class Bucket(Enum):
    """
    GEKO Sample Buckets - The core innovation.

    Samples are dynamically classified based on:
    - Correctness: Did the model get it right?
    - Confidence: How sure was the model?
    - Quality: How good is the sample itself?

    Training Priority: HARD > FOCUS > LIGHT >> FREEZE (never)
    """
    FREEZE = "FREEZE"  # Correct + confident + high-quality (NEVER train)
    LIGHT = "LIGHT"    # Correct but uncertain or low-quality (low priority)
    FOCUS = "FOCUS"    # Incorrect + low confidence (medium priority)
    HARD = "HARD"      # Incorrect + HIGH confidence (HIGHEST priority)

    def __str__(self):
        return self.value


@dataclass
class SampleState:
    """
    Tracks the learning state of a single sample.

    GEKO maintains state per-sample to enable:
    - Dynamic bucket reclassification
    - Q-value based difficulty estimation
    - Curriculum progression tracking

    Attributes:
        sample_id: Unique identifier for the sample
        bucket: Current GEKO bucket assignment
        q_value: Estimated "learnability" score (0-1)
        confidence: Model's confidence on this sample
        loss_history: Recent losses for trend analysis
        times_seen: How many times trained on this sample
        last_loss: Most recent loss value
        frozen_at_epoch: When sample moved to FREEZE (if applicable)
    """
    sample_id: str
    bucket: Bucket = Bucket.FOCUS  # Start in FOCUS (assume uncertain)
    q_value: float = 0.5  # Initial Q-value (neutral)
    confidence: float = 0.0
    quality: float = 1.0
    loss_history: List[float] = field(default_factory=list)
    times_seen: int = 0
    last_loss: float = float('inf')
    frozen_at_epoch: Optional[int] = None
    correct: bool = False

    def update(self, loss: float, confidence: float, correct: bool, epoch: int = 0):
        """
        Update sample state after a training step.

        Args:
            loss: The loss value for this sample
            confidence: Model's confidence (0-1)
            correct: Whether model got it right
            epoch: Current training epoch
        """
        self.times_seen += 1
        self.last_loss = loss
        self.confidence = confidence
        self.correct = correct

        # Maintain rolling history (last 5 losses)
        self.loss_history.append(loss)
        if len(self.loss_history) > 5:
            self.loss_history.pop(0)

        # Update Q-value using exponential moving average
        # Q increases when loss decreases (learning happening)
        loss_trend = self._compute_loss_trend()
        self.q_value = 0.9 * self.q_value + 0.1 * (1.0 - min(loss / 10.0, 1.0))

        # Track when frozen
        if self.bucket == Bucket.FREEZE and self.frozen_at_epoch is None:
            self.frozen_at_epoch = epoch

    def _compute_loss_trend(self) -> float:
        """Compute loss trend: negative = improving, positive = worsening."""
        if len(self.loss_history) < 2:
            return 0.0
        return self.loss_history[-1] - self.loss_history[0]

    @property
    def is_improving(self) -> bool:
        """Check if sample shows learning progress."""
        return self._compute_loss_trend() < -0.1

    @property
    def is_stuck(self) -> bool:
        """Check if sample is stuck (no improvement after many tries)."""
        return self.times_seen > 10 and not self.is_improving

    @property
    def priority_score(self) -> float:
        """
        Compute training priority score.

        Higher = more important to train on.
        HARD samples with high confidence-wrong get highest priority.
        """
        bucket_weights = {
            Bucket.FREEZE: 0.0,
            Bucket.LIGHT: 0.2,
            Bucket.FOCUS: 0.6,
            Bucket.HARD: 1.0,
        }

        base = bucket_weights[self.bucket]

        # Boost HARD samples that are confidently wrong
        if self.bucket == Bucket.HARD:
            base *= (1 + self.confidence)

        # Reduce priority for stuck samples
        if self.is_stuck:
            base *= 0.5

        return base


@dataclass
class GEKOConfig:
    """
    Configuration for GEKO training.

    Thresholds control bucket assignment:
    - freeze_confidence: Confidence threshold for FREEZE (default: 0.85)
    - freeze_quality: Quality threshold for FREEZE (default: 0.80)
    - focus_confidence: Confidence threshold for FOCUS vs HARD (default: 0.60)

    Bucket weights control training emphasis:
    - bucket_weights: (HARD, FOCUS, LIGHT) weights (default: 3, 1, 0)

    Example:
        config = GEKOConfig(
            freeze_confidence=0.90,  # Stricter FREEZE threshold
            bucket_weights=(5, 2, 1)  # Heavy emphasis on HARD
        )
    """
    # Bucket thresholds
    freeze_confidence: float = 0.85
    freeze_quality: float = 0.80
    focus_confidence: float = 0.60

    # Training weights (HARD, FOCUS, LIGHT) - FREEZE is always 0
    bucket_weights: Tuple[int, int, int] = (3, 1, 0)

    # Curriculum settings
    use_curriculum: bool = True
    curriculum_phases: int = 5  # Mountain curriculum phases
    warmup_epochs: int = 1

    # Q-value settings
    q_value_lr: float = 0.1  # Learning rate for Q-value updates
    min_q_for_freeze: float = 0.8  # Minimum Q-value to allow FREEZE

    # Training settings
    repartition_every: int = 1  # Re-partition every N epochs
    log_bucket_stats: bool = True

    # Early stopping for samples
    max_times_seen: int = 50  # Stop training on sample after this

    def validate(self):
        """Validate configuration values."""
        assert 0 < self.freeze_confidence <= 1.0
        assert 0 < self.freeze_quality <= 1.0
        assert 0 < self.focus_confidence < self.freeze_confidence
        assert len(self.bucket_weights) == 3
        assert all(w >= 0 for w in self.bucket_weights)

    def get_bucket_weight(self, bucket: Bucket) -> int:
        """Get training weight for a bucket."""
        weights = {
            Bucket.HARD: self.bucket_weights[0],
            Bucket.FOCUS: self.bucket_weights[1],
            Bucket.LIGHT: self.bucket_weights[2],
            Bucket.FREEZE: 0,  # Never train on FREEZE
        }
        return weights[bucket]


# Preset configurations
GEKO_AGGRESSIVE = GEKOConfig(
    freeze_confidence=0.90,
    freeze_quality=0.85,
    bucket_weights=(5, 2, 0),  # Heavy on HARD
)

GEKO_BALANCED = GEKOConfig(
    freeze_confidence=0.85,
    freeze_quality=0.80,
    bucket_weights=(3, 2, 1),  # Include LIGHT
)

GEKO_CONSERVATIVE = GEKOConfig(
    freeze_confidence=0.80,
    freeze_quality=0.75,
    bucket_weights=(2, 2, 1),  # More balanced
)
