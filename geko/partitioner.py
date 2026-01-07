"""
GEKO Sample Partitioner

The heart of GEKO: classifies samples into confidence-gated buckets.

Bucket Logic:
- FREEZE: correct AND confidence > 0.85 AND quality > 0.80 (never train)
- LIGHT:  correct AND (confidence <= 0.85 OR quality <= 0.80)
- FOCUS:  incorrect AND confidence <= 0.60
- HARD:   incorrect AND confidence > 0.60 (confident-wrong = highest priority)
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .core import Bucket, SampleState, GEKOConfig


@dataclass
class PartitionStats:
    """Statistics from a partitioning operation."""
    total: int
    freeze_count: int
    light_count: int
    focus_count: int
    hard_count: int

    @property
    def freeze_ratio(self) -> float:
        return self.freeze_count / self.total if self.total > 0 else 0

    @property
    def light_ratio(self) -> float:
        return self.light_count / self.total if self.total > 0 else 0

    @property
    def focus_ratio(self) -> float:
        return self.focus_count / self.total if self.total > 0 else 0

    @property
    def hard_ratio(self) -> float:
        return self.hard_count / self.total if self.total > 0 else 0

    @property
    def trainable_count(self) -> int:
        """Samples that will be trained on (excludes FREEZE)."""
        return self.light_count + self.focus_count + self.hard_count

    @property
    def accuracy(self) -> float:
        """Model accuracy (FREEZE + LIGHT are correct)."""
        return (self.freeze_count + self.light_count) / self.total if self.total > 0 else 0

    def __str__(self) -> str:
        return (
            f"FREEZE: {self.freeze_count} ({self.freeze_ratio:.1%}) | "
            f"LIGHT: {self.light_count} ({self.light_ratio:.1%}) | "
            f"FOCUS: {self.focus_count} ({self.focus_ratio:.1%}) | "
            f"HARD: {self.hard_count} ({self.hard_ratio:.1%})"
        )


class SamplePartitioner:
    """
    Partitions samples into GEKO buckets based on model performance.

    This is the core GEKO algorithm:
    1. Evaluate model on each sample (get correctness, confidence)
    2. Classify into FREEZE/LIGHT/FOCUS/HARD
    3. Train only on HARD, FOCUS, (optionally LIGHT)
    4. Re-partition after each epoch to track learning progress

    Usage:
        partitioner = SamplePartitioner(config)

        # After evaluation
        for sample_id, (correct, confidence) in results.items():
            states[sample_id].correct = correct
            states[sample_id].confidence = confidence

        # Partition
        stats = partitioner.partition(states)
        print(stats)  # FREEZE: 30% | LIGHT: 20% | FOCUS: 35% | HARD: 15%

        # Get trainable samples
        trainable = partitioner.get_trainable_samples(states)
    """

    def __init__(self, config: Optional[GEKOConfig] = None):
        self.config = config or GEKOConfig()

    def classify(self, state: SampleState) -> Bucket:
        """
        Classify a single sample into a bucket.

        Decision tree:
        1. Is it correct?
           - Yes: Is it confident AND high-quality? → FREEZE, else → LIGHT
           - No: Is it low confidence? → FOCUS, else → HARD
        """
        if state.correct:
            # Correct samples
            is_confident = state.confidence > self.config.freeze_confidence
            is_high_quality = state.quality > self.config.freeze_quality
            is_high_q = state.q_value >= self.config.min_q_for_freeze

            if is_confident and is_high_quality and is_high_q:
                return Bucket.FREEZE
            else:
                return Bucket.LIGHT
        else:
            # Incorrect samples
            is_low_confidence = state.confidence <= self.config.focus_confidence

            if is_low_confidence:
                return Bucket.FOCUS
            else:
                return Bucket.HARD  # Confident-wrong (highest priority!)

    def partition(self, states: Dict[str, SampleState], epoch: int = 0) -> PartitionStats:
        """
        Partition all samples into buckets.

        Args:
            states: Dict mapping sample_id to SampleState
            epoch: Current epoch (for tracking when samples freeze)

        Returns:
            PartitionStats with counts for each bucket
        """
        counts = {b: 0 for b in Bucket}

        for sample_id, state in states.items():
            new_bucket = self.classify(state)

            # Track bucket transitions
            old_bucket = state.bucket
            state.bucket = new_bucket

            # Record freeze time
            if new_bucket == Bucket.FREEZE and old_bucket != Bucket.FREEZE:
                state.frozen_at_epoch = epoch

            counts[new_bucket] += 1

        return PartitionStats(
            total=len(states),
            freeze_count=counts[Bucket.FREEZE],
            light_count=counts[Bucket.LIGHT],
            focus_count=counts[Bucket.FOCUS],
            hard_count=counts[Bucket.HARD],
        )

    def get_trainable_samples(
        self,
        states: Dict[str, SampleState],
        weighted: bool = True
    ) -> List[str]:
        """
        Get list of sample IDs to train on.

        Args:
            states: Dict mapping sample_id to SampleState
            weighted: If True, repeat samples based on bucket weights

        Returns:
            List of sample_ids (possibly with repeats if weighted)
        """
        samples = []

        for sample_id, state in states.items():
            if state.bucket == Bucket.FREEZE:
                continue  # Never train on FREEZE

            if state.times_seen >= self.config.max_times_seen:
                continue  # Sample has been seen enough

            weight = self.config.get_bucket_weight(state.bucket) if weighted else 1
            samples.extend([sample_id] * weight)

        return samples

    def get_bucket_samples(
        self,
        states: Dict[str, SampleState],
        bucket: Bucket
    ) -> List[str]:
        """Get all sample IDs in a specific bucket."""
        return [sid for sid, state in states.items() if state.bucket == bucket]

    def compute_efficiency(self, stats: PartitionStats) -> float:
        """
        Compute GEKO efficiency score.

        Efficiency = (FREEZE + LIGHT) / Total
        Higher = more samples mastered = less compute needed
        """
        return stats.freeze_ratio + stats.light_ratio

    def should_stop_early(self, stats: PartitionStats, threshold: float = 0.95) -> bool:
        """
        Check if training can stop early.

        If FREEZE ratio exceeds threshold, model has mastered the dataset.
        """
        return stats.freeze_ratio >= threshold
