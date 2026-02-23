"""
Unit tests for geko/partitioner.py

Covers: SamplePartitioner.classify(), partition(), get_bucket_samples(),
        compute_efficiency(), should_stop_early(), PartitionStats
"""

import pytest
from geko.core import Bucket, SampleState, GEKOConfig
from geko.partitioner import SamplePartitioner, PartitionStats


def make_state(
    sample_id: str = "0",
    correct: bool = False,
    confidence: float = 0.5,
    quality: float = 1.0,
    q_value: float = 0.9,
    bucket: Bucket = Bucket.FOCUS,
) -> SampleState:
    """Helper to build a SampleState with explicit fields."""
    s = SampleState(sample_id=sample_id, bucket=bucket, q_value=q_value)
    s.correct = correct
    s.confidence = confidence
    s.quality = quality
    return s


# ---------------------------------------------------------------------------
# SamplePartitioner.classify()
# ---------------------------------------------------------------------------

class TestClassify:
    def setup_method(self):
        self.config = GEKOConfig(
            freeze_confidence=0.85,
            freeze_quality=0.80,
            focus_confidence=0.60,
            min_q_for_freeze=0.8,
        )
        self.partitioner = SamplePartitioner(self.config)

    def test_classify_freeze(self):
        # Correct + confident + high quality + high Q → FREEZE
        s = make_state(correct=True, confidence=0.90, quality=0.95, q_value=0.90)
        assert self.partitioner.classify(s) == Bucket.FREEZE

    def test_classify_light_low_confidence(self):
        # Correct but confidence below threshold → LIGHT
        s = make_state(correct=True, confidence=0.50, quality=0.95, q_value=0.90)
        assert self.partitioner.classify(s) == Bucket.LIGHT

    def test_classify_light_low_quality(self):
        # Correct, high confidence, but low quality → LIGHT
        s = make_state(correct=True, confidence=0.90, quality=0.50, q_value=0.90)
        assert self.partitioner.classify(s) == Bucket.LIGHT

    def test_classify_light_low_q_value(self):
        # Correct, high confidence, high quality, but Q too low → LIGHT
        s = make_state(correct=True, confidence=0.90, quality=0.95, q_value=0.50)
        assert self.partitioner.classify(s) == Bucket.LIGHT

    def test_classify_focus(self):
        # Incorrect + low confidence → FOCUS
        s = make_state(correct=False, confidence=0.30)
        assert self.partitioner.classify(s) == Bucket.FOCUS

    def test_classify_hard(self):
        # Incorrect + high confidence → HARD (confident-wrong)
        s = make_state(correct=False, confidence=0.90)
        assert self.partitioner.classify(s) == Bucket.HARD

    def test_confidence_boundary_freeze_exact(self):
        # confidence == freeze_confidence threshold (not strictly greater) → LIGHT
        s = make_state(correct=True, confidence=0.85, quality=0.95, q_value=0.90)
        assert self.partitioner.classify(s) == Bucket.LIGHT

    def test_confidence_boundary_focus_exact(self):
        # confidence == focus_confidence threshold → FOCUS (not strictly greater)
        s = make_state(correct=False, confidence=0.60)
        assert self.partitioner.classify(s) == Bucket.FOCUS

    def test_confidence_just_above_focus_threshold_is_hard(self):
        s = make_state(correct=False, confidence=0.61)
        assert self.partitioner.classify(s) == Bucket.HARD


# ---------------------------------------------------------------------------
# SamplePartitioner.partition()
# ---------------------------------------------------------------------------

class TestPartition:
    def setup_method(self):
        self.config = GEKOConfig()
        self.partitioner = SamplePartitioner(self.config)

    def _make_states(self) -> dict:
        return {
            "freeze": make_state("freeze", correct=True, confidence=0.90, quality=0.95, q_value=0.90),
            "light": make_state("light", correct=True, confidence=0.50, quality=0.95, q_value=0.90),
            "focus": make_state("focus", correct=False, confidence=0.30),
            "hard": make_state("hard", correct=False, confidence=0.90),
        }

    def test_partition_total_count(self):
        states = self._make_states()
        stats = self.partitioner.partition(states)
        assert stats.total == 4

    def test_partition_counts_sum_to_total(self):
        states = self._make_states()
        stats = self.partitioner.partition(states)
        assert (stats.freeze_count + stats.light_count +
                stats.focus_count + stats.hard_count) == stats.total

    def test_partition_mutates_bucket_on_state(self):
        states = {
            "hard": make_state("hard", correct=False, confidence=0.90, bucket=Bucket.FOCUS),
        }
        self.partitioner.partition(states)
        assert states["hard"].bucket == Bucket.HARD

    def test_partition_records_frozen_at_epoch(self):
        states = {
            "s": make_state("s", correct=True, confidence=0.90, quality=0.95, q_value=0.90),
        }
        assert states["s"].frozen_at_epoch is None
        self.partitioner.partition(states, epoch=3)
        assert states["s"].frozen_at_epoch == 3

    def test_partition_does_not_overwrite_existing_freeze_epoch(self):
        states = {
            "s": make_state("s", correct=True, confidence=0.90, quality=0.95, q_value=0.90),
        }
        self.partitioner.partition(states, epoch=2)
        self.partitioner.partition(states, epoch=5)
        # frozen_at_epoch should stay 2 (first freeze)
        assert states["s"].frozen_at_epoch == 2

    def test_partition_correct_bucket_assignments(self):
        states = self._make_states()
        self.partitioner.partition(states)
        assert states["freeze"].bucket == Bucket.FREEZE
        assert states["light"].bucket == Bucket.LIGHT
        assert states["focus"].bucket == Bucket.FOCUS
        assert states["hard"].bucket == Bucket.HARD


# ---------------------------------------------------------------------------
# PartitionStats properties
# ---------------------------------------------------------------------------

class TestPartitionStats:
    def _make_stats(self, freeze=1, light=2, focus=3, hard=4) -> PartitionStats:
        return PartitionStats(
            total=freeze + light + focus + hard,
            freeze_count=freeze,
            light_count=light,
            focus_count=focus,
            hard_count=hard,
        )

    def test_freeze_ratio(self):
        stats = self._make_stats(freeze=1, light=0, focus=0, hard=9)
        assert stats.freeze_ratio == pytest.approx(0.1)

    def test_accuracy_is_freeze_plus_light(self):
        stats = self._make_stats(freeze=2, light=3, focus=3, hard=2)
        # 2+3 correct out of 10
        assert stats.accuracy == pytest.approx(0.5)

    def test_trainable_count_excludes_freeze(self):
        stats = self._make_stats(freeze=2, light=1, focus=3, hard=4)
        assert stats.trainable_count == 1 + 3 + 4

    def test_str_contains_all_buckets(self):
        stats = self._make_stats()
        s = str(stats)
        for name in ["FREEZE", "LIGHT", "FOCUS", "HARD"]:
            assert name in s

    def test_zero_total_ratios_are_zero(self):
        stats = PartitionStats(total=0, freeze_count=0, light_count=0, focus_count=0, hard_count=0)
        assert stats.freeze_ratio == 0
        assert stats.accuracy == 0

    def test_to_dict_roundtrip(self):
        stats = self._make_stats(freeze=2, light=3, focus=1, hard=4)
        d = stats.to_dict()
        assert d == {
            "total": 10,
            "freeze_count": 2,
            "light_count": 3,
            "focus_count": 1,
            "hard_count": 4,
        }
        restored = PartitionStats.from_dict(d)
        assert restored.total == stats.total
        assert restored.freeze_count == stats.freeze_count
        assert restored.light_count == stats.light_count
        assert restored.focus_count == stats.focus_count
        assert restored.hard_count == stats.hard_count
        assert restored.freeze_ratio == stats.freeze_ratio
        assert restored.accuracy == stats.accuracy


# ---------------------------------------------------------------------------
# SamplePartitioner.get_bucket_samples()
# ---------------------------------------------------------------------------

class TestGetBucketSamples:
    def setup_method(self):
        self.partitioner = SamplePartitioner(GEKOConfig())

    def test_returns_correct_ids(self):
        states = {
            "a": make_state("a", bucket=Bucket.HARD),
            "b": make_state("b", bucket=Bucket.FOCUS),
            "c": make_state("c", bucket=Bucket.HARD),
        }
        hard_ids = self.partitioner.get_bucket_samples(states, Bucket.HARD)
        assert set(hard_ids) == {"a", "c"}

    def test_returns_empty_for_no_matches(self):
        states = {
            "a": make_state("a", bucket=Bucket.FOCUS),
        }
        assert self.partitioner.get_bucket_samples(states, Bucket.FREEZE) == []


# ---------------------------------------------------------------------------
# SamplePartitioner.compute_efficiency() / should_stop_early()
# ---------------------------------------------------------------------------

class TestEfficiencyAndEarlyStopping:
    def setup_method(self):
        self.partitioner = SamplePartitioner(GEKOConfig())

    def test_efficiency_is_freeze_plus_light(self):
        stats = PartitionStats(total=10, freeze_count=4, light_count=2, focus_count=3, hard_count=1)
        assert self.partitioner.compute_efficiency(stats) == pytest.approx(0.6)

    def test_should_stop_early_above_threshold(self):
        stats = PartitionStats(total=10, freeze_count=10, light_count=0, focus_count=0, hard_count=0)
        assert self.partitioner.should_stop_early(stats, threshold=0.95) is True

    def test_should_not_stop_early_below_threshold(self):
        stats = PartitionStats(total=10, freeze_count=8, light_count=0, focus_count=1, hard_count=1)
        assert self.partitioner.should_stop_early(stats, threshold=0.95) is False
