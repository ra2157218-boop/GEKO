"""
Unit tests for geko/core.py

Covers: Bucket, SampleState, GEKOConfig
"""

import pytest
from collections import deque
from geko.core import Bucket, SampleState, GEKOConfig


# ---------------------------------------------------------------------------
# Bucket
# ---------------------------------------------------------------------------

class TestBucket:
    def test_all_four_buckets_exist(self):
        names = {b.value for b in Bucket}
        assert names == {"FREEZE", "LIGHT", "FOCUS", "HARD"}

    def test_str_returns_value(self):
        assert str(Bucket.HARD) == "HARD"
        assert str(Bucket.FREEZE) == "FREEZE"


# ---------------------------------------------------------------------------
# SampleState — initialization
# ---------------------------------------------------------------------------

class TestSampleStateInit:
    def test_default_bucket_is_focus(self):
        s = SampleState(sample_id="0")
        assert s.bucket == Bucket.FOCUS

    def test_default_q_value(self):
        s = SampleState(sample_id="0")
        assert s.q_value == 0.5

    def test_default_quality_is_one(self):
        s = SampleState(sample_id="0")
        assert s.quality == 1.0

    def test_loss_history_is_deque(self):
        s = SampleState(sample_id="0")
        assert isinstance(s.loss_history, deque)
        assert s.loss_history.maxlen == 5

    def test_times_seen_starts_at_zero(self):
        s = SampleState(sample_id="0")
        assert s.times_seen == 0


# ---------------------------------------------------------------------------
# SampleState — update()
# ---------------------------------------------------------------------------

class TestSampleStateUpdate:
    def test_update_increments_times_seen(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.7, correct=True)
        assert s.times_seen == 1

    def test_update_stores_last_loss(self):
        s = SampleState(sample_id="0")
        s.update(loss=2.5, confidence=0.5, correct=False)
        assert s.last_loss == 2.5

    def test_update_stores_confidence(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.9, correct=True)
        assert s.confidence == 0.9

    def test_update_stores_correct(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.9, correct=True)
        assert s.correct is True

    def test_loss_history_grows(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.5, correct=False)
        s.update(loss=2.0, confidence=0.5, correct=False)
        assert list(s.loss_history) == [1.0, 2.0]

    def test_loss_history_maxlen_is_five(self):
        s = SampleState(sample_id="0")
        for i in range(7):
            s.update(loss=float(i), confidence=0.5, correct=False)
        assert len(s.loss_history) == 5
        # Oldest entries were dropped; last 5 should be 2..6
        assert list(s.loss_history) == [2.0, 3.0, 4.0, 5.0, 6.0]


# ---------------------------------------------------------------------------
# SampleState — Q-value
# ---------------------------------------------------------------------------

class TestQValue:
    def test_q_value_increases_on_low_loss(self):
        s = SampleState(sample_id="0", q_value=0.5)
        # loss=0 → contribution = lr * 1.0 → Q goes up
        initial = s.q_value
        s.update(loss=0.0, confidence=0.5, correct=True, lr=0.1)
        assert s.q_value > initial

    def test_q_value_decreases_on_high_loss(self):
        s = SampleState(sample_id="0", q_value=0.5)
        # loss=10 → contribution = lr * 0.0 → Q goes down
        initial = s.q_value
        s.update(loss=10.0, confidence=0.5, correct=False, lr=0.1)
        assert s.q_value < initial

    def test_q_value_uses_lr_parameter(self):
        s1 = SampleState(sample_id="0", q_value=0.5)
        s2 = SampleState(sample_id="1", q_value=0.5)
        # Higher LR → bigger update
        s1.update(loss=0.0, confidence=0.5, correct=True, lr=0.5)
        s2.update(loss=0.0, confidence=0.5, correct=True, lr=0.1)
        assert s1.q_value > s2.q_value

    def test_q_value_uses_loss_scale(self):
        # A loss of 2.0 is 100% of loss_scale=2 → Q contribution = 0
        # But only 20% of loss_scale=10 → Q contribution = 0.8
        s_small = SampleState(sample_id="0", q_value=0.5)
        s_large = SampleState(sample_id="1", q_value=0.5)
        s_small.update(loss=2.0, confidence=0.5, correct=False, lr=0.1, loss_scale=2.0)
        s_large.update(loss=2.0, confidence=0.5, correct=False, lr=0.1, loss_scale=10.0)
        # With loss_scale=2, loss/scale=1.0 → contribution=0 → Q goes DOWN
        # With loss_scale=10, loss/scale=0.2 → contribution=0.8 → Q goes UP
        assert s_small.q_value < s_large.q_value

    def test_q_value_loss_scale_default_is_ten(self):
        s1 = SampleState(sample_id="0", q_value=0.5)
        s2 = SampleState(sample_id="1", q_value=0.5)
        # Explicit loss_scale=10 should match implicit default
        s1.update(loss=5.0, confidence=0.5, correct=False, lr=0.1)
        s2.update(loss=5.0, confidence=0.5, correct=False, lr=0.1, loss_scale=10.0)
        assert s1.q_value == pytest.approx(s2.q_value)


# ---------------------------------------------------------------------------
# SampleState — quality auto-computation
# ---------------------------------------------------------------------------

class TestQualityAutoCompute:
    def test_quality_stays_one_with_single_update(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.5, correct=True)
        # Only one loss → quality unchanged (needs >= 2 losses)
        assert s.quality == 1.0

    def test_quality_high_for_consistent_losses(self):
        s = SampleState(sample_id="0")
        for _ in range(5):
            s.update(loss=1.0, confidence=0.5, correct=True)  # zero variance
        # std=0 → quality = 1.0
        assert s.quality == pytest.approx(1.0)

    def test_quality_low_for_volatile_losses(self):
        s = SampleState(sample_id="0")
        for l in [0.0, 10.0, 0.0, 10.0, 0.0]:
            s.update(loss=l, confidence=0.5, correct=True)
        # High variance → quality near 0
        assert s.quality < 0.5

    def test_quality_between_zero_and_one(self):
        s = SampleState(sample_id="0")
        for l in [0.0, 5.0, 2.0, 8.0, 1.0]:
            s.update(loss=l, confidence=0.5, correct=True)
        assert 0.0 <= s.quality <= 1.0


# ---------------------------------------------------------------------------
# SampleState — loss trend (linear regression slope)
# ---------------------------------------------------------------------------

class TestLossTrend:
    def test_trend_zero_with_single_loss(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.5, correct=True)
        assert s._compute_loss_trend() == 0.0

    def test_trend_negative_for_decreasing_losses(self):
        s = SampleState(sample_id="0")
        for l in [5.0, 4.0, 3.0, 2.0, 1.0]:
            s.update(loss=l, confidence=0.5, correct=True)
        assert s._compute_loss_trend() < 0

    def test_trend_positive_for_increasing_losses(self):
        s = SampleState(sample_id="0")
        for l in [1.0, 2.0, 3.0, 4.0, 5.0]:
            s.update(loss=l, confidence=0.5, correct=True)
        assert s._compute_loss_trend() > 0

    def test_trend_zero_for_flat_losses(self):
        s = SampleState(sample_id="0")
        for _ in range(5):
            s.update(loss=3.0, confidence=0.5, correct=True)
        assert s._compute_loss_trend() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SampleState — is_improving / is_stuck
# ---------------------------------------------------------------------------

class TestImprovingAndStuck:
    def test_is_improving_with_decreasing_losses(self):
        s = SampleState(sample_id="0")
        for l in [5.0, 3.0, 1.0, 0.5, 0.1]:
            s.update(loss=l, confidence=0.5, correct=True)
        assert s.is_improving is True

    def test_not_improving_with_flat_losses(self):
        s = SampleState(sample_id="0")
        for _ in range(5):
            s.update(loss=3.0, confidence=0.5, correct=True)
        assert s.is_improving is False

    def test_is_stuck_requires_many_steps(self):
        s = SampleState(sample_id="0")
        for _ in range(5):
            s.update(loss=3.0, confidence=0.5, correct=True)
        # times_seen=5, not > 10
        assert s.is_stuck is False

    def test_is_stuck_when_seen_enough_and_not_improving(self):
        s = SampleState(sample_id="0")
        for _ in range(15):
            s.update(loss=3.0, confidence=0.5, correct=True)
        assert s.is_stuck is True


# ---------------------------------------------------------------------------
# SampleState — priority_score
# ---------------------------------------------------------------------------

class TestPriorityScore:
    def _make_state(self, bucket: Bucket, confidence: float = 0.5) -> SampleState:
        s = SampleState(sample_id="0", bucket=bucket, confidence=confidence)
        return s

    def test_freeze_priority_is_zero(self):
        s = self._make_state(Bucket.FREEZE)
        assert s.priority_score == 0.0

    def test_hard_priority_highest(self):
        hard = self._make_state(Bucket.HARD, confidence=0.5)
        focus = self._make_state(Bucket.FOCUS)
        light = self._make_state(Bucket.LIGHT)
        assert hard.priority_score > focus.priority_score
        assert focus.priority_score > light.priority_score

    def test_hard_confidence_boosts_priority(self):
        low_conf = self._make_state(Bucket.HARD, confidence=0.1)
        high_conf = self._make_state(Bucket.HARD, confidence=0.9)
        assert high_conf.priority_score > low_conf.priority_score


# ---------------------------------------------------------------------------
# SampleState — to_dict()
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_is_json_serializable(self):
        import json
        s = SampleState(sample_id="42")
        s.update(loss=1.5, confidence=0.7, correct=True)
        d = s.to_dict()
        # Should not raise
        json.dumps(d)

    def test_to_dict_loss_history_is_list(self):
        s = SampleState(sample_id="0")
        s.update(loss=1.0, confidence=0.5, correct=True)
        assert isinstance(s.to_dict()['loss_history'], list)

    def test_to_dict_bucket_is_string(self):
        s = SampleState(sample_id="0")
        assert isinstance(s.to_dict()['bucket'], str)

    def test_to_dict_contains_all_fields(self):
        s = SampleState(sample_id="99")
        d = s.to_dict()
        for key in ['sample_id', 'bucket', 'q_value', 'confidence', 'quality',
                    'loss_history', 'times_seen', 'last_loss', 'frozen_at_epoch', 'correct']:
            assert key in d


# ---------------------------------------------------------------------------
# GEKOConfig
# ---------------------------------------------------------------------------

class TestGEKOConfig:
    def test_default_config_validates(self):
        # Should not raise
        config = GEKOConfig()
        config.validate()

    def test_post_init_validates(self):
        with pytest.raises(ValueError, match="focus_confidence"):
            # focus_confidence > freeze_confidence — invalid
            GEKOConfig(focus_confidence=0.90, freeze_confidence=0.50)

    def test_invalid_freeze_confidence_zero(self):
        with pytest.raises(ValueError, match="freeze_confidence"):
            GEKOConfig(freeze_confidence=0.0)

    def test_invalid_bucket_weights_wrong_length(self):
        with pytest.raises(ValueError, match="bucket_weights"):
            GEKOConfig(bucket_weights=(3, 1))

    def test_get_bucket_weight_freeze_always_zero(self):
        config = GEKOConfig(bucket_weights=(5, 2, 1))
        assert config.get_bucket_weight(Bucket.FREEZE) == 0

    def test_get_bucket_weight_hard(self):
        config = GEKOConfig(bucket_weights=(5, 2, 1))
        assert config.get_bucket_weight(Bucket.HARD) == 5

    def test_get_bucket_weight_focus(self):
        config = GEKOConfig(bucket_weights=(5, 2, 1))
        assert config.get_bucket_weight(Bucket.FOCUS) == 2

    def test_get_bucket_weight_light(self):
        config = GEKOConfig(bucket_weights=(5, 2, 1))
        assert config.get_bucket_weight(Bucket.LIGHT) == 1
