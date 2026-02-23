"""
Unit tests for geko/curriculum.py

Covers: MountainCurriculum phases, weights, progress, LR adjustment
"""

import pytest
from geko.core import GEKOConfig
from geko.curriculum import MountainCurriculum, CurriculumPhase


TOTAL = 1000  # convenient total for phase boundary tests


def make_curriculum(total: int = TOTAL) -> MountainCurriculum:
    return MountainCurriculum(total_samples=total, config=GEKOConfig())


def advance_to(curriculum: MountainCurriculum, fraction: float):
    """Advance curriculum to a given fraction of total_samples."""
    target = int(fraction * curriculum.total_samples)
    curriculum.step(target - curriculum.samples_seen)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_starts_in_warmup(self):
        c = make_curriculum()
        assert c.current_phase == CurriculumPhase.WARMUP

    def test_initial_progress_is_zero(self):
        c = make_curriculum()
        assert c.get_total_progress() == 0.0

    def test_initial_phase_changed_is_false(self):
        c = make_curriculum()
        assert c.phase_changed is False


# ---------------------------------------------------------------------------
# Phase weights
# ---------------------------------------------------------------------------

class TestPhaseWeights:
    def test_warmup_weights(self):
        c = make_curriculum()
        assert c.get_current_weights() == (1, 2, 3)

    def test_ascent_weights(self):
        c = make_curriculum()
        advance_to(c, 0.20)  # into ASCENT (starts at 15%)
        assert c.current_phase == CurriculumPhase.ASCENT
        assert c.get_current_weights() == (2, 3, 1)

    def test_peak_weights(self):
        c = make_curriculum()
        advance_to(c, 0.50)  # into PEAK (starts at 35%)
        assert c.current_phase == CurriculumPhase.PEAK
        assert c.get_current_weights() == (5, 2, 0)

    def test_descent_weights(self):
        c = make_curriculum()
        advance_to(c, 0.75)  # into DESCENT (starts at 65%)
        assert c.current_phase == CurriculumPhase.DESCENT
        assert c.get_current_weights() == (2, 3, 1)

    def test_consolidate_weights(self):
        c = make_curriculum()
        advance_to(c, 0.90)  # into CONSOLIDATE (starts at 85%)
        assert c.current_phase == CurriculumPhase.CONSOLIDATE
        assert c.get_current_weights() == (1, 2, 3)


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_phases_in_order(self):
        c = make_curriculum()
        seen_phases = [c.current_phase]

        fractions = [0.20, 0.50, 0.75, 0.90]
        for frac in fractions:
            advance_to(c, frac)
            if c.current_phase not in seen_phases:
                seen_phases.append(c.current_phase)

        expected = [
            CurriculumPhase.WARMUP,
            CurriculumPhase.ASCENT,
            CurriculumPhase.PEAK,
            CurriculumPhase.DESCENT,
            CurriculumPhase.CONSOLIDATE,
        ]
        assert seen_phases == expected

    def test_phase_changed_flag_set_on_transition(self):
        c = make_curriculum()
        # Step just past WARMUP boundary (15%)
        advance_to(c, 0.16)
        assert c.phase_changed is True
        assert c.current_phase == CurriculumPhase.ASCENT

    def test_phase_changed_flag_false_within_phase(self):
        c = make_curriculum()
        # Stay inside WARMUP
        c.step(50)  # 5% of 1000
        assert c.phase_changed is False


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class TestProgress:
    def test_total_progress_after_half(self):
        c = make_curriculum()
        c.step(500)
        assert c.get_total_progress() == pytest.approx(0.5)

    def test_total_progress_capped_at_one(self):
        c = make_curriculum()
        c.step(2000)  # beyond total
        assert c.get_total_progress() == pytest.approx(1.0)

    def test_phase_progress_within_bounds(self):
        c = make_curriculum()
        c.step(100)
        p = c.get_phase_progress()
        assert 0.0 <= p <= 1.0

    def test_is_complete_false_initially(self):
        c = make_curriculum()
        assert c.is_complete() is False

    def test_is_complete_true_at_end(self):
        c = make_curriculum()
        c.step(TOTAL)
        assert c.is_complete() is True


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_to_warmup(self):
        c = make_curriculum()
        advance_to(c, 0.50)
        c.reset()
        assert c.current_phase == CurriculumPhase.WARMUP

    def test_reset_clears_samples_seen(self):
        c = make_curriculum()
        c.step(400)
        c.reset()
        assert c.samples_seen == 0

    def test_reset_clears_phase_changed(self):
        c = make_curriculum()
        advance_to(c, 0.20)
        c.reset()
        assert c.phase_changed is False


# ---------------------------------------------------------------------------
# Difficulty multiplier
# ---------------------------------------------------------------------------

class TestDifficultyMultiplier:
    def test_warmup_is_low(self):
        c = make_curriculum()
        assert c.get_difficulty_multiplier() == pytest.approx(0.2)

    def test_peak_is_one(self):
        c = make_curriculum()
        advance_to(c, 0.50)
        assert c.get_difficulty_multiplier() == pytest.approx(1.0)

    def test_consolidate_is_low(self):
        c = make_curriculum()
        advance_to(c, 0.90)
        assert c.get_difficulty_multiplier() == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Learning rate adjustment
# ---------------------------------------------------------------------------

class TestAdjustLearningRate:
    def test_peak_lr_is_base_lr(self):
        c = make_curriculum()
        advance_to(c, 0.50)
        assert c.adjust_learning_rate(1e-4) == pytest.approx(1e-4)

    def test_warmup_lr_is_half(self):
        c = make_curriculum()
        assert c.adjust_learning_rate(1e-4) == pytest.approx(0.5e-4)

    def test_consolidate_lr_is_30_percent(self):
        c = make_curriculum()
        advance_to(c, 0.90)
        assert c.adjust_learning_rate(1e-4) == pytest.approx(0.3e-4)

    def test_lr_always_positive(self):
        c = make_curriculum()
        for frac in [0.05, 0.20, 0.50, 0.75, 0.90]:
            advance_to(c, frac)
            assert c.adjust_learning_rate(1e-4) > 0
