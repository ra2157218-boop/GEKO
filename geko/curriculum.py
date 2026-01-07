"""
GEKO Mountain Curriculum

A novel curriculum learning strategy that follows a "mountain" pattern:
Easy → Medium → Hard → Medium → Easy

Unlike traditional curriculum (easy to hard), Mountain Curriculum:
1. Starts easy to build foundation
2. Peaks at hard samples for maximum learning
3. Returns to easy for consolidation

This prevents catastrophic forgetting while maximizing learning efficiency.
"""

from enum import Enum
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from .core import Bucket, SampleState, GEKOConfig


class CurriculumPhase(Enum):
    """Mountain Curriculum phases."""
    WARMUP = "warmup"       # Phase 1: Easy samples (LIGHT + some FOCUS)
    ASCENT = "ascent"       # Phase 2: Medium difficulty (FOCUS heavy)
    PEAK = "peak"           # Phase 3: Hard samples (HARD heavy)
    DESCENT = "descent"     # Phase 4: Back to medium (FOCUS + LIGHT)
    CONSOLIDATE = "consolidate"  # Phase 5: Easy again (reinforce learning)


@dataclass
class CurriculumState:
    """Tracks curriculum progression."""
    current_phase: CurriculumPhase
    phase_progress: float  # 0.0 to 1.0 within current phase
    total_progress: float  # 0.0 to 1.0 overall
    samples_this_phase: int
    phase_target: int


class MountainCurriculum:
    """
    Implements the Mountain Curriculum strategy.

    The "mountain" shape:

         PEAK
          /\\
         /  \\
        /    \\
       /      \\
    WARMUP    CONSOLIDATE

    Each phase adjusts bucket weights to control difficulty:
    - WARMUP: Focus on LIGHT samples (easy)
    - ASCENT: Shift toward FOCUS samples
    - PEAK: Maximum weight on HARD samples
    - DESCENT: Reduce HARD, increase FOCUS
    - CONSOLIDATE: Back to LIGHT for reinforcement

    Usage:
        curriculum = MountainCurriculum(total_samples=1000000, config=config)

        for batch in dataloader:
            weights = curriculum.get_current_weights()
            # Use weights for sampling...

            curriculum.step(batch_size)

            if curriculum.phase_changed:
                print(f"Entered phase: {curriculum.current_phase}")
    """

    def __init__(
        self,
        total_samples: int,
        config: Optional[GEKOConfig] = None,
        num_phases: int = 5
    ):
        self.config = config or GEKOConfig()
        self.total_samples = total_samples
        self.num_phases = num_phases

        # Phase distribution (as fraction of total training)
        self.phase_fractions = {
            CurriculumPhase.WARMUP: 0.15,      # 15%
            CurriculumPhase.ASCENT: 0.20,      # 20%
            CurriculumPhase.PEAK: 0.30,        # 30% (longest)
            CurriculumPhase.DESCENT: 0.20,     # 20%
            CurriculumPhase.CONSOLIDATE: 0.15  # 15%
        }

        # Bucket weights per phase: (HARD, FOCUS, LIGHT)
        self.phase_weights = {
            CurriculumPhase.WARMUP: (1, 2, 3),       # Easy heavy
            CurriculumPhase.ASCENT: (2, 3, 1),       # Medium
            CurriculumPhase.PEAK: (5, 2, 0),         # Hard heavy
            CurriculumPhase.DESCENT: (2, 3, 1),      # Back to medium
            CurriculumPhase.CONSOLIDATE: (1, 2, 3),  # Easy again
        }

        # State
        self.samples_seen = 0
        self.current_phase = CurriculumPhase.WARMUP
        self.phase_changed = False
        self._last_phase = CurriculumPhase.WARMUP

    def step(self, num_samples: int = 1):
        """
        Advance curriculum by num_samples.

        Call this after each batch to track progress.
        """
        self.samples_seen += num_samples
        self._update_phase()

    def _update_phase(self):
        """Update current phase based on progress."""
        self._last_phase = self.current_phase
        progress = self.samples_seen / self.total_samples

        cumulative = 0.0
        for phase in CurriculumPhase:
            cumulative += self.phase_fractions[phase]
            if progress < cumulative:
                self.current_phase = phase
                break
        else:
            self.current_phase = CurriculumPhase.CONSOLIDATE

        self.phase_changed = (self.current_phase != self._last_phase)

    def get_current_weights(self) -> Tuple[int, int, int]:
        """
        Get bucket weights (HARD, FOCUS, LIGHT) for current phase.

        Returns:
            Tuple of weights for (HARD, FOCUS, LIGHT) sampling
        """
        return self.phase_weights[self.current_phase]

    def get_phase_progress(self) -> float:
        """Get progress within current phase (0.0 to 1.0)."""
        progress = self.samples_seen / self.total_samples

        # Find start of current phase
        cumulative = 0.0
        phase_start = 0.0
        for phase in CurriculumPhase:
            if phase == self.current_phase:
                phase_start = cumulative
                break
            cumulative += self.phase_fractions[phase]

        phase_size = self.phase_fractions[self.current_phase]
        phase_progress = (progress - phase_start) / phase_size

        return min(max(phase_progress, 0.0), 1.0)

    def get_total_progress(self) -> float:
        """Get overall curriculum progress (0.0 to 1.0)."""
        return min(self.samples_seen / self.total_samples, 1.0)

    def get_state(self) -> CurriculumState:
        """Get current curriculum state."""
        progress = self.samples_seen / self.total_samples

        # Calculate phase target
        cumulative = 0.0
        for phase in CurriculumPhase:
            cumulative += self.phase_fractions[phase]
            if phase == self.current_phase:
                break
        phase_target = int(cumulative * self.total_samples)

        return CurriculumState(
            current_phase=self.current_phase,
            phase_progress=self.get_phase_progress(),
            total_progress=progress,
            samples_this_phase=self.samples_seen,
            phase_target=phase_target,
        )

    def get_difficulty_multiplier(self) -> float:
        """
        Get current difficulty multiplier (0.0 = easy, 1.0 = hard).

        Follows the mountain shape:
        - Starts at 0.2 (warmup)
        - Peaks at 1.0 (peak phase)
        - Returns to 0.2 (consolidate)
        """
        multipliers = {
            CurriculumPhase.WARMUP: 0.2,
            CurriculumPhase.ASCENT: 0.5,
            CurriculumPhase.PEAK: 1.0,
            CurriculumPhase.DESCENT: 0.5,
            CurriculumPhase.CONSOLIDATE: 0.2,
        }
        return multipliers[self.current_phase]

    def adjust_learning_rate(self, base_lr: float) -> float:
        """
        Adjust learning rate based on curriculum phase.

        - Higher LR during PEAK (aggressive learning)
        - Lower LR during CONSOLIDATE (gentle refinement)
        """
        lr_multipliers = {
            CurriculumPhase.WARMUP: 0.5,
            CurriculumPhase.ASCENT: 0.8,
            CurriculumPhase.PEAK: 1.0,
            CurriculumPhase.DESCENT: 0.8,
            CurriculumPhase.CONSOLIDATE: 0.3,
        }
        return base_lr * lr_multipliers[self.current_phase]

    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.samples_seen >= self.total_samples

    def reset(self):
        """Reset curriculum to beginning."""
        self.samples_seen = 0
        self.current_phase = CurriculumPhase.WARMUP
        self.phase_changed = False
        self._last_phase = CurriculumPhase.WARMUP

    def __str__(self) -> str:
        return (
            f"MountainCurriculum("
            f"phase={self.current_phase.value}, "
            f"progress={self.get_total_progress():.1%}, "
            f"weights={self.get_current_weights()})"
        )


def visualize_mountain(num_points: int = 50) -> str:
    """
    Generate ASCII visualization of mountain curriculum.

    Returns a string showing the difficulty curve.
    """
    lines = []
    height = 10

    # Generate mountain shape
    difficulties = []
    for i in range(num_points):
        progress = i / num_points

        if progress < 0.15:  # WARMUP
            d = 0.2 + (progress / 0.15) * 0.3
        elif progress < 0.35:  # ASCENT
            d = 0.5 + ((progress - 0.15) / 0.20) * 0.5
        elif progress < 0.65:  # PEAK
            d = 1.0
        elif progress < 0.85:  # DESCENT
            d = 1.0 - ((progress - 0.65) / 0.20) * 0.5
        else:  # CONSOLIDATE
            d = 0.5 - ((progress - 0.85) / 0.15) * 0.3

        difficulties.append(d)

    # Draw
    for row in range(height, 0, -1):
        line = ""
        threshold = row / height
        for d in difficulties:
            if d >= threshold:
                line += "█"
            else:
                line += " "
        lines.append(f"{row:2d}│{line}│")

    lines.append("  └" + "─" * num_points + "┘")
    lines.append("   WARMUP→ASCENT→PEAK→DESCENT→CONSOLIDATE")

    return "\n".join(lines)
