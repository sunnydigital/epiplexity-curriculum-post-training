"""
Curriculum scheduling strategies for epiplexity-weighted training.

Controls how sampling weights evolve over the course of training.
Each strategy defines a mapping:
    (step, total_steps, raw_scores) → sampling_weights

Strategies:
    uniform:          Equal weights throughout (baseline)
    high_first:       Start sharply focused on high-epi datasets, anneal to uniform
    low_first:        Start sharply focused on low-epi datasets, anneal to uniform
    high_constant:    Fixed sharp focus on high-epi datasets
    low_constant:     Fixed sharp focus on low-epi datasets
    anneal_to_high:   Start uniform, progressively sharpen toward high-epi
    single:           Train on a single dataset only (set via --single-dataset flag)

All strategies use a mixture kernel:
    final_weights = (1 - uniform_mix) * curriculum_weights + uniform_mix * uniform_weights

This prevents any dataset from being completely starved at any training stage.
"""
from __future__ import annotations

import math
from enum import Enum


class CurriculumStrategy(str, Enum):
    UNIFORM = "uniform"
    HIGH_FIRST = "high_first"
    LOW_FIRST = "low_first"
    HIGH_CONSTANT = "high_constant"
    LOW_CONSTANT = "low_constant"
    ANNEAL_TO_HIGH = "anneal_to_high"
    SINGLE = "single"


class CurriculumScheduler:
    """
    Computes per-dataset sampling weights at each training step based on
    the chosen curriculum strategy.

    Args:
        strategy: Which curriculum strategy to use.
        raw_scores: Dict of {dataset_name: epiplexity_score}.
        total_steps: Total training steps for annealing schedules.
        temp_start: Starting temperature for annealing strategies.
        temp_end: Ending temperature for annealing strategies.
        floor_weight: Minimum sampling probability per dataset.
        uniform_mix: Fraction of uniform distribution mixed in (prevents starvation).
        single_dataset: Dataset name for SINGLE strategy.
    """

    def __init__(
        self,
        strategy: CurriculumStrategy | str,
        raw_scores: dict[str, float],
        total_steps: int = 500,
        temp_start: float = 0.1,
        temp_end: float = 5.0,
        floor_weight: float = 0.02,
        uniform_mix: float = 0.1,
        single_dataset: str | None = None,
    ):
        if isinstance(strategy, str):
            strategy = CurriculumStrategy(strategy)
        self.strategy = strategy
        self.raw_scores = raw_scores
        self.total_steps = max(total_steps, 1)
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.floor_weight = floor_weight
        self.uniform_mix = uniform_mix
        self.single_dataset = single_dataset
        self._names = sorted(raw_scores.keys())
        self._n = len(self._names)

        if strategy == CurriculumStrategy.SINGLE and single_dataset:
            if single_dataset not in raw_scores:
                raise ValueError(
                    f"single_dataset='{single_dataset}' not in scores: {list(raw_scores)}"
                )

    def get_weights(self, step: int) -> dict[str, float]:
        """
        Compute sampling probabilities for the given training step.

        Args:
            step: Current training step (0-indexed).

        Returns:
            Dict mapping dataset name to sampling probability (sums to 1.0).
        """
        if self.strategy == CurriculumStrategy.UNIFORM:
            return self._uniform()

        if self.strategy == CurriculumStrategy.SINGLE:
            return self._single()

        if self.strategy == CurriculumStrategy.HIGH_CONSTANT:
            return self._softmax_weights(temperature=self.temp_start, direction=1.0)

        if self.strategy == CurriculumStrategy.LOW_CONSTANT:
            return self._softmax_weights(temperature=self.temp_start, direction=-1.0)

        # Annealing strategies: interpolate temperature over training
        progress = min(step / self.total_steps, 1.0)

        if self.strategy == CurriculumStrategy.HIGH_FIRST:
            # Sharp on high-epi → uniform: temperature low → high
            temp = self.temp_start + progress * (self.temp_end - self.temp_start)
            return self._softmax_weights(temperature=temp, direction=1.0)

        if self.strategy == CurriculumStrategy.LOW_FIRST:
            # Sharp on low-epi → uniform: temperature low → high
            temp = self.temp_start + progress * (self.temp_end - self.temp_start)
            return self._softmax_weights(temperature=temp, direction=-1.0)

        if self.strategy == CurriculumStrategy.ANNEAL_TO_HIGH:
            # Uniform → sharp on high-epi: temperature high → low
            temp = self.temp_end - progress * (self.temp_end - self.temp_start)
            return self._softmax_weights(temperature=temp, direction=1.0)

        return self._uniform()

    def _uniform(self) -> dict[str, float]:
        w = 1.0 / self._n
        return {name: w for name in self._names}

    def _single(self) -> dict[str, float]:
        weights = {name: 0.0 for name in self._names}
        if self.single_dataset:
            weights[self.single_dataset] = 1.0
        else:
            return self._uniform()
        return weights

    def _softmax_weights(self, temperature: float, direction: float) -> dict[str, float]:
        """
        Compute softmax weights with floor and uniform mixture.

        direction = +1.0: high scores get high weight
        direction = -1.0: low scores get high weight (inverted curriculum)
        """
        temperature = max(temperature, 1e-6)  # prevent division by zero
        scores = [direction * self.raw_scores[n] / temperature for n in self._names]
        max_s = max(scores)
        exp_vals = [math.exp(s - max_s) for s in scores]
        total = sum(exp_vals)
        softmax_probs = [e / total for e in exp_vals]

        # Apply floor
        floored = [max(p, self.floor_weight) for p in softmax_probs]
        total_f = sum(floored)
        curriculum_weights = [p / total_f for p in floored]

        # Mix with uniform distribution to prevent starvation
        uniform_w = 1.0 / self._n
        final = [
            (1.0 - self.uniform_mix) * cw + self.uniform_mix * uniform_w
            for cw in curriculum_weights
        ]
        total_final = sum(final)
        final = [f / total_final for f in final]

        return dict(zip(self._names, final))

    def describe(self) -> str:
        """Human-readable description of the curriculum strategy."""
        if self.strategy == CurriculumStrategy.UNIFORM:
            return "Uniform sampling (baseline)"
        if self.strategy == CurriculumStrategy.SINGLE:
            return f"Single dataset: {self.single_dataset}"
        if self.strategy == CurriculumStrategy.HIGH_FIRST:
            return f"High-epi first → uniform (temp {self.temp_start}→{self.temp_end})"
        if self.strategy == CurriculumStrategy.LOW_FIRST:
            return f"Low-epi first → uniform (temp {self.temp_start}→{self.temp_end})"
        if self.strategy == CurriculumStrategy.HIGH_CONSTANT:
            return f"Constant high-epi focus (temp={self.temp_start})"
        if self.strategy == CurriculumStrategy.LOW_CONSTANT:
            return f"Constant low-epi focus (temp={self.temp_start})"
        if self.strategy == CurriculumStrategy.ANNEAL_TO_HIGH:
            return f"Uniform → high-epi focus (temp {self.temp_end}→{self.temp_start})"
        return str(self.strategy)

    def log_weights_at_step(self, step: int) -> None:
        weights = self.get_weights(step)
        progress = step / self.total_steps * 100
        print(f"  [step {step}/{self.total_steps} ({progress:.0f}%)] weights: ", end="")
        print("  ".join(f"{n}={w:.3f}" for n, w in weights.items()))
