"""
Epiplexity-weighted dataset sampler with curriculum scheduling.

EpiplexityWeightedSampler reads per-dataset scores from a JSON file produced
by the upstream epiplexity measurement stage (e.g., using Qwen2.5-0.5B as
probe model). Higher epiplexity score = more sampling weight for that dataset.

The CurriculumScheduler (see data/curriculum.py) controls how sampling weights
evolve over training — from deterministic high-epi focus to uniform to
inverted orderings.

Scores are model-agnostic — this class only reads the JSON and normalizes
weights, so the upstream probe can be swapped without changes here.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, concatenate_datasets

from data.curriculum import CurriculumScheduler, CurriculumStrategy


def load_epiplexity_scores(scores_path: str | Path) -> dict[str, float]:
    """Load epiplexity scores from JSON, stripping metadata keys."""
    with open(scores_path) as f:
        raw = json.load(f)
    return {k: float(v) for k, v in raw.items() if not k.startswith("_")}


class EpiplexityWeightedSampler:
    """
    Combines multiple datasets into a single weighted dataset for curriculum
    training. Sampling probability for each dataset is controlled by a
    CurriculumScheduler which evolves weights over training steps.

    Args:
        dataset_dict: Mapping of dataset name -> HuggingFace Dataset.
            Each dataset must have columns: prompt, answer, dataset, category.
        scheduler: CurriculumScheduler controlling weight evolution.
    """

    def __init__(
        self,
        dataset_dict: dict[str, Dataset],
        scheduler: CurriculumScheduler,
    ):
        self.dataset_dict = dataset_dict
        self.scheduler = scheduler
        self._combined: Optional[Dataset] = None
        self._current_weights: dict[str, float] = scheduler.get_weights(0)
        # Map each example index → its dataset name (for weight updates)
        self._index_to_dataset: list[str] = []

    def build_combined_dataset(self) -> Dataset:
        """
        Concatenate all datasets with a per-example 'sample_weight' column.

        Returns a Dataset with columns:
            prompt, answer, dataset, category, sample_weight
        """
        weights = self._current_weights
        parts = []
        self._index_to_dataset = []
        for name, ds in self.dataset_dict.items():
            prob = weights.get(name, 1.0 / len(self.dataset_dict))
            # Assign weight proportional to sampling probability, normalized by dataset size
            weight_per_example = prob / max(len(ds), 1)
            ds_with_weight = ds.map(
                lambda ex, w=weight_per_example: {**ex, "sample_weight": w},
                desc=f"Attaching weights for {name}",
            )
            parts.append(ds_with_weight)
            self._index_to_dataset.extend([name] * len(ds))

        self._combined = concatenate_datasets(parts)
        return self._combined

    def get_torch_sampler(
        self, combined_dataset: Optional[Dataset] = None
    ) -> torch.utils.data.WeightedRandomSampler:
        """
        Build a torch WeightedRandomSampler from the combined dataset's
        sample_weight column. Use this with a DataLoader for weighted sampling.
        """
        ds = combined_dataset or self._combined
        if ds is None:
            raise RuntimeError("Call build_combined_dataset() before get_torch_sampler().")
        weights = torch.tensor(ds["sample_weight"], dtype=torch.float)
        return torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

    def update_weights_for_step(self, step: int) -> torch.Tensor:
        """
        Recompute sampling weights for a given training step using the
        curriculum scheduler. Returns updated weight tensor suitable for
        reassigning to an existing WeightedRandomSampler.
        """
        self._current_weights = self.scheduler.get_weights(step)
        # Rebuild per-example weights
        new_weights = []
        for ds_name in self._index_to_dataset:
            prob = self._current_weights.get(ds_name, 1.0 / len(self.dataset_dict))
            ds_size = sum(1 for n in self._index_to_dataset if n == ds_name)
            new_weights.append(prob / max(ds_size, 1))
        return torch.tensor(new_weights, dtype=torch.float)

    def print_weights(self, step: int = 0) -> None:
        weights = self.scheduler.get_weights(step)
        print(f"Curriculum sampling weights (step={step}, strategy={self.scheduler.strategy.value}):")
        for name, prob in sorted(weights.items()):
            score = self.scheduler.raw_scores.get(name, "—")
            print(f"  {name}: {prob:.4f} (raw score: {score})")
