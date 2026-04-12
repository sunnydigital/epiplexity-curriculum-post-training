"""
Epiplexity-weighted dataset sampler.

EpiplexityWeightedSampler reads per-dataset scores from a JSON file produced
by the upstream epiplexity measurement stage (e.g., using Qwen2.5-0.5B as
probe model). Higher epiplexity score = more sampling weight for that dataset.

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


class EpiplexityWeightedSampler:
    """
    Combines multiple datasets into a single weighted dataset for curriculum
    training. Sampling probability for each dataset is proportional to its
    epiplexity score after softmax normalization.

    Args:
        dataset_dict: Mapping of dataset name -> HuggingFace Dataset.
            Each dataset must have columns: prompt, answer, dataset, category.
        scores_path: Path to JSON file with {dataset_name: float} scores.
        temperature: Softmax temperature for weight sharpening/smoothing.
            temperature < 1.0 sharpens (more curriculum effect),
            temperature > 1.0 flattens (more uniform sampling).
        floor_weight: Minimum sampling probability per dataset (before normalization).
            Prevents any dataset from being starved to near-zero.
    """

    def __init__(
        self,
        dataset_dict: dict[str, Dataset],
        scores_path: str | Path,
        temperature: float = 1.0,
        floor_weight: float = 0.05,
    ):
        self.dataset_dict = dataset_dict
        self.scores_path = Path(scores_path)
        self.temperature = temperature
        self.floor_weight = floor_weight

        self._raw_scores: dict[str, float] = {}
        self._sampling_probs: dict[str, float] = {}
        self._combined: Optional[Dataset] = None

        self._load_scores()

    def _load_scores(self) -> None:
        with open(self.scores_path) as f:
            raw = json.load(f)
        # Strip metadata keys starting with underscore
        self._raw_scores = {k: float(v) for k, v in raw.items() if not k.startswith("_")}
        self._sampling_probs = self._normalize(self._raw_scores)

    def _normalize(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Convert raw scores to sampling probabilities via softmax with floor.
        Only normalizes datasets present in both scores and dataset_dict.
        """
        names = [n for n in self.dataset_dict if n in scores]
        if not names:
            raise ValueError(
                "No overlap between scores JSON keys and dataset_dict keys. "
                f"Scores: {list(scores)}, datasets: {list(self.dataset_dict)}"
            )

        raw = [scores[n] / self.temperature for n in names]
        max_raw = max(raw)
        exp_vals = [math.exp(v - max_raw) for v in raw]  # numerically stable softmax
        total = sum(exp_vals)
        softmax_probs = [e / total for e in exp_vals]

        # Apply floor weight and renormalize
        floored = [max(p, self.floor_weight) for p in softmax_probs]
        total_floored = sum(floored)
        normalized = [p / total_floored for p in floored]

        return dict(zip(names, normalized))

    def build_combined_dataset(self) -> Dataset:
        """
        Concatenate all datasets with a per-example 'sample_weight' column
        reflecting the epiplexity-derived sampling probability.

        Returns a Dataset with columns: prompt, answer, dataset, category, sample_weight.
        The 'sample_weight' column can be used with torch WeightedRandomSampler.
        """
        parts = []
        for name, ds in self.dataset_dict.items():
            prob = self._sampling_probs.get(name, 1.0 / len(self.dataset_dict))
            ds_with_weight = ds.map(
                lambda ex, p=prob: {**ex, "sample_weight": p},
                desc=f"Attaching weights for {name}",
            )
            parts.append(ds_with_weight)

        self._combined = concatenate_datasets(parts)
        return self._combined

    def get_torch_sampler(self, combined_dataset: Optional[Dataset] = None) -> torch.utils.data.WeightedRandomSampler:
        """
        Build a torch WeightedRandomSampler from the combined dataset's
        sample_weight column. Use this with a DataLoader for weighted sampling.

        Args:
            combined_dataset: Pass the output of build_combined_dataset(), or
                leave None to use the last built dataset.
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

    def update_weights(self, new_scores: dict[str, float]) -> None:
        """
        Update sampling weights mid-training from new epiplexity scores.
        Call this between epochs or at a scheduled step interval.
        Rebuilds internal probabilities; call build_combined_dataset() again
        to propagate to the dataset (or rebuild the DataLoader sampler).

        Args:
            new_scores: Fresh {dataset_name: epiplexity_score} dict.
        """
        self._raw_scores.update(new_scores)
        self._sampling_probs = self._normalize(self._raw_scores)
        print("Updated epiplexity sampling weights:")
        for name, prob in self._sampling_probs.items():
            print(f"  {name}: {prob:.4f}")

    def print_weights(self) -> None:
        print("Current epiplexity sampling weights:")
        for name, prob in self._sampling_probs.items():
            print(f"  {name}: {prob:.4f} (raw score: {self._raw_scores.get(name, '—')})")
