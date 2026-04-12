"""
Epiplexity probe pipeline.

Measures per-dataset "epiplexity" (perplexity-based difficulty) using a small
probe model (default: Qwen2.5-0.5B-Instruct). Writes scores to a JSON file
consumed by the GRPO training pipeline's EpiplexityWeightedSampler.

The score for each dataset is the mean token-level perplexity of the probe
model on that dataset's formatted prompts+answers. Higher perplexity indicates
the dataset is "harder" for the model family, and should receive more
curriculum weight during training.

Usage:
    python probe_epiplexity.py \\
        --probe-model Qwen/Qwen2.5-0.5B-Instruct \\
        --output data/epiplexity_scores.json \\
        [--max-samples 500] \\
        [--batch-size 8] \\
        [--max-length 1024]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.registry import get_registry_with_formatters
from data.datasets import load_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-dataset epiplexity with a probe model"
    )
    parser.add_argument(
        "--probe-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model to use as perplexity probe",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/epiplexity_scores.json",
        help="Output path for the scores JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Max samples per dataset to evaluate (speeds up probing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for perplexity computation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
        help="Max token length for tokenized prompt+answer sequences",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run probe on (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset of datasets to probe (default: all). E.g. --datasets gsm8k math",
    )
    return parser.parse_args()


def get_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def compute_dataset_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> dict:
    """
    Compute mean per-token perplexity over a list of texts.

    Returns dict with:
        mean_perplexity:   geometric mean of per-example perplexities
        median_perplexity: median per-example perplexity
        std_perplexity:    standard deviation
        num_examples:      number of examples evaluated
        total_tokens:      total tokens processed
    """
    model.eval()
    all_ppls: list[float] = []
    total_tokens = 0

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]

        encodings = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Shift labels: we want loss only on non-padding tokens
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        # Compute per-example loss by manually computing cross-entropy
        # outputs.loss is the mean over the whole batch; we need per-example
        logits = outputs.logits[:, :-1, :]  # (B, seq_len-1, vocab)
        shift_labels = labels[:, 1:]         # (B, seq_len-1)

        loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_token_loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1),
        ).reshape(shift_labels.shape)  # (B, seq_len-1)

        # Mean loss per example (only over non-padding tokens)
        mask = (shift_labels != -100).float()
        tokens_per_example = mask.sum(dim=1)
        example_losses = (per_token_loss * mask).sum(dim=1) / tokens_per_example.clamp(min=1)

        for i, (loss_val, n_tokens) in enumerate(
            zip(example_losses.cpu().tolist(), tokens_per_example.cpu().tolist())
        ):
            if n_tokens > 0:
                ppl = math.exp(min(loss_val, 100))  # clamp to avoid inf
                all_ppls.append(ppl)
                total_tokens += int(n_tokens)

    if not all_ppls:
        return {
            "mean_perplexity": float("inf"),
            "median_perplexity": float("inf"),
            "std_perplexity": 0.0,
            "num_examples": 0,
            "total_tokens": 0,
        }

    sorted_ppls = sorted(all_ppls)
    n = len(sorted_ppls)
    mean_ppl = sum(sorted_ppls) / n
    median_ppl = (
        sorted_ppls[n // 2]
        if n % 2 == 1
        else (sorted_ppls[n // 2 - 1] + sorted_ppls[n // 2]) / 2
    )
    variance = sum((p - mean_ppl) ** 2 for p in sorted_ppls) / n
    std_ppl = math.sqrt(variance)

    return {
        "mean_perplexity": round(mean_ppl, 4),
        "median_perplexity": round(median_ppl, 4),
        "std_perplexity": round(std_ppl, 4),
        "num_examples": n,
        "total_tokens": total_tokens,
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load probe model
    print(f"Loading probe model: {args.probe_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.probe_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.probe_model,
        torch_dtype=dtype,
        device_map=device if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    # Load datasets via the shared registry
    registry = get_registry_with_formatters()
    if args.datasets:
        registry = {k: v for k, v in registry.items() if k in args.datasets}
        missing = set(args.datasets) - set(registry)
        if missing:
            print(f"Warning: unknown datasets ignored: {missing}")

    dataset_dict = load_all_datasets(registry, max_samples_per_dataset=args.max_samples)

    # Compute perplexity per dataset
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for name, ds in dataset_dict.items():
        print(f"\nProbing {name} ({len(ds)} examples)...")
        start_time = time.time()

        # Concatenate prompt + answer as the full sequence the probe evaluates
        texts = [f"{ex['prompt']} {ex['answer']}" for ex in ds]

        result = compute_dataset_perplexity(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )

        elapsed = time.time() - start_time
        scores[name] = result["mean_perplexity"]
        details[name] = {**result, "elapsed_seconds": round(elapsed, 1)}

        print(
            f"  mean_ppl={result['mean_perplexity']:.2f}  "
            f"median_ppl={result['median_perplexity']:.2f}  "
            f"std={result['std_perplexity']:.2f}  "
            f"tokens={result['total_tokens']:,}  "
            f"time={elapsed:.1f}s"
        )

    # Write scores JSON (consumed by EpiplexityWeightedSampler)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        **scores,
        "_metadata": {
            "probe_model": args.probe_model,
            "max_samples_per_dataset": args.max_samples,
            "max_length": args.max_length,
            "details": details,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Epiplexity scores written to: {output_path}")
    print(f"{'='*60}")
    print("\nScores (higher = harder for probe → more training weight):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: {score:.2f}")


if __name__ == "__main__":
    main()
