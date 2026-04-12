"""
Epiplexity probe pipeline.

Estimates per-dataset epiplexity following Finzi et al. (2026),
"From Entropy to Epiplexity: Rethinking Information for Computationally
Bounded Intelligence" (arXiv:2601.03220).

Epiplexity measures the *learnable structural content* in data for a
compute-bounded observer, NOT just perplexity (model confusion). It is
estimated via the prequential coding two-part code length:

    K(X) = K(M) + K(X|M)

Where:
    K(X|M) = eval_loss * num_tokens / log(2)   [data given model, in bits]
    K(M)   ≈ K_auc = AUC of training loss curve above final loss
             [model description length, in bits — how much the model learned]

This requires *training* the probe on each dataset (not just inference).
The probe is fine-tuned separately per dataset, tracking the loss curve,
then K_auc is computed as the integral of (loss_t - loss_final).

Higher epiplexity = more learnable structure = more curriculum weight.

Usage:
    python probe_epiplexity.py \\
        --probe-model Qwen/Qwen2.5-0.5B-Instruct \\
        --output data/epiplexity_scores.json \\
        [--max-samples 500] \\
        [--batch-size 4] \\
        [--train-steps 200] \\
        [--lr 5e-5]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.registry import get_registry_with_formatters
from data.datasets import load_all_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate per-dataset epiplexity (arXiv:2601.03220)"
    )
    parser.add_argument(
        "--probe-model",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model to use as probe (same family as training target)",
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
        help="Max samples per dataset for training the probe",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for probe training",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for tokenized sequences",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Number of training steps per dataset for K_auc estimation",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate for probe fine-tuning",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=5,
        help="Log loss every N steps for K_auc curve",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Subset of datasets to probe (default: all)",
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


class TokenizedTextDataset(Dataset):
    """Wraps pre-tokenized input_ids for DataLoader iteration."""

    def __init__(self, input_ids_list: list[torch.Tensor]):
        self.input_ids_list = input_ids_list

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        return self.input_ids_list[idx]


def collate_fn(batch: list[torch.Tensor]) -> dict:
    """Pad a batch of variable-length token sequences."""
    max_len = max(t.size(0) for t in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, t in enumerate(batch):
        input_ids[i, : t.size(0)] = t
        attention_mask[i, : t.size(0)] = 1
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


@torch.no_grad()
def evaluate_loss(
    model: AutoModelForCausalLM,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, int]:
    """
    Compute mean per-token loss (nats) over the dataset.
    Returns (mean_loss, total_tokens).
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        # Count non-masked label tokens (shifted internally by HF)
        n_tokens = (labels[:, 1:] != -100).sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    return mean_loss, total_tokens


def estimate_epiplexity(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    train_steps: int,
    lr: float,
    log_interval: int,
    device: torch.device,
) -> dict:
    """
    Estimate epiplexity for a single dataset via prequential K_auc.

    1. Clone the probe model (fresh copy per dataset)
    2. Tokenize the dataset
    3. Fine-tune for `train_steps`, logging loss at each `log_interval`
    4. Compute K_auc = integral of (loss_t - loss_final) over training
    5. Compute K(X|M) = final_loss * total_tokens / log(2)
    6. Epiplexity = K(M) + K(X|M) = K_auc + K(X|M)

    Returns dict with all metrics.
    """
    # Clone the model so each dataset starts from the same base weights
    probe = copy.deepcopy(base_model).to(device)
    probe.train()

    # Tokenize
    encoded = [
        tokenizer.encode(t, truncation=True, max_length=max_length, return_tensors="pt").squeeze(0)
        for t in texts
    ]
    # Filter out empty sequences
    encoded = [e for e in encoded if e.numel() > 1]
    if not encoded:
        return _empty_result()

    dataset = TokenizedTextDataset(encoded)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.01)

    # Training loop with loss curve tracking
    loss_curve: list[tuple[int, float, int]] = []  # (step, loss_nats, tokens_seen)
    tokens_seen = 0
    step = 0
    data_iter = iter(dataloader)

    # Initial loss (before any training)
    init_loss, init_tokens = evaluate_loss(probe, dataloader, device)
    loss_curve.append((0, init_loss, 0))
    print(f"    step 0: loss={init_loss:.4f} (initial)")

    probe.train()
    while step < train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = probe(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()

        n_tokens = (labels[:, 1:] != -100).sum().item()
        tokens_seen += n_tokens
        step += 1

        if step % log_interval == 0 or step == train_steps:
            eval_loss, _ = evaluate_loss(probe, dataloader, device)
            loss_curve.append((step, eval_loss, tokens_seen))
            print(f"    step {step}: loss={eval_loss:.4f}  tokens={tokens_seen:,}")
            probe.train()

    # Final evaluation
    final_loss, total_eval_tokens = evaluate_loss(probe, dataloader, device)

    # Total tokens in dataset (for K(X|M) computation)
    total_dataset_tokens = sum(e.numel() - 1 for e in encoded)  # -1 for shift

    # ── Compute K_auc (model description length via prequential coding) ──
    # K_auc = integral of (loss_t - loss_final) dt, measured in nats * tokens,
    # then converted to bits by dividing by log(2).
    # We approximate the integral via trapezoidal rule over the loss curve.
    k_auc_nats = 0.0
    for i in range(1, len(loss_curve)):
        step_prev, loss_prev, tok_prev = loss_curve[i - 1]
        step_curr, loss_curr, tok_curr = loss_curve[i]
        delta_tokens = tok_curr - tok_prev
        avg_excess = ((loss_prev - final_loss) + (loss_curr - final_loss)) / 2.0
        k_auc_nats += max(0.0, avg_excess) * delta_tokens

    k_auc_bits = k_auc_nats / math.log(2)

    # ── K(X|M): data given model (bits) ──
    k_data_given_model_bits = final_loss * total_dataset_tokens / math.log(2)

    # ── Epiplexity: two-part code length ──
    epiplexity = k_auc_bits + k_data_given_model_bits

    # Normalized epiplexity (per token, for comparability across dataset sizes)
    epiplexity_per_token = epiplexity / max(total_dataset_tokens, 1)

    # Clean up cloned model
    del probe
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "epiplexity": round(epiplexity, 2),
        "epiplexity_per_token": round(epiplexity_per_token, 6),
        "k_auc_bits": round(k_auc_bits, 2),
        "k_data_given_model_bits": round(k_data_given_model_bits, 2),
        "initial_loss_nats": round(loss_curve[0][1], 4),
        "final_loss_nats": round(final_loss, 4),
        "loss_reduction": round(loss_curve[0][1] - final_loss, 4),
        "total_dataset_tokens": total_dataset_tokens,
        "train_steps": train_steps,
        "loss_curve": [(s, round(l, 4)) for s, l, _ in loss_curve],
    }


def _empty_result() -> dict:
    return {
        "epiplexity": 0.0,
        "epiplexity_per_token": 0.0,
        "k_auc_bits": 0.0,
        "k_data_given_model_bits": 0.0,
        "initial_loss_nats": 0.0,
        "final_loss_nats": 0.0,
        "loss_reduction": 0.0,
        "total_dataset_tokens": 0,
        "train_steps": 0,
        "loss_curve": [],
    }


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    print(f"Using device: {device}")
    print(f"Epiplexity estimation via K_auc (Finzi et al., 2026)")

    # Load probe model
    print(f"\nLoading probe model: {args.probe_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.probe_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        args.probe_model,
        torch_dtype=dtype,
    )
    base_model.to(device)

    # Load datasets via the shared registry
    registry = get_registry_with_formatters()
    if args.datasets:
        registry = {k: v for k, v in registry.items() if k in args.datasets}
        missing = set(args.datasets) - set(registry)
        if missing:
            print(f"Warning: unknown datasets ignored: {missing}")

    dataset_dict = load_all_datasets(registry, max_samples_per_dataset=args.max_samples)

    # Estimate epiplexity per dataset
    scores: dict[str, float] = {}
    details: dict[str, dict] = {}

    for name, ds in dataset_dict.items():
        print(f"\n{'─'*60}")
        print(f"Probing {name} ({len(ds)} examples, {args.train_steps} train steps)...")
        start_time = time.time()

        texts = [f"{ex['prompt']} {ex['answer']}" for ex in ds]

        result = estimate_epiplexity(
            base_model=base_model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            train_steps=args.train_steps,
            lr=args.lr,
            log_interval=args.log_interval,
            device=device,
        )

        elapsed = time.time() - start_time
        # Use epiplexity_per_token as the score (comparable across dataset sizes)
        scores[name] = result["epiplexity_per_token"]
        details[name] = {**result, "elapsed_seconds": round(elapsed, 1)}

        print(
            f"  epiplexity={result['epiplexity']:.1f} bits  "
            f"per_token={result['epiplexity_per_token']:.4f}  "
            f"K_auc={result['k_auc_bits']:.1f}  "
            f"K(X|M)={result['k_data_given_model_bits']:.1f}  "
            f"loss: {result['initial_loss_nats']:.3f}→{result['final_loss_nats']:.3f}  "
            f"time={elapsed:.1f}s"
        )

    # Write scores JSON (consumed by EpiplexityWeightedSampler)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        **scores,
        "_metadata": {
            "method": "k_auc_prequential",
            "reference": "arXiv:2601.03220 (Finzi et al., 2026)",
            "probe_model": args.probe_model,
            "train_steps": args.train_steps,
            "lr": args.lr,
            "max_samples_per_dataset": args.max_samples,
            "max_length": args.max_length,
            "score_type": "epiplexity_per_token",
            "details": details,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Epiplexity scores written to: {output_path}")
    print(f"{'='*60}")
    print("\nScores (higher epiplexity = more learnable structure → more training weight):")
    for name, score in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"  {name:12s}: {score:.4f} bits/token")

    print(f"\nBreakdown:")
    for name in sorted(details, key=lambda n: -scores[n]):
        d = details[name]
        print(
            f"  {name:12s}: K_auc={d['k_auc_bits']:8.1f} bits  "
            f"K(X|M)={d['k_data_given_model_bits']:8.1f} bits  "
            f"total={d['epiplexity']:8.1f} bits  "
            f"({d['total_dataset_tokens']:,} tokens)"
        )


if __name__ == "__main__":
    main()
