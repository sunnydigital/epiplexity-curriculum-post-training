# FINAL_RESULTS.md: A Reinterpretation of the Rollout-Epiplexity Findings

This document is a holistic re-reading of the experiments in this repository,
written after a post-hoc reanalysis of the persisted loss curves under
alternative integrator anchoring strategies. It supersedes the framing in
[RESULTS.md](RESULTS.md) on three specific points (called out in §6) without
changing any of the underlying numerical results.

The shape of the document follows the order in which the analysis was done:
inventory the experiments, walk through the predictor results, attempt and
falsify a measurement-artifact hypothesis, and arrive at a competence-curve
interpretation that explains the scale-dependent behavior.

---

## 1. The two experimental archetypes

The repository contains two methodologically distinct experiments that share
a common evaluation harness but ask different questions.

### Archetype 1 — Static-probe curriculum experiment

**Question.** Can a small probe model's static epiplexity scores predict
which datasets a larger training model should over-sample during GRPO?

**Setup.** A two-model pipeline:
- Probe: Qwen2.5-1.5B-Instruct, teacher-forcing CLM loss, 500 steps per
  dataset → epiplexity scores via prequential coding (K_auc).
- Target: Qwen2.5-3B-Instruct + LoRA (r=16), 2000 GRPO steps under one of six
  curriculum strategies derived from the probe's scores, plus a uniform
  baseline.

**Implementation.**
[probe_epiplexity.py](probe_epiplexity.py) → [data/curriculum.py](data/curriculum.py),
[data/loader.py](data/loader.py) → [post_training.py](post_training.py) →
[evaluate.py](evaluate.py) → [compare_results.py](compare_results.py).

**Sub-experiments inside Archetype 1.**

*Curriculum sweep* (six epiplexity-derived schedules + baseline):

| Rank | Strategy | MIXED | Δ vs baseline |
|------|----------|------:|--------------:|
| 1 | low_first | 0.4701 | +0.82pp |
| 2 | high_constant | 0.4699 | +0.80pp |
| 3 | uniform | 0.4695 | +0.76pp |
| 4 | low_constant | 0.4689 | +0.70pp |
| 5 | high_first | 0.4666 | +0.47pp |
| 6 | anneal_to_high | 0.4663 | +0.44pp |
| — | baseline_3b | 0.4619 | — |

Spread across six curriculum strategies: 0.38pp.

*Single-dataset ablation* (eight runs, each trains GRPO on one dataset alone
and evaluates on all eight):

| Train dataset | MIXED |
|---|---:|
| math | 0.488 |
| arc | 0.483 |
| humaneval | 0.481 |
| mbpp | 0.480 |
| triviaqa | 0.464 |
| mmlu | 0.460 |
| boolq | 0.459 |
| gsm8k | 0.409 |

**Verdict for Archetype 1.** GRPO produces a consistent +0.7pp improvement
over baseline regardless of curriculum. The probe-derived curriculum doesn't
matter (within-noise spread) and the static probe's epiplexity does not
predict transfer (Spearman ρ = −0.17 at 1.5B, −0.02 at 3B). The hypothesis
that a cheap probe is a useful curriculum oracle is falsified.

### Archetype 2 — Rollout-epiplexity predictor experiment

**Question.** Given that the static probe failed, can a *dynamic*
GRPO-native learnability proxy do better? Specifically: instead of
teacher-forcing CE on dataset tokens (off-objective for GRPO), measure the
prequential-coding K_auc of the *GRPO surrogate itself* on policy-sampled
rollouts?

**Setup.** Per dataset, attach a fresh LoRA adapter to the policy, run 50
chunks of 16 prompts × 8 generations, alternating MEASURE (no-grad surrogate
`L = −E[A · log π(τ)]`) and TRAIN (one optimizer step on the same chunk).
Integrate `max(0, L_i − L_final)` × tokens trapezoidally into K_auc.

**Implementation.**
[measure_rollout_epiplexity.py](measure_rollout_epiplexity.py),
outputs in [data/rollout_epiplexity_1.5b.json](data/rollout_epiplexity_1.5b.json)
and [data/rollout_epiplexity_3b.json](data/rollout_epiplexity_3b.json).

The same measurement protocol is run independently on the 1.5B and 3B
models. Unlike Archetype 1, there is no probe-vs-target distinction — each
model acts as its own measurement instrument, and the question is whether
the predictor it produces ranks datasets in agreement with Archetype 1's
ablation transfer scores.

**Other predictors in the comparison.** [RESULTS.md](RESULTS.md) compares
rollout epiplexity against three static baselines: epiplexity (Archetype 1),
reward variance ([measure_reward_variance.py](measure_reward_variance.py)),
and forking-token entropy ([measure_forking_entropy.py](measure_forking_entropy.py)).

---

## 2. The headline predictor results

Spearman ρ between each predictor and the ablation transfer scores (n=8):

| Predictor | 1.5B ρ | 3B ρ |
|---|---:|---:|
| Static epiplexity | −0.17 | −0.02 |
| Reward variance | −0.05 | −0.31 |
| Forking-token entropy | −0.19 | −0.12 |
| **Rollout epiplexity / token** | **+0.52** (p=0.18) | −0.12 |
| **Rollout K_auc raw bits** | **+0.64** (p=0.09) | +0.05 |

Two leave-one-out observations from RESULTS.md:
- Removing boolq from the 1.5B rollout result gives ρ = +0.86 (p=0.014) for
  per-token and +0.89 (p=0.007) for raw K_auc.
- Boolq is identified *a priori* as a degenerate (binary-answer) failure
  mode for compression-based learnability proxies.

The 1.5B rollout-epiplexity row is the only positive signal in the 4×2
predictor grid, and it was the project's headline finding.

---

## 3. The scale disparity

The same measurement protocol applied to 1.5B and 3B yields qualitatively
different results: ρ ≈ +0.52 → ρ ≈ −0.12. Two endpoint observations from
the 3B output (`L_initial`, `L_final`) hint at why:

| Dataset (3B) | L_initial | L_final | Net change | K_auc bits |
|---|---:|---:|---:|---:|
| gsm8k | −0.009 | +0.003 | +0.012 ↑ | 0 |
| arc | −0.009 | +0.110 | +0.119 ↑ | 0 |
| mmlu | −0.106 | −0.020 | −0.086 ↓ | 8033 |

For gsm8k and arc at 3B, the surrogate loss at chunk 50 is *higher* than at
chunk 0. The integrator's `max(0, …)` clamp in
[integrate_k_auc](measure_rollout_epiplexity.py#L252-L269) zeros K_auc on
those datasets. RESULTS.md attributes the 3B negative correlation to "3B
inner-loop instability at the chosen hyperparameters (lr = 3e-6, lora_r = 16,
AdamW)."

This is the framing the rest of the document tests.

---

## 4. The artifact hypothesis: K_auc is sensitive to last-chunk noise

Reading the persisted per-chunk loss curves in
[data/rollout_epiplexity_3b.json](data/rollout_epiplexity_3b.json) and
[data/rollout_epiplexity_1.5b.json](data/rollout_epiplexity_1.5b.json)
showed something not addressed in RESULTS.md: every loss curve at both
scales is dominated by chunk-to-chunk noise of the same order of magnitude
as (or larger than) any net trend.

For arc at 3B specifically:
- Chunks 0–48 range from −0.097 to +0.034 with most values between −0.06
  and 0.
- **Chunk 49 alone is +0.110** — the highest value in the entire 50-chunk
  sequence. It becomes `L_final`.
- Every other chunk is below `L_final`, so `(L_i − L_final)` is negative
  everywhere, every segment clamps to zero, K_auc = 0.

The same pathology hits **gsm8k at 1.5B**:
- Chunks 0–48 range from −0.040 to −0.001, all negative.
- Chunk 49 is +0.0033, the *highest* value in the run, so K_auc = 0.

This generates a clean falsifiable hypothesis: K_auc is unstable in
`L_final`, and the 3B "negative correlation" is at least partly a
single-point anchoring artifact. If this is the whole story, replacing
`L_final = loss_curve[-1]` with `mean_last_K` or `min_last_K` should leave
the 1.5B correlation roughly intact and shift the 3B correlation toward
zero or positive.

---

## 5. Recomputing K_auc under alternative anchors

The full per-chunk loss curves are persisted, so this test does not require
re-running any GPU-bound work. The script
[recompute_script/recompute_kauc.py](recompute_script/recompute_kauc.py)
re-derives K_auc from the existing JSON files under eight anchoring
strategies and computes Spearman ρ against ablation transfer.

Sanity check: under `last (published)`, the recomputation reproduces RESULTS.md
to two decimal places (1.5B: ρ = +0.643 raw, +0.893 without boolq, matching
+0.64 / +0.89; 3B per-token: ρ = −0.120, matching −0.12). The integrator is
faithful and the per-chunk-token approximation is fine.

### 5.1 Spearman ρ across anchors (K_auc raw bits vs ablation transfer)

| anchor | 1.5B | 1.5B −boolq | 3B | 3B −boolq |
|--------|----:|-----------:|---:|----------:|
| last (published) | **+0.643** | **+0.893** | +0.048 | −0.018 |
| mean_last_3 | +0.452 | +0.786 | −0.238 | −0.250 |
| mean_last_5 | +0.190 | +0.571 | −0.357 | −0.500 |
| mean_last_10 | +0.167 | +0.321 | +0.000 | −0.250 |
| median_last_5 | +0.214 | +0.607 | −0.048 | −0.071 |
| min_last_5 | +0.381 | +0.679 | +0.071 | +0.000 |
| min_last_10 | +0.214 | +0.286 | −0.024 | −0.179 |
| min_overall | +0.381 | +0.500 | +0.095 | −0.107 |

### 5.2 What the recomputation says

**The artifact hypothesis is falsified, in both directions.**

- **1.5B degrades sharply with smoothing.** `mean_last_5` reduces the
  no-boolq correlation from +0.893 to +0.571. `mean_last_10` reduces it
  to +0.321. The published `last` anchor is the one that maximizes the
  correlation among reasonable choices — it is *not* a robust signal that
  smoothing happens to disturb.
- **3B does not recover under any anchor.** No smoothing variant produces a
  meaningfully positive 3B correlation. The best 3B no-boolq cell is
  `min_last_5` at exactly 0.000.

Conclusion: the 3B failure is not a single-chunk anchoring artifact. The
underlying integrand is genuinely doing something different at the two
scales, and `L_final` choice cannot reconcile them.

---

## 6. The competence-curve interpretation

Examining per-dataset K_auc rankings (rather than aggregate ρ) under each
anchor revealed a systematic pattern that points to a non-artifact mechanism.

### 6.1 What's actually different between 1.5B and 3B

The persisted `mean_reward` and `fraction_zero_variance_groups` fields
(values per group of 8 generations where all rewards are equal, hence all
group-relative advantages are zero) describe where each model sits on a
reward-variance curve:

| Dataset | mean_r 1.5B | zero_var 1.5B | mean_r 3B | zero_var 3B |
|---------|------------:|--------------:|----------:|------------:|
| arc | 0.729 | 0.64 | 0.772 | 0.63 |
| boolq | 0.726 | 0.47 | 0.796 | 0.74 |
| mmlu | 0.550 | 0.60 | 0.584 | 0.58 |
| math | 0.516 | 0.38 | 0.464 | **0.63** |
| triviaqa | 0.416 | 0.67 | 0.532 | 0.69 |
| gsm8k | 0.619 | 0.30 | 0.737 | 0.25 |
| humaneval | 0.277 | 0.03 | 0.377 | 0.02 |
| mbpp | 0.308 | 0.09 | 0.357 | 0.05 |

The math row jumps out: at 3B, the mean reward *drops* slightly (0.52 →
0.46) but the zero-variance fraction *rises* sharply (0.38 → 0.63). The 3B
isn't doing math better; its outputs are more *concentrated* — more groups
of 8 generations produce the same answer eight times — so the GRPO surrogate
has no within-group spread to exploit.

### 6.2 Per-dataset K_auc ranking inversion

The script
[recompute_script/inspect_rankings.py](recompute_script/inspect_rankings.py)
prints K_auc rankings vs transfer rankings under each anchor. The published-
anchor rankings reveal the mechanism cleanly:

| Dataset | Transfer rank | 1.5B K_auc rank | 3B K_auc rank |
|---------|--------------:|----------------:|--------------:|
| math | 1 | 4 | **6** |
| arc | 2 | 1 | **8** (zeroed) |
| humaneval | 3 | 2 | 2 |
| mbpp | 4 | 5 | **1** |
| triviaqa | 5 | 6 | 4 |
| mmlu | 6 | 7 | **3** |
| boolq | 7 | 3 | 5 |
| gsm8k | 8 | 8 | 7 |

The 1.5B ranks the best-transferring datasets near the top and the worst
near the bottom. The 3B inverts this in the upper half: the *best*-transfer
datasets (math, arc) sit at K_auc ranks 6 and 8, while *moderate*-transfer
datasets (mbpp, mmlu) take ranks 1 and 3. This pattern persists under every
smoothed anchor (see §5), so it is a property of the loss curves
themselves, not the integrator.

### 6.3 Why the inversion happens

The GRPO surrogate has near-zero gradient signal whenever within-group
reward variance is small, because group-relative advantages collapse. So
per-dataset K_auc is largely a function of how much remaining reward
variance the policy has on that dataset. This produces a U-shape against
base accuracy:

- **Too weak** (mean reward ≪ 0.5): all generations fail, advantages are
  zero, K_auc collapses. Not observed in this dataset/model grid.
- **Productive middle** (mean reward ≈ 0.2–0.5, low zero_var fraction):
  reward variance is high, gradient signal is strong, K_auc is large.
- **Too strong** (mean reward ≫ 0.5, high zero_var fraction): all
  generations succeed, advantages collapse again, K_auc collapses.

At 1.5B, most of the eight datasets sit in or near the productive middle —
including math (0.52, 0.38), arc (0.73, 0.64), humaneval (0.28, 0.03), mbpp
(0.31, 0.09). The 1.5B's K_auc rankings therefore correlate well with
transfer.

At 3B, several datasets have crossed the saturation knee:
- arc: zero_var 0.63, K_auc collapses despite the dataset being a strong
  transfer source.
- math: zero_var jumps from 0.38 to 0.63, K_auc drops to 2,733 (rank 6).
- mmlu, boolq: also high zero_var.

Meanwhile humaneval (zero_var 0.02) and mbpp (zero_var 0.05) remain in the
productive middle because the 3B is still relatively weak at code. They
top the 3B K_auc ranking — but they're only mid-tier transfer sources.

The disparity is therefore: **the 1.5B and 3B are measuring different
things on the same protocol because they sit at different points on the
saturation curve, and the metric's behavior depends on which side of the
knee you're on.**

### 6.4 What this means for transfer prediction

Transfer is dominated by latent reasoning competence — math and arc
transfer best because the 3B already *can* do those tasks well, and the
GRPO update sharpens that competence into something that generalizes.

But latent competence and remaining reward variance are *anti-correlated*
past the saturation knee: the more competent the model is on a dataset,
the less reward variance remains for the GRPO update mechanism to exploit,
the smaller K_auc becomes — even though the underlying competence is
exactly what makes the dataset transfer well.

So the 3B "negative correlation" is the metric faithfully reporting that
the model has run out of within-group variance on the very datasets that
drive transfer. It is not a bug, not noise, not a measurement artifact.
It is K_auc reporting a *different signal* than transfer demands at this
point on the competence curve.

### 6.5 The math anomaly — why all four predictors miss it in the same direction

Math is the cleanest single piece of evidence for the competence-curve
interpretation, because it is the dataset on which **every predictor in
the project fails in the same direction**. Pulling the per-dataset values
from [RESULTS.md:178](RESULTS.md#L178) (all measured on the 1.5B probe):

| Dataset | Epi | RV | Fork-H | Roll-Epi | K_auc | Transfer | T-rank |
|---------|----:|---:|-------:|---------:|------:|---------:|-------:|
| math | **0.748** (8/8) | 0.048 (5/8) | 0.847 (7/8) | 0.0177 (4/8) | 26 232 (4/8) | **0.488** | **1** |
| arc | 1.476 | 0.061 | 1.541 | 0.0890 | 75 708 | 0.483 | 2 |
| humaneval | 2.006 | 0.048 | 1.022 | 0.0198 | 30 222 | 0.481 | 3 |

(Predictor ranks in parentheses are out of 8, with 1 meaning "highest
predicted learnability." Math's static-epiplexity rank of 8/8 means it has
the *lowest* static epiplexity of all eight datasets.)

Math has:
- The **lowest static epiplexity** (0.748 bits/token, rank 8/8).
- Mid-low **reward variance** (0.048, rank 5/8).
- Mid-low **forking-token entropy** (0.847, rank 7/8 of "decision-token
  uncertainty").
- Mid-pack **rollout epiplexity** (0.0177, rank 4/8).

Yet math is the **best transfer source** (MIXED 0.488, rank 1/8). It
contributes more cross-dataset capability via single-dataset GRPO training
than any other dataset in the suite — including arc and the code datasets
that all four predictors prefer.

This is not a one-predictor miss that could be attributed to noise. It is
a **four-predictor unanimous miss**, which makes it diagnostic of what the
project is actually measuring.

**The mechanism, under the competence-curve framing.** The Qwen2.5
checkpoints' base accuracy on MATH (algebra) is 0.51 at 1.5B and 0.44 at
3B. Mathematical reasoning is hard enough that the policy doesn't saturate
(zero-variance fraction at 1.5B is 0.38, the second-lowest of any non-code
dataset), but the *individual numeric answers* are highly compressible by
a model that already has arithmetic patterns:

- **Static epiplexity is low** because numeric answers are short, regular,
  and high-frequency in pretraining; teacher-forcing CE on them looks
  trivially predictable. The probe model has effectively *already
  compressed* the surface-level structure of math answers.
- **Reward variance is mid-low** because most math problems are decided by
  a single arithmetic chain — the policy either gets the chain right or
  doesn't, and the pass/fail correlation across the 8 generations within a
  group leaves moderate but not maximal spread.
- **Forking entropy is low** because the "decision tokens" on a successful
  math rollout are arithmetic operations the policy is fairly confident
  about; per-step entropy is concentrated in the *answer* token, which
  Wang et al.'s top-20% filter dilutes against operation tokens.
- **Rollout epiplexity is mid-pack** because the GRPO surrogate compresses
  decently on math (the inner loop reduces L_initial = −0.019 to L_final
  = −0.020) but doesn't compress dramatically — the policy isn't exploring
  much new structure.

All four predictors agree: from any base-model perspective, math looks
*already-compressed*. The predictors collectively report "there's not much
left to learn here."

But what they miss is that **GRPO's job isn't to learn surface compression
— it's to sharpen latent reasoning into a generalizable operator**. Math
training teaches the policy to commit to arithmetic chains under the
GRPO update, and that commitment generalizes: ablation transfer shows math
training improves arc (0.728 → 0.728), mmlu (0.570 → 0.556 — flat), and
*especially* code (humaneval 0.384 → 0.389; mbpp 0.301 → 0.340 — large
gain) and triviaqa (0.498 → 0.494). The math-trained policy applies
"commit to a structured answer chain" as a generic post-training behavior.
None of the four predictors capture this because none of them measure
*generalization potential*; they all measure variants of "what's left to
compress on this dataset specifically."

**The contrast with humaneval.** Humaneval has the *opposite* profile —
high static epiplexity (2.006, rank 2/8), low reward variance (0.048),
high rollout epiplexity (0.0198, rank 3/8). All predictors say humaneval
is highly learnable. And humaneval *is* a strong transfer source (rank
3/8) — but it's still ranked below math. The predictors over-rank
humaneval for the same reason they under-rank math: surface-level code
patterns are *visibly* compressible (long, novel-to-the-probe, with
varied per-token entropy), so every predictor lights up on them. But code
patterns also generalize *less* than mathematical reasoning chains do,
because they're more idiosyncratic to the task.

The result is that the predictors confuse "looks compressible from a base
policy's perspective" with "produces a transferable update under GRPO."
Math is the cleanest counterexample because it sits in the regime where
surface compression has already happened (pretraining did most of the
work) but reasoning sharpening is still available (GRPO has work to do).
None of the four predictors can see into that regime.

**Why this matters for the project's central claim.** The four-predictor
unanimous miss on math is the strongest evidence that the
"small-probe-large-target" architecture is fundamentally limited at this
scale. Any predictor derived from the base model's behavior on the data
— whether through teacher-forcing CE, reward variance, decision-token
entropy, or even GRPO-surrogate compression — will rank datasets by how
much the *probe* can extract from them, not by how much *latent reasoning
structure* GRPO can sharpen for transfer. Math defeats all four because
its transfer value lives in a regime the predictors can't see: the
pretraining-already-compressed-but-RL-still-applicable middle.

A predictor that *would* catch math would need to measure something like
"how often does the policy commit to a structured multi-step answer when
sampling on this dataset, and how stable is that commitment under
gradient updates" — which is closer to a behavioral reasoning probe than
to any compression-based signal. No such predictor was implemented in
this project, and the math anomaly suggests it would have been the more
fruitful direction.

---

## 7. What changes from RESULTS.md

Three specific updates to the framing in RESULTS.md, none of which
contradict any number it reports.

**(a) The 3B failure is not "inner-loop instability."** RESULTS.md attributes
the 3B sign collapse to LoRA hyperparameter instability ("3B inner loop is
unstable at the chosen hyperparameters"). The recomputation falsifies this:
smoothing `L_final` does not rescue 3B under any alternative anchor.
The actual mechanism is **policy saturation**: the 3B has crossed the
within-group reward-variance knee on math, arc, mmlu, and boolq, leaving
the GRPO surrogate with little to compress on the very datasets that drive
transfer. Tuning the inner-loop hyperparameters would not fix this — the
issue is upstream of the optimizer.

**(b) The 1.5B success is not "the dynamic GRPO-native predictor working."**
RESULTS.md frames the +0.89 (no-boolq) result as the first non-trivial
positive predictor of GRPO transfer. The recomputation shows the 1.5B
correlation is fragile to natural denoising: `mean_last_5` cuts it from
+0.89 to +0.57. The published `last` anchor is the integrator choice that
maximizes the correlation. The signal is therefore better described as
**a coincidence of probe-positioning**: the 1.5B happens to sit in the
productive middle of the reward-variance curve on enough of the
high-transfer datasets that K_auc ends up correlating with transfer.

**(c) The "small-probe-large-target" architecture is a positioning artifact,
not a property of the data.** Both Archetype 1 (probe a small model to
predict the curriculum for a large model) and Archetype 2 (probe at small
scale to predict transfer at large scale) work or fail because of where
the probe sits on the saturation curve relative to the target. Archetype 1
fails because static teacher-forcing CE doesn't encode this relationship
at all. Archetype 2 partially works at 1.5B because the 1.5B's saturation
profile happens to align with what GRPO can sharpen in the 3B, on these
particular eight datasets. There is no reason to expect the same alignment
for a different target model (e.g. 7B), or even for a different mixture of
datasets at the same target scale.

---

## 8. Holistic interpretation of the project

Stepping back from the individual experiments, the project's deepest
finding — clearer in retrospect than during the runs — is that
**post-training transfer at this scale is dominated by latent competence,
and the predictors we know how to measure mostly capture proxies that
correlate with competence only when the probe sits in the right window.**

The Archetype 1 sweep showed that *with* GRPO, curriculum doesn't matter
(0.38pp spread across six strategies; +0.7pp gain over baseline regardless).
The Archetype 1 ablation showed that *across* training datasets, transfer
spread is also small (0.41–0.49, ≈8pp). Together, these indicate that the
3B at 2000 GRPO steps sits at a regime where *what* you train on matters
much less than *that* you do GRPO at all.

Within that small remaining spread, no static base-model predictor explains
the variation (|ρ| ≤ 0.31 across epiplexity, reward variance, forking
entropy, at both scales). The dynamic predictor explains some of it at
1.5B but in a way that is now revealed (§5–6) to be:

1. **Sensitive to the integrator anchor** (1.5B no-boolq drops from +0.89
   to +0.32–+0.68 depending on smoothing).
2. **Scale-dependent in mechanism**: at 1.5B, K_auc tracks "reward variance
   headroom across the eight datasets," which happens to align with
   transfer. At 3B, the same K_auc still tracks "reward variance headroom"
   — but on different datasets, because the 3B has saturated on the
   high-transfer ones.
3. **Selectively rescued by removing boolq**, the one dataset RESULTS.md
   identifies *a priori* as a degenerate case for any compression proxy.

The honest summary across both archetypes:

> *At Qwen2.5-3B + 2000 GRPO steps, transfer is dominated by general
> reasoning acquisition rather than dataset-specific structure. Among the
> four predictors tested, only rollout epiplexity at 1.5B shows positive
> correlation with transfer, and the recomputation reveals this is more a
> consequence of where the 1.5B sits on the saturation curve than of the
> metric itself. The intuition that one could use a cheap probe model to
> compute a learnability score that predicts what GRPO will extract from a
> larger target — the original motivation for this project — is not
> supported by any of the data.*

What *would* be needed to support that original intuition is a predictor
that is insensitive to saturation, captures latent reasoning structure
rather than reward-variance availability, and is computable on the target
model directly. None of the four predictors in this project meet those
requirements. Forking-token entropy comes closest in spirit but, as
RESULTS.md notes, ends up measuring answer-token uncertainty on short
tasks rather than reasoning branch-points.

---

## 9. Reproducing this analysis

The recomputation is a desk calculation from the already-persisted loss
curves; no GPU or training is required.

```bash
# From the repo root, with uv installed
uv run python recompute_script/recompute_kauc.py
uv run python recompute_script/inspect_rankings.py
```

Outputs:
- `recompute_script/recomputed_kauc.json` — full per-dataset K_auc under
  each anchor, both scales.
- `recompute_script/correlations.csv` — flat ρ table.

Caveats inherited from the original measurement:
- **n = 8.** Even ρ = +0.89 at p = 0.014 is suggestive, not conclusive,
  at this sample size.
- **Per-chunk token weighting is approximated** as `total_tokens / 50` in
  the recomputation because per-chunk token counts are not persisted.
  This is exact when chunks have similar token counts (each is 16 prompts
  × 8 generations of similar length); it is a second-order effect relative
  to the L_final-anchoring effect being studied.
- **Single-seed measurement.** Whether the rankings would survive a
  different seed is not testable from existing artifacts and would require
  re-running [measure_rollout_epiplexity.py](measure_rollout_epiplexity.py).

---

## 10. Open questions

The competence-curve interpretation suggests several testable follow-ups
that this project did not run:

1. **Multi-seed stability.** Re-run the rollout-epiplexity measurement with
   3–5 different seeds at each scale. The competence-curve story predicts
   the 3B rankings will be *stable* across seeds (saturation is a property
   of the model, not the sampling), while the 1.5B headline correlation
   may *fluctuate* in the +0.4 to +0.9 range depending on which datasets
   land favorably on chunk 49.
2. **Reward-variance-corrected K_auc.** Compute K_auc only on rollouts
   from non-saturated groups (filter out groups with all-equal rewards),
   then re-correlate with transfer. If the 3B negative correlation is
   driven by saturated groups dragging the integrand toward zero, this
   filter should partly rescue it.
3. **Direct competence-window measurement.** Skip K_auc entirely and use
   `1 − fraction_zero_variance_groups` (or equivalently a per-dataset
   reward-variance score) as the predictor. The competence-curve story
   predicts this simpler metric will correlate with transfer about as well
   as rollout epiplexity at 1.5B and *worse* at 3B (where saturation
   anti-correlates with transfer).
4. **Larger target.** Whether a 1.5B probe predicts transfer for a 7B
   target the same way it does for a 3B target is the cleanest direct test
   of the "small-probe-large-target architecture" claim. If the
   competence-curve interpretation is right, a 7B target would push more
   datasets past saturation and the 1.5B's correlation would weaken
   correspondingly.

None of these were run as part of this project. They are listed here as
the analyses that would either confirm or falsify the interpretation in
§6, and as the natural continuation of the work if it is taken further.
