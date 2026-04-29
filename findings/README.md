# Findings

Companion notes and analysis scripts. Each file in this directory is a deeper or single-topic dive that did not fit cleanly into the deck or into the main pipeline code.

## Index

- [`RESULTS.md`](RESULTS.md) — Full experiment writeup with tables and analysis (4 predictors × 2 scales × 8 datasets).
- [`FINAL_RESULTS.md`](FINAL_RESULTS.md) — Holistic re-reading of the experiments after the post-hoc anchor-sensitivity reanalysis. Supersedes `RESULTS.md` on the points called out in §6.
- [`anchor_sensitivity.md`](anchor_sensitivity.md) — Single-page reference for how the headline rollout-K_auc → transfer correlation varies with the choice of `L_final` integrator anchor.
- [`recompute_script/`](recompute_script/) — Standalone re-derivation of K_auc from the persisted loss curves under alternative anchoring strategies. No GPU required. Run from the repo root: `uv run python findings/recompute_script/recompute_kauc.py`.
