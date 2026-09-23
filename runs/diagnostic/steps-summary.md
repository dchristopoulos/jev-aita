# Step-question setups, 300 development posts

Exploratory, not part of the final results. Posts come from the 2023 pool (`data/final-clean-20260923.jsonl`, never sent to Jev before; its provenance, and why it can't be re-drawn from this repo, is in [METHODOLOGY.md](../../docs/METHODOLOGY.md#choosing-jevs-question-setup)): 100 drawn by `python -m jevbench.steps sample`, then 200 more by `sample --size 200 --out data/steps-dev-extra-200.jsonl --exclude data/steps-dev-100.jsonl`. Step answers are combined by logistic regression scored out of fold. Regenerate with `python -m jevbench.steps fit runs/diagnostic/steps-dev-*.jsonl [--balanced]`.

## Weighted to the 2025 population mix

300 posts {'nta': 105, 'yta': 105, 'esh': 45, 'nah': 45} · weighted to the 2025 population mix · no-model baseline Brier 0.415

| Jev setup | Q | Weighted Brier ↓ [95% CI] | Δ vs direct refitted [95% CI] | Weighted top-1 | YTA / NTA / ESH / NAH recall | p50 | $/1k |
|---|---:|---:|---:|---:|---|---:|---:|
| direct, raw (as in the benchmark) | 1 | 0.354 [0.307, 0.404] |  | 75.8% | 59% / 87% / 7% / 13% | 525 ms | $0.034 |
| direct, refitted | 1 | 0.320 [0.293, 0.350] |  | 79.3% | 44% / 96% / 0% / 0% | 525 ms | $0.034 |
| yes/no ×5 | 5 | 0.312 [0.286, 0.338] | -0.008 [-0.026, +0.010] | 80.6% | 43% / 98% / 0% / 0% | 529 ms | $0.040 |
| severity scores ×2 | 2 | 0.329 [0.303, 0.356] | +0.009 [-0.006, +0.024] | 80.0% | 40% / 98% / 0% / 0% | 526 ms | $0.035 |
| steps ×5 (no verdict) | 5 | 0.322 [0.291, 0.354] | +0.001 [-0.022, +0.026] | 80.9% | 45% / 98% / 0% / 0% | 531 ms | $0.046 |
| steps ×5 + verdict | 6 | 0.322 [0.289, 0.357] | +0.002 [-0.016, +0.023] | 79.8% | 46% / 96% / 0% / 2% | 520 ms | $0.052 |

## Four verdicts weighted equally

300 posts {'nta': 105, 'yta': 105, 'esh': 45, 'nah': 45} · weighted to equal verdict shares · no-model baseline Brier 0.750

| Jev setup | Q | Weighted Brier ↓ [95% CI] | Δ vs direct refitted [95% CI] | Weighted top-1 | YTA / NTA / ESH / NAH recall | p50 | $/1k |
|---|---:|---:|---:|---:|---|---:|---:|
| direct, raw (as in the benchmark) | 1 | 0.736 [0.686, 0.790] |  | 41.4% | 59% / 87% / 7% / 13% | 525 ms | $0.034 |
| direct, refitted | 1 | 0.569 [0.531, 0.611] |  | 58.5% | 50% / 66% / 58% / 60% | 525 ms | $0.034 |
| yes/no ×5 | 5 | 0.575 [0.537, 0.614] | +0.006 [-0.027, +0.036] | 54.1% | 47% / 54% / 51% / 64% | 529 ms | $0.040 |
| severity scores ×2 | 2 | 0.590 [0.554, 0.629] | +0.021 [-0.006, +0.046] | 53.5% | 41% / 49% / 62% / 62% | 526 ms | $0.035 |
| steps ×5 (no verdict) | 5 | 0.563 [0.518, 0.612] | -0.006 [-0.033, +0.022] | 52.7% | 50% / 46% / 51% / 64% | 531 ms | $0.046 |
| steps ×5 + verdict | 6 | 0.560 [0.512, 0.612] | -0.009 [-0.033, +0.016] | 55.7% | 51% / 51% / 56% / 64% | 520 ms | $0.052 |
