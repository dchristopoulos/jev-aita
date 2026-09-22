## Calibration against human rater fractions

600 comments from `google/civil_comments`. Target is the fraction of human raters who called the comment toxic, so a perfectly calibrated model's mean prediction equals it in every bin.

| Model | n | ECE ↓ | Brier ↓ | mean pred | mean truth | p50 latency |
|---|---|---|---|---|---|---|
| `openai/gpt-5-nano#minimal` | 300 | 0.125 | 0.078 | 0.485 | 0.385 | 1182 ms |
| `~typesafe/jev-latest` | 300 | 0.171 | 0.089 | 0.554 | 0.385 | 378 ms |

### `openai/gpt-5-nano#minimal` reliability

| predicted | n | mean pred | mean truth | gap |
|---|---|---|---|---|
| 0.0–0.1 | 36 | 0.045 | 0.085 | +0.040 |
| 0.1–0.2 | 47 | 0.147 | 0.195 | +0.048 |
| 0.2–0.3 | 24 | 0.252 | 0.251 | -0.001 |
| 0.3–0.4 | 7 | 0.346 | 0.321 | -0.025 |
| 0.4–0.5 | 12 | 0.428 | 0.272 | -0.157 |
| 0.5–0.6 | 27 | 0.545 | 0.476 | -0.069 |
| 0.6–0.7 | 50 | 0.644 | 0.528 | -0.116 |
| 0.7–0.8 | 62 | 0.744 | 0.470 | -0.274 |
| 0.8–0.9 | 22 | 0.837 | 0.648 | -0.189 |
| 0.9–1.0 | 13 | 0.925 | 0.702 | -0.223 |

### `~typesafe/jev-latest` reliability

| predicted | n | mean pred | mean truth | gap |
|---|---|---|---|---|
| 0.0–0.1 | 17 | 0.068 | 0.061 | -0.007 |
| 0.1–0.2 | 26 | 0.145 | 0.160 | +0.015 |
| 0.2–0.3 | 41 | 0.250 | 0.192 | -0.058 |
| 0.3–0.4 | 21 | 0.341 | 0.210 | -0.130 |
| 0.4–0.5 | 19 | 0.440 | 0.318 | -0.122 |
| 0.5–0.6 | 23 | 0.549 | 0.346 | -0.203 |
| 0.6–0.7 | 33 | 0.646 | 0.392 | -0.254 |
| 0.7–0.8 | 33 | 0.749 | 0.494 | -0.255 |
| 0.8–0.9 | 54 | 0.852 | 0.614 | -0.238 |
| 0.9–1.0 | 33 | 0.929 | 0.657 | -0.272 |

### Does a constant fix it?

Two parameters (slope and intercept on the log-odds) fitted on a random half and scored on the held-out half.

| Model | fit / test | ECE raw → cal | Brier raw → cal | a | b |
|---|---|---|---|---|---|
| `openai/gpt-5-nano#minimal` | 150 / 150 | 0.124 → **0.027** | 0.074 → **0.054** | 0.55 | -0.50 |
| `~typesafe/jev-latest` | 150 / 150 | 0.163 → **0.039** | 0.082 → **0.050** | 0.60 | -0.80 |
