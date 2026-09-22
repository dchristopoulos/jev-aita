"""Turn saved run logs into the tables and charts the README shows.

Everything in the report is derived from JSONL records, not hand-typed.

Reddit labels a verdict, not a probability. The reliability chart bins each
model's probability for its chosen verdict, then checks how often that verdict
matched the flair.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .data import VERDICTS
from .metrics import (bin_support, ece, gated_coverage, multiclass_scores, percentile,
                      reliability_bins)
from .client import UNPARSED_CHOICE
from .questions import balanced_distribution

COLORS = ["#2f81f7", "#d29922", "#8957e5", "#3fb950", "#db6d28", "#e34c26", "#00bcd4"]
AXIS = "#8b949e"
DRAWS = 2000   # bootstrap resamples; seeded, so every interval is reproducible

NAMES = {
    "~typesafe/jev-latest": "Jev",
    "anthropic/claude-sonnet-5": "Sonnet 5",
    "openai/gpt-5-nano": "GPT-5 nano (low effort)",
    "openai/gpt-5-nano#minimal": "GPT-5 nano (minimal effort)",
    "local/qwen/qwen3.6-35b-a3b#none": "Qwen 3.6 35B-A3B (local)",
    "local/google/gemma-4-26b-a4b-qat#none": "Gemma 4 26B-A4B (local)",
}
ARM_NAMES = {"monolithic": "direct", "balanced": "two questions",
             "standard_choice": "label only"}


def label(model: str, arm: str) -> str:
    """A readable row name; unknown model IDs fall back to the raw ID."""
    return f"{NAMES.get(model, model)} · {ARM_NAMES.get(arm, arm)}"


def ordered(groups: dict[tuple[str, str], list[dict]]) -> list[tuple[tuple[str, str], list[dict]]]:
    """Rows in a fixed reading order: Jev first, then the table in NAMES."""
    names, arms = list(NAMES), list(ARM_NAMES)
    return sorted(groups.items(), key=lambda kv: (
        names.index(kv[0][0]) if kv[0][0] in names else len(names), kv[0][0],
        arms.index(kv[0][1]) if kv[0][1] in arms else len(arms), kv[0][1]))


def load(*paths: Path) -> list[dict]:
    """Read one or more run logs into a single set of records.

    Several logs matter because an expensive baseline is usually added in a
    later pass, against the same cached sample. Merging them puts every model
    in one table instead of leaving two reports to compare by eye.
    """
    recs: list[dict] = []
    for path in paths:
        recs += [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if not recs:
        raise SystemExit(f"no records in {', '.join(str(p) for p in paths)}")
    return recs


def check_same_sample(recs: list[dict]) -> str:
    """Warn if merged logs judged different posts.

    Comparing a model scored on 200 posts against one scored on 40 different
    ones is not a comparison. Cheap to check, expensive to miss.
    """
    per_model: dict[str, set[str]] = defaultdict(set)
    for r in recs:
        # Keyed on (model, arm): an arm that stopped early on budget judges
        # fewer posts than its siblings, and that is exactly the silent
        # mismatch this guard exists to catch.
        per_model[f"{r['model']} {r['arm']}"].add(r["item_id"])
    sizes = {m: len(ids) for m, ids in per_model.items()}
    if len(set(map(frozenset, per_model.values()))) <= 1:
        return ""
    overlap = set.intersection(*per_model.values()) if per_model else set()
    return (f"\n> **Note:** these models were not all scored on the same posts "
            f"({sizes}); {len(overlap)} posts are common to all. Compare with care.\n")


def by_arm(recs: list[dict], include_errors: bool = False) -> dict[tuple[str, str], list[dict]]:
    """Group by model and arm.

    Failed requests are excluded by default. Scoring them would count an API
    timeout as a wrong verdict, which silently punishes whichever model was
    flakiest rather than whichever was least accurate. They are reported
    separately in the operations table instead.
    """
    out: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        if not include_errors and r.get("error"):
            continue
        out[(r["model"], r["arm"])].append(r)
    return {k: v for k, v in out.items() if v}


def ops_table(recs: list[dict]) -> str:
    """What the run actually cost, in money, time and failures."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["arm"])].append(r)

    rows = []
    for (model, arm), rs in ordered(groups):
        ok = [r for r in rs if not r.get("error")]
        errs = len(rs) - len(ok)
        retried = sum(1 for r in ok if r.get("attempts", 1) > 1)
        tok = sum(r.get("total_tokens", 0) for r in ok)
        spend = sum(r["cost_usd"] for r in ok)
        secs = sum(r["latency_s"] for r in ok)
        per_1k = (spend / len(ok) * 1000) if ok else float("nan")
        rows.append(
            f"| {label(model, arm)} | {len(rs)} | {errs} | {retried} | {tok:,} "
            f"| ${spend:.4f} | ${per_1k:.3f} | {secs:.0f}s |"
        )
    head = ("| Model and arm | Requests | Failed | Retried | Tokens | Spend "
            "| $/1k posts | API time |")
    sep = "|---" * 8 + "|"
    return "\n".join([head, sep, *rows])


def run_summary(recs: list[dict], meta: dict | None) -> str:
    """One paragraph a reader can check the rest of the report against."""
    ok = [r for r in recs if not r.get("error")]
    spend = sum(r["cost_usd"] for r in ok)
    tok = sum(r.get("total_tokens", 0) for r in ok)
    served = sorted({r.get("resolved_model") or r["model"] for r in ok})
    bits = [
        f"**{len(ok)} requests**, {len(recs) - len(ok)} failed",
        f"**${spend:.4f}** total",
        f"{tok:,} tokens",
    ]
    if meta and meta.get("sample_sha256"):
        bits.append(f"sample SHA-256 `{meta['sample_sha256'][:12]}`")
    return " · ".join(bits) + f"\n\nServed by: {', '.join(f'`{m}`' for m in served)}"


def predicted(r: dict) -> tuple[str, float]:
    """The arm's verdict and the probability assigned to that verdict.

    For a direct Choice we use its chosen-label probability, which is
    comparable across Jev and chat. Jev also returns a separate confidence
    field; it is logged but is not the same quantity.
    """
    if "verdict" in r.get("choices", {}):
        c = r["choices"]["verdict"]
        p = distribution(r)
        return c["choice"], p.get(c["choice"], 0.0) if p else c["confidence"]
    p = balanced_distribution(r["probs"])
    verdict = max(p, key=p.get)
    return verdict, p[verdict]


def baseline_accuracy(recs: list[dict]) -> float:
    """Always answering the most common verdict in the sample."""
    counts = Counter(r["verdict_true"] for r in recs)
    return counts.most_common(1)[0][1] / len(recs)


def distribution(r: dict) -> dict[str, float] | None:
    """Return only a complete, usable four-verdict distribution."""
    probs = (r.get("choices") or {}).get("verdict", {}).get("probabilities")
    if probs is None and {"poster_at_fault", "other_at_fault"} <= set(r.get("probs") or {}):
        try:
            probs = balanced_distribution(r["probs"])
        except (TypeError, ValueError):
            return None
    if not isinstance(probs, dict):
        return None
    try:
        total = sum(probs.values())
        # Jev's saved Choice probabilities can be rounded to 2 decimals.
        if abs(total - 1.0) > 0.021:
            return None
        normalised = {k: v / total for k, v in probs.items()}
        multiclass_scores([normalised], [r["verdict_true"]], tuple(VERDICTS))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return normalised


def brier_one(p: dict[str, float], truth: str) -> float:
    return sum((p[v] - (v == truth)) ** 2 for v in VERDICTS)


def log_loss_one(p: dict[str, float], truth: str) -> float:
    return -math.log(max(p[truth], 1e-15))


def stratified(parts: dict[str, list[float]], priors: dict[str, float] | None,
               seed: int = 0) -> tuple[float, float, float]:
    """Class-weighted mean of per-post values, with a 95% bootstrap interval.

    `parts` maps each true class to per-post values. Posts are resampled within
    their class, because the sample was drawn within class, and class means are
    weighted by `priors` (or pooled, when no priors are given).
    """
    if priors:
        total = sum(priors.values())
        weights = {c: priors[c] / total for c in parts}
    else:
        parts = {"all": [x for values in parts.values() for x in values]}
        weights = {"all": 1.0}

    def mean(p: dict[str, list[float]]) -> float:
        return sum(weights[c] * sum(p[c]) / len(p[c]) for c in weights)

    rng = random.Random(seed)
    draws = [mean({c: rng.choices(values, k=len(values)) for c, values in parts.items()})
             for _ in range(DRAWS)]
    return mean(parts), percentile(draws, 0.025), percentile(draws, 0.975)


def _by_class(pairs: list[tuple[str, float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for truth, value in pairs:
        out[truth].append(value)
    return out


def macro_recall(recs: list[dict]) -> float:
    """Mean of the four per-class recalls; always-NTA scores 25% here."""
    hits = _by_class([(r["verdict_true"], float(predicted(r)[0] == r["verdict_true"]))
                      for r in recs])
    return sum(sum(v) / len(v) for v in hits.values()) / len(hits)


def probability_table(groups: dict[tuple[str, str], list[dict]],
                      priors: dict[str, float] | None = None) -> str:
    """Proper scores from the full choice distribution, with visible coverage."""
    rows = []
    for (model, arm), recs in ordered(groups):
        scored = [(r, distribution(r)) for r in recs]
        scored = [(r, p) for r, p in scored if p is not None]
        macro = f"{macro_recall(recs):.1%}"
        if scored:
            rs, probs = zip(*scored)
            truth = [r["verdict_true"] for r in rs]
            bs, ll = multiclass_scores(list(probs), truth, tuple(VERDICTS))
            top1 = sum(predicted(r)[0] == r["verdict_true"] for r in recs) / len(recs)
            if priors is not None and set(truth) == set(VERDICTS):
                wb, lo, hi = stratified(
                    _by_class([(t, brier_one(p, t)) for t, p in zip(truth, probs)]), priors)
                _, wl = multiclass_scores(list(probs), truth, tuple(VERDICTS), priors)
                wa, alo, ahi = stratified(
                    _by_class([(r["verdict_true"], float(predicted(r)[0] == r["verdict_true"]))
                               for r in recs]), priors)
                weighted = (f"{wb:.3f} [{lo:.3f}, {hi:.3f}] | {wl:.3f} "
                            f"| {wa:.1%} [{alo:.1%}, {ahi:.1%}] | ")
            elif priors is not None:
                weighted = "n/a | n/a | n/a | "
            else:
                weighted = ""
            cells = f"{weighted}{macro} | {bs:.3f} | {ll:.3f} | {top1:.1%}"
        else:
            cells = ("n/a | n/a | n/a | " if priors is not None else "") + f"{macro} | n/a | n/a | n/a"
        rows.append(f"| {label(model, arm)} | {len(scored)}/{len(recs)} | {cells} |")
    head = "| Model and arm | Scored / successful |"
    sep = "|---|---:|"
    if priors is not None:
        head += " Weighted Brier ↓ [95% CI] | Weighted log loss ↓ | Weighted top-1 [95% CI] |"
        sep += "---:|---:|---:|"
    head += " Macro recall | Sample Brier ↓ | Sample log loss ↓ | Sample top-1 |"
    sep += "---:|---:|---:|---:|"
    return "\n".join([head, sep, *rows])


def _jev_direct(groups: dict[tuple[str, str], list[dict]]) -> tuple[str, str] | None:
    return next((k for k in groups if k[0].startswith("~typesafe/") and k[1] == "monolithic"), None)


def paired_table(groups: dict[tuple[str, str], list[dict]],
                 priors: dict[str, float] | None = None) -> str:
    """Paired differences from Jev's direct arm on the same posts.

    Three views of the same comparison, so a reader can see whether a ranking
    depends on the weighting or the scoring rule. Positive means worse than Jev.
    """
    key = _jev_direct(groups)
    if key is None:
        return "_No Jev direct arm for paired comparison._"

    def usable(recs: list[dict]) -> dict[str, dict]:
        return {r["item_id"]: r for r in recs if distribution(r) is not None}
    baseline = usable(groups[key])
    views = [(brier_one, priors), (log_loss_one, priors), (brier_one, None)] if priors else \
        [(brier_one, None), (log_loss_one, None)]
    rows = []
    for (model, arm), recs in ordered(groups):
        if arm != "monolithic" or model == key[0]:
            continue
        other = usable(recs)
        shared = sorted(set(baseline) & set(other))
        if (not shared or {r["item_id"] for r in recs} !=
                {r["item_id"] for r in groups[key]} or
                any(baseline[i]["verdict_true"] != other[i]["verdict_true"] for i in shared)):
            rows.append(f"| {label(model, arm)} | n/a |" + " n/a |" * len(views)
                        + " samples or distributions differ |")
            continue
        cells = []
        for loss, weights in views:
            diffs = _by_class([(baseline[i]["verdict_true"],
                                loss(distribution(other[i]), other[i]["verdict_true"])
                                - loss(distribution(baseline[i]), baseline[i]["verdict_true"]))
                               for i in shared])
            if weights and set(diffs) != set(VERDICTS):
                cells.append("n/a")
                continue
            d, lo, hi = stratified(diffs, weights)
            cells.append(f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        omitted = len(groups[key]) - len(shared)
        rows.append(f"| {label(model, arm)} | {len(shared)} | " + " | ".join(cells)
                    + f" | {omitted} unpaired or invalid |")
    if not rows:
        return "_No other direct-arm model for paired comparison._"
    heads = (["Weighted Brier Δ", "Weighted log loss Δ", "Unweighted Brier Δ"] if priors
             else ["Brier Δ", "Log loss Δ"])
    return ("| Model and arm | Paired posts | " + " | ".join(f"{h} [95% CI]" for h in heads)
            + " | Check |\n|---|---:|" + "---:|" * len(heads) + "---|\n" + "\n".join(rows))


CALIBRATION_NOTE = ("\n_The reliability plot, ECE, and confidence-gating table are unweighted "
                    "descriptive results for the stratified sample. They are not estimates for "
                    "the eligible-source class mix._\n")


def constant_baselines(recs: list[dict], priors: dict[str, float]) -> str:
    """No-model bars under the same eligible-source class mix."""
    truth = [r["verdict_true"] for r in recs]
    total = sum(priors.values())
    mix = {c: priors[c] / total for c in VERDICTS}
    always_nta = {c: float(c == "nta") for c in VERDICTS}
    rows = []
    for name, probs in (("Source-prior probabilities", mix), ("Always NTA", always_nta)):
        brier_score, log_loss = multiclass_scores([probs] * len(truth), truth,
                                                   tuple(VERDICTS), priors)
        accuracy = mix[max(probs, key=probs.get)]
        rows.append(f"| {name} | {brier_score:.3f} | {log_loss:.3f} | {accuracy:.1%} | 25.0% |")
    return ("| No-model baseline | Weighted Brier ↓ | Weighted log loss ↓ | Weighted accuracy "
            "| Macro recall |\n|---|---:|---:|---:|---:|\n" + "\n".join(rows))


def consensus_check(groups: dict[tuple[str, str], list[dict]], source: list[dict]) -> str:
    """Compare official flair with comment pluralities, then rescore agreeing posts."""
    key = _jev_direct(groups) or next(iter(groups))
    sample = {r["item_id"]: r["verdict_true"] for r in groups[key]}
    raw = {str(r["id"]): r for r in source if str(r["id"]) in sample}
    if set(raw) != set(sample):
        raise ValueError("source rows do not cover every post in the run")
    counts = {}
    for name, prefix in (("vote-weighted", "comments_prop_weighted_"),
                         ("unweighted", "comments_prop_")):
        matches = set()
        ties = 0
        for id, truth in sample.items():
            values = {v: raw[id][prefix + v.upper()] for v in VERDICTS}
            top = {v for v, p in values.items() if p == max(values.values())}
            ties += len(top) > 1
            if top == {truth}:
                matches.add(id)
        counts[name] = (matches, ties)
    agreeing = counts["vote-weighted"][0]
    mix = Counter(sample[id] for id in agreeing)
    rows = []
    for (model, arm), recs in ordered(groups):
        subset = [r for r in recs if r["item_id"] in agreeing]
        valid = [(r, distribution(r)) for r in subset]
        valid = [(r, p) for r, p in valid if p is not None]
        bs = (multiclass_scores([p for _, p in valid],
                                [r["verdict_true"] for r, _ in valid], tuple(VERDICTS))[0]
              if valid else float("nan"))
        accuracy = sum(predicted(r)[0] == r["verdict_true"] for r in subset) / len(subset)
        rows.append(f"| {label(model, arm)} | {len(subset)} | {accuracy:.1%} "
                    f"| {len(valid)}/{len(subset)} | {bs:.3f} |")
    return (f"Official flair matches the vote-weighted four-verdict plurality on "
            f"{len(agreeing)}/{len(sample)} posts ({_mix(mix)}); "
            f"{counts['vote-weighted'][1]} ties. The unweighted plurality matches "
            f"on {len(counts['unweighted'][0])}/{len(sample)} posts, with "
            f"{counts['unweighted'][1]} ties. INFO is excluded.\n\n"
            "| Model and arm | Posts | Sample accuracy | Brier coverage | Sample Brier ↓ |\n"
            "|---|---:|---:|---:|---:|\n" + "\n".join(rows))


def _mix(counts: Counter) -> str:
    return ", ".join(f"{counts[v]} {v.upper()}" for v in VERDICTS if counts[v])


def headline_table(groups: dict[tuple[str, str], list[dict]]) -> str:
    head = ("| Model and arm | Q/call | Sample accuracy | vs. sample majority | Stratified-sample chosen-label ECE ↓ "
            "| ECE support | p50 | p95 | $/1k posts | Unparseable |\n"
            "|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for (model, arm), recs in ordered(groups):
        got = [predicted(r) for r in recs]
        correct = [v == r["verdict_true"] for (v, _), r in zip(got, recs)]
        acc = sum(correct) / len(correct)
        base = baseline_accuracy(recs)
        lat = [r["latency_s"] for r in recs]
        per_1k = sum(r["cost_usd"] for r in recs) / len(recs) * 1000
        bad = sum(1 for r in recs if r.get("parse_failed"))
        # A malformed answer has no probability to calibrate; it is already
        # counted wrong in the accuracy column.
        parsed = [(c, float(ok)) for (v, c), ok in zip(got, correct) if v != UNPARSED_CHOICE]
        conf, truth = [c for c, _ in parsed], [t for _, t in parsed]
        n_bins, min_n = bin_support(conf, truth)
        rows.append(
            f"| {label(model, arm)} | {recs[0]['n_questions']} | {acc:.1%} "
            f"| {(acc - base) * 100:+.1f} pp | {ece(conf, truth):.3f} "
            f"| {n_bins} bins, min {min_n} "
            f"| {percentile(lat, 0.5)*1000:.0f} ms | {percentile(lat, 0.95)*1000:.0f} ms "
            f"| ${per_1k:.3f} | {bad} |"
        )
    return head + "\n".join(rows)


def confusion_table(groups: dict[tuple[str, str], list[dict]]) -> str:
    """Which verdicts get confused for which.

    The failure worth watching: collapsing everything into NTA, which scores
    well on Reddit's lopsided distribution while understanding nothing.
    """
    out = []
    for (model, arm), recs in ordered(groups):
        pairs = Counter((r["verdict_true"], predicted(r)[0]) for r in recs)
        order = list(VERDICTS) + sorted({p for _, p in pairs} - set(VERDICTS))
        out.append(f"\n**{label(model, arm)}.** Rows are Reddit's verdict; columns are the model's.\n")
        out.append("| | " + " | ".join(v.upper() if v in VERDICTS else "UNPARSED"
                                       for v in order) + " | recall |")
        out.append("|---" * (len(order) + 2) + "|")
        for t in order:
            total = sum(pairs[(t, p)] for p in order)
            if not total:
                continue
            cells = []
            for p in order:
                n = pairs[(t, p)]
                cells.append(f"**{n}**" if p == t and n else str(n))
            out.append(f"| **{t.upper()}** | " + " | ".join(cells)
                       + f" | {pairs[(t, t)] / total:.0%} |")
    return "\n".join(out)


def fanout_note(groups: dict[tuple[str, str], list[dict]]) -> str:
    """Test the Speculative Fan-Out claim: extra questions are nearly free."""
    rows = []
    for model in sorted({m for m, _ in groups}):
        base = groups.get((model, "monolithic"))
        recs = groups.get((model, "balanced"))
        if not base or not recs:
            continue
        b50 = percentile([r["latency_s"] for r in base], 0.5) * 1000
        bc = sum(r["cost_usd"] for r in base) / len(base)
        a50 = percentile([r["latency_s"] for r in recs], 0.5) * 1000
        ac = sum(r["cost_usd"] for r in recs) / len(recs)
        rows.append(
            f"| {NAMES.get(model, model)} | {base[0]['n_questions']}→{recs[0]['n_questions']} "
            f"| {b50:.0f} → {a50:.0f} ms | {a50/b50:.2f}× "
            f"| ${bc*1000:.3f} → ${ac*1000:.3f} | {ac/bc if bc else float('nan'):.2f}× |"
        )
    if not rows:
        return "_Needs the direct arm plus the two-question arm for the same model._"
    return ("| Model | Questions | p50 latency | Latency ratio | $/1k | Cost ratio |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows))


def confidence_gate_table(groups: dict[tuple[str, str], list[dict]]) -> str:
    """Routing on chosen-label probability: accept high values, review the rest.

    A useful confidence signal buys accuracy as coverage falls. A flat or
    inverted column means the number is decorative -- the specific failure to
    watch for in an LLM's self-reported confidence.
    """
    rows = []
    for (model, arm), recs in ordered(groups):
        got = [predicted(r) for r in recs]
        conf = [c for _, c in got]
        correct = [v == r["verdict_true"] for (v, _), r in zip(got, recs)]
        for c in gated_coverage(conf, correct, [0.5, 0.7, 0.9, 0.95]):
            acc = "n/a" if c.accuracy != c.accuracy else f"{c.accuracy:.1%}"
            rows.append(f"| {label(model, arm)} | {c.threshold:.2f} | {c.coverage:.1%} | {acc} |")
    return ("| Model and arm | Chosen-label p ≥ | Coverage | Accuracy when it acts |\n"
            "|---|---|---|---|\n" + "\n".join(rows))


def _svg_open(w: int, h: int) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="system-ui,-apple-system,Segoe UI,sans-serif" '
            f'font-size="13">')


def reliability_svg(groups: dict[tuple[str, str], list[dict]], w: int = 700, h: int = 460) -> str:
    """Chosen-label probability (x) against observed accuracy (y).

    The diagonal is perfect calibration. Below it the model is overconfident --
    the usual failure. Dot area scales with bin population.
    """
    pad = 52
    plot = min(w, h) - pad * 2
    px = lambda v: pad + v * plot          # noqa: E731
    py = lambda v: h - pad - v * plot      # noqa: E731

    parts = [
        _svg_open(w, h),
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" '
        f'stroke="{AXIS}" stroke-dasharray="4 4"/>',
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(0)}" stroke="{AXIS}"/>',
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(0)}" y2="{py(1)}" stroke="{AXIS}"/>',
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{px(t)}" y="{py(0)+16}" fill="{AXIS}" text-anchor="middle">{t:g}</text>')
        parts.append(f'<text x="{px(0)-8}" y="{py(t)+4}" fill="{AXIS}" text-anchor="end">{t:g}</text>')
    parts.append(f'<text x="{px(0.5)}" y="{h-8}" fill="{AXIS}" text-anchor="middle">chosen-label probability</text>')
    parts.append(f'<text x="14" y="{py(0.5)}" fill="{AXIS}" text-anchor="middle" '
                 f'transform="rotate(-90 14 {py(0.5)})">actually right this often</text>')

    for i, ((model, arm), recs) in enumerate(ordered(groups)):
        color = COLORS[i % len(COLORS)]
        got = [(v, c, v == r["verdict_true"]) for r in recs for v, c in [predicted(r)]
               if v != UNPARSED_CHOICE]
        bins = reliability_bins([c for _, c, _ in got], [float(ok) for *_, ok in got])
        biggest = max((b.n for b in bins), default=1)
        pts = " ".join(f"{px(b.mean_pred):.1f},{py(b.mean_truth):.1f}" for b in bins)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for b in bins:
            parts.append(f'<circle cx="{px(b.mean_pred):.1f}" cy="{py(b.mean_truth):.1f}" '
                         f'r="{2.5 + 5*(b.n/biggest)**0.5:.1f}" fill="{color}"/>')
        y = pad + 18 * i
        parts.append(f'<circle cx="{h-20}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{h-10}" y="{y+4}" fill="{AXIS}">{label(model, arm)}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def headline_svg(groups: dict[tuple[str, str], list[dict]], priors: dict[str, float],
                 w: int = 820) -> str:
    """Weighted Brier with 95% intervals, best first, against the no-model forecast.

    Each row also prints cost and median latency, so the chart carries the
    whole trade-off the README describes.
    """
    rows = []
    for (model, arm), recs in groups.items():
        scored = [(r["verdict_true"], distribution(r)) for r in recs]
        scored = [(t, p) for t, p in scored if p is not None]
        if {t for t, _ in scored} != set(VERDICTS):
            continue
        mean, lo, hi = stratified(_by_class([(t, brier_one(p, t)) for t, p in scored]), priors)
        cost = sum(r["cost_usd"] for r in recs) / len(recs) * 1000
        p50 = percentile([r["latency_s"] for r in recs], 0.5)
        rows.append((mean, lo, hi, label(model, arm), cost, p50, model.startswith("local/")))
    rows.sort()
    any_recs = next(iter(groups.values()))
    total = sum(priors.values())
    prior = {c: priors[c] / total for c in VERDICTS}
    base = multiclass_scores([prior] * len(any_recs), [r["verdict_true"] for r in any_recs],
                             tuple(VERDICTS), priors)[0]

    top, row_h, left, right = 92, 34, 250, 270
    n = len(rows)
    h = top + row_h * n + 56
    x0, x1 = 0.30, 0.65
    px = lambda v: left + (v - x0) / (x1 - x0) * (w - left - right)   # noqa: E731
    cols = [(w - right + 40, "Brier"), (w - right + 140, "$ / 1k posts"), (w - right + 235, "median call")]
    parts = [
        _svg_open(w, h),
        f'<text x="16" y="26" fill="{AXIS}" font-size="16" font-weight="600">'
        f'Which model forecasts the Reddit verdict best? Lower is better.</text>',
        f'<text x="16" y="46" fill="{AXIS}" font-size="12">Population-weighted four-class '
        f'Brier score with 95% bootstrap interval, {len(any_recs)} Reddit AITA posts '
        f'from 2025.</text>',
        f'<circle cx="22" cy="63" r="5" fill="{COLORS[0]}"/>'
        f'<text x="32" y="67" fill="{AXIS}" font-size="12">whole interval beats the no-model '
        f'forecast</text>'
        f'<circle cx="292" cy="63" r="5" fill="{AXIS}"/>'
        f'<text x="302" y="67" fill="{AXIS}" font-size="12">does not</text>',
    ]
    for x, name in cols:
        parts.append(f'<text x="{x}" y="{top - 8}" fill="{AXIS}" font-size="12" '
                     f'text-anchor="end">{name}</text>')
    for t in (0.3, 0.4, 0.5, 0.6):
        parts.append(f'<line x1="{px(t):.1f}" y1="{top}" x2="{px(t):.1f}" '
                     f'y2="{top + row_h * n}" stroke="{AXIS}" stroke-opacity="0.25"/>')
        parts.append(f'<text x="{px(t):.1f}" y="{top + row_h * n + 18}" fill="{AXIS}" '
                     f'text-anchor="middle" font-size="12">{t:.1f}</text>')
    parts.append(f'<line x1="{px(base):.1f}" y1="{top - 4}" x2="{px(base):.1f}" '
                 f'y2="{top + row_h * n}" stroke="{AXIS}" stroke-dasharray="5 4" stroke-width="1.5"/>')
    parts.append(f'<text x="{px(base):.1f}" y="{top + row_h * n + 38}" fill="{AXIS}" '
                 f'text-anchor="middle" font-size="12">no-model forecast from base rates '
                 f'({base:.3f})</text>')
    for i, (mean, lo, hi, name, cost, p50, local) in enumerate(rows):
        y = top + row_h * i + row_h / 2
        color = COLORS[0] if hi < base else AXIS
        weight = ' font-weight="700"' if name.startswith("Jev") else ""
        money = "not billed" if local else f"${cost:.3f}"
        cells = (f"{mean:.3f}", money, f"{p50:.2f} s")
        parts += [
            f'<text x="{left - 14}" y="{y + 4:.1f}" fill="{AXIS}" text-anchor="end"{weight}>'
            f'{name}</text>',
            f'<line x1="{px(lo):.1f}" y1="{y:.1f}" x2="{px(hi):.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="3" stroke-linecap="round"/>',
            f'<circle cx="{px(mean):.1f}" cy="{y:.1f}" r="6" fill="{color}"/>',
            *(f'<text x="{x}" y="{y + 4:.1f}" fill="{AXIS}" font-size="13" text-anchor="end">'
              f'{cell}</text>' for (x, _), cell in zip(cols, cells)),
        ]
    parts.append("</svg>")
    return "\n".join(parts)


def render(groups: dict[tuple[str, str], list[dict]], recs: list[dict], svg: Path,
           meta: dict | None = None, priors: dict[str, float] | None = None,
           source: list[dict] | None = None) -> str:
    """The whole report as one markdown string, so it can be printed and saved."""
    any_recs = next(iter(groups.values()))
    mix = Counter(r["verdict_true"] for r in any_recs)
    out = [
        f"_Generated by `jevbench.report`. {len(any_recs)} posts ({_mix(mix)}) · "
        f"{len(recs)} requests across {len(groups)} model and arm pairs._\n",
        run_summary(recs, meta),
        check_same_sample(recs),
        "\n### Primary score: four-way probabilities\n",
        probability_table(groups, priors),
        ("\n_Weighted scores use class frequencies among eligible four-flair source posts. "
         "Intervals resample posts within each class. Macro recall is the unweighted mean "
         "of the four class recalls. Malformed answers count as wrong for top-1 and recall "
         "and are left out of Brier and log loss. Log loss clips the true-class probability "
         "at 1e-15, so a zero on the true class costs about 34.5 instead of infinity._\n" if priors
         else "\n_Population weights unavailable; scores reflect this sample's class mix._\n"),
        "\n### Paired comparison with Jev's direct arm\n",
        paired_table(groups, priors),
        "\n_Positive means worse than Jev. Only posts where both models gave a valid "
        "distribution are paired._\n",
        ("\n### No-model baselines\n\n" + constant_baselines(
            groups[_jev_direct(groups) or next(iter(groups))], priors) if priors else ""),
        "\n### Accuracy, latency and cost\n",
        headline_table(groups),
        ("\n### Flair and comment disagreement\n\n" + consensus_check(groups, source)
         if source is not None else ""),
        f"\n![Chosen-label probability versus observed accuracy]({svg})\n",
        CALIBRATION_NOTE,
        "### Two-question cost and latency\n",
        fanout_note(groups),
        "\n### Where it goes wrong\n",
        confusion_table(groups),
        "\n### Confidence-gated routing\n",
        confidence_gate_table(groups),
        CALIBRATION_NOTE,
        "\n### Cost, speed and failures\n",
        ops_table(recs),
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render tables and charts from run logs.")
    ap.add_argument("run", type=Path, nargs="+",
                    help="one or more run logs; several are merged into one report")
    ap.add_argument("--svg", type=Path, default=Path("docs/reliability.svg"))
    ap.add_argument("--chart", type=Path,
                    help="also write the headline Brier chart here (needs --priors)")
    ap.add_argument("--md", type=Path, default=Path("docs/results.md"),
                    help="also write the tables here, so a run's output is kept")
    ap.add_argument("--priors", type=Path,
                    help="JSON file with exact population counts for all four verdicts")
    ap.add_argument("--source-raw", type=Path,
                    help="original dilemma rows for the comment-plurality label check")
    args = ap.parse_args()

    recs = load(*args.run)
    groups = by_arm(recs)
    metas = [json.loads(p.with_suffix(".meta.json").read_text())
             for p in args.run if p.with_suffix(".meta.json").exists()]
    meta = metas[0] if metas else None
    priors = json.loads(args.priors.read_text()) if args.priors else None
    if isinstance(priors, dict) and "eligible_mix" in priors:
        priors = priors["eligible_mix"]
    if priors is not None:
        if (not isinstance(priors, dict) or set(priors) != set(VERDICTS) or
                any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0
                    for v in priors.values())):
            ap.error("--priors needs positive counts for nta, yta, esh, and nah")
    if args.chart and priors is None:
        ap.error("--chart needs --priors")
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(reliability_svg(groups))
    if args.chart:
        args.chart.write_text(headline_svg(groups, priors))

    source = ([json.loads(line) for line in args.source_raw.read_text().splitlines() if line]
              if args.source_raw else None)
    svg_link = Path(os.path.relpath(args.svg, args.md.parent)) if args.md else args.svg
    md = render(groups, recs, svg_link, meta, priors, source)
    print(md)

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n")
    print(f"\nwrote {args.svg}" + (f", {args.chart}" if args.chart else "")
          + (f" and {args.md}" if args.md else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
