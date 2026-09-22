"""Turn a run log into the tables and chart the README shows.

Everything is derived from the committed JSONL, so the README regenerates from
scratch and nothing in it is hand-typed.

Reddit labels a verdict, not a probability, so calibration here is the standard
confidence kind: bin answers by how confident the model was, and check whether
it was right that often. "When it says 80% sure, is it right 80% of the time?"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .data import VERDICTS
from .metrics import (bin_support, brier, ece, gated_coverage, percentile,
                      reliability_bins)
from .questions import (DECIDING, SEVERITY_LEVELS, compose_verdict,
                        severity_target)

COLORS = ["#2f81f7", "#d29922", "#8957e5", "#3fb950", "#db6d28", "#e34c26"]
AXIS = "#8b949e"


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
    """What the run actually cost, in money, time and failures.

    Cost per 1k is the number people quote, but the totals matter for anyone
    deciding whether to reproduce this, and the failure and retry columns are
    how you tell a cheap model from an unreliable one.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in recs:
        groups[(r["model"], r["arm"])].append(r)

    rows = []
    for (model, arm), rs in sorted(groups.items()):
        ok = [r for r in rs if not r.get("error")]
        errs = len(rs) - len(ok)
        retried = sum(1 for r in ok if r.get("attempts", 1) > 1)
        tok = sum(r.get("total_tokens", 0) for r in ok)
        spend = sum(r["cost_usd"] for r in ok)
        secs = sum(r["latency_s"] for r in ok)
        per_1k = (spend / len(ok) * 1000) if ok else float("nan")
        rows.append(
            f"| `{model}` | {arm} | {len(rs)} | {errs} | {retried} | {tok:,} "
            f"| ${spend:.4f} | ${per_1k:.3f} | {secs:.0f}s |"
        )
    head = ("| Model | Arm | Requests | Failed | Retried | Tokens | Spend "
            "| $/1k posts | API time |")
    sep = "|---" * 9 + "|"
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
    if meta:
        if meta.get("wall_s"):
            bits.append(f"{meta['wall_s']:.0f}s wall")
        if meta.get("git_sha"):
            bits.append(f"commit `{meta['git_sha']}`")
        if meta.get("seed") is not None:
            bits.append(f"seed {meta['seed']}")
    return " · ".join(bits) + f"\n\nServed by: {', '.join(f'`{m}`' for m in served)}"


def predicted(r: dict) -> tuple[str, float]:
    """The arm's verdict and the confidence behind it.

    A direct Choice carries its own confidence. A verdict composed from nouls
    has none -- Noul returns no confidence field -- so we synthesise one from
    how far the deciding noul sat from 0.5. That gap is itself a finding:
    composition costs you the confidence signal.

    Only DECIDING questions count. Including the context questions would let
    `op_omits` landing near 0 or 1 inflate the confidence behind a verdict it
    played no part in, which would corrupt exactly the numbers the
    decomposition hypothesis rests on.
    """
    if "verdict" in r.get("choices", {}):
        c = r["choices"]["verdict"]
        return c["choice"], c["confidence"]
    v = compose_verdict(r["probs"])
    deciding = [p for k, p in r["probs"].items() if k in DECIDING]
    spread = max((abs(p - 0.5) for p in deciding), default=0.0)
    return v, min(1.0, spread * 2)


def baseline_accuracy(recs: list[dict]) -> float:
    """Always answering the most common verdict in the sample."""
    counts = Counter(r["verdict_true"] for r in recs)
    return counts.most_common(1)[0][1] / len(recs)


def headline_table(groups: dict[tuple[str, str], list[dict]]) -> str:
    head = ("| Model | Arm | Q/call | Verdict accuracy | vs. majority verdict | Confidence ECE ↓ "
            "| ECE support | p50 | p95 | $/1k posts | Unparseable |\n"
            "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for (model, arm), recs in sorted(groups.items()):
        got = [predicted(r) for r in recs]
        correct = [v == r["verdict_true"] for (v, _), r in zip(got, recs)]
        conf = [c for _, c in got]
        acc = sum(correct) / len(correct)
        base = baseline_accuracy(recs)
        lat = [r["latency_s"] for r in recs]
        per_1k = sum(r["cost_usd"] for r in recs) / len(recs) * 1000
        bad = sum(1 for r in recs if r.get("parse_failed"))
        truth = [float(c) for c in correct]
        n_bins, min_n = bin_support(conf, truth)
        rows.append(
            f"| `{model}` | {arm} | {recs[0]['n_questions']} | {acc:.1%} "
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
    for (model, arm), recs in sorted(groups.items()):
        pairs = Counter((r["verdict_true"], predicted(r)[0]) for r in recs)
        order = list(VERDICTS)
        out.append(f"\n**`{model}` · {arm}** — rows are Reddit's verdict, columns the model's\n")
        out.append("| | " + " | ".join(v.upper() for v in order) + " | recall |")
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
        if not base:
            continue
        b50 = percentile([r["latency_s"] for r in base], 0.5) * 1000
        bc = sum(r["cost_usd"] for r in base) / len(base)
        for arm in ("decomposed", "full"):
            recs = groups.get((model, arm))
            if not recs:
                continue
            a50 = percentile([r["latency_s"] for r in recs], 0.5) * 1000
            ac = sum(r["cost_usd"] for r in recs) / len(recs)
            rows.append(
                f"| `{model}` | {arm} | {base[0]['n_questions']}→{recs[0]['n_questions']} "
                f"| {b50:.0f} → {a50:.0f} ms | {a50/b50:.2f}× "
                f"| ${bc*1000:.3f} → ${ac*1000:.3f} | {ac/bc if bc else float('nan'):.2f}× |"
            )
    if not rows:
        return "_Needs the monolithic arm plus at least one other for the same model._"
    return ("| Model | Arm | Questions | p50 latency | Latency ratio | $/1k | Cost ratio |\n"
            "|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def score_floor(recs: list[dict]) -> float:
    """Whether this run's Score levels are 0- or 1-indexed.

    The docs never pin it. Decided once from the smallest raw score in the whole
    run rather than per answer, so every score in a run is scaled the same way.
    A single 0.x anywhere means the levels start at 0.
    """
    raws = [r["scores"]["severity"]["raw_score"]
            for r in recs if "severity" in r.get("scores", {})]
    return 0.0 if raws and min(raws) < 1.0 else 1.0


def renormalise(raw: float, floor: float, n_levels: int) -> float:
    hi = floor + n_levels - 1
    return min(max((raw - floor) / (hi - floor), 0.0), 1.0) if hi > floor else 0.0


def severity_table(groups: dict[tuple[str, str], list[dict]], floor: float = 1.0) -> str:
    """The Score question, graded as a continuous position in [0,1].

    The target is the verdict mapped onto the rubric, not an independent human
    severity rating -- Reddit does not publish one. Directional only.
    """
    n_levels = len(SEVERITY_LEVELS)
    rows = []
    for (model, arm), recs in sorted(groups.items()):
        usable = [r for r in recs if "severity" in r.get("scores", {})]
        if not usable:
            continue
        preds = [renormalise(r["scores"]["severity"]["raw_score"], floor, n_levels)
                 for r in usable]
        truth = [severity_target(r["verdict_true"]) for r in usable]
        conf = [r["scores"]["severity"]["confidence"] for r in usable]
        rows.append(f"| `{model}` | {arm} | {ece(preds, truth):.3f} "
                    f"| {brier(preds, truth):.3f} | {sum(conf)/len(conf):.2f} |")
    if not rows:
        return "_No Score question in this run._"
    note = (f"\n\n_Levels read as {'0' if floor == 0 else '1'}-indexed, "
            f"derived from the smallest raw score in this run._")
    return ("| Model | Arm | ECE ↓ | Brier ↓ | Mean confidence |\n|---|---|---|---|---|\n"
            + "\n".join(rows) + note)


def confidence_gate_table(groups: dict[tuple[str, str], list[dict]]) -> str:
    """Routing on confidence: auto-accept the sure calls, review the rest.

    A useful confidence signal buys accuracy as coverage falls. A flat or
    inverted column means the number is decorative -- the specific failure to
    watch for in an LLM's self-reported confidence.
    """
    rows = []
    for (model, arm), recs in sorted(groups.items()):
        got = [predicted(r) for r in recs]
        conf = [c for _, c in got]
        correct = [v == r["verdict_true"] for (v, _), r in zip(got, recs)]
        for c in gated_coverage(conf, correct, [0.5, 0.7, 0.9, 0.95]):
            acc = "n/a" if c.accuracy != c.accuracy else f"{c.accuracy:.1%}"
            rows.append(f"| `{model}` | {arm} | {c.threshold:.2f} | {c.coverage:.1%} | {acc} |")
    return ("| Model | Arm | Confidence ≥ | Coverage | Accuracy when it acts |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows))


def reliability_svg(groups: dict[tuple[str, str], list[dict]], w: int = 460, h: int = 460) -> str:
    """Confidence (x) against observed accuracy (y).

    The diagonal is perfect calibration. Below it the model is overconfident --
    the usual failure. Dot area scales with bin population.
    """
    pad = 52
    plot = w - pad * 2
    px = lambda v: pad + v * plot          # noqa: E731
    py = lambda v: h - pad - v * plot      # noqa: E731

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="system-ui,sans-serif" font-size="11">',
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(1)}" '
        f'stroke="{AXIS}" stroke-dasharray="4 4"/>',
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(1)}" y2="{py(0)}" stroke="{AXIS}"/>',
        f'<line x1="{px(0)}" y1="{py(0)}" x2="{px(0)}" y2="{py(1)}" stroke="{AXIS}"/>',
    ]
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{px(t)}" y="{py(0)+16}" fill="{AXIS}" text-anchor="middle">{t:g}</text>')
        parts.append(f'<text x="{px(0)-8}" y="{py(t)+4}" fill="{AXIS}" text-anchor="end">{t:g}</text>')
    parts.append(f'<text x="{px(0.5)}" y="{h-8}" fill="{AXIS}" text-anchor="middle">stated confidence</text>')
    parts.append(f'<text x="14" y="{py(0.5)}" fill="{AXIS}" text-anchor="middle" '
                 f'transform="rotate(-90 14 {py(0.5)})">actually right this often</text>')

    for i, ((model, arm), recs) in enumerate(sorted(groups.items())):
        color = COLORS[i % len(COLORS)]
        got = [predicted(r) for r in recs]
        conf = [c for _, c in got]
        correct = [float(v == r["verdict_true"]) for (v, _), r in zip(got, recs)]
        bins = reliability_bins(conf, correct)
        biggest = max((b.n for b in bins), default=1)
        pts = " ".join(f"{px(b.mean_pred):.1f},{py(b.mean_truth):.1f}" for b in bins)
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
        for b in bins:
            parts.append(f'<circle cx="{px(b.mean_pred):.1f}" cy="{py(b.mean_truth):.1f}" '
                         f'r="{2.5 + 5*(b.n/biggest)**0.5:.1f}" fill="{color}"/>')
        y = pad + 14 * i
        parts.append(f'<circle cx="{pad+6}" cy="{y}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{pad+16}" y="{y+4}" fill="{AXIS}">{model} · {arm}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def render(groups: dict[tuple[str, str], list[dict]], recs: list[dict], svg: Path,
           meta: dict | None = None) -> str:
    """The whole report as one markdown string, so it can be printed and saved.

    Building it as a value rather than printing as we go is what lets the run
    output be committed instead of scrolling past in a terminal.
    """
    any_recs = next(iter(groups.values()))
    mix = Counter(r["verdict_true"] for r in any_recs)
    models = sorted({r["model"] for r in recs})
    out = [
        f"_Generated by `jevbench.report`. {len(any_recs)} posts · {dict(mix)} · "
        f"{len(recs)} requests across {len(models)} model(s)._\n",
        run_summary(recs, meta),
        check_same_sample(recs),
        "\n### Headline\n",
        headline_table(groups),
        f"\n![calibration]({svg})\n",
        "### Does decomposition pay? (Speculative Fan-Out)\n",
        fanout_note(groups),
        "\n### Where it goes wrong\n",
        confusion_table(groups),
        "\n### Score question: severity\n",
        severity_table(groups, score_floor(recs)),
        "\n### Confidence-gated routing\n",
        confidence_gate_table(groups),
        "\n### Cost, speed and failures\n",
        ops_table(recs),
    ]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Render tables and chart from a run log.")
    ap.add_argument("run", type=Path, nargs="+",
                    help="one or more run logs; several are merged into one report")
    ap.add_argument("--svg", type=Path, default=Path("docs/reliability.svg"))
    ap.add_argument("--md", type=Path, default=Path("docs/results.md"),
                    help="also write the tables here, so a run's output is kept")
    args = ap.parse_args()

    recs = load(*args.run)
    groups = by_arm(recs)
    metas = [json.loads(p.with_suffix(".meta.json").read_text())
             for p in args.run if p.with_suffix(".meta.json").exists()]
    meta = metas[0] if metas else None
    args.svg.parent.mkdir(parents=True, exist_ok=True)
    args.svg.write_text(reliability_svg(groups))

    md = render(groups, recs, args.svg, meta)
    print(md)

    if args.md:
        args.md.parent.mkdir(parents=True, exist_ok=True)
        args.md.write_text(md + "\n")
    print(f"\nwrote {args.svg}" + (f" and {args.md}" if args.md else ""), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
