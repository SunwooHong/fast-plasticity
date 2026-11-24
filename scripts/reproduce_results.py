"""Paired comparisons for the representative subset in results/episode_returns.csv.

The CSV covers four confirmation conditions, not every row of the submitted
Table 1. The SVG is a subset bar chart, not the manuscript figure package.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results" / "paper"
T = {20: 2.093024054408263, 10: 2.2621571627409915}


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load():
    with (ROOT / "results" / "episode_returns.csv").open() as handle:
        return list(csv.DictReader(handle))


def paired(group, left, right):
    def by_method(name):
        found = {}
        for row in group:
            if row["method"] != name:
                continue
            seed = int(row["seed"])
            if seed in found:
                raise SystemExit(f"duplicate seed {seed} for {name}")
            found[seed] = float(row["return"])
        return found

    a = by_method(left)
    b = by_method(right)
    if set(a) != set(b):
        raise SystemExit(f"seed sets differ for {left} vs {right}")
    seeds = sorted(a)
    if len(seeds) != 20:
        raise SystemExit(f"expected 20 paired seeds, found {len(seeds)}")
    d = [a[s] - b[s] for s in seeds]
    n = len(d)
    m = mean(d)
    s = sd(d)
    half = T[n] * s / math.sqrt(n) if n in T else float("nan")
    pos = sum(x > 1e-9 for x in d)
    neg = sum(x < -1e-9 for x in d)
    return {
        "n": n,
        "left": mean(a[s] for s in seeds),
        "right": mean(b[s] for s in seeds),
        "diff": m,
        "lo": m - half,
        "hi": m + half,
        "pos": pos,
        "neg": neg,
        "zero": n - pos - neg,
    }


def figure(rows, path):
    groups = []
    for key in (
        ("walker-walk", "0.6", "1.0"),
        ("cheetah-run", "0.7", "1.0"),
        ("hopper-hop", "1.0", "1.3"),
    ):
        subset = [r for r in rows if (r["task"], r["gain"], r["mass"]) == key]
        groups.append((key, paired(subset, "bias", "frozen"), paired(subset, "keyed", "frozen")))
    width, height = 420, 220
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<line x1="40" y1="180" x2="400" y2="180" stroke="#E5E7EB"/>',
        '<text x="40" y="18" font-size="11" fill="#111827">Subset only: three confirmation contrasts</text>',
        '<line x1="40" y1="30" x2="40" y2="180" stroke="#E5E7EB"/>',
    ]
    for i, (key, bias, keyed) in enumerate(groups):
        x = 80 + i * 110
        for stat, color, dx in ((keyed, "#76AEDB", -12), (bias, "#0B6FB0", 12)):
            h = max(0.0, stat["diff"]) * 0.4
            parts.append(f'<rect x="{x+dx}" y="{180-h:.1f}" width="16" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x}" y="200" font-size="10" fill="#6B7280">{key[0].split("-")[0]}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def main():
    rows = load()
    groups = defaultdict(list)
    for row in rows:
        groups[(
            row["task"],
            row["perturbation"],
            row["gain"],
            row["mass"],
            row["checkpoint"],
            row["cohort"],
        )].append(row)
    lines = [
        "task,perturbation_family,gain,mass,checkpoint,cohort,contrast,n,mean_a,mean_b,diff,ci_low,ci_high,pos,neg,zero"
    ]
    for key, group in sorted(groups.items()):
        for left, right in (("keyed", "frozen"), ("bias", "frozen")):
            stat = paired(group, left, right)
            lines.append(
                ",".join([
                    key[0], key[1], key[2], key[3], key[4], key[5],
                    f"{left}-{right}", str(stat["n"]),
                    f"{stat['left']:.6f}", f"{stat['right']:.6f}", f"{stat['diff']:.6f}",
                    f"{stat['lo']:.6f}", f"{stat['hi']:.6f}",
                    str(stat["pos"]), str(stat["neg"]), str(stat["zero"]),
                ])
            )
            print(key, left, f"{stat['diff']:.2f}", f"[{stat['lo']:.2f}, {stat['hi']:.2f}]", f"{stat['pos']}/{stat['neg']}/{stat['zero']}")
    text = "\n".join(lines) + "\n"
    (ROOT / "results" / "summary.csv").write_text(text)
    PAPER.mkdir(parents=True, exist_ok=True)
    (PAPER / "paired_summary.csv").write_text(text)
    figure(rows, ROOT / "results" / "paired_bias_keyed.svg")
    print("wrote", ROOT / "results" / "summary.csv")


if __name__ == "__main__":
    main()
