"""Figures + effect sizes from results/*.jsonl.  python -m analysis.make_all

Writes figures/*.png and analysis/effects.md. Study data = the coarse/zoom/ablation runs listed in STUDY_FILES
(the pilot and smoke runs are excluded; the pilot repeats sequences 0-4 of the coarse run).
Breaking point = first tested N whose accuracy is below THRESHOLD (see REPORT.md for why 0.75).
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from eval.score import apply_truncation_rule  # noqa: E402

STUDY_FILES = ["coarse_gemma4_default", "zoom1_gemma4_default", "zoom2_n12_gemma4_default",
               "borders_count_gemma4_default", "max1024_ts0_gemma4", "best_bd1_ts1_gemma4_default"]
THRESHOLD = 0.75
OUT = Path("figures")
TASKS = ["moving_dot", "tile_probe", "count_events"]
NS = [4, 9, 12, 16, 25, 36, 49, 64]

# dataviz reference palette, light mode (validated: scripts/validate_palette.js, 4 slots all PASS)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SERIES = {"no labels": "#2a78d6", "borders": "#eb6834", "timestamps": "#1baf7a", "timestamps + borders": "#eda100"}
LIMIT_COLOUR = {"default": "#2a78d6", "1024": "#eb6834"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def load() -> list[dict]:
    rows, seen = [], set()
    for name in STUDY_FILES:
        for line in open(Path("results") / f"{name}.jsonl", encoding="utf-8"):
            r = apply_truncation_rule(json.loads(line))
            c = r["condition"]
            limit = "1024" if r["server"]["image_max_tokens"] == 1024 else "default"
            key = (limit, r["task"], r["n"], c["timestamps"], c["borders"], c["order"], c["aligned"], c["fmt"],
                   c["budget"], r["seq_index"])
            if key in seen or "error" in r:
                continue
            seen.add(key)
            truth_frame = r["truth"] if r["task"] in ("tile_probe", "moving_dot") else None
            cell = r["cells"][truth_frame - 1] if truth_frame else None
            rows.append({"limit": limit, "task": r["task"], "n": r["n"], "ts": c["timestamps"], "bd": c["borders"],
                         "seq": r["seq_index"], "correct": bool(r["correct"]), "within1": bool(r["within1"]),
                         "eff_tile": r["predicted"]["eff_tile_px"],
                         "tok_per_frame": (r["predicted"]["total_tokens"] - 2) / r["n"],
                         "cell": cell, "rows": max(c2["row"] for c2 in r["cells"]) + 1,
                         "cols": max(c2["col"] for c2 in r["cells"]) + 1})
    return rows


def condition(r) -> str:
    return {(False, False): "no labels", (False, True): "borders", (True, False): "timestamps",
            (True, True): "timestamps + borders"}[(r["ts"], r["bd"])]


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def curve(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["n"]].append(r["correct"])
    ns = sorted(g)
    acc = [sum(g[n]) / len(g[n]) for n in ns]
    ci = [wilson(sum(g[n]), len(g[n])) for n in ns]
    return ns, acc, ci, [len(g[n]) for n in ns]


def breaking_point(ns, acc):
    return next((n for n, a in zip(ns, acc) if a < THRESHOLD), None)


def fig_accuracy_vs_n(rows, table):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, TASKS):
        style(ax)
        for name, colour in SERIES.items():
            sub = [r for r in rows if r["task"] == task and r["limit"] == "default" and condition(r) == name]
            if not sub:
                continue
            ns, acc, ci, cnt = curve(sub)
            x = np.log2(ns)
            ax.fill_between(x, [c[0] for c in ci], [c[1] for c in ci], color=colour, alpha=0.12, linewidth=0)
            ax.plot(x, acc, color=colour, linewidth=2, marker="o", markersize=5, label=name,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            bp = breaking_point(ns, acc)
            if bp:
                i = ns.index(bp)
                ax.plot(x[i], acc[i], marker="o", markersize=13, markerfacecolor="none", markeredgecolor=colour,
                        markeredgewidth=1.5)
            ax.annotate(name, (x[-1], acc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8,
                        color=INK2, va="center")
            for n, a, c, k in zip(ns, acc, ci, cnt):
                table.append(f"| {task} | {name} | {n} | {k} | {a:.2f} | {c[0]:.2f}-{c[1]:.2f} |")
            table.append(f"| {task} | {name} | breaking point | | {bp or 'none (>= 0.75 at all N)'} | |")
        ax.axhline(THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
        ax.set_xticks(np.log2(NS), [str(n) for n in NS])
        ax.set_xlim(np.log2(4) - 0.2, np.log2(64) + 1.6)
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(task, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("frames in mosaic (N), 1536² budget", color=INK2, fontsize=9)
    axes[0].set_ylabel("accuracy (exact)", color=INK2, fontsize=9)
    axes[0].text(np.log2(4), THRESHOLD + 0.02, f"breaking threshold {THRESHOLD}", color=MUTED, fontsize=8)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK2)
    fig.suptitle("Gemma 4 12B, default image limits (mosaic seen at 768²): accuracy vs frames per mosaic. "
                 "Ring = breaking point; band = 95% CI", color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(OUT / "accuracy_vs_n.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_accuracy_vs_tokens(rows, table):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, facecolor=SURFACE)
    for ax, task in zip(axes, ["moving_dot", "tile_probe"]):
        style(ax)
        for limit, colour in LIMIT_COLOUR.items():
            sub = [r for r in rows if r["task"] == task and r["limit"] == limit and condition(r) == "no labels"]
            g = defaultdict(list)
            for r in sub:
                g[(r["n"], r["tok_per_frame"], r["eff_tile"])].append(r["correct"])
            pts = sorted(g, key=lambda k: k[1])
            x = [p[1] for p in pts]
            acc = [sum(g[p]) / len(g[p]) for p in pts]
            ci = [wilson(sum(g[p]), len(g[p])) for p in pts]
            ax.errorbar(x, acc, yerr=[[a - c[0] for a, c in zip(acc, ci)], [c[1] - a for a, c in zip(acc, ci)]],
                        color=colour, linewidth=2, marker="o", markersize=5, capsize=0, elinewidth=1,
                        label=f"{limit} image limit")
            for p, a in zip(pts, acc):
                ax.annotate(f"N={p[0]}", (p[1], a), xytext=(4, 6) if limit == "default" else (-8, -14),
                            textcoords="offset points", fontsize=7, color=MUTED,
                            ha="left" if limit == "default" else "right")
                table.append(f"| {task} | {limit} | {p[0]} | {p[1]:.1f} | {p[2]} | {len(g[p])} | "
                             f"{sum(g[p]) / len(g[p]):.2f} |")
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(task + " (no timestamps, no borders)", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("vision tokens per frame (log)", color=INK2, fontsize=9)
    axes[0].set_ylabel("accuracy (exact)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    fig.suptitle("More tokens per frame barely helps: accuracy vs vision tokens per frame (Gemma 4 12B)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "accuracy_vs_tokens_per_frame.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_position_heatmap(rows, table):
    """Error rate by where the target frame sits in the grid (normalised to 3x3 bins), N >= 9, no labels."""
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), facecolor=SURFACE)
    for ax, task in zip(axes, ["moving_dot", "tile_probe"]):
        sub = [r for r in rows if r["task"] == task and condition(r) == "no labels" and r["n"] >= 9]
        err, cnt = np.zeros((3, 3)), np.zeros((3, 3))
        for r in sub:
            i = min(2, r["cell"]["row"] * 3 // r["rows"])
            j = min(2, r["cell"]["col"] * 3 // r["cols"])
            cnt[i, j] += 1
            err[i, j] += not r["correct"]
        rate = np.divide(err, cnt, out=np.full((3, 3), np.nan), where=cnt > 0)
        ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                v = rate[i, j]
                txt = "–" if np.isnan(v) else f"{v:.0%}\nn={int(cnt[i, j])}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                        color="#ffffff" if (not np.isnan(v) and v > 0.5) else INK)
                table.append(f"| {task} | {['top', 'middle', 'bottom'][i]} | {['left', 'centre', 'right'][j]} | "
                             f"{int(cnt[i, j])} | {'' if np.isnan(v) else f'{v:.2f}'} |")
        ax.set_xticks(range(3), ["left", "centre", "right"])
        ax.set_yticks(range(3), ["top", "middle", "bottom"])
        ax.tick_params(colors=INK2, labelsize=9, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{task}: error rate by target position", color=INK, fontsize=10, loc="left")
    fig.suptitle("Where in the grid the target frame was (N >= 9, no timestamps/borders, both image limits)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "position_heatmap.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def paired_effect(rows, label, sel_a, sel_b, key=("limit", "task", "n", "seq")):
    """Mean paired difference (b - a) in exact accuracy over matched sequences, seeded bootstrap 95% CI."""
    a = {tuple(r[k] for k in key): r["correct"] for r in rows if sel_a(r)}
    b = {tuple(r[k] for k in key): r["correct"] for r in rows if sel_b(r)}
    common = sorted(set(a) & set(b))
    if not common:
        return f"| {label} | 0 | | | | not measured |"
    d = np.array([int(b[k]) - int(a[k]) for k in common])
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (10000, len(d))).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    acc_a, acc_b = np.mean([a[k] for k in common]), np.mean([b[k] for k in common])
    return f"| {label} | {len(common)} | {acc_a:.2f} | {acc_b:.2f} | {d.mean():+.2f} | {lo:+.2f} to {hi:+.2f} |"


def effects(rows) -> list[str]:
    out = ["| effect (b vs a) | matched calls | acc a | acc b | b - a | 95% bootstrap CI |", "|---|---|---|---|---|---|"]
    d = lambda r: r["limit"] == "default"  # noqa: E731
    for task in TASKS:
        out.append(paired_effect(rows, f"timestamps on vs off, no borders: {task}",
                                 lambda r, t=task: d(r) and r["task"] == t and not r["ts"] and not r["bd"],
                                 lambda r, t=task: d(r) and r["task"] == t and r["ts"] and not r["bd"]))
    out.append(paired_effect(rows, "borders on vs off, no timestamps: count_events",
                             lambda r: d(r) and r["task"] == "count_events" and not r["ts"] and not r["bd"],
                             lambda r: d(r) and r["task"] == "count_events" and not r["ts"] and r["bd"]))
    for task in TASKS:
        out.append(paired_effect(rows, f"borders on vs off, with timestamps: {task}",
                                 lambda r, t=task: d(r) and r["task"] == t and r["ts"] and not r["bd"],
                                 lambda r, t=task: d(r) and r["task"] == t and r["ts"] and r["bd"]))
    out.append(paired_effect(rows, "image limit 1024 vs default, no labels: moving_dot + tile_probe, N 9-16",
                             lambda r: r["limit"] == "default" and not r["ts"] and not r["bd"],
                             lambda r: r["limit"] == "1024" and not r["ts"] and not r["bd"],
                             key=("task", "n", "seq")))
    for name in ("snake vs row-major order", "patch-misaligned vs aligned tiles",
                 "separate images vs one mosaic"):
        out.append(f"| {name} | 0 | | | | not measured yet |")
    return out


def main():
    OUT.mkdir(exist_ok=True)
    rows = load()
    t1 = ["| task | condition | N | calls | accuracy | 95% CI |", "|---|---|---|---|---|---|"]
    t2 = ["| task | image limit | N | tokens/frame | eff tile px | calls | accuracy |", "|---|---|---|---|---|---|---|"]
    t3 = ["| task | row band | column band | calls | error rate |", "|---|---|---|---|---|"]
    fig_accuracy_vs_n(rows, t1)
    fig_accuracy_vs_tokens(rows, t2)
    fig_position_heatmap(rows, t3)
    md = [f"# Analysis tables (generated by analysis/make_all.py from {len(rows)} calls)", "",
          "## Accuracy vs N (default image limits) — data for figures/accuracy_vs_n.png", "", *t1, "",
          "## Accuracy vs vision tokens per frame — data for figures/accuracy_vs_tokens_per_frame.png", "", *t2, "",
          "## Error rate by target position — data for figures/position_heatmap.png", "", *t3, "",
          "## Effect sizes (paired on identical sequences)", "", *effects(rows), ""]
    Path("analysis/effects.md").write_text("\n".join(md), encoding="utf-8")
    print(f"{len(rows)} calls -> figures/*.png, analysis/effects.md")


if __name__ == "__main__":
    main()
