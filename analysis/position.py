"""Controlled-position analysis (N=16, 4x4).  python -m analysis.position

Reads results/position_n16_gemma4_default.jsonl, writes figures/position_4x4.png, figures/error_offsets.png and
analysis/position.md. Error vector of a wrong answer = (answered row - true row, answered col - true col) in the
row-major 4x4 grid.
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from eval.score import apply_truncation_rule  # noqa: E402

from .make_all import BLUE_RAMP, INK, INK2, MUTED, SURFACE  # noqa: E402

SRC = Path("results/position_n16_gemma4_default.jsonl")
N, COLS = 16, 4
TASKS = ["moving_dot", "tile_probe"]
CMAP = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)


def rc(frame):
    return (frame - 1) // COLS, (frame - 1) % COLS


def classify(t, a):
    if a is None or not 1 <= a <= N:
        return "invalid / unparsed"
    (tr, tc), (ar, ac) = rc(t), rc(a)
    dr, dc = ar - tr, ac - tc
    if dr == 0 and abs(dc) == 1:
        return "column slip (same row, ±1 column)"
    if dc == 0 and abs(dr) == 1:
        return "row slip (same column, ±1 row = ±4 frames)"
    if abs(a - t) == 1:
        return "±1 frame across a row break"
    if abs(dr) == 1 and abs(dc) == 1:
        return "diagonal neighbour"
    return "scattered (2+ cells away)"


def load():
    return [apply_truncation_rule(json.loads(line)) for line in SRC.open(encoding="utf-8")]


def heat(ax, grid, counts, title, vmax=1.0):
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=vmax)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            ax.text(j, i, f"{v:.0%}\n" + (f"{int(counts[i, j])}" if counts is not None else ""),
                    ha="center", va="center", fontsize=9, color="#ffffff" if v > 0.5 * vmax else INK)
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rows = load()
    md = [f"# Controlled-position run (N=16, 4x4, default image limits, no borders) — {len(rows)} calls", ""]
    # --- 1. error rate per position -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE)
    md += ["## Error rate per target position (no timestamps, 5 sequences per position)", "",
           "| task | " + " | ".join(f"pos {p}" for p in range(1, N + 1)) + " | overall |",
           "|---|" + "---|" * (N + 1)]
    for ax, task in zip(axes, TASKS):
        sub = [r for r in rows if r["task"] == task and not r["condition"]["timestamps"]]
        err, cnt = np.zeros((4, 4)), np.zeros((4, 4))
        for r in sub:
            i, j = rc(r["target_position"])
            cnt[i, j] += 1
            err[i, j] += not r["correct"]
        rate = err / cnt
        heat(ax, rate, None, f"{task}: error rate by target position")
        for i in range(4):
            for j in range(4):
                ax.text(j + 0.42, i - 0.38, str(i * 4 + j + 1), ha="right", va="top", fontsize=7, color=MUTED)
        ax.set_xticks(range(4), [f"col {c + 1}" for c in range(4)])
        ax.set_yticks(range(4), [f"row {r + 1}" for r in range(4)])
        md.append(f"| {task} | " + " | ".join(f"{rate.flat[p]:.1f}" for p in range(N)) +
                  f" | {err.sum() / cnt.sum():.2f} ({int(err.sum())}/{int(cnt.sum())} wrong) |")
    fig.suptitle("Target frame placed at every cell of a 4x4 mosaic (5 sequences each, no timestamps). "
                 "Small grey number = frame index", color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("figures/position_4x4.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # control with timestamps
    md += ["", "## Control: timestamps on (1 sequence per position)", "", "| task | accuracy | wrong positions |",
           "|---|---|---|"]
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task and r["condition"]["timestamps"]]
        wrong = [r["target_position"] for r in sub if not r["correct"]]
        md.append(f"| {task} | {sum(r['correct'] for r in sub) / len(sub):.2f} ({len(sub)} calls) | {wrong or 'none'} |")

    # --- 2. error vectors ----------------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE)
    md += ["", "## Error vectors of wrong answers (no timestamps)", "",
           "| task | wrong | " + " | ".join(["row slip", "column slip", "±1 frame across row break",
                                                 "diagonal", "scattered", "invalid"]) + " |", "|---|---|" + "---|" * 6]
    for ax, task in zip(axes, TASKS):
        sub = [r for r in rows if r["task"] == task and not r["condition"]["timestamps"] and not r["correct"]]
        kinds = Counter(classify(r["truth"], r["parsed"]) for r in sub)
        order = ["row slip (same column, ±1 row = ±4 frames)", "column slip (same row, ±1 column)",
                 "±1 frame across a row break", "diagonal neighbour", "scattered (2+ cells away)", "invalid / unparsed"]
        md.append(f"| {task} | {len(sub)} | " + " | ".join(f"{kinds[k]} ({kinds[k] / len(sub):.0%})" for k in order) + " |")
        off = np.zeros((7, 7))
        for r in sub:
            if r["parsed"] is not None and 1 <= r["parsed"] <= N:
                (tr, tc), (ar, ac) = rc(r["truth"]), rc(r["parsed"])
                off[ar - tr + 3, ac - tc + 3] += 1
        ax.imshow(off, cmap=CMAP, vmin=0, vmax=max(1, off.max()))
        for i in range(7):
            for j in range(7):
                if off[i, j]:
                    ax.text(j, i, int(off[i, j]), ha="center", va="center", fontsize=9,
                            color="#ffffff" if off[i, j] > 0.5 * off.max() else INK)
        ax.plot(3, 3, marker="+", color=MUTED, markersize=12)
        ax.set_xticks(range(7), [str(d) for d in range(-3, 4)])
        ax.set_yticks(range(7), [str(d) for d in range(-3, 4)])
        ax.set_xlabel("column offset (answered − true)", color=INK2, fontsize=9)
        ax.set_ylabel("row offset (answered − true)", color=INK2, fontsize=9)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{task}: where wrong answers landed ({len(sub)} errors)", color=INK, fontsize=10, loc="left")
    fig.suptitle("Error offsets in grid cells; + = the true cell", color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("figures/error_offsets.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    # signed frame differences
    md += ["", "## Signed frame difference of wrong answers (answered − true)", "", "| task | differences |", "|---|---|"]
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task and not r["condition"]["timestamps"] and not r["correct"]]
        diffs = Counter(r["parsed"] - r["truth"] for r in sub if r["parsed"] is not None)
        md.append(f"| {task} | " + ", ".join(f"{d:+d}: {c}" for d, c in sorted(diffs.items())) + " |")

    # --- 3. moving_dot confound ------------------------------------------------------------------------------
    sub = [r for r in rows if r["task"] == "moving_dot" and not r["condition"]["timestamps"] and r["parsed"] is not None]
    later_red = [r for r in sub if r["parsed"] > r["truth"] and r["parsed"] <= N]
    earlier_blue = [r for r in sub if r["parsed"] < r["truth"]]
    last = [r for r in later_red if r["parsed"] == N]
    pos_err = {}
    for half, sel in (("positions 1-8", lambda p: p <= 8), ("positions 9-16", lambda p: p > 8)):
        h = [r for r in sub if sel(r["target_position"])]
        pos_err[half] = f"{sum(not r['correct'] for r in h) / len(h):.2f}"
    md += ["", "## moving_dot confound check (no timestamps)", "",
           f"- Answers naming a red frame that is not the first red frame: **{len(later_red)}/{len(sub)} "
           f"({len(later_red) / len(sub):.0%})**; of these, the last frame (16): {len(last)}.",
           f"- Answers naming a blue frame (before the change): {len(earlier_blue)}/{len(sub)} "
           f"({len(earlier_blue) / len(sub):.0%}).",
           f"- Error rate, change in positions 1-8: {pos_err['positions 1-8']}; positions 9-16: {pos_err['positions 9-16']}.",
           "- Position 1 has no blue frame (dot red from the start)."]

    # --- 4. attractor answers: wrong answers snapping to the first cell of a row --------------------------------
    md += ["", "## Where wrong answers land: first cell of a row (frames 1, 5, 9, 13)", "",
           "Chance share = 4 of the 15 other frames (27%) if wrong answers were spread evenly.", "",
           "| task | wrong answers | on a row's first cell | same row as truth | top 3 answered frames (all answers) |",
           "|---|---|---|---|---|"]
    for task in TASKS:
        sub = [r for r in rows if r["task"] == task and not r["condition"]["timestamps"]]
        wrong = [r for r in sub if not r["correct"] and r["parsed"] is not None and 1 <= r["parsed"] <= N]
        first_col = sum(rc(r["parsed"])[1] == 0 for r in wrong)
        same_row = sum(rc(r["parsed"])[0] == rc(r["truth"])[0] for r in wrong)
        top = Counter(r["parsed"] for r in sub).most_common(3)
        md.append(f"| {task} | {len(wrong)} | {first_col} ({first_col / len(wrong):.0%}) | "
                  f"{same_row} ({same_row / len(wrong):.0%}) | " + ", ".join(f"{f} ({c}x)" for f, c in top) + " |")
    md += snake_section(rows)
    md += conditions_section(rows)
    Path("analysis/position.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote analysis/position.md, figures/position_4x4.png, figures/error_offsets.png")


SNAKE_SRC = Path("results/position_n16_snake_gemma4_default.jsonl")
COND_SRC = {  # all on the same 3 sequences per position (seq_index = pos*100 + rep, rep < 3), row-major, N=16
    "number only, no labels": None,  # taken from SRC (first 3 reps)
    "reasoning, no labels": Path("results/position_n16_reason_gemma4_default.jsonl"),
    "number only, frame numbers": Path("results/position_n16_ts_gemma4_default.jsonl"),
    "reasoning, frame numbers": Path("results/position_n16_ts_reason_gemma4_default.jsonl"),
    "number only, column labels (c1-c4)": Path("results/position_n16_collabel_gemma4_default.jsonl"),
    "number only, row labels (r1-r4)": Path("results/position_n16_rowlabel_gemma4_default.jsonl"),
}


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar(a: dict, b: dict) -> str:
    """Exact two-sided McNemar on paired correctness dicts keyed by sequence."""
    from math import comb
    keys = set(a) & set(b)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    m = only_a + only_b
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2 ** m) if m else 1.0
    return f"{only_b} vs {only_a}, p={p:.3g}"


def conditions_section(row_rows) -> list[str]:
    """2x2 (frame numbers x reasoning), column/row-only labels, token + latency cost, and the reasoning-trace check."""
    data = {}
    for name, path in COND_SRC.items():
        if path is None:
            rs = [r for r in row_rows if not r["condition"]["timestamps"] and r["seq_index"] % 100 < 3]
        elif path.exists():
            rs = [apply_truncation_rule(json.loads(line)) for line in path.open(encoding="utf-8")]
        else:
            continue
        data[name] = rs
    md = ["", "## Frame numbers x reasoning, and row/column-only labels (N=16, same 3 sequences per position)", "",
          "Latency is inflated for the reasoning-without-labels run (a second agent shared the one-slot server for "
          "part of it); output tokens are unaffected.", "",
          "| task | condition | calls | accuracy (95% CI) | unparsed | cut off | mean output tokens | median latency s |",
          "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        for name, rs in data.items():
            s = [r for r in rs if r["task"] == task]
            if not s:
                continue
            k = sum(bool(r["correct"]) for r in s)
            lo, hi = wilson(k, len(s))
            lat = sorted(r["latency_s"] for r in s if "latency_s" in r)
            md.append(f"| {task} | {name} | {len(s)} | {k / len(s):.2f} ({lo:.2f}-{hi:.2f}) | "
                      f"{sum(r['parsed'] is None for r in s)} | {sum(r.get('finish_reason') == 'length' for r in s)} | "
                      f"{sum(r['completion_tokens'] for r in s) / len(s):.0f} | {lat[len(lat) // 2]:.2f} |")

    def corr(name, task):
        return {r["seq_index"]: bool(r["correct"]) for r in data.get(name, []) if r["task"] == task}

    md += ["", "Paired comparisons (exact McNemar; 'x vs y' = sequences only the second / only the first got right):", ""]
    pairs = [("number only, no labels", "reasoning, no labels"),
             ("number only, no labels", "number only, frame numbers"),
             ("reasoning, no labels", "number only, frame numbers"),
             ("number only, frame numbers", "reasoning, frame numbers"),
             ("number only, no labels", "number only, column labels (c1-c4)"),
             ("number only, no labels", "number only, row labels (r1-r4)")]
    for task in TASKS:
        for a_, b_ in pairs:
            ca, cb = corr(a_, task), corr(b_, task)
            if ca and cb:
                md.append(f"- {task}: **{b_}** vs {a_}: {mcnemar(ca, cb)}")

    # where do errors land under row/column-only labels? (moving_dot)
    md += ["", "| moving_dot condition | wrong (parsed) | right row, wrong column | right column, wrong row | "
               "both wrong | answer = first cell of the true row |", "|---|---|---|---|---|---|"]
    for name in ("number only, no labels", "number only, column labels (c1-c4)", "number only, row labels (r1-r4)",
                 "number only, frame numbers"):
        w = [r for r in data.get(name, []) if r["task"] == "moving_dot" and not r["correct"]
             and r["parsed"] is not None and 1 <= r["parsed"] <= N]
        if not w:
            md.append(f"| {name} | 0 | | | | |")
            continue
        rr = sum(rc(r["parsed"])[0] == rc(r["truth"])[0] for r in w)
        cc = sum(rc(r["parsed"])[1] == rc(r["truth"])[1] for r in w)
        fr = sum(rc(r["parsed"]) == (rc(r["truth"])[0], 0) for r in w)
        md.append(f"| {name} | {len(w)} | {rr} | {cc} | {len(w) - rr - cc} | {fr} |")

    # reasoning-trace check: is the frame numbering in the trace right, and where does it put the target?
    md += ["", "### Reasoning traces (no labels): numbering vs localisation", "",
           "For each reasoning answer we read the trace's own row headings and the frame numbers listed under them, "
           "and check them against row-major numbering (row k holds frames 4k-3..4k).", "",
           "| task | answers | trace numbering consistent | wrong answers | wrong with consistent numbering | "
           "wrong answer's offset from truth (rows, cols) |", "|---|---|---|---|---|---|"]
    for task in TASKS:
        s = [r for r in data.get("reasoning, no labels", []) if r["task"] == task and r.get("raw_answer")]
        consistent = {}
        for r in s:
            ok, row = True, None
            for line in r["raw_answer"].splitlines():
                m = re.search(r"\bRow\s+(\d)\b", line)
                if m and not re.search(r"Frame\s+\d+", line):
                    row = int(m.group(1))
                for f in re.findall(r"Frame\s+(\d+)", line):
                    if row is not None and not (4 * row - 3 <= int(f) <= 4 * row):
                        ok = False
            consistent[r["seq_index"]] = ok
        w = [r for r in s if not r["correct"] and r["parsed"] is not None and 1 <= r["parsed"] <= N]
        offs = Counter((rc(r["parsed"])[0] - rc(r["truth"])[0], rc(r["parsed"])[1] - rc(r["truth"])[1]) for r in w)
        md.append(f"| {task} | {len(s)} | {sum(consistent.values())} | {len(w)} | "
                  f"{sum(consistent[r['seq_index']] for r in w)} | "
                  + ", ".join(f"({dr:+d},{dc:+d}) x{c}" for (dr, dc), c in offs.most_common()) + " |")
    return md


def landing(r) -> str:
    """Which hypothesis a wrong answer fits. Uses the actual layout of this call (row-major or snake)."""
    a, t = r["parsed"], r["truth"]
    if a is None or not 1 <= a <= N:
        return "invalid"
    cell = {c["frame"]: (c["row"], c["col"]) for c in r["cells"]}
    tr, tc = cell[t]
    if a == tr * COLS + tc + 1 and a != t:
        return "row-major number of the true cell"
    reading_start = a in (1, 5, 9, 13)
    left_edge = cell[a][1] == 0
    if reading_start and left_edge:
        return "reading-order row start AND left edge"
    if reading_start:
        return "reading-order row start (not left edge)"
    if left_edge:
        return "left-edge cell (not a row start)"
    return "other"


def snake_section(row_rows) -> list[str]:
    """Snake vs row-major on identical sequences (reps 0-2), and where wrong snake answers land."""
    if not SNAKE_SRC.exists():
        return []
    snake = [apply_truncation_rule(json.loads(line)) for line in SNAKE_SRC.open(encoding="utf-8")]
    snake_keys = {(r["task"], r["seq_index"]) for r in snake}
    row = [r for r in row_rows if not r["condition"]["timestamps"] and (r["task"], r["seq_index"]) in snake_keys]
    kinds = ["reading-order row start AND left edge", "reading-order row start (not left edge)",
             "left-edge cell (not a row start)", "row-major number of the true cell", "other", "invalid"]
    md = ["", "## Snake order vs row-major (N=16, no timestamps, same 3 sequences per position)", "",
          "In snake order the reading-order row starts 5 and 13 sit on the RIGHT edge, and frames 8 and 16 sit on "
          "the left edge, so the two explanations for the row-start snap make different predictions.", "",
          "| task | order | accuracy | wrong | " + " | ".join(kinds) + " |", "|---|---|---|---|" + "---|" * len(kinds)]
    for task in TASKS:
        for name, rs in (("row-major", row), ("snake", snake)):
            sub = [r for r in rs if r["task"] == task]
            wrong = [r for r in sub if not r["correct"]]
            c = Counter(landing(r) for r in wrong)
            md.append(f"| {task} | {name} | {sum(r['correct'] for r in sub) / len(sub):.2f} ({len(sub)}) | {len(wrong)} | "
                      + " | ".join(f"{c[k]} ({c[k] / max(1, len(wrong)):.0%})" for k in kinds) + " |")
    md += ["", "| task | most common answers, snake (all answers) | most common answers, row-major (same sequences) |",
           "|---|---|---|"]
    for task in TASKS:
        top = lambda rs: ", ".join(f"{f} ({c}x)" for f, c in Counter(  # noqa: E731
            r["parsed"] for r in rs if r["task"] == task).most_common(4))
        md.append(f"| {task} | {top(snake)} | {top(row)} |")

    # same sequence in both layouts: is the wrong answer the same NUMBER (error in frame-number space)
    # or does it follow the cell on screen?
    md += ["", "| task | pairs where both layouts answered wrong (parsed) | same number in both | "
               "snake answer = frame drawn where the row-major answer's cell is |", "|---|---|---|---|"]
    for task in TASKS:
        rm = {r["seq_index"]: r for r in row if r["task"] == task}
        pairs = [(rm[s["seq_index"]], s) for s in snake if s["task"] == task and s["seq_index"] in rm]
        both = [(a, b) for a, b in pairs if not a["correct"] and not b["correct"]
                and a["parsed"] and b["parsed"] and 1 <= a["parsed"] <= N and 1 <= b["parsed"] <= N]
        same_num = sum(a["parsed"] == b["parsed"] for a, b in both)
        same_cell = 0
        for a, b in both:
            cell_a = {c["frame"]: (c["row"], c["col"]) for c in a["cells"]}[a["parsed"]]
            frame_at = {(c["row"], c["col"]): c["frame"] for c in b["cells"]}[cell_a]
            same_cell += b["parsed"] == frame_at and frame_at != a["parsed"]
        md.append(f"| {task} | {len(both)} | {same_num} ({same_num / max(1, len(both)):.0%}) | "
                  f"{same_cell} ({same_cell / max(1, len(both)):.0%}) |")
    trunc = [r for r in snake if r.get("finish_reason") == "length"]
    md += ["", f"Snake answers cut off by the 16-token answer cap (model started explaining instead of answering): "
               f"{len(trunc)} ({', '.join(sorted({r['task'] for r in trunc}))}, target positions "
               f"{sorted({r['target_position'] for r in trunc})}). Counted as wrong. "
               f"moving_dot snake accuracy among parsed answers: "
               f"{sum(r['correct'] for r in snake if r['task'] == 'moving_dot' and r['parsed'] is not None)}/"
               f"{sum(1 for r in snake if r['task'] == 'moving_dot' and r['parsed'] is not None)}."]

    # figure: snake error rate drawn at each cell's real place in the grid
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE)
    for ax, task in zip(axes, TASKS):
        err, cnt, lab = np.zeros((4, 4)), np.zeros((4, 4)), {}
        for r in snake:
            if r["task"] != task:
                continue
            cell = {c["frame"]: (c["row"], c["col"]) for c in r["cells"]}
            i, j = cell[r["target_position"]]
            cnt[i, j] += 1
            err[i, j] += not r["correct"]
            lab[(i, j)] = r["target_position"]
        heat(ax, err / cnt, None, f"{task}, snake order: error rate by cell")
        for (i, j), f in lab.items():
            ax.text(j + 0.42, i - 0.38, str(f), ha="right", va="top", fontsize=7, color=MUTED)
        ax.set_xticks(range(4), [f"col {c + 1}" for c in range(4)])
        ax.set_yticks(range(4), [f"row {r + 1}" for r in range(4)])
    fig.suptitle("Snake order, 3 sequences per cell, no timestamps. Small grey number = frame index in snake order",
                 color=INK, fontsize=9.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig("figures/position_4x4_snake.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return md


if __name__ == "__main__":
    main()
