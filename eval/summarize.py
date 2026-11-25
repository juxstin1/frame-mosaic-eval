"""Accuracy table per condition from results/*.jsonl, with Wilson 95% intervals.

python -m eval.summarize results/pilot_gemma4.jsonl --by server.image_max_tokens task n condition.timestamps
"""
import argparse
import json
import math
from collections import defaultdict

from .score import apply_truncation_rule


def get(rec, path):
    for p in path.split("."):
        rec = rec.get(p) if isinstance(rec, dict) else None
    return rec


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", nargs="+", default=["server.image_max_tokens", "task", "n", "condition.timestamps"])
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in a.files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                r = apply_truncation_rule(json.loads(line))
                groups[tuple(get(r, b) for b in a.by)].append(r)
    hdr = [b.split(".")[-1] for b in a.by] + ["calls", "acc", "95% CI", "within1", "partial", "parse_fail",
                                              "errors", "img_tok", "eff_tile", "lat_s"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for key in sorted(groups, key=lambda k: tuple((x is None, x) for x in k)):
        rs = groups[key]
        n = len(rs)
        k = sum(bool(r.get("correct")) for r in rs)
        lo, hi = wilson(k, n)
        ok = [r for r in rs if "error" not in r]
        row = list(map(str, key)) + [
            str(n), f"{k / n:.2f}", f"{lo:.2f}-{hi:.2f}",
            f"{sum(bool(r.get('within1')) for r in rs) / n:.2f}",
            f"{sum(r.get('partial') or 0 for r in rs) / n:.2f}",
            str(sum(r.get("parsed") is None and "error" not in r for r in rs)), str(n - len(ok)),
            str(rs[0]["predicted"]["total_tokens"]), str(rs[0]["predicted"]["eff_tile_px"]),
            f"{sum(r.get('latency_s', 0) for r in ok) / max(1, len(ok)):.2f}"]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
