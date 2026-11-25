"""Build model inputs from a saved frame sequence, and the budget table.

python -m mosaic.make data/frames/counter/n16/seq000 --budget fixed_total --timestamps --out docs/mosaics
python -m mosaic.make --table docs/mosaic_budget_table.md
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

from PIL import Image

from .builder import GRIDS, Condition, build
from .profiles import vision_plan

SETTINGS = [("gemma4uv", None), ("gemma4uv", 1024), ("qwen3vl", None)]  # (profile, --image-max-tokens)


def load_frames(seq_dir: Path) -> tuple[list[Image.Image], dict]:
    meta = json.loads((seq_dir / "answer.json").read_text())
    return [Image.open(seq_dir / f).convert("RGB") for f in meta["files"]], meta


def plan(cond: Condition, profile: str, max_tokens: int | None) -> dict:
    """Encoder view of a condition without rendering: per-image plan for the mosaic or each separate image."""
    rows, cols = GRIDS[cond.n]
    tile, pad = cond.tile_px(), (0 if cond.aligned else cond.align // 2)
    if cond.fmt == "mosaic":
        p = vision_plan(cols * tile + 2 * pad, rows * tile + 2 * pad, profile, max_tokens)
        return {**p, "n_images": 1, "total_tokens": p["tokens"] + 2, "eff_tile_px": round(tile * p["scale"], 1)}
    p = vision_plan(tile + 2 * pad, tile + 2 * pad, profile, max_tokens)
    return {**p, "n_images": cond.n, "total_tokens": (p["tokens"] + 2) * cond.n, "eff_tile_px": round(tile * p["scale"], 1)}


def table() -> str:
    lines = ["# What the encoder sees, per condition (predicted by mosaic.profiles; verified against b9555 in logs/probe_*.json)",
             "", "eff tile = tile px after llama.cpp resize. tokens include 2 marker tokens per image. "
             "**bold** = downscaled.", ""]
    for fmt in ("mosaic", "separate"):
        for budget in ("fixed_total", "fixed_tile"):
            lines += [f"## {fmt}, {budget} ({'1536^2 area' if budget == 'fixed_total' else '384 px tiles'})", "",
                      "| N | grid | tile px | image px | " + " | ".join(
                          f"{p}{'' if m is None else f' max{m}'}: eff tile / tokens" for p, m in SETTINGS) + " |",
                      "|---|---|---|---|" + "---|" * len(SETTINGS)]
            for n, (r, c) in GRIDS.items():
                cells = []
                for prof, mx in SETTINGS:
                    cond = Condition(n=n, budget=budget, fmt=fmt, align=48 if prof == "gemma4uv" else 32)
                    pl = plan(cond, prof, mx)
                    s = f"{pl['eff_tile_px']:g} / {pl['total_tokens']}"
                    cells.append(f"**{s}**" if pl["downscaled"] else s)
                cg = Condition(n=n, budget=budget, fmt=fmt)
                w, h = (c * cg.tile_px(), r * cg.tile_px()) if fmt == "mosaic" else (cg.tile_px(), cg.tile_px())
                lines.append(f"| {n} | {r}x{c} | {cg.tile_px()} | {w}x{h} | " + " | ".join(cells) + " |")
            lines.append("")
    lines.append("Tile px column is for align=48 (gemma); qwen (align=32) tiles can differ by rounding.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=Path("docs/mosaics"))
    ap.add_argument("--table", type=Path)
    for f in fields(Condition):
        if f.name == "n":
            continue
        if f.type in (bool, "bool"):
            ap.add_argument(f"--{f.name}", action=argparse.BooleanOptionalAction, default=f.default)
        else:
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    a = ap.parse_args()
    if a.table:
        a.table.write_text(table())
        print(a.table)
        return
    frames, meta = load_frames(a.seq_dir)
    cond = Condition(n=len(frames), **{f.name: getattr(a, f.name) for f in fields(Condition) if f.name != "n"})
    images, m = build(frames, cond)
    stem = f"{meta['task']}_seq{meta['index']:03d}_{cond.key()}"
    a.out.mkdir(parents=True, exist_ok=True)
    for k, im in enumerate(images):
        im.save(a.out / (f"{stem}.png" if len(images) == 1 else f"{stem}_{k + 1:02d}.png"))
    (a.out / f"{stem}.json").write_text(json.dumps({**m, "source": str(a.seq_dir)}, indent=1))
    print(a.out / f"{stem}.png", m["image_wh"][0], f"tile={m['tile_px']}")


if __name__ == "__main__":
    main()
