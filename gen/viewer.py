"""Save a contact sheet of one sequence with its ground truth overlaid, for eyeballing.

python -m gen.viewer data/frames/counter/n16/seq000 --out docs/sheets/counter_n16.png
Highlighted (green border) = frames the answer refers to. Captions under each thumbnail.
"""
import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw

from .common import font


def annotate(meta: dict) -> tuple[dict[int, str], set[int], str]:
    """Per-frame captions, highlighted frames, and a one-line header for the sheet."""
    a, n, task = meta["answer"], meta["n_frames"], meta["task"]
    cap = {i: f"#{i}" for i in range(1, n + 1)}
    hi: set[int] = set()
    if task == "moving_dot":
        hi = {a["change_frame"]}
        for i, c in enumerate(a["colours"], 1):
            cap[i] += f" {c}"
        head = f"change_frame={a['change_frame']}"
    elif task == "counter":
        for i, v in enumerate(a["values"], 1):
            cap[i] += f" ={v}"
        head = f"values={a['values']}"
    elif task == "event_order":
        for e in a["events"]:
            hi.add(e["frame"])
            cap[e["frame"]] += f" {e['shape']}"
        head = f"order={a['order']}"
    elif task == "count_events":
        hi = set(a["flash_frames"])
        head = f"K={a['k']} flash_frames={a['flash_frames']}"
    elif task == "tile_probe":
        hi = {a["probe_frame"]}
        head = f"probe_frame={a['probe_frame']}"
    else:
        head = ""
    return cap, hi, head


def contact_sheet(seq_dir: Path, thumb: int = 160, cols: int | None = None) -> Image.Image:
    meta = json.loads((seq_dir / "answer.json").read_text())
    n = meta["n_frames"]
    cols = cols or math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    pad, cap_h, head_h = 8, 22, 34
    W = cols * (thumb + pad) + pad
    H = head_h + rows * (thumb + cap_h + pad) + pad
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    cap, hi, head = annotate(meta)
    d.text((pad, 8), f"{meta['task']}  n={n}  seq={meta['index']}  {head}"[:160], fill=(0, 0, 0), font=font(15))
    f = font(13)
    for i, name in enumerate(meta["files"], 1):
        r, c = divmod(i - 1, cols)
        x, y = pad + c * (thumb + pad), head_h + r * (thumb + cap_h + pad)
        sheet.paste(Image.open(seq_dir / name).convert("RGB").resize((thumb, thumb), Image.LANCZOS), (x, y))
        if i in hi:
            d.rectangle([x - 3, y - 3, x + thumb + 2, y + thumb + 2], outline=(0, 170, 0), width=4)
        d.text((x, y + thumb + 3), cap[i], fill=(0, 0, 0), font=f)
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_dir", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, help="output png (single seq) or directory (several)")
    ap.add_argument("--thumb", type=int, default=160)
    a = ap.parse_args()
    for s in a.seq_dir:
        out = a.out if (a.out and len(a.seq_dir) == 1) else (a.out or s) / f"sheet_{s.parent.parent.name}_{s.parent.name}_{s.name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        contact_sheet(s, a.thumb).save(out)
        print(out)


if __name__ == "__main__":
    main()
