"""Turn a frame sequence into a model input (one mosaic, or separate images) under controlled knobs."""
import math
from dataclasses import asdict, dataclass

from PIL import Image, ImageDraw

from gen.common import font

GRIDS = {4: (2, 2), 6: (2, 3), 9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5),
         25: (5, 5), 36: (6, 6), 49: (7, 7), 64: (8, 8)}  # N -> (rows, cols)
PAD_COLOUR = (255, 255, 255)


@dataclass(frozen=True)
class Condition:
    n: int
    budget: str = "fixed_total"  # fixed_total: total pixel area fixed, tiles shrink | fixed_tile: tile fixed, image grows
    total_side: int = 1536       # fixed_total: area = total_side**2
    tile_side: int = 384         # fixed_tile: tile edge in px
    timestamps: bool = False     # frame number burned into each tile
    borders: bool = False
    order: str = "row"           # row | snake
    aligned: bool = True         # False: half-patch margin so every tile edge falls mid-patch
    align: int = 48              # model px per vision token side (gemma4uv 48, qwen3vl 32)
    fmt: str = "mosaic"          # mosaic | separate
    label_mode: str = "frame"    # what timestamps=True burns in: frame (frame number) | col ("c1".."cC") | row ("r1"..)

    def tile_px(self) -> int:
        raw = math.sqrt(self.total_side ** 2 / self.n) if self.budget == "fixed_total" else self.tile_side
        return max(self.align, int(raw // self.align) * self.align)

    def key(self) -> str:
        b = f"tot{self.total_side}" if self.budget == "fixed_total" else f"tile{self.tile_side}"
        return (f"{self.fmt}_n{self.n}_{b}_ts{int(self.timestamps)}_bd{int(self.borders)}_{self.order}"
                f"_{'al' if self.aligned else 'mis'}{self.align}"
                + ("" if self.label_mode == "frame" else f"_lab{self.label_mode}"))


def cell_of(i: int, rows: int, cols: int, order: str) -> tuple[int, int]:
    r, c = divmod(i, cols)
    if order == "snake" and r % 2 == 1:
        c = cols - 1 - c
    return r, c


def label_text(frame_no: int, row: int, col: int, mode: str) -> str:
    return {"frame": str(frame_no), "col": f"c{col + 1}", "row": f"r{row + 1}"}[mode]


def render_tile(frame: Image.Image, frame_no: int, tile: int, cond: Condition, label: str | None = None) -> Image.Image:
    label = label if label is not None else str(frame_no)
    t = frame.convert("RGB").resize((tile, tile), Image.LANCZOS)
    d = ImageDraw.Draw(t)
    if cond.borders:
        bw = max(2, round(tile / 96))
        d.rectangle([0, 0, tile - 1, tile - 1], outline=(0, 0, 0), width=bw)
    if cond.timestamps:
        f = font(max(10, round(tile * 0.14)))
        x0, y0, x1, y1 = d.textbbox((0, 0), label, font=f)
        m = max(2, round(tile * 0.03))
        off = (max(2, round(tile / 96)) if cond.borders else 0) + 1
        d.rectangle([off, off, off + (x1 - x0) + 2 * m, off + (y1 - y0) + 2 * m], fill=(255, 255, 255))
        d.text((off + m - x0, off + m - y0), label, fill=(0, 0, 0), font=f)
    return t


def build(frames: list[Image.Image], cond: Condition) -> tuple[list[Image.Image], dict]:
    """Returns (images, meta). meta['cells'][k] = where frame k+1 sits (row, col, pixel box)."""
    n = len(frames)
    assert n == cond.n and n in GRIDS, (n, cond.n)
    rows, cols = GRIDS[n]
    tile = cond.tile_px()
    pad = 0 if cond.aligned else cond.align // 2   # pad on each side; 2*pad == align keeps canvas aligned
    assert cond.label_mode == "frame" or cond.fmt == "mosaic", "row/column labels only make sense in a grid"
    tiles = [render_tile(f, i + 1, tile, cond, label_text(i + 1, *cell_of(i, rows, cols, cond.order), cond.label_mode))
             for i, f in enumerate(frames)]
    cells = []
    if cond.fmt == "mosaic":
        canvas = Image.new("RGB", (cols * tile + 2 * pad, rows * tile + 2 * pad), PAD_COLOUR)
        for i, t in enumerate(tiles):
            r, c = cell_of(i, rows, cols, cond.order)
            x, y = pad + c * tile, pad + r * tile
            canvas.paste(t, (x, y))
            cells.append({"frame": i + 1, "row": r, "col": c, "box": [x, y, x + tile, y + tile]})
        images = [canvas]
    elif cond.fmt == "separate":
        images = []
        for i, t in enumerate(tiles):
            im = Image.new("RGB", (tile + 2 * pad, tile + 2 * pad), PAD_COLOUR)
            im.paste(t, (pad, pad))
            images.append(im)
            cells.append({"frame": i + 1, "image": i, "box": [pad, pad, pad + tile, pad + tile]})
    else:
        raise ValueError(cond.fmt)
    meta = {"condition": asdict(cond), "key": cond.key(), "rows": rows, "cols": cols, "tile_px": tile,
            "pad_px": pad, "image_wh": [list(im.size) for im in images], "cells": cells}
    return images, meta
