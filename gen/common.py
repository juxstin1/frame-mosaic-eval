"""Shared helpers: seeding, fonts, shape drawing, saving a sequence + answer key."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FRAME_SIZE = 512
SHAPES = ["circle", "square", "triangle", "diamond", "star", "cross", "hexagon"]


def rng_for(task: str, n_frames: int, index: int, base_seed: int = 0) -> np.random.Generator:
    """Deterministic RNG per (task, n_frames, index, base_seed); stable across processes."""
    task_id = int.from_bytes(hashlib.sha256(task.encode()).digest()[:4], "little")
    return np.random.default_rng(np.random.SeedSequence([base_seed, task_id, n_frames, index]))


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled font, so renders are identical on any machine.
    return ImageFont.load_default(size=size)


def new_frame(size: int, bg) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (size, size), tuple(bg))
    return img, ImageDraw.Draw(img)


def shape_points(name: str, cx: float, cy: float, r: float):
    if name == "triangle":
        a = np.deg2rad([-90, 30, 150])
    elif name == "diamond":
        a = np.deg2rad([-90, 0, 90, 180])
    elif name == "hexagon":
        a = np.deg2rad(np.arange(6) * 60)
    elif name == "star":
        a = np.deg2rad(np.arange(10) * 36 - 90)
        rr = np.where(np.arange(10) % 2 == 0, r, r * 0.42)
        return [(cx + q * np.cos(t), cy + q * np.sin(t)) for q, t in zip(rr, a)]
    else:
        raise ValueError(name)
    return [(cx + r * np.cos(t), cy + r * np.sin(t)) for t in a]


def draw_shape(draw: ImageDraw.ImageDraw, name: str, cx: float, cy: float, r: float, fill):
    fill = tuple(int(c) for c in fill)
    if name == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif name == "square":
        s = r * 0.85
        draw.rectangle([cx - s, cy - s, cx + s, cy + s], fill=fill)
    elif name == "cross":
        w = r * 0.32
        draw.rectangle([cx - r, cy - w, cx + r, cy + w], fill=fill)
        draw.rectangle([cx - w, cy - r, cx + w, cy + r], fill=fill)
    else:
        draw.polygon(shape_points(name, cx, cy, r), fill=fill)


def save_sequence(out_dir: Path, frames: list[Image.Image], meta: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for i, img in enumerate(frames, start=1):
        name = f"frame_{i:03d}.png"
        img.save(out_dir / name, optimize=True)
        files.append(name)
    meta = {**meta, "frame_indexing": "1-based", "files": files}
    (out_dir / "answer.json").write_text(json.dumps(meta, indent=2))
    return out_dir


# Mosaic timestamps are burned into each tile's top-left corner; answer-bearing content must stay out of it.
LABEL_ZONE = 0.24  # fraction of frame side


def clear_of_label(cx: float, cy: float, r: float, size: int) -> bool:
    z = LABEL_ZONE * size
    return cx - r > z or cy - r > z
