"""The five synthetic tasks. Each returns (frames, answer, params) with exact ground truth.

All frame numbers in answers are 1-based. Every random draw comes from the rng passed in.
"""
import numpy as np

from .common import SHAPES, clear_of_label, draw_shape, font, new_frame

DOT_BEFORE = (40, 90, 220)   # blue
DOT_AFTER = (220, 40, 40)    # red


def moving_dot(n: int, rng: np.random.Generator, size: int, change_frame: int | None = None):
    """A dot moves along a quadratic Bezier; from frame `change_frame` on it is red instead of blue."""
    r = 0.055 * size
    lo, hi = r * 1.5, size - r * 1.5
    def path(p0, p1, ctrl):
        ts = np.linspace(0, 1, n)[:, None]
        return (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * ctrl + ts ** 2 * p1

    while True:  # resample until no frame puts the dot under the timestamp label
        pts = path(*(rng.uniform(lo, hi, 2) for _ in range(3)))
        if all(clear_of_label(x, y, r, size) for x, y in pts):
            break
    change = int(rng.integers(2, n + 1))  # frame 1 always shows the original colour
    if change_frame is not None:  # controlled placement; drawn value still consumed so the stream is unchanged
        change = change_frame     # change_frame=1 means red from the first frame
    frames, positions, colours = [], [], []
    for i, (x, y) in enumerate(pts):
        colour = "red" if i + 1 >= change else "blue"
        img, d = new_frame(size, (236, 236, 236))
        draw_shape(d, "circle", x, y, r, DOT_AFTER if colour == "red" else DOT_BEFORE)
        frames.append(img)
        positions.append([round(float(x) / size, 4), round(float(y) / size, 4)])
        colours.append(colour)
    answer = {"change_frame": change, "positions_norm": positions, "colours": colours}
    return frames, answer, {"dot_radius_px": round(r, 1), "colour_change": "blue->red, persists"}


def counter(n: int, rng: np.random.Generator, size: int, p_repeat: float = 0.3):
    """A big two-digit number per frame; with probability p_repeat a frame repeats the previous value."""
    values = [int(rng.integers(0, 100))]
    for _ in range(n - 1):
        if rng.random() < p_repeat:
            values.append(values[-1])
        else:
            v = int(rng.integers(0, 99))
            values.append(v if v < values[-1] else v + 1)  # uniform over 0..99 minus previous
    f = font(int(size * 0.55))
    frames = []
    for v in values:
        img, d = new_frame(size, (250, 250, 250))
        d.text((size / 2, size / 2), f"{v:02d}", fill=(20, 20, 20), font=f, anchor="mm")
        frames.append(img)
    return frames, {"values": values}, {"p_repeat": p_repeat, "format": "two digits, zero-padded"}


def event_order(n: int, rng: np.random.Generator, size: int):
    """3-5 distinct shapes, each visible in exactly one frame (distinct frames); other frames empty."""
    k = int(rng.integers(3, min(5, n) + 1))
    shapes = [SHAPES[j] for j in rng.choice(len(SHAPES), k, replace=False)]
    event_frames = sorted(int(f) + 1 for f in rng.choice(n, k, replace=False))
    r = 0.16 * size
    by_frame = dict(zip(event_frames, shapes))
    frames = []
    for i in range(1, n + 1):
        img, d = new_frame(size, (236, 236, 236))
        if i in by_frame:
            cx, cy = rng.uniform(r * 1.2, size - r * 1.2, 2)
            while not clear_of_label(cx, cy, r, size):
                cx, cy = rng.uniform(r * 1.2, size - r * 1.2, 2)
            draw_shape(d, by_frame[i], cx, cy, r, (30, 30, 30))
        frames.append(img)
    events = [{"shape": s, "frame": f} for f, s in zip(event_frames, shapes)]
    return frames, {"events": events, "order": shapes}, {"n_shapes": k, "shape_pool": SHAPES}


def count_events(n: int, rng: np.random.Generator, size: int, k_cap: int = 10):
    """K flash frames (never adjacent, so each flash is one unambiguous event) on a dark background."""
    k_max = min((n + 1) // 2, k_cap)
    k = int(rng.integers(1, k_max + 1))
    # choose k of n-k+1 slots then spread by index => no two flashes adjacent
    slots = sorted(int(s) for s in rng.choice(n - k + 1, k, replace=False))
    flash = [s + j + 1 for j, s in enumerate(slots)]
    frames = []
    for i in range(1, n + 1):
        img, d = new_frame(size, (30, 30, 40))
        if i in flash:
            draw_shape(d, "star", size / 2, size / 2, size * 0.42, (255, 225, 0))
        frames.append(img)
    return frames, {"k": k, "flash_frames": flash}, {"k_max": k_max, "k_cap": k_cap, "flashes_adjacent": False}


def tile_probe(n: int, rng: np.random.Generator, size: int, n_clutter: int = 8, jitter_px: int = 3,
               probe_frame: int | None = None):
    """A static grey scene with tiny per-frame jitter; one frame also contains a red star."""
    clutter = [
        (SHAPES[int(rng.integers(len(SHAPES)))], *rng.uniform(0.1 * size, 0.9 * size, 2),
         rng.uniform(0.04, 0.09) * size, int(rng.integers(110, 190)))
        for _ in range(n_clutter)
    ]
    probe = int(rng.integers(1, n + 1))
    if probe_frame is not None:  # controlled placement; drawn value still consumed so the stream is unchanged
        probe = probe_frame
    pr = 0.08 * size
    px, py = rng.uniform(pr * 1.3, size - pr * 1.3, 2)
    while not clear_of_label(px, py, pr, size):
        px, py = rng.uniform(pr * 1.3, size - pr * 1.3, 2)
    frames = []
    for i in range(1, n + 1):
        img, d = new_frame(size, (236, 236, 236))
        for name, cx, cy, r, g in clutter:
            dx, dy = rng.integers(-jitter_px, jitter_px + 1, 2)
            g2 = int(np.clip(g + rng.integers(-6, 7), 0, 255))
            draw_shape(d, name, cx + dx, cy + dy, r, (g2, g2, g2))
        if i == probe:
            draw_shape(d, "star", px, py, pr, (220, 30, 30))
        frames.append(img)
    return frames, {"probe_frame": probe}, {"n_clutter": n_clutter, "jitter_px": jitter_px, "probe": "red star"}


TASKS = {f.__name__: f for f in (moving_dot, counter, event_order, count_events, tile_probe)}
