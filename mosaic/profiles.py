"""How llama.cpp b9555 resizes an image before the vision encoder (see docs/preprocessing.md).

`smart_resize` is a port of img_tool::calc_size_preserved_ratio (tools/mtmd/mtmd-image.cpp L145-169),
used by mtmd_image_preprocessor_dyn_size for both gemma4uv and qwen3vl projectors.
"""
import math

import numpy as np

# align = patch_size * n_merge (px per vision token side); token limits = llama.cpp defaults
PROFILES = {
    "gemma4uv": {"align": 48, "min_tokens": 40, "max_tokens": 280},   # clip.cpp L1400-1415
    "qwen3vl": {"align": 32, "min_tokens": 8, "max_tokens": 4096},    # clip.cpp L1426-1442
}


def smart_resize(w: int, h: int, align: int, min_px: int, max_px: int) -> tuple[int, int]:
    f32 = np.float32
    rnd = lambda x: int(round(float(x) / align)) * align  # noqa: E731  (C++ std::round: half away from zero)
    rnd = lambda x: int(math.floor(float(x) / align + 0.5)) * align  # noqa: E731  C++ std::round (x>0)
    h_bar, w_bar = max(align, rnd(h)), max(align, rnd(w))
    if h_bar * w_bar > max_px:
        beta = np.sqrt(f32(h * w) / f32(max_px))
        h_bar = max(align, int(math.floor(float(f32(h) / beta) / align)) * align)
        w_bar = max(align, int(math.floor(float(f32(w) / beta) / align)) * align)
    elif h_bar * w_bar < min_px:
        beta = np.sqrt(f32(min_px) / f32(h * w))
        h_bar = int(math.ceil(float(f32(h) * beta) / align)) * align
        w_bar = int(math.ceil(float(f32(w) * beta) / align)) * align
    return w_bar, h_bar


def vision_plan(w: int, h: int, profile: str, max_tokens: int | None = None, min_tokens: int | None = None) -> dict:
    """What the encoder will actually see for a w x h image, and how many vision tokens it costs."""
    p = PROFILES[profile]
    a = p["align"]
    mx, mn = (max_tokens or p["max_tokens"]), (min_tokens or p["min_tokens"])
    tw, th = smart_resize(w, h, a, mn * a * a, mx * a * a)
    return {"profile": profile, "max_tokens": mx, "input_wh": [w, h], "encoder_wh": [tw, th],
            "tokens": (tw // a) * (th // a), "scale": round(min(tw / w, th / h), 4),
            "downscaled": tw * th < 0.98 * w * h}
