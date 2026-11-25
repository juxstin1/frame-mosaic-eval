"""Mosaic builder geometry + llama.cpp resize port. Run: python -m pytest tests -q"""
import numpy as np
import pytest
from PIL import Image

from mosaic.builder import GRIDS, Condition, build, cell_of
from mosaic.profiles import vision_plan


def frames(n, size=64):
    return [Image.new("RGB", (size, size), (i * 3 % 256, 100, 200)) for i in range(n)]


@pytest.mark.parametrize("n", list(GRIDS))
@pytest.mark.parametrize("aligned", [True, False])
def test_geometry(n, aligned):
    c = Condition(n=n, aligned=aligned)
    (img,), m = build(frames(n), c)
    t, a = m["tile_px"], c.align
    assert t % a == 0 and img.size[0] % a == 0 and img.size[1] % a == 0
    xs = {cell["box"][0] for cell in m["cells"]}
    assert all((x % a == 0) == aligned for x in xs)          # every tile edge on / off the patch grid
    assert sorted((cl["row"], cl["col"]) for cl in m["cells"]) == sorted(
        (r, cc) for r in range(m["rows"]) for cc in range(m["cols"]))   # grid fully and uniquely filled
    if c.budget == "fixed_total":
        assert t * t * n <= 1536 ** 2


def test_snake():
    assert [cell_of(i, 2, 3, "snake") for i in range(6)] == [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
    assert [cell_of(i, 2, 3, "row") for i in range(6)] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]


def test_pixels_land_in_their_cell():
    fs = frames(9)
    (img,), m = build(fs, Condition(n=9, order="snake"))
    arr = np.asarray(img)
    for cell in m["cells"]:
        x0, y0, x1, y1 = cell["box"]
        assert tuple(arr[(y0 + y1) // 2, (x0 + x1) // 2]) == fs[cell["frame"] - 1].getpixel((32, 32))


def test_separate_matches_tile():
    imgs, m = build(frames(4), Condition(n=4, fmt="separate"))
    assert len(imgs) == 4 and all(im.size == (m["tile_px"],) * 2 for im in imgs)


@pytest.mark.parametrize("args,enc,tok", [  # measured on llama-server b9555 (logs/probe_gemma4_*.json)
    ((512, 512, "gemma4uv"), [528, 528], 121), ((256, 256, "gemma4uv"), [336, 336], 49),
    ((1536, 1536, "gemma4uv"), [768, 768], 256), ((1024, 1024, "gemma4uv", 1024), [1008, 1008], 441),
    ((2048, 2048, "gemma4uv", 1024), [1536, 1536], 1024)])
def test_resize_port_matches_server(args, enc, tok):
    p = vision_plan(*args)
    assert p["encoder_wh"] == enc and p["tokens"] == tok


def test_label_modes_text_key_and_prompt():
    from eval.questions import prompt
    from mosaic.builder import label_text
    assert label_text(7, 1, 2, "frame") == "7" and label_text(7, 1, 2, "col") == "c3" and label_text(7, 1, 2, "row") == "r2"
    assert Condition(n=16).key() == Condition(n=16, label_mode="frame").key()          # old keys unchanged
    assert Condition(n=16, timestamps=True, label_mode="col").key().endswith("_labcol")
    assert "column number" in prompt("moving_dot", Condition(n=16, timestamps=True, label_mode="col"))
    assert "row number" in prompt("moving_dot", Condition(n=16, timestamps=True, label_mode="row"))
    a, _ = build(frames(16), Condition(n=16, timestamps=True, label_mode="col"))
    b, _ = build(frames(16), Condition(n=16, timestamps=True, label_mode="frame"))
    assert np.asarray(a[0]).tobytes() != np.asarray(b[0]).tobytes()
