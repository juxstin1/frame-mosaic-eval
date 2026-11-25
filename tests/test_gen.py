"""Ground-truth invariants and seed reproducibility for gen/. Run: python -m pytest tests -q"""
import hashlib

import numpy as np
import pytest

from gen.common import rng_for
from gen.tasks import TASKS

NS = [4, 6, 9, 12, 16, 20, 25, 36, 49, 64]
SIZE = 128  # small frames keep the test fast; logic is size-independent


def run(task, n, i, seed=0):
    return TASKS[task](n, rng_for(task, n, i, seed), SIZE)


def digest(frames):
    return hashlib.sha256(b"".join(np.asarray(f).tobytes() for f in frames)).hexdigest()


@pytest.mark.parametrize("task", list(TASKS))
def test_reproducible(task):
    a, b, c = run(task, 16, 3), run(task, 16, 3), run(task, 16, 4)
    assert digest(a[0]) == digest(b[0]) and a[1] == b[1]
    assert digest(a[0]) != digest(c[0])


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("i", range(5))
def test_invariants(n, i):
    fr, a, _ = run("moving_dot", n, i)
    assert len(fr) == n and 2 <= a["change_frame"] <= n
    assert a["colours"].index("red") + 1 == a["change_frame"]

    fr, a, _ = run("counter", n, i)
    assert len(a["values"]) == n and all(0 <= v <= 99 for v in a["values"])

    fr, a, p = run("event_order", n, i)
    fs = [e["frame"] for e in a["events"]]
    assert 3 <= len(fs) <= min(5, n) and fs == sorted(set(fs)) and len(set(a["order"])) == len(fs)
    blank = np.asarray(fr[0]) if 1 not in fs else None
    for j, f in enumerate(fr, 1):  # shape pixels only in event frames
        has_ink = (np.asarray(f) < 100).any()
        assert has_ink == (j in fs)

    fr, a, p = run("count_events", n, i)
    ff = a["flash_frames"]
    assert a["k"] == len(ff) and 1 <= a["k"] <= p["k_max"]
    assert all(b - a_ >= 2 for a_, b in zip(ff, ff[1:])) and ff[-1] <= n

    fr, a, _ = run("tile_probe", n, i)
    red = [j for j, f in enumerate(fr, 1) if ((np.asarray(f)[..., 0] > 180) & (np.asarray(f)[..., 1] < 80)).any()]
    assert red == [a["probe_frame"]]


@pytest.mark.parametrize("n", NS)
@pytest.mark.parametrize("i", range(5))
def test_answer_content_clear_of_timestamp_zone(n, i):
    from gen.common import LABEL_ZONE
    z = int(LABEL_ZONE * SIZE)
    for task in ("moving_dot", "event_order", "count_events", "counter"):
        for f in run(task, n, i)[0]:
            corner = np.asarray(f)[:z, :z].astype(int)
            assert (np.abs(corner - corner[0, 0]).max(axis=-1) < 10).all(), (task, n, i)
    fr, a, _ = run("tile_probe", n, i)
    corner = np.asarray(fr[a["probe_frame"] - 1])[:z, :z]
    assert not ((corner[..., 0] > 180) & (corner[..., 1] < 80)).any()
