"""Generate frame sequences + answer keys.

python -m gen.make --task all --n 4 9 16 --count 5 --seed 0 --out data/frames
Layout: <out>/<task>/n<N>/seq<idx>/frame_001.png ... answer.json
"""
import argparse
from pathlib import Path

from .common import FRAME_SIZE, rng_for, save_sequence
from .tasks import TASKS


def make_one(task: str, n: int, index: int, seed: int, size: int, out: Path) -> Path:
    rng = rng_for(task, n, index, seed)
    frames, answer, params = TASKS[task](n, rng, size)
    meta = {"task": task, "n_frames": n, "index": index, "base_seed": seed,
            "frame_size": size, "params": params, "answer": answer}
    return save_sequence(out / task / f"n{n}" / f"seq{index:03d}", frames, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", nargs="+", default=["all"])
    ap.add_argument("--n", nargs="+", type=int, default=[4, 9, 16])
    ap.add_argument("--count", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=FRAME_SIZE)
    ap.add_argument("--out", type=Path, default=Path("data/frames"))
    a = ap.parse_args()
    tasks = list(TASKS) if a.task == ["all"] else a.task
    for t in tasks:
        for n in a.n:
            for i in range(a.start, a.start + a.count):
                make_one(t, n, i, a.seed, a.size, a.out)
        print(f"{t}: {len(a.n) * a.count} sequences")


if __name__ == "__main__":
    main()
