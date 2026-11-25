"""Run a grid of conditions against the running llama-server and append one JSON line per call.

python -m eval.run --name pilot_gemma4 --tasks moving_dot count_events --n 4 9 16 36 --timestamps 0 1 --seqs 5 --dry-run
Server details (exact command, version, model/mmproj sha256) are read from logs/server/current.json.
"""
import argparse
import itertools
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from gen.common import FRAME_SIZE, rng_for
from gen.tasks import TASKS
from mosaic.builder import Condition, build
from mosaic.make import plan
from runner.client import chat

from .questions import max_answer_tokens, prompt
from .score import apply_truncation_rule, score

PROFILE_ALIGN = {"gemma4uv": 48, "qwen3vl": 32}
SEED = 0  # sampling seed sent with every request (temperature is 0 anyway)
MAX_CALLS_WITHOUT_OK = 300


def server_info() -> dict:
    p = Path("logs/server/current.json")
    if not p.exists():
        sys.exit("no running server recorded (python -m runner.server start ...)")
    s = json.loads(p.read_text())
    cmd = s["cmd"]

    def flag(f):
        return cmd[cmd.index(f) + 1] if f in cmd else None

    model = flag("-m")
    return {"label": s["label"], "url": f"http://127.0.0.1:{s['port']}", "cmdline": s["cmdline"],
            "version": s["version"], "model_file": Path(model).name, "mmproj_file": Path(flag("--mmproj")).name,
            "quant": next((q for q in ("Q4_0", "Q4_K_M", "Q8_0", "IQ4_XS", "F16") if q in Path(model).name), None),
            "model_sha256": s.get("model_sha256"), "mmproj_sha256": s.get("mmproj_sha256"),
            "image_max_tokens": int(flag("--image-max-tokens")) if flag("--image-max-tokens") else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>.jsonl (appended)")
    ap.add_argument("--profile", default="gemma4uv", choices=list(PROFILE_ALIGN))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--n", nargs="+", type=int, default=[4, 9, 16, 36])
    ap.add_argument("--budget", nargs="+", default=["fixed_total"])
    ap.add_argument("--timestamps", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--borders", nargs="+", type=int, default=[0])
    ap.add_argument("--order", nargs="+", default=["row"])
    ap.add_argument("--aligned", nargs="+", type=int, default=[1])
    ap.add_argument("--fmt", nargs="+", default=["mosaic"])
    ap.add_argument("--seqs", type=int, default=5)
    ap.add_argument("--seq-start", type=int, default=0, help="first sequence index (zoom-ins use fresh sequences)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-big", action="store_true", help=f"needed for >{MAX_CALLS_WITHOUT_OK} calls")
    a = ap.parse_args()

    grid = list(itertools.product(a.tasks, a.n, a.budget, a.timestamps, a.borders, a.order, a.aligned, a.fmt,
                                  range(a.seq_start, a.seq_start + a.seqs)))
    print(f"{len(grid)} calls planned -> results/{a.name}.jsonl")
    if len(grid) > MAX_CALLS_WITHOUT_OK and not a.allow_big:
        sys.exit(f"over {MAX_CALLS_WITHOUT_OK} calls: needs the operator's OK (--allow-big)")
    if a.dry_run:
        return
    srv = server_info()
    print(f"server: {srv['label']}  {srv['model_file']}  image_max_tokens={srv['image_max_tokens']}")
    out = Path("results") / f"{a.name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    t_start, n_ok = time.time(), 0
    with out.open("a", encoding="utf-8") as fh:
        for k, (task, n, budget, ts, bd, order, al, fmt, i) in enumerate(grid, 1):
            frames, answer, _params = TASKS[task](n, rng_for(task, n, i, a.base_seed), FRAME_SIZE)
            cond = Condition(n=n, budget=budget, timestamps=bool(ts), borders=bool(bd), order=order,
                             aligned=bool(al), align=PROFILE_ALIGN[a.profile], fmt=fmt)
            rec = call_and_record(srv, a.profile, task, frames, answer, cond,
                                  {"run_id": run_id, "seq_index": i, "base_seed": a.base_seed})
            n_ok += "error" not in rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 10 == 0 or k == len(grid):
                print(f"  {k}/{len(grid)}  ok={n_ok}  {time.time() - t_start:.0f}s")


def call_and_record(srv: dict, profile: str, task: str, frames, answer: dict, cond: Condition, extra: dict,
                    max_tokens: int | None = None, reasoning: bool = False) -> dict:
    """Build the input, ask, score. Returns the full log record (errors are recorded, never raised)."""
    images, meta = build(frames, cond)
    q = prompt(task, cond, reasoning)
    pl = plan(cond, profile, srv["image_max_tokens"])
    rec = {**extra, "time": datetime.now().isoformat(timespec="seconds"), "server": srv,
           "task": task, "n": cond.n, "frame_size": FRAME_SIZE,
           "condition": meta["condition"], "cond_key": meta["key"], "tile_px": meta["tile_px"],
           "image_wh": meta["image_wh"], "cells": meta["cells"],
           "predicted": {f: pl[f] for f in ("encoder_wh", "tokens", "total_tokens", "scale",
                                            "downscaled", "eff_tile_px")},
           "prompt": q, "temperature": 0, "seed": SEED, "thinking": False, "reasoning_allowed": reasoning,
           "max_answer_tokens": max_tokens or max_answer_tokens(task, cond.n)}
    try:
        r = chat(srv["url"], q, images, max_tokens=max_tokens or max_answer_tokens(task, cond.n), seed=SEED)
        msg = r["choices"][0]["message"]
        raw = msg.get("content") or ""
        rec.update({"raw_answer": raw, "reasoning_chars": len(msg.get("reasoning_content") or ""),
                    "finish_reason": r["choices"][0].get("finish_reason"),
                    "prompt_tokens": r["usage"]["prompt_tokens"],
                    "completion_tokens": r["usage"]["completion_tokens"], "latency_s": r["latency_s"],
                    **score(task, answer, raw, reasoning)})
        apply_truncation_rule(rec)
    except Exception as e:  # noqa: BLE001  -- log the failure and keep going
        rec.update({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-800:],
                    "correct": False, "parsed": None})
    return rec


if __name__ == "__main__":
    main()
