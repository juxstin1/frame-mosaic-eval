"""Controlled-position run: the target frame is placed at every grid position on purpose.

python -m eval.position_run --name position_n16_gemma4_default --dry-run
Default design (192 calls): tile_probe + moving_dot, N=16 (4x4), no borders, default image limits;
5 sequences per position without timestamps, 1 per position with timestamps (control).
moving_dot at position 1 means the dot is red from the first frame (there is no blue frame).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from gen.common import FRAME_SIZE, rng_for
from gen.tasks import moving_dot, tile_probe
from mosaic.builder import Condition

from .run import MAX_CALLS_WITHOUT_OK, PROFILE_ALIGN, call_and_record, server_info

TARGET_KW = {"tile_probe": ("probe_frame", tile_probe), "moving_dot": ("change_frame", moving_dot)}
BASE_SEED = 1000  # fresh sequences, disjoint from the random-position runs (base seed 0)


def plan_calls(n: int, reps_no_ts: int, reps_ts: int) -> list[tuple]:
    calls = []
    for task in TARGET_KW:
        for ts, reps in ((0, reps_no_ts), (1, reps_ts)):
            for pos in range(1, n + 1):
                for rep in range(reps):
                    calls.append((task, ts, pos, rep))
    return calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--profile", default="gemma4uv", choices=list(PROFILE_ALIGN))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--reps", type=int, default=5, help="sequences per position, no timestamps")
    ap.add_argument("--reps-ts", type=int, default=1, help="sequences per position, timestamps on (control)")
    ap.add_argument("--order", default="row", choices=["row", "snake"],
                    help="reading order; sequences are identical across orders, so runs pair by seq_index")
    ap.add_argument("--reasoning", action="store_true",
                    help="allow working before a final 'Answer: N' line (visible reasoning; hidden thinking stays off)")
    ap.add_argument("--max-tokens", type=int, default=None, help="answer cap (default: the strict-format cap)")
    ap.add_argument("--tasks", nargs="+", default=list(TARGET_KW), choices=list(TARGET_KW))
    ap.add_argument("--label-mode", default="frame", choices=["frame", "col", "row"],
                    help="what the burned-in label shows when timestamps are on")
    ap.add_argument("--ts-same-seqs", action="store_true",
                    help="run only timestamps-on calls, on the same sequences as the no-timestamp runs (--reps each)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--allow-big", action="store_true")
    a = ap.parse_args()
    if a.ts_same_seqs:  # timestamps ON, on exactly the no-timestamp sequences (paired comparison)
        calls = [(task, 1, pos, rep) for task in TARGET_KW for pos in range(1, a.n + 1) for rep in range(a.reps)]
    else:
        calls = plan_calls(a.n, a.reps, a.reps_ts)
    calls = [c for c in calls if c[0] in a.tasks]
    out = Path("results") / f"{a.name}.jsonl"
    if out.exists():  # resume: skip calls already logged (same task, timestamps, sequence)
        done = {(r["task"], int(r["condition"]["timestamps"]), r["seq_index"])
                for r in map(json.loads, out.open(encoding="utf-8"))}
        calls = [c for c in calls if (c[0], c[1], c[2] * 100 + c[3] + (50 if c[1] and not a.ts_same_seqs else 0))
                 not in done]
    print(f"{len(calls)} calls planned -> {out}")
    if len(calls) > MAX_CALLS_WITHOUT_OK and not a.allow_big:
        sys.exit(f"over {MAX_CALLS_WITHOUT_OK} calls: needs the operator's OK (--allow-big)")
    if a.dry_run:
        return
    srv = server_info()
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    t0, n_ok = time.time(), 0
    with out.open("a", encoding="utf-8") as fh:
        for k, (task, ts, pos, rep) in enumerate(calls, 1):
            kw, fn = TARGET_KW[task]
            # unique per (position, rep, timestamps); --ts-same-seqs reuses the no-timestamp sequences
            seq_index = pos * 100 + rep + (50 if ts and not a.ts_same_seqs else 0)
            frames, answer, _ = fn(a.n, rng_for(task, a.n, seq_index, BASE_SEED), FRAME_SIZE, **{kw: pos})
            cond = Condition(n=a.n, timestamps=bool(ts), order=a.order, align=PROFILE_ALIGN[a.profile],
                             label_mode=a.label_mode)
            rec = call_and_record(srv, a.profile, task, frames, answer, cond,
                                  {"run_id": run_id, "seq_index": seq_index, "base_seed": BASE_SEED,
                                   "target_position": pos, "design": "controlled_position",
                                   "colours": answer.get("colours")},
                                  max_tokens=a.max_tokens, reasoning=a.reasoning)
            n_ok += "error" not in rec
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 20 == 0 or k == len(calls):
                print(f"  {k}/{len(calls)}  ok={n_ok}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
