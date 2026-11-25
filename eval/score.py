"""Parse raw answers and score them against the answer key.

Primary metric `correct` (exact); secondary `within1` (frame/count off by one) and `partial`
(counter: fraction of frames right; event_order: fraction of shape pairs in the right relative order).
`parsed` is None when the answer can't be parsed (counted as wrong, and reported separately).
"""
import re

from gen.common import SHAPES


def _ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s or "")]


def final_answer(raw: str) -> str:
    """Reasoning-allowed mode: the text after the LAST 'Answer:' marker; '' if there is none (= unparsed)."""
    hits = re.findall(r"answer\s*:\s*\**\s*([^\n]*)", raw or "", flags=re.IGNORECASE)
    return hits[-1] if hits else ""


def score(task: str, answer: dict, raw: str, reasoning: bool = False) -> dict:
    raw = final_answer(raw) if reasoning else (raw or "")
    raw = raw.strip()
    if task in ("moving_dot", "tile_probe", "count_events"):
        truth = {"moving_dot": answer.get("change_frame"), "tile_probe": answer.get("probe_frame"),
                 "count_events": answer.get("k")}[task]
        nums = _ints(raw)
        if not nums:
            return {"parsed": None, "truth": truth, "correct": False, "within1": False, "partial": 0.0}
        v = nums[0]
        return {"parsed": v, "truth": truth, "correct": v == truth, "within1": abs(v - truth) <= 1,
                "partial": float(v == truth)}
    if task == "counter":
        truth, got = answer["values"], _ints(raw)
        if not got:
            return {"parsed": None, "truth": truth, "correct": False, "within1": False, "partial": 0.0}
        hits = sum(a == b for a, b in zip(got, truth))
        return {"parsed": got, "truth": truth, "correct": got == truth, "within1": False,
                "partial": hits / len(truth)}
    if task == "event_order":
        truth = answer["order"]
        got = [w for w in re.findall(r"[a-z]+", raw.lower()) if w in SHAPES]
        if not got:
            return {"parsed": None, "truth": truth, "correct": False, "within1": False, "partial": 0.0}
        pos = {s: i for i, s in enumerate(got)}
        pairs = [(a, b) for i, a in enumerate(truth) for b in truth[i + 1:]]
        ok = sum(a in pos and b in pos and pos[a] < pos[b] for a, b in pairs)
        return {"parsed": got, "truth": truth, "correct": got == truth, "within1": False,
                "partial": ok / len(pairs)}
    raise ValueError(task)


def apply_truncation_rule(rec: dict) -> dict:
    """An answer cut off by max_tokens (finish_reason 'length') is not a final answer: unparsed and wrong.

    Without this, the scorer would take the first number out of half-finished reasoning text.
    Applied at scoring time for new calls and when loading older logs (the logs themselves are never rewritten).
    """
    if rec.get("finish_reason") == "length":
        rec.update(parsed=None, correct=False, within1=False, partial=0.0, truncated=True)
    return rec
