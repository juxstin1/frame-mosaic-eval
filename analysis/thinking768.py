"""Validate and summarize the separate thinking-on round; no network access."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = set(itertools.product(["moving_dot", "tile_probe", "count_events"],
                                 [4, 9, 16, 36, 64], [False, True], range(5)))
BASELINE = ROOT / "results/deepinfra_expansion_matched768_schema_v1"


def watch(folder):
    """Wait for saved records, then validate once; never make model requests."""
    folder = folder.resolve()
    assert folder.is_relative_to(ROOT / "results")
    manifest = json.loads((folder / "protocol.json").read_text())
    deadline = time.monotonic() + 6 * 3600
    while time.monotonic() < deadline:
        counts = {}
        for alias in manifest["models"]:
            path = folder / f"{alias}_sweep.jsonl"
            # Only consume flushed complete lines while another process writes.
            raw = path.read_text() if path.exists() else ""
            lines = raw.splitlines()
            if raw and not raw.endswith("\n"):
                lines = lines[:-1]
            rows = [json.loads(line) for line in lines]
            assert not any("error" in r for r in rows), (alias, "run recorded an API error; no retry performed")
            assert len(rows) <= 150, (alias, "unexpected extra rows")
            counts[alias] = len(rows)
        print(json.dumps({"waiting_for_saved_results": counts}), flush=True)
        if all(n == 150 for n in counts.values()):
            summarize(folder)
            return
        time.sleep(30)
    raise TimeoutError("Thinking round did not complete within six hours; saved records retained")


def case(row):
    return row["task"], row["n"], row["labels"], row["seq_index"]


def accuracy(rows):
    hits = sum(r["correct"] for r in rows)
    return f"{hits}/{len(rows)} ({hits / len(rows):.1%})"


def summarize(folder):
    folder = folder.resolve()
    assert folder.is_relative_to(ROOT / "results"), "Expected a repo results directory"
    manifest = json.loads((folder / "protocol.json").read_text())
    assert manifest["protocol"] == "matched768_thinking4096_v1"
    assert manifest["output_cap"] == 4096
    assert hashlib.sha256((folder / "runner_source.py").read_bytes()).hexdigest() == manifest["runner_sha256"]
    data = {}
    for alias, model in manifest["models"].items():
        rows = [json.loads(line) for line in (folder / f"{alias}_sweep.jsonl").read_text().splitlines()]
        assert len(rows) == 150 and {case(r) for r in rows} == EXPECTED, (alias, "coverage", len(rows))
        baseline = {case(r): r for r in map(json.loads, (BASELINE / f"{alias}_sweep.jsonl").read_text().splitlines())}
        assert len(baseline) == 150 and set(baseline) == EXPECTED
        for r in rows:
            assert "error" not in r, (alias, case(r), r.get("error"))
            assert r["protocol"] == manifest["protocol"]
            assert r["response"]["model"] == r["model"] == model
            assert r["request_options"] == manifest["request_options"][alias] == {"reasoning_effort": "high"}
            assert r["max_tokens"] == 4096 and r["system_prompt"] == manifest["system_prompt"]
            choice = r["response"]["choices"][0]
            assert r["raw_answer"] == (choice["message"].get("content") or "")
            assert r["finish_reason"] == choice["finish_reason"]
            assert r["hit_output_cap"] == (r["finish_reason"] == "length")
            try:
                parsed = json.loads(r["raw_answer"])
                valid = type(parsed) is int and r["finish_reason"] == "stop"
            except ValueError:
                valid = False
            assert r["single_integer"] == valid
            assert r["parsed"] == (parsed if valid else None)
            assert r["correct"] == (valid and parsed == r["truth"])
            for field in ["image_sha256", "image_wh", "condition", "truth", "prompt",
                          "response_format", "temperature", "base_seed"]:
                assert r[field] == baseline[case(r)][field], (alias, case(r), field)
        data[alias] = (rows, list(baseline.values()))

    lines = ["# Thinking-on round: matched 768 x 768 inputs", "",
             "Separate exploratory round: 150 cases per model, five sequences per cell,",
             "three tasks, N=4/9/16/36/64, labels off/on. No smoke calls or automatic retries.", "",
             "Every case was checked against the prior round for identical PNG hash, task prompt,",
             "ground truth, condition, seed, temperature and integer response schema.",
             "Changes: reasoning_effort=high, max_tokens=4096, and a system instruction permitting",
             "internal reasoning while requesting only an integer final answer. Thus any difference",
             "is an effect of the configuration bundle, not an isolated estimate of thinking alone.", "",
             "A cutoff means finish_reason=length. Cutoffs and invalid final answers count as wrong;",
             "the cap covers output including reasoning. A 4096-token cap does not ensure completion.", "",
             "The prior round requested reasoning_effort=none. GLM Flash nevertheless returned",
             "reasoning on 88/150 calls: its prior score is not a verified thinking-off baseline.",
             "Other models returning no reasoning fields does not prove the absence of internal reasoning.", "",
             "Mistral, Scout and Qwen3-VL Instruct are excluded. This is not an all-model ranking.",
             "Local Gemma 12B is not rerun in this hosted round. Prior result files are unchanged.", "",
             "| Model | Prior 32-token score | Thinking 4096 score | Labels off | Labels on | Cutoffs | Invalid final answers |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for alias, (rows, baseline) in data.items():
        lines.append(f"| {manifest['models'][alias]} | {accuracy(baseline)} | {accuracy(rows)} | "
                     f"{accuracy([r for r in rows if not r['labels']])} | {accuracy([r for r in rows if r['labels']])} | "
                     f"{sum(r['hit_output_cap'] for r in rows)} | {sum(not r['single_integer'] for r in rows)} |")
    lines += ["", "## Reported reasoning, tokens, cost and timing", "",
              "Reasoning counts describe returned fields, not independently observed internal computation.", "",
              "| Model | Reasoning text present | Positive reasoning tokens reported | Mean completion tokens | Median seconds | Estimated USD |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    total = 0
    for alias, (rows, _) in data.items():
        usages = [r["response"]["usage"] for r in rows]
        cost = sum(u["estimated_cost"] for u in usages)
        total += cost
        text_count = sum(bool(r["response"]["choices"][0]["message"].get("reasoning_content")) for r in rows)
        tokens_count = sum((u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) > 0 for u in usages)
        lines.append(f"| {manifest['models'][alias]} | {text_count}/150 | {tokens_count}/150 | "
                     f"{statistics.mean(u['completion_tokens'] for u in usages):.2f} | "
                     f"{statistics.median(r['latency_s'] for r in rows):.3f} | {cost:.8f} |")
    lines += ["", f"Total provider-estimated cost: USD {total:.8f}. Runs overlapped; latency is descriptive.", "",
              "## Task results", "", "| Model | Task | Labels | Accuracy | Cutoffs |", "| --- | --- | --- | ---: | ---: |"]
    for alias, (rows, _) in data.items():
        for task, labels in itertools.product(["moving_dot", "tile_probe", "count_events"], [False, True]):
            subset = [r for r in rows if r["task"] == task and r["labels"] == labels]
            lines.append(f"| {manifest['models'][alias]} | {task} | {'on' if labels else 'off'} | "
                         f"{accuracy(subset)} | {sum(r['hit_output_cap'] for r in subset)} |")
    relative = folder.relative_to(ROOT).as_posix()
    lines += ["", "## Reproduce", "", "Offline validation and table generation:", "", "```powershell",
              f"python -B -m runner.matched_768 --summarize-thinking {relative}", "```", "",
              "Run a new paid round (choose an unused results directory):", "", "```powershell",
              "python -B -u -m runner.matched_768 --thinking-on --models " + " ".join(data) + " --out results/new_thinking_round",
              "```", "", "The run directory preserves JSONL responses, exact request settings, provider catalog and runner source.",
              "[DeepInfra reasoning controls](https://docs.deepinfra.com/chat/reasoning).", ""]
    out = ROOT / "analysis/thinking768.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validated {150 * len(data)} cases against prior input hashes and scoring; wrote analysis/thinking768.md")
    for alias, (rows, _) in data.items():
        print(f"{alias}: {accuracy(rows)}; cap hits {sum(r['hit_output_cap'] for r in rows)}; invalid {sum(not r['single_integer'] for r in rows)}")
    print(f"Total estimated USD: {total:.8f}")
