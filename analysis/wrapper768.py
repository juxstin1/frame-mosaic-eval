"""Offline paired comparison of Mistral's bare-integer and answer-object runs."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Mistral bare integer": "results/deepinfra_matched768_schema_v1/mistral_sweep.jsonl",
    "Mistral answer object": "results/deepinfra_mistral_matched768_object_v1/mistral_sweep.jsonl",
    "Haiku answer object": "results/claude_matched768_haiku_object_v1/haiku55_sweep.jsonl",
}
TASKS = ["moving_dot", "tile_probe", "count_events"]
EXPECTED = set(itertools.product(TASKS, [4, 9, 16, 36, 64], [False, True], range(5)))


def key(r):
    return r["task"], r["n"], r["labels"], r["seq_index"]


def cell(rows):
    hits = sum(r["correct"] for r in rows)
    return f"{hits}/{len(rows)} ({hits / len(rows):.1%})"


def summarize():
    data = {}
    for name, filename in FILES.items():
        path = ROOT / filename
        manifest = json.loads((path.parent / "protocol.json").read_text())
        assert hashlib.sha256((path.parent / "runner_source.py").read_bytes()).hexdigest() == manifest["runner_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 150 and {key(r) for r in rows} == EXPECTED, (name, "coverage")
        for r in rows:
            assert "error" not in r and r["response"]["model"] == r["model"]
            assert r["max_tokens"] == 32 and r["image_wh"] == [[768, 768]]
            if name.startswith("Haiku"):
                raw = "".join(b["text"] for b in r["response"]["content"] if b["type"] == "text")
                finish = {"end_turn": "stop", "max_tokens": "length"}.get(r["response"]["stop_reason"], r["response"]["stop_reason"])
                assert r["request_options"] == {"thinking": {"type": "disabled"}}
            else:
                choice = r["response"]["choices"][0]
                raw = choice["message"].get("content") or ""
                finish = choice["finish_reason"]
                assert r["temperature"] == 0
            assert raw == r["raw_answer"] and finish == r["finish_reason"]
            try:
                value = json.loads(raw)
                if name != "Mistral bare integer":
                    value = value["answer"] if type(value) is dict and list(value) == ["answer"] else None
                valid = type(value) is int and finish == "stop"
            except ValueError:
                valid = False
            assert r["single_integer"] == valid
            assert r["parsed"] == (value if valid else None)
            assert r["correct"] == (valid and value == r["truth"])
        data[name] = {key(r): r for r in rows}
    bare, wrapped, haiku = data.values()
    schema = {"type": "object", "properties": {"answer": {"type": "integer"}},
              "required": ["answer"], "additionalProperties": False}
    for k in EXPECTED:
        for field in ["image_sha256", "image_wh", "condition", "truth", "prompt", "system_prompt", "base_seed", "max_tokens"]:
            assert bare[k][field] == wrapped[k][field] == haiku[k][field], (k, field)
        assert wrapped[k]["response_format"]["json_schema"]["schema"] == haiku[k]["response_format"]["schema"] == schema
        assert bare[k]["temperature"] == wrapped[k]["temperature"]
        assert bare[k]["model"] == wrapped[k]["model"]
        assert bare[k].get("request_options", {}) == wrapped[k].get("request_options", {})
        assert wrapped[k]["protocol"] == "matched768_answer_object_v1"
        assert bare[k]["response_format"]["json_schema"]["schema"] == {"type": "integer"}
        for field in ["name", "strict"]:
            assert bare[k]["response_format"]["json_schema"][field] == wrapped[k]["response_format"]["json_schema"][field]
    gained = sum(not bare[k]["correct"] and wrapped[k]["correct"] for k in EXPECTED)
    lost = sum(bare[k]["correct"] and not wrapped[k]["correct"] for k in EXPECTED)
    both = sum(bare[k]["correct"] and wrapped[k]["correct"] for k in EXPECTED)
    neither = 150 - gained - lost - both
    changed = sum(bare[k]["parsed"] != wrapped[k]["parsed"] for k in EXPECTED)
    lines = ["# Mistral answer-wrapper control", "",
             "150 identical cases per run: three tasks, N=4/9/16/36/64, labels off/on, five sequences per cell.",
             "Source hashes, all case IDs, image hashes, task/system prompts, ground truths and 32-token caps were verified.",
             "Mistral retained temperature=0 and the same model ID and request options. Only the schema",
             "changed from a bare integer to Haiku's exact {answer: integer} object; scoring extracts that field.",
             "Haiku uses a different provider and omits temperature; its request explicitly disables thinking.", "",
             "| Run | Overall | Labels off | Labels on | Cutoffs | Invalid answers | Mean output tokens |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, indexed in data.items():
        rows = list(indexed.values())
        token_field = "output_tokens" if name.startswith("Haiku") else "completion_tokens"
        lines.append(f"| {name} | {cell(rows)} | {cell([r for r in rows if not r['labels']])} | "
                     f"{cell([r for r in rows if r['labels']])} | {sum(r['finish_reason'] == 'length' for r in rows)} | "
                     f"{sum(not r['single_integer'] for r in rows)} | {statistics.mean(r['response']['usage'][token_field] for r in rows):.2f} |")
    lines += ["", "## Paired Mistral changes", "",
              f"- Wrong to correct: {gained}", f"- Correct to wrong: {lost}",
              f"- Correct in both: {both}", f"- Wrong in both: {neither}",
              f"- Parsed answer changed: {changed}",
              f"- Net accuracy change: {gained - lost:+d}/150 ({(gained - lost) / 150 * 100:+.2f} percentage points)", "",
              "## By task and frame-number labels", "", "| Task | Labels | Bare Mistral | Wrapped Mistral | Wrapped Haiku |",
              "| --- | --- | ---: | ---: | ---: |"]
    for task, labels in itertools.product(TASKS, [False, True]):
        values = [cell([r for r in indexed.values() if r["task"] == task and r["labels"] == labels]) for indexed in data.values()]
        lines.append(f"| {task} | {'on' if labels else 'off'} | " + " | ".join(values) + " |")
    cost = sum(r["response"]["usage"]["estimated_cost"] for r in wrapped.values())
    cutoffs = sum(r["finish_reason"] == "length" for r in wrapped.values())
    brace_only = sum(r["finish_reason"] == "length" and r["raw_answer"].strip() == "{" for r in wrapped.values())
    lines += ["", f"The wrapped Mistral run hit the cap {cutoffs}/150 times; {brace_only} of those outputs",
              "contained only an opening brace and whitespace. These are retained as failures.",
              "When output failures dominate, this control cannot isolate a wrapper benefit to task accuracy.", "",
              f"Provider-estimated cost of the new Mistral control: USD {cost:.8f}.", "",
              "This is one exploratory run per configuration. It measures the observed wrapper difference",
              "for this Mistral deployment; it cannot establish that format explains all differences between",
              "models or isolate Haiku's visual perception. Sampling and provider changes are not controlled by",
              "temperature=0 alone. Low output-token counts alone do not prove absence of internal reasoning.", "",
              "## Reproduce", "", "Offline summary (no model calls):", "", "```powershell",
              "python -B -m runner.matched_768 --summarize-wrapper", "```", "",
              "New paid control (use a fresh directory):", "", "```powershell",
              "python -B -u -m runner.matched_768 --models mistral --answer-object --out results/new_mistral_object_run", "```", "",
              "Inputs:", ""]
    lines.extend(f"- `{path}`" for path in FILES.values())
    (ROOT / "analysis/wrapper768.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Validated 450 saved records and 150 matched cases; wrote analysis/wrapper768.md")
    for name, indexed in data.items():
        print(f"{name}: {cell(list(indexed.values()))}")
    print(f"Paired gains={gained}, losses={lost}, both correct={both}, both wrong={neither}; cost USD {cost:.8f}")
