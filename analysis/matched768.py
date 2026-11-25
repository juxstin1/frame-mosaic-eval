"""Regenerate the matched-input comparison from saved calls; no model access."""
import hashlib
import itertools
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "Gemma 4 12B QAT Q4_0": "results/gemma_matched768_schema_v1/gemma_sweep.jsonl",
    "Mistral Small 3.2 24B": "results/deepinfra_matched768_schema_v1/mistral_sweep.jsonl",
    "Llama 4 Scout": "results/deepinfra_matched768_schema_v1/scout_sweep.jsonl",
    "Qwen3-VL-30B-A3B": "results/deepinfra_qwen30_matched768_schema_v1/qwen30_sweep.jsonl",
    "GLM-5.3-Flash": "results/deepinfra_expansion_matched768_schema_v1/glmflash_sweep.jsonl",
    "Gemma 4 26B A4B": "results/deepinfra_expansion_matched768_schema_v1/gemma26_sweep.jsonl",
    "Nemotron 3.5 Super VL": "results/deepinfra_expansion_matched768_schema_v1/nemotron_sweep.jsonl",
    "Qwen3.8-27B": "results/deepinfra_expansion_matched768_schema_v1/qwen38_sweep.jsonl",
}
TASKS = ["moving_dot", "tile_probe", "count_events"]
NS = [4, 9, 16, 36, 64]


def key(r):
    return r["task"], r["n"], r["labels"], r["seq_index"]


def cell(rows):
    hits = sum(r["correct"] for r in rows)
    return f"{hits}/{len(rows)} ({hits / len(rows):.1%})"


def main():
    data = {}
    expected = set(itertools.product(TASKS, NS, [False, True], range(5)))
    for model, file in FILES.items():
        path = ROOT / file
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 150, (model, "incomplete", len(rows))
        indexed = {key(r): r for r in rows}
        assert len(indexed) == 150 and set(indexed) == expected, (model, "case coverage")
        protocol = json.loads((path.parent / "protocol.json").read_text())
        assert hashlib.sha256((path.parent / "runner_source.py").read_bytes()).hexdigest() == protocol["runner_sha256"]
        for r in rows:
            assert "error" not in r and r["phase"] == "sweep"
            assert r["protocol"] == "matched768_integer_schema_v1"
            if "request_options" in protocol:
                alias = next(a for a, model_id in protocol["models"].items() if model_id == r["model"])
                assert r["request_options"] == protocol["request_options"][alias]
            assert r["response"]["model"] == r["model"]
            try:
                parsed = json.loads(r["raw_answer"])
                valid = type(parsed) is int and r["finish_reason"] == "stop"
            except ValueError:
                valid = False
            assert r["single_integer"] == valid, (model, key(r), "format classification")
            assert r["parsed"] == (parsed if valid else None), (model, key(r), "numeric parsing")
            assert r["correct"] == (valid and parsed == r["truth"]), (model, key(r), "score")
            assert r["image_wh"] == [[768, 768]] and r["max_tokens"] == 32
        data[model] = indexed
    reference = next(iter(data.values()))
    equal_fields = ["image_sha256", "prompt", "system_prompt", "response_format", "truth",
                    "condition", "image_wh", "temperature", "max_tokens", "base_seed"]
    for model, indexed in data.items():
        for k in expected:
            for field in equal_fields:
                assert indexed[k][field] == reference[k][field], (model, k, field)

    lines = ["# Matched 768 x 768 input comparison", "",
             f"{150 * len(data)} scored calls: 150 per model, three tasks, five grid sizes (4, 9, 16, 36, 64),",
             "labels off/on, and five sequences per cell (indices 0-4, base seed 0).",
             "The same 150 PNG hashes, task/system prompts, ground truths, integer JSON schema,",
             "temperature 0 and 32-token output cap were verified across all models.", "",
             "Every response returned the requested model identity. Scores and format failures",
             "were independently checked; unfinished or non-integer outputs count as wrong.", "",
             "| Model | Overall | Labels off | Labels on |",
             "| --- | ---: | ---: | ---: |"]
    for model, indexed in data.items():
        rows = list(indexed.values())
        lines.append(f"| {model} | {cell(rows)} | {cell([r for r in rows if not r['labels']])} | {cell([r for r in rows if r['labels']])} |")
    lines += ["", "## Output failures", "", "Counts below overlap: a cutoff is also an invalid final answer.", "",
              "| Model | Cutoffs | Invalid final answers | Responses with reasoning text |",
              "| --- | ---: | ---: | ---: |"]
    for model, indexed in data.items():
        rows = list(indexed.values())
        cutoffs = sum(r["finish_reason"] == "length" for r in rows)
        invalid = sum(not r["single_integer"] for r in rows)
        reasoning = sum(bool(r["response"]["choices"][0]["message"].get("reasoning_content")) for r in rows)
        lines.append(f"| {model} | {cutoffs} | {invalid} | {reasoning} |")
    lines += ["", "## Task results", "", "| Task | Labels | " + " | ".join(data) + " |",
              "| --- | --- | " + " | ".join(["---:"] * len(data)) + " |"]
    for task, labels in itertools.product(TASKS, [False, True]):
        values = [cell([r for r in indexed.values() if r["task"] == task and r["labels"] == labels]) for indexed in data.values()]
        lines.append(f"| {task} | {'on' if labels else 'off'} | " + " | ".join(values) + " |")
    lines += ["", "## All cells", "", "Each cell contains only five sequences; percentages move in 20-point steps.", "",
              "| Task | N | Labels | " + " | ".join(data) + " |",
              "| --- | ---: | --- | " + " | ".join(["---:"] * len(data)) + " |"]
    for task, n, labels in itertools.product(TASKS, NS, [False, True]):
        values = [cell([r for r in indexed.values() if (r["task"], r["n"], r["labels"]) == (task, n, labels)]) for indexed in data.values()]
        lines.append(f"| {task} | {n} | {'on' if labels else 'off'} | " + " | ".join(values) + " |")
    lines += ["", "## Usage and timing", "", "| Model | Mean prompt tokens | Median request seconds | Provider-estimated USD for 150 scored calls |",
              "| --- | ---: | ---: | ---: |"]
    for model, indexed in data.items():
        rows = list(indexed.values())
        usages = [r["response"]["usage"] for r in rows]
        cost = sum(u.get("estimated_cost", 0) for u in usages)
        price = f"{cost:.8f}" if all("estimated_cost" in u for u in usages) else "local; electricity not measured"
        lines.append(f"| {model} | {statistics.mean(u['prompt_tokens'] for u in usages):.2f} | {statistics.median(r['latency_s'] for r in rows):.3f} | {price} |")
    lines += ["", "These are prompt-token totals including task text and image overhead, not isolated",
              "vision-token counts. Timings were collected while runs overlapped and are descriptive.", "",
              "## Scope and limitations", "",
              "This is a matched-input-resolution comparison of deployed configurations, not a",
              "token-matched comparison or an isolated model effect. Hosted preprocessing remains",
              "unobserved. Local Gemma used llama.cpp b9555, QAT Q4_0 weights, BF16 projector,",
              "an 8192-token context, image_max_tokens=256, Vulkan, and thinking disabled.",
              "Model/projector/binary hashes and exact hosted IDs are retained in the run protocols.", "",
              "All mosaics were built with the original generator, align=48 and the default 1536",
              "source-side budget, then resized to exactly 768 x 768 with Pillow LANCZOS. These",
              "are new measurements; the original report's tables have not been changed.", "",
              "The requested output schema specifies an integer but does not restrict it to the",
              "correct answer range. Out-of-range integers are scored as wrong, never removed.", "",
              "Smoke sequence 99, the earlier unconstrained Mistral sweep, and failed Scout format",
              "checks are excluded. They remain in their own files. Scout's prose/truncation in",
              "those checks prompted the same integer-schema constraint for all models.", "",
              "Five sequences per cell support an exploratory comparison, not a precise ranking",
              "or a new estimate of the 90% breaking point. No confidence or significance claim is made.", "",
              "Qwen3-VL-30B-A3B-Instruct was added on DeepInfra using the saved sweep runner,",
              "without new smoke tests. It is distinct from the local Qwen 3.5 4B model.",
              "The 30B-A3B option was selected over Qwen3-VL-235B-A22B-Instruct because the",
              "catalog prices were USD 0.15/0.60 versus 0.20/0.88 per million input/output",
              "tokens. Both have Apache 2.0 model licenses and use hosted hardware; measured",
              "request latency is reported above. The Qwen run retains its live catalog entry.", "",
              "The four expansion models were requested with reasoning_effort=none; the earlier",
              "hosted runs omitted that option. Exact options and catalog entries are archived.",
              "GLM Flash nevertheless returned reasoning text on some calls; Gemma 26B returned",
              "long digit strings on some calls. Their fixed-32-token scores include these failures",
              "and cannot be interpreted as isolated vision accuracy. No requests were retried.", "",
              "DeepInfra documents reasoning_effort=none as disabling reasoning:",
              "[reasoning controls](https://docs.deepinfra.com/chat/reasoning). The saved response",
              "fields, rather than that documented expectation, determine the failure counts above.", "",
              "The exact GLM-5.3 and DeepSeek-V4-Flash IDs were omitted because DeepInfra's",
              "catalog listed text-only inputs. No different model ID was silently substituted.", "",
              "Sources: [30B provider](https://deepinfra.com/Qwen/Qwen3-VL-30B-A3B-Instruct),",
              "[235B provider](https://deepinfra.com/Qwen/Qwen3-VL-235B-A22B-Instruct),",
              "[30B license](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct),",
              "[235B license](https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct).", "",
              "## Reproduce this summary", "", "```powershell", "python -B -m runner.matched_768 --summarize", "```", "",
              "This command reads saved JSONL files and makes no model calls. It fails if any run",
              "is incomplete, duplicated, has mismatched input hashes/prompts, or inconsistent scores.", "",
              "Inputs:", ""]
    lines += [f"- `{file}`" for file in FILES.values()]
    out = ROOT / "analysis/matched768.md"
    fig, axes = plt.subplots(3, 2, figsize=(10, 10), sharex=True, sharey=True)
    colors = list(plt.get_cmap("tab10").colors)
    short = list(data)
    for row, task in enumerate(TASKS):
        for col, labels in enumerate([False, True]):
            ax = axes[row, col]
            for (model, indexed), color, label in zip(data.items(), colors, short):
                ys = [sum(r["correct"] for r in indexed.values() if (r["task"], r["n"], r["labels"]) == (task, n, labels)) / 5 for n in NS]
                ax.plot(NS, ys, marker="o", color=color, label=label)
            ax.set_title(task + (" | frame numbers" if labels else " | no labels"))
            ax.set_ylim(-0.04, 1.04)
            ax.set_xticks(NS)
            ax.set_yticks([0, .2, .4, .6, .8, 1])
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel("Accuracy")
            if row == 2:
                ax.set_xlabel("Frames in the mosaic (N)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5, .025), fontsize=8)
    fig.suptitle("Identical 768 x 768 inputs and integer output constraint", fontsize=15)
    fig.text(.5, .008, "Exploratory: 5 sequences per cell; 150 calls per model. Native preprocessing and quantization differ.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, .96))
    fig.savefig(ROOT / "figures/matched768_accuracy.png", dpi=160)
    plt.close(fig)
    lines += ["", "![Accuracy by task, grid size and labels](../figures/matched768_accuracy.png)"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{150 * len(data)} rows validated; 150 matched cases; output failures counted as wrong.")
    for model, indexed in data.items():
        print(model + ": " + cell(list(indexed.values())))
    print("Wrote analysis/matched768.md")


if __name__ == "__main__":
    main()
