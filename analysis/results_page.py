"""Generate the indexed matched-768 results page and charts; no model calls.

Run: python -B -m analysis.results_page
Only table_cells.selected_path supplies scores. Failed attempts are never pooled.
"""
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

NAMES = {
    "gemma": "Gemma 4 12B (local)", "mistral": "Mistral Small 3.2",
    "scout": "Llama 4 Scout", "qwen30": "Qwen3-VL-30B",
    "gemma26": "Gemma 4 26B", "nemotron": "Nemotron 3.5 VL",
    "qwen38": "Qwen3.8-27B", "glmflash": "GLM-5.3-Flash",
    "haiku55": "Haiku 5.5", "sonnet55": "Sonnet 5.5", "opus55": "Opus 5.5",
    "kimi": "Kimi K3.1 Pro", "qwen4max": "Qwen4-Max",
    "dsflash41": "DeepSeek V4.1 Flash", "mimo26pro": "MiMo V2.6 Pro",
    "inkling2m": "Inkling 2 Mini", "gemini37f": "Gemini 3.7 Flash",
    "hy4": "Hy4", "qwen35": "Qwen3.5-397B",
}
TASKS = ["moving_dot", "tile_probe", "count_events"]
EXPECTED = set(itertools.product(TASKS, [4, 9, 16, 36, 64], [False, True], range(5)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def case(row):
    return row["task"], row["n"], row["labels"], row["seq_index"]


def token_stats(rows):
    completions, reasoning = [], []
    text_count = 0
    for row in rows:
        response = row.get("response") or {}
        usage = response.get("usage") or {}
        output = usage.get("completion_tokens", usage.get("output_tokens"))
        if output is not None:
            completions.append(output)
        details = usage.get("completion_tokens_details") or {}
        reason = details.get("reasoning_tokens")
        if reason is None:
            reason = (usage.get("output_tokens_details") or {}).get("thinking_tokens")
        if reason is not None:
            reasoning.append(reason)
        if response.get("choices"):
            text_count += bool(response["choices"][0]["message"].get("reasoning_content"))
        else:
            text_count += any(b.get("type") == "thinking" and b.get("thinking")
                              for b in response.get("content", []))
    median = statistics.median(completions) if completions else None
    reasons = []
    if median is not None and median > 9:
        reasons.append("median completion tokens > 9 (screening rule, not proof)")
    if any(n > 0 for n in reasoning):
        reasons.append("positive provider-reported reasoning tokens")
    if len(reasoning) < len(rows):
        reasons.append(f"reasoning counter missing on {len(rows) - len(reasoning)} responses")
    if text_count:
        reasons.append("reasoning text returned")
    return {
        "completion_reported": len(completions),
        "completion_total": sum(completions) if completions else None,
        "completion_median": median,
        "completion_min": min(completions) if completions else None,
        "completion_max": max(completions) if completions else None,
        "reasoning_reported": len(reasoning),
        "reasoning_total": sum(reasoning) if reasoning else None,
        "reasoning_positive_responses": sum(n > 0 for n in reasoning),
        "reasoning_text_responses": text_count,
        "thinking_note": "thinking unconfirmed" if reasons else "provider reports zero reasoning tokens",
        "note_reasons": reasons,
    }


def check_score(row, fmt):
    try:
        value = json.loads(row["raw_answer"])
        if fmt == "object":
            value = value["answer"] if type(value) is dict and list(value) == ["answer"] else None
        valid = type(value) is int and row["finish_reason"] == "stop"
    except ValueError:
        value, valid = None, False
    require(row["correct"] == (valid and value == row["truth"]), "Saved score does not match raw answer")
    require(row["parsed"] == (value if valid else None), "Saved parsed answer mismatch")
    require(row["single_integer"] == valid, "Saved format validity mismatch")


def load_selected(index):
    entries = {entry["path"]: entry for entry in index["runs"]}
    ref = {case(r): r for r in read_rows(ROOT / index["reference_path"])}
    selected, summaries = {}, []
    for cell in index["table_cells"]:
        path_string = cell.get("selected_path")
        if not path_string:
            continue
        path = ROOT / path_string
        entry = entries[path_string]
        require(digest(path) == entry["jsonl_sha256"], f"Changed result: {path_string}")
        rows = read_rows(path)
        require(len(rows) == 150 and {case(r) for r in rows} == EXPECTED, f"Case coverage: {path_string}")
        protocol = json.loads((path.parent / "protocol.json").read_text(encoding="utf-8-sig"))
        require(digest(path.parent / "runner_source.py") == protocol["runner_sha256"] == entry["runner_sha256"],
                f"Runner snapshot mismatch: {path_string}")
        if entry.get("config_sha256"):
            require(digest(path.parent / "models.toml") == entry["config_sha256"], f"Config snapshot mismatch: {path_string}")
        if entry.get("launch_source_path"):
            require(digest(ROOT / entry["launch_source_path"]) == entry["launch_source_sha256_at_indexing"],
                    "Gemma launcher changed since indexing")
        for row in rows:
            require(not row.get("error") and row.get("response"), f"Failed call selected: {path_string}")
            require(row["max_tokens"] == 32 and row["image_wh"] == [[768, 768]], "Input size / cap mismatch")
            require(row["response"]["model"] == row["model"] == entry["model"], "Model identity mismatch")
            require(row["temperature"] == (None if row["model"].startswith("claude-") else 0), "Temperature mismatch")
            fmt = row["response_format"]
            schema = fmt.get("json_schema", fmt)["schema"]
            require(schema["type"] == cell["answer_format"], "Actual schema disagrees with index")
            for field in ("image_sha256", "prompt", "system_prompt", "truth", "condition", "base_seed"):
                require(row[field] == ref[case(row)][field], f"Matched input mismatch: {path_string}: {field}")
            if "request_options" in row:
                require(row["request_options"] == entry.get("requested_options"), "Request options disagree with index")
            response = row["response"]
            if response.get("choices"):
                choice = response["choices"][0]
                raw = choice["message"].get("content") or ""
                finish = choice["finish_reason"]
            else:
                raw = "".join(b["text"] for b in response["content"] if b["type"] == "text")
                finish = {"end_turn": "stop", "max_tokens": "length"}.get(response["stop_reason"], response["stop_reason"])
            require(raw == row["raw_answer"] and finish == row["finish_reason"], "Raw response disagrees with score fields")
            check_score(row, cell["answer_format"])
        stat = token_stats(rows)
        require(stat["completion_reported"] == 150, "Missing completion usage")
        stat.update(model_alias=cell["model_alias"], model=entry["model"], answer_format=cell["answer_format"],
                    path=path_string, responses=150, correct=sum(r["correct"] for r in rows),
                    cutoffs=sum(r["finish_reason"] == "length" for r in rows),
                    invalid_answers=sum(not r["single_integer"] for r in rows),
                    request_options=entry.get("requested_options"), historical_notes=entry.get("notes", []))
        require(stat["correct"] == entry["correct"] and stat["cutoffs"] == entry["cutoffs"], "Index counts stale")
        summaries.append(stat)
        selected[cell["model_alias"], cell["answer_format"]] = (rows, stat)
    return selected, summaries


def number(value):
    return "missing" if value is None else f"{value:g}"


def make_charts(selected):
    aliases = sorted({a for a, _ in selected}, key=lambda a: (
        -selected.get((a, "object"), (None, {"correct": -1}))[1]["correct"], NAMES[a]))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    colors = {"integer": "#486b91", "object": "#d07836"}
    fig, ax = plt.subplots(figsize=(12, 10))
    y = np.arange(len(aliases))
    for fmt, offset in (("integer", -.19), ("object", .19)):
        for i, alias in enumerate(aliases):
            item = selected.get((alias, fmt))
            if item is None:
                continue
            s = item[1]
            ax.barh(i + offset, s["correct"], height=.34, color=colors[fmt])
            marker = "*" if s["thinking_note"] == "thinking unconfirmed" else ""
            ax.text(s["correct"] + 1, i + offset, f'{s["correct"]}{marker}', va="center", fontsize=9)

    ax.set_yticks(y, [NAMES[a] for a in aliases])
    ax.invert_yaxis()
    ax.set_xlim(0, 163)
    ax.set_xticks([0, 30, 60, 90, 120, 150])
    ax.set_xlabel("Correct answers out of 150; cutoffs and invalid answers count as wrong")
    ax.set_title("Same 768 x 768 mosaics, two answer formats", loc="left", fontsize=17, pad=16)
    ax.legend(handles=[Patch(facecolor=colors["integer"], label="Bare integer"),
                       Patch(facecolor=colors["object"], label='{"answer": N}')], loc="lower right")
    fig.text(.02, .018, "* thinking unconfirmed: median >9, positive reasoning tokens/text, or missing reasoning counter.\n"
             "Exploratory deployed configurations; provider preprocessing, sampling and reasoning controls differ.\n"
             "Scout wrapper and both Gemini formats incomplete; no score plotted for those cells.", fontsize=9)
    fig.tight_layout(rect=(0, .085, 1, 1))
    fig.savefig(ROOT / "figures/matched768_formats.png", dpi=160)
    plt.close(fig)

    paired = [a for a in aliases if all((a, f) in selected for f in ("integer", "object"))]
    paired.sort(key=lambda a: selected[a, "object"][1]["correct"] - selected[a, "integer"][1]["correct"])
    deltas = [selected[a, "object"][1]["correct"] - selected[a, "integer"][1]["correct"] for a in paired]
    fig, ax = plt.subplots(figsize=(11, 9))
    labels = [NAMES[a] + (" *" if any(selected[a, f][1]["thinking_note"] == "thinking unconfirmed"
                                     for f in ("integer", "object")) else "") for a in paired]
    ax.barh(labels, deltas, color=[colors["object"] if d >= 0 else colors["integer"] for d in deltas])
    for i, d in enumerate(deltas):
        ax.text(d + (1 if d >= 0 else -1), i, f"{d:+d}", ha="left" if d >= 0 else "right", va="center")
    ax.axvline(0, color="#334155", linewidth=.8)
    ax.set_xlim(min(deltas) - 15, max(deltas) + 15)
    ax.set_xlabel("Wrapper correct count minus integer correct count (same 150 cases)")
    ax.set_title("Answer format changes the measured score", loc="left", fontsize=17)
    fig.text(.02, .02, "* thinking unconfirmed on at least one format; see token audit.\n"
             "Descriptive paired counts, not a significance claim. Differences include format failures.", fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, 1))
    fig.savefig(ROOT / "figures/matched768_format_delta.png", dpi=160)
    plt.close(fig)

    wrappers = [a for a in aliases if (a, "object") in selected]
    cells = [(t, label) for t in TASKS for label in (False, True)]
    values = [[sum(r["correct"] for r in selected[a, "object"][0] if (r["task"], r["labels"]) == c)
               for c in cells] for a in wrappers]
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(values, vmin=0, vmax=25, cmap="Blues", aspect="auto")
    ax.set_yticks(range(len(wrappers)), [NAMES[a] + (" *" if selected[a, "object"][1]["thinking_note"] == "thinking unconfirmed" else "")
                                       for a in wrappers])
    ax.set_xticks(range(6), [t.replace("_", " ") + "\nlabels " + ("on" if l else "off") for t, l in cells])
    ax.tick_params(axis="x", length=0)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v}/25", ha="center", va="center", color="white" if v > 14 else "#172b4d", fontsize=9)
    ax.set_title('Wrapper format: task and frame-number breakdown', loc="left", fontsize=15, pad=16)
    fig.text(.02, .02, "* thinking unconfirmed; see token audit. Each cell pools five grid sizes and five sequences.\n"
             "Incomplete models omitted. Same images; native provider preprocessing remains unobserved.", fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(ROOT / "figures/matched768_wrapper_tasks.png", dpi=160)
    plt.close(fig)


def main():
    index = json.loads((ROOT / "analysis/run_index.json").read_text(encoding="utf-8-sig"))
    selected, summaries = load_selected(index)
    summaries.sort(key=lambda s: (NAMES[s["model_alias"]], s["answer_format"]))
    make_charts(selected)
    lines = ["# Matched-768: answer format comparison", "",
             f"{len(summaries)} complete runs; {len(summaries) * 150:,} scored responses. Each run uses the same 150 cases:",
             "three tasks, five grid sizes (4, 9, 16, 36, 64), labels off/on, five sequences per cell.",
             "Images are 768 x 768; the output cap is 32 tokens. Cutoffs and invalid answers count as wrong.", "",
             "This compares deployed configurations, not matched reasoning compute or isolated vision accuracy.",
             "Claude omits temperature and uses model-specific thinking controls; Opus requests low effort.",
             "Other selected runs use temperature 0. Requested reasoning controls do not prove reasoning was disabled.", "",
             "## Reading the notes", "",
             '**"thinking unconfirmed"** flags any run with median completion tokens >9, any positive reported reasoning',
             "tokens/text, or an unavailable reasoning counter. The >9 threshold is a conservative screening rule,",
             "not proof of hidden reasoning: a wrapper or a repeated digit string can also use more tokens.",
             "A zero is shown only when the provider explicitly reports it; missing counters stay missing.",
             'Otherwise the note is "provider reports zero reasoning tokens", not a claim about unobserved computation.', "",
             "Qwen4-Max has no reasoning-token counter on either format. Kimi explicitly reports zero on all 300",
             "responses; its integer run nevertheless reaches 32 tokens on all 150 calls. These scores describe",
             "the requested output protocol, not a verified all-model thinking-off comparison.", "",
             "## Scores and token audit", "",
             "Completion columns include final output and any reasoning counted by the provider. Claude uses",
             "output_tokens and output_tokens_details.thinking_tokens; other APIs use completion_tokens and",
             "completion_tokens_details.reasoning_tokens. Reasoning totals cover only reported counters.", "",
             "| Model | Format | Correct /150 | Cutoffs | Completion median [min-max] | Completion total | Reasoning total | Counter present /150 | Positive reasoning calls | Note |",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |"]
    for s in summaries:
        token_range = f'{number(s["completion_median"])} [{s["completion_min"]}-{s["completion_max"]}]'
        note = s["thinking_note"]
        if s["note_reasons"]:
            note += ": " + "; ".join(s["note_reasons"])
        lines.append(f'| {NAMES[s["model_alias"]]} | {s["answer_format"]} | {s["correct"]} | {s["cutoffs"]} | {token_range} | {s["completion_total"]} | {number(s["reasoning_total"])} | {s["reasoning_reported"]} | {s["reasoning_positive_responses"]} | {note} |')
    lines += ["", "## Charts", ""]
    for filename, caption in (
        ("matched768_formats.png", "Correct answers by format"),
        ("matched768_format_delta.png", "Change in correct answers with the wrapper"),
        ("matched768_wrapper_tasks.png", "Wrapper accuracy by task and labels"),
    ):
        lines += [f"![{caption}](../figures/{filename})", ""]
    lines += ["## Incomplete cells and retries", "",
              "Scout wrapper and both Gemini 3.7 Flash formats have no selected complete 150-case run.",
              "Incomplete attempts are not scored as 150-case results. Retries and earlier batches stay separate.",
              "Selection uses the index, not the highest observed score, and no attempts are concatenated.", "",
              "For the new models, the earliest verified complete run per format is selected. Qwen4-Max wrapper",
              "and both DeepSeek V4.1 Flash / MiMo formats come from new3; their later new6 attempts failed.",
              "The selected paths below make this explicit.", "",
              "### Gemini usage from incomplete attempts", "",
              "| Attempt | Responses | Completion total | Reasoning total | Positive reasoning calls | Reasoning text calls |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for entry in index["runs"]:
        if entry["model_alias"] == "gemini37f":
            rows = [r for r in read_rows(ROOT / entry["path"]) if r.get("response")]
            s = token_stats(rows)
            lines.append(f'| [{entry["path"]}](../{entry["path"]}) | {len(rows)} | {s["completion_total"]} | {number(s["reasoning_total"])} | {s["reasoning_positive_responses"]} | {s["reasoning_text_responses"]} |')
    lines += ["", "These are separate failed attempts, not independent additional benchmark cases.",
              "Empty reasoning_content does not establish zero reasoning usage.", "",
              "## Provenance and scope", "",
              "Only table_cells.selected_path in analysis/run_index.json supplies the main scores. This generator",
              "checks result hashes, archived runner/config hashes, 150 unique cases, model identities, image hashes,",
              "questions, system prompts, truth, input resolution, cap, temperature and raw-answer scoring.",
              "Historical protocol-name errors remain in the raw files; saved response_format determines the actual format.", "",
              "Gemma's 91/150 integer run requires both runner_source.py and launch_source.py from its result folder.",
              "The launcher overrides runner settings. Its hash was recorded at indexing, not in the original manifest.",
              "The original server launch command was not saved; recorded settings and binary/model hashes remain.", "",
              "Two Gemma server logs had private absolute paths redacted after the run. The index records the paths,",
              "redaction scope and current hashes; these logs are not untouched originals. Result JSONL files were not edited.", "",
              "The 4096-token thinking round, the earlier Opus 148/150 run at cap 4096, raw text-only tag tests,",
              "smoke tests and failed attempts are excluded from the two main format tables.",
              "Five sequences per cell support exploratory descriptions, not a precise model ranking.",
              "Identical input pixels do not imply equal provider image tokens or equal internal preprocessing.", "",
              "## Reproduce", "", "```sh", "python -B -m analysis.results_page", "```", "",
              "Reads existing logs only. Writes this Markdown page, docs/matched768_results.html,",
              "analysis/matched768_tokens.json and three figures. No API keys, GPU or model calls are required.", "",
              "## Selected source files", "", "| Model | Format | Source |", "| --- | --- | --- |"]
    lines += [f'| {NAMES[s["model_alias"]]} | {s["answer_format"]} | [{s["path"]}](../{s["path"]}) |' for s in summaries]
    markdown = "\n".join(lines) + "\n"
    (ROOT / "analysis/matched768_results.md").write_text(markdown, encoding="utf-8")
    # A small renderer for this generator's known Markdown subset; no extra dependency.
    parts, in_table, in_code = [], False, False
    import re
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
        return re.sub(r"\[([^]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)
    for line in lines:
        if in_table and not line.startswith("|"):
            parts.append("</tbody></table></div>")
            in_table = False
        if line.startswith("```"):
            parts.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
        elif in_code:
            parts.append(html.escape(line) + "\n")
        elif line.startswith("|"):
            columns = [v.strip() for v in line.strip("|").split("|")]
            if all(re.fullmatch(r"[-: ]+", v) for v in columns):
                continue
            if not in_table:
                parts.append('<div class="table"><table><thead><tr>' + "".join(f"<th>{inline(v)}</th>" for v in columns) + "</tr></thead><tbody>")
                in_table = True
            else:
                parts.append("<tr>" + "".join(f"<td>{inline(v)}</td>" for v in columns) + "</tr>")
        elif line.startswith("!["):
            match = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
            parts.append(f'<img alt="{html.escape(match[1])}" src="{html.escape(match[2])}">')
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            parts.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line:
            parts.append(f"<p>{inline(line)}</p>")
    if in_table:
        parts.append("</tbody></table></div>")
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frame Mosaic: matched-768 results</title><style>
body{max-width:1300px;margin:40px auto;padding:0 24px;background:#f8fafc;color:#172b4d;font:16px/1.6 system-ui}
h1{font-size:2.4rem;line-height:1.15}h2{margin-top:2.5rem}a{color:#235b91}
.table{overflow:auto}table{border-collapse:collapse;background:white;font-size:14px;width:100%}
th,td{padding:10px;text-align:left;border-bottom:1px solid #dce3eb}th{background:#e9eff5}
img{width:100%;height:auto;margin:20px 0}pre{padding:18px;background:#e9eff5}
</style><main>""" + "\n".join(parts) + "</main></html>\n"
    (ROOT / "docs/matched768_results.html").write_text(page, encoding="utf-8")
    (ROOT / "analysis/matched768_tokens.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(summaries)} runs / {150 * len(summaries)} responses; generated page, token audit and 3 charts.")
    for s in summaries:
        print(f'{s["model_alias"]} {s["answer_format"]}: median={s["completion_median"]:g}; completion={s["completion_total"]}; reasoning={number(s["reasoning_total"])}; reported={s["reasoning_reported"]}/150; {s["thinking_note"]}')


if __name__ == "__main__":
    main()
