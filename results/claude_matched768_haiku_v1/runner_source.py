"""Single entry point for matched-768 sweeps, offline summaries and receipt audits.

python -B -m runner.matched_768 --models mistral scout gemma --out results/new_run
python -B -m runner.matched_768 --models haiku55 sonnet55 opus55 --out results/new_claude_run
python -B -m runner.matched_768 --summarize

No smoke tests or automatic retries. Existing results are never overwritten.
Keys come from the private key file: DEEPINFRA_API_KEY for hosted open models,
ANTHROPIC_API_KEY for Claude (official `anthropic` SDK, no refusal fallbacks, since a
fallback would score another model's answer). Claude protocol differences are listed in
DEVIATIONS and written into protocol.json and every record.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import urllib.request

import numpy
import PIL
from PIL import Image

from gen.common import FRAME_SIZE, rng_for
from gen.tasks import TASKS
from mosaic.builder import Condition, build
from eval.questions import prompt
from eval.score import score, apply_truncation_rule

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"mistral": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
          "scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
          "qwen30": "Qwen/Qwen3-VL-30B-A3B-Instruct",
          "glmflash": "zai-org/GLM-5.3-Flash",
          "gemma26": "google/gemma-4-26B-A4B-it",
          "nemotron": "nvidia/NVIDIA-Nemotron-3.5-Super-VL",
          "qwen38": "Qwen/Qwen3.8-27B",
          "gemma": "gemma4-12b-qat-q4_0-matched768",
          "haiku55": "claude-haiku-5-5",
          "sonnet55": "claude-sonnet-5-5",
          "opus55": "claude-opus-5-5"}
CLAUDE = ("haiku55", "sonnet55", "opus55")
MODEL_OPTIONS = {**{alias: {"reasoning_effort": "none"}
                    for alias in ("glmflash", "gemma26", "nemotron", "qwen38")},
                 # Claude: thinking off where the model allows it; Opus 5.5 rejects every way of turning it off.
                 "haiku55": {"thinking": {"type": "disabled"}},
                 "sonnet55": {"thinking": {"type": "between_tools"}},
                 "opus55": {"output_config": {"effort": "low"}}}
OUTPUT_CAP = {"opus55": 4096}  # everything else: 32
DEVIATIONS = {alias: ["temperature omitted (current Claude models reject sampling parameters)"]
              for alias in CLAUDE}
DEVIATIONS["opus55"] += ["thinking cannot be disabled; adaptive thinking at effort low",
                         "output cap 4096 instead of 32 so thinking is not cut off"]
SYSTEM = "Respond with a single integer only. Do not explain or show working."
THINKING_MODELS = ("glmflash", "gemma26", "nemotron", "qwen38")
THINKING_SYSTEM = "Reason internally as needed. Return only a single integer as the final answer."
THINKING_PROTOCOL = "matched768_thinking4096_v1"
PROTOCOL = "matched768_integer_schema_v1"
FORMAT = {"type": "json_schema", "json_schema": {"name": "integer_answer", "strict": True,
                                                   "schema": {"type": "integer"}}}
CLAUDE_FORMAT = {"type": "json_schema", "schema": {"type": "integer"}}
ANTHROPIC_API = "https://api.anthropic.com"
FINISH = {"end_turn": "stop", "max_tokens": "length"}  # Claude stop_reason -> finish_reason
CASES = list(itertools.product(["moving_dot", "tile_probe", "count_events"],
                              [4, 9, 16, 36, 64], [False, True], range(5)))


def make_case(task, n, labels, seq):
    frames, answer, _ = TASKS[task](n, rng_for(task, n, seq, 0), FRAME_SIZE)
    cond = Condition(n=n, timestamps=labels, align=48)
    images, meta = build(frames, cond)
    im = images[0].resize((768, 768), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue(), answer, meta, prompt(task, cond)


def read_key(path, required=True, name="DEEPINFRA_API_KEY"):
    value = ""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            field, sep, content = line.partition("=")
            if sep and field.strip() == name:
                value = content.strip().strip("\"'")
    else:
        value = os.environ.get(name, "").strip()
    if required and not value:
        raise ValueError(f"Missing {name} in private key file or environment")
    return value


def claude_client(key):
    import anthropic  # only needed when a Claude model is selected
    # Explicit base_url so an ANTHROPIC_BASE_URL in the environment cannot redirect the eval.
    return anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_API, max_retries=0, timeout=300)


def run_model(alias, key, folder, local_url, thinking=False):
    model = MODELS[alias]
    claude = alias in CLAUDE
    cap = 4096 if thinking else OUTPUT_CAP.get(alias, 32)
    system = THINKING_SYSTEM if thinking else SYSTEM
    options = {"reasoning_effort": "high"} if thinking else MODEL_OPTIONS.get(alias, {})
    client = claude_client(key) if claude else None
    endpoint = (local_url.rstrip("/") + "/v1/chat/completions" if alias == "gemma"
                else "https://api.deepinfra.com/v1/openai/chat/completions")
    with (folder / (alias + "_sweep.jsonl")).open("x", encoding="utf-8") as fh:
        for index, (task, n, labels, seq) in enumerate(CASES, 1):
            png, answer, meta, question = make_case(task, n, labels, seq)
            image = base64.b64encode(png).decode()
            if claude:
                options = MODEL_OPTIONS[alias]
                body = {"model": model, "max_tokens": cap, "system": SYSTEM, "messages": [
                    {"role": "user", "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image}},
                        {"type": "text", "text": question}]}],
                    "output_config": {"format": CLAUDE_FORMAT, **options.get("output_config", {})}}
                if "thinking" in options:
                    body["thinking"] = options["thinking"]
            else:
                body = {"model": model, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
                        {"type": "text", "text": question}]}],
                    "temperature": 0, "max_tokens": cap, "stream": False, "response_format": FORMAT}
                if alias == "gemma":
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                body.update(options)
            rec = {"protocol": THINKING_PROTOCOL if thinking else PROTOCOL, "response_format": CLAUDE_FORMAT if claude else FORMAT, "model": model,
                   "phase": "sweep", "task": task, "n": n, "labels": labels,
                   "seq_index": seq, "base_seed": 0, "condition": meta["condition"],
                   "source_image_wh": meta["image_wh"], "image_wh": [[768, 768]],
                   "image_sha256": hashlib.sha256(png).hexdigest(), "system_prompt": system,
                   "prompt": question, "temperature": None if claude else 0, "max_tokens": cap,
                   "time_utc": datetime.now(timezone.utc).isoformat()}
            rec["request_options"] = options
            if claude:
                rec["deviations"] = DEVIATIONS[alias]
            start = time.perf_counter()
            try:
                if claude:
                    message = client.messages.create(**body)
                    result = message.to_dict(mode="json")
                    rec["request_id"] = message._request_id
                    raw = "".join(block.text for block in message.content if block.type == "text")
                    finish = FINISH.get(message.stop_reason, message.stop_reason)
                else:
                    headers = {"Content-Type": "application/json"}
                    if alias != "gemma":
                        headers["Authorization"] = "Bearer " + key
                    req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
                    with urllib.request.urlopen(req, timeout=300 if thinking else 90) as response:
                        result = json.load(response)
                    choice = result["choices"][0]
                    raw = choice["message"].get("content") or ""
                    finish = choice.get("finish_reason")
                rec["response"] = result
                rec["latency_s"] = round(time.perf_counter() - start, 3)
                rec.update(raw_answer=raw, finish_reason=finish, **score(task, answer, raw))
                rec["hit_output_cap"] = finish == "length"
                apply_truncation_rule(rec)
                if result.get("model") != model:
                    raise ValueError("Returned model identity does not match request")
                # Decode the entire integer, never the first digit in an explanation.
                try:
                    value = json.loads(raw)
                    valid = type(value) is int and rec["finish_reason"] == "stop"
                except ValueError:
                    valid = False
                rec["single_integer"] = valid
                if valid:
                    rec.update(parsed=value, correct=value == rec["truth"],
                               within1=abs(value - rec["truth"]) <= 1,
                               partial=float(value == rec["truth"]))
                else:
                    rec.update(parsed=None, correct=False, within1=False, partial=0.0)
            except Exception as exc:
                rec["error"] = {"type": type(exc).__name__,
                                "http_status": getattr(exc, "status_code", None) or getattr(exc, "code", None)}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(json.dumps({"model": alias, "done": index - 1, "error": rec["error"]}), flush=True)
                return False
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if index % 10 == 0:
                print(json.dumps({"model": alias, "done": index, "total": len(CASES)}), flush=True)
    return True


def audit(key_file, forbidden):
    keys = [k.encode() for k in (read_key(key_file, False), read_key(key_file, False, "ANTHROPIC_API_KEY")) if k]
    terms = [word.lower().encode() for word in forbidden]
    hits = []
    count = 0
    for path in ROOT.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            count += 1
            if any(t in raw.lower() for t in terms) or any(k in raw for k in keys):
                hits.append(path.relative_to(ROOT).as_posix())
    objects = subprocess.run(["git", "cat-file", "--batch", "--batch-all-objects"], cwd=ROOT,
                             input=b"", capture_output=True, check=True).stdout
    object_hit = any(t in objects.lower() for t in terms) or any(k in objects for k in keys)
    print(json.dumps({"files_scanned": count, "file_hits": hits, "decompressed_git_object_hit": object_hit}))
    if hits or object_hit:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["mistral", "scout"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/frame-mosaic-eval/.env")
    parser.add_argument("--local-url", default="http://127.0.0.1:8098")
    parser.add_argument("--local-receipt", type=Path, default=ROOT / "logs/gemma_matched768_server.json")
    parser.add_argument("--thinking-on", action="store_true", help="Separate DeepInfra reasoning round, high effort, 4096 output tokens")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--summarize-thinking", type=Path, metavar="RESULTS_DIR")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--forbid", nargs="*", default=[])
    args = parser.parse_args()
    if args.summarize_thinking:
        from analysis.thinking768 import summarize
        summarize(args.summarize_thinking)
        return
    if args.summarize:
        from analysis.matched768 import main as summarize
        summarize()
        return
    if args.audit:
        audit(args.key_file, args.forbid)
        return
    selected = list(dict.fromkeys(args.models))
    if args.thinking_on and any(m not in THINKING_MODELS for m in selected):
        parser.error("--thinking-on supports only: " + ", ".join(THINKING_MODELS))
    hosted = [m for m in selected if m != "gemma" and m not in CLAUDE]
    claude = [m for m in selected if m in CLAUDE]
    if args.dry_run:
        print(json.dumps({"models": selected, "calls_per_model": len(CASES),
                          "total_calls": len(CASES) * len(selected), "smoke_calls": 0,
                          "protocol": THINKING_PROTOCOL if args.thinking_on else PROTOCOL,
                          "output_cap": 4096 if args.thinking_on else 32, "input_size": [768, 768],
                          "deviations": {m: DEVIATIONS[m] for m in claude}}))
        return
    if args.out is None:
        parser.error("--out is required for a live sweep; choose a new directory")
    folder = args.out.resolve()
    if not folder.is_relative_to(ROOT / "results"):
        parser.error("--out must be a new directory under results/")
    if folder.exists():
        parser.error("output already exists; results will not be overwritten")
    keys = {"hosted": read_key(args.key_file) if hosted else "",
            "claude": read_key(args.key_file, name="ANTHROPIC_API_KEY") if claude else ""}
    local = None
    if "gemma" in selected:
        local = json.loads(args.local_receipt.read_text())
        if local["alias"] != MODELS["gemma"]:
            parser.error("local server receipt alias differs from selected Gemma model")
        with urllib.request.urlopen(args.local_url.rstrip("/") + "/v1/models", timeout=10) as response:
            served = json.load(response)
        if MODELS["gemma"] not in [m["id"] for m in served["data"]]:
            parser.error("verified Gemma alias is not served at --local-url")
    catalog = {}
    if hosted:
        with urllib.request.urlopen("https://api.deepinfra.com/v1/openai/models", timeout=20) as response:
            current = {item["id"]: item for item in json.load(response)["data"]}
        catalog = {name: current[MODELS[name]] for name in hosted}
        if args.thinking_on:
            for name, item in catalog.items():
                tags = item.get("metadata", {}).get("tags", [])
                if "reasoning" not in tags:
                    parser.error(name + " is not tagged as reasoning-capable in the current catalog")
    if claude:
        client = claude_client(keys["claude"])
        catalog.update({name: client.models.retrieve(MODELS[name]).to_dict(mode="json") for name in claude})
    folder.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    manifest = {"protocol": THINKING_PROTOCOL if args.thinking_on else PROTOCOL, "response_format": FORMAT,
                "models": {m: MODELS[m] for m in selected}, "system_prompt": THINKING_SYSTEM if args.thinking_on else SYSTEM,
                "request_options": {m: {"reasoning_effort": "high"} if args.thinking_on else MODEL_OPTIONS.get(m, {}) for m in selected},
                "smoke_per_model": 0, "sweep_per_model": 150, "seed": 0,
                "tasks": ["moving_dot", "tile_probe", "count_events"], "n": [4, 9, 16, 36, 64],
                "seqs": list(range(5)), "labels": [False, True], "output_cap": 4096 if args.thinking_on else 32,
                "temperature": 0, "input_size": [768, 768], "server": local, "provider_catalog": catalog,
                "runner_sha256": hashlib.sha256(source).hexdigest(),
                "versions": {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": numpy.__version__}}
    if claude:
        import anthropic
        manifest.update(claude_endpoint=ANTHROPIC_API, claude_response_format=CLAUDE_FORMAT,
                        output_cap_overrides={m: OUTPUT_CAP[m] for m in claude if m in OUTPUT_CAP},
                        deviations={m: DEVIATIONS[m] for m in claude})
        manifest["versions"]["anthropic"] = anthropic.__version__
    (folder / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "runner_source.py").write_bytes(source)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [pool.submit(run_model, name, keys["claude" if name in CLAUDE else "hosted"], folder, args.local_url, args.thinking_on)
                   for name in selected]
        outcomes = {name: future.result() for name, future in zip(selected, futures)}
    print(json.dumps({"completed": outcomes}), flush=True)
    if not all(outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
