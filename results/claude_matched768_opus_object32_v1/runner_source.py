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
import tomllib
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
OUTPUT_CAP = {}  # per-model output-cap overrides; none, so every direct run gets 32
CONFIG_SOURCE = (ROOT / "runner/models.toml").read_bytes()
CONFIGURED_ALIASES = []
for _alias, _entry in tomllib.loads(CONFIG_SOURCE.decode("utf-8-sig")).get("models", {}).items():
    if _alias in MODELS:
        continue  # Preserve archived settings for existing aliases.
    if _entry.get("provider") != "deepinfra" or "direct" not in _entry:
        raise ValueError("New config aliases currently require DeepInfra direct mode")
    _options = _entry["direct"]
    if set(_options) - {"reasoning_effort", "reasoning", "chat_template_kwargs"}:
        raise ValueError("Config may only add reasoning controls, not override request fields")
    _cap = _entry.get("max_tokens", 32)
    if type(_cap) is not int or not 1 <= _cap <= 4096:
        raise ValueError("Configured output cap must be between 1 and 4096")
    MODELS[_alias] = _entry["id"]
    MODEL_OPTIONS[_alias] = _options
    OUTPUT_CAP[_alias] = _cap
    CONFIGURED_ALIASES.append(_alias)
DEVIATIONS = {alias: ["temperature omitted (current Claude models reject sampling parameters)"] for alias in CLAUDE}
DEVIATIONS["opus55"] += ["thinking cannot be disabled; adaptive thinking at effort low"]
SYSTEM = "Respond with a single integer only. Do not explain or show working."
THINKING_MODELS = ("glmflash", "gemma26", "nemotron", "qwen38")
THINKING_SYSTEM = "Reason internally as needed. Return only a single integer as the final answer."
THINKING_PROTOCOL = "matched768_thinking4096_v1"
PROTOCOL = "matched768_integer_schema_v1"
FORMAT = {"type": "json_schema", "json_schema": {"name": "integer_answer", "strict": True,
                                                   "schema": {"type": "integer"}}}
CLAUDE_FORMAT = {"type": "json_schema", "schema": {"type": "object", "properties": {"answer": {"type": "integer"}},
                                                   "required": ["answer"], "additionalProperties": False}}
CLAUDE_INTEGER_FORMAT = {"type": "json_schema", "schema": {"type": "integer"}}
OBJECT_FORMAT = {"type": "json_schema", "json_schema": {"name": "integer_answer", "strict": True,
                                                        "schema": CLAUDE_FORMAT["schema"]}}
OBJECT_PROTOCOL = "matched768_answer_object_v1"
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


def run_model(alias, key, folder, local_url, thinking=False, answer_object=False):
    model = MODELS[alias]
    claude = alias in CLAUDE
    cap = 4096 if thinking else OUTPUT_CAP.get(alias, 32)
    system = THINKING_SYSTEM if thinking else SYSTEM
    options = {"reasoning_effort": "high"} if thinking else MODEL_OPTIONS.get(alias, {})
    response_format = ((CLAUDE_FORMAT if answer_object else CLAUDE_INTEGER_FORMAT) if claude
                       else OBJECT_FORMAT if answer_object else FORMAT)
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
                    "output_config": {"format": response_format, **options.get("output_config", {})}}
                if "thinking" in options:
                    body["thinking"] = options["thinking"]
            else:
                body = {"model": model, "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
                        {"type": "text", "text": question}]}],
                    "temperature": 0, "max_tokens": cap, "stream": False, "response_format": response_format}
                if alias == "gemma":
                    body["chat_template_kwargs"] = {"enable_thinking": False}
                body.update(options)
            rec = {"protocol": OBJECT_PROTOCOL if answer_object else THINKING_PROTOCOL if thinking else PROTOCOL, "response_format": response_format, "model": model,
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
                    if answer_object:  # Require exactly {"answer": integer}.
                        value = value["answer"] if type(value) is dict and list(value) == ["answer"] else None
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


def run_raw_test(config_path, folder, key_file):
    """Run a small saved text-only prompt comparison; never score it as a vision sweep."""
    config_path = config_path.resolve()
    folder = folder.resolve()
    if not config_path.is_relative_to(ROOT) or not folder.is_relative_to(ROOT / "results"):
        raise ValueError("Test config and results must stay inside the repo")
    source = config_path.read_bytes()
    config = json.loads(source.decode("utf-8-sig"))
    model = config["model"]
    if not model.startswith("zai-org/") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-./_" for c in model) or ".." in model:
        raise ValueError("Expected an explicit zai-org model ID")
    cases = config["requests"]
    if not 1 <= len(cases) <= 12 or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Expected 1 to 12 uniquely named test requests")
    for case in cases:
        body = case["body"]
        if set(body) - {"input", "max_new_tokens", "temperature", "stop", "stream"}:
            raise ValueError("Raw test accepts only text input, cap, sampling, stop and stream settings")
        if not isinstance(body.get("input"), str) or type(body.get("max_new_tokens")) is not int or not 1 <= body["max_new_tokens"] <= 4096 or body.get("stream", False):
            raise ValueError("Expected text input, bounded output cap and non-streaming requests")
    key = read_key(key_file)
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "test_config.json").write_bytes(source)
    (folder / "runner_source.py").write_bytes(Path(__file__).read_bytes())
    endpoint = "https://api.deepinfra.com/v1/inference/" + model
    with (folder / "responses.jsonl").open("x", encoding="utf-8") as fh:
        for case in cases:
            record = {"id": case["id"], "model": model, "endpoint": endpoint,
                      "text_only": True, "request": case["body"],
                      "config_sha256": hashlib.sha256(source).hexdigest(),
                      "time_utc": datetime.now(timezone.utc).isoformat()}
            start = time.perf_counter()
            try:
                req = urllib.request.Request(endpoint, data=json.dumps(case["body"]).encode(),
                    headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
                with urllib.request.urlopen(req, timeout=180) as response:
                    record["http_status"] = response.status
                    record["response"] = json.load(response)
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
            record["latency_s"] = round(time.perf_counter() - start, 3)
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            reply = record.get("response", {})
            print(json.dumps({"id": case["id"], "error": record.get("error"),
                              "tokens": reply.get("num_tokens"),
                              "status": reply.get("inference_status", {}).get("status")}), flush=True)
            if "error" in record or reply.get("inference_status", {}).get("status") == "failed":
                raise SystemExit(1)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=["mistral", "scout"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--key-file", type=Path, default=Path.home() / ".config/frame-mosaic-eval/.env")
    parser.add_argument("--local-url", default="http://127.0.0.1:8098")
    parser.add_argument("--local-receipt", type=Path, default=ROOT / "logs/gemma_matched768_server.json")
    parser.add_argument("--thinking-on", action="store_true", help="Separate DeepInfra reasoning round, high effort, 4096 output tokens")
    parser.add_argument("--answer-object", action="store_true", help="Use the {\"answer\": N} schema (hosted or Claude models) in a separate direct-answer run")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--raw-test-config", type=Path, help="Small saved text-only DeepInfra raw-template test")
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--summarize-thinking", type=Path, metavar="RESULTS_DIR")
    action.add_argument("--summarize-wrapper", action="store_true", help="Offline Mistral integer vs answer-object comparison")
    action.add_argument("--watch-thinking", type=Path, metavar="RESULTS_DIR", help="Wait for saved results and generate the separate table; no model calls")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--forbid", nargs="*", default=[])
    args = parser.parse_args()
    if args.raw_test_config:
        if args.out is None:
            parser.error("--out is required for a raw template test")
        run_raw_test(args.raw_test_config, args.out, args.key_file)
        return
    if args.summarize_wrapper:
        from analysis.wrapper768 import summarize
        summarize()
        return
    if args.watch_thinking:
        from analysis.thinking768 import watch
        watch(args.watch_thinking)
        audit(args.key_file, args.forbid)
        return
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
    if args.answer_object and (args.thinking_on or "gemma" in selected):
        parser.error("--answer-object is a separate direct-answer control; cannot combine with thinking or the local model")
    if args.thinking_on and any(m not in THINKING_MODELS for m in selected):
        parser.error("--thinking-on supports only: " + ", ".join(THINKING_MODELS))
    hosted = [m for m in selected if m != "gemma" and m not in CLAUDE]
    claude = [m for m in selected if m in CLAUDE]
    if args.dry_run:
        print(json.dumps({"models": selected, "calls_per_model": len(CASES),
                          "total_calls": len(CASES) * len(selected), "smoke_calls": 0,
                          "protocol": OBJECT_PROTOCOL if args.answer_object else THINKING_PROTOCOL if args.thinking_on else PROTOCOL,
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
    manifest = {"protocol": OBJECT_PROTOCOL if args.answer_object else THINKING_PROTOCOL if args.thinking_on else PROTOCOL,
                "response_format": OBJECT_FORMAT if args.answer_object else FORMAT,
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
        manifest.update(claude_endpoint=ANTHROPIC_API,
                        claude_response_format=CLAUDE_FORMAT if args.answer_object else CLAUDE_INTEGER_FORMAT,
                        output_cap_overrides={m: OUTPUT_CAP[m] for m in claude if m in OUTPUT_CAP},
                        deviations={m: DEVIATIONS[m] for m in claude})
        manifest["versions"]["anthropic"] = anthropic.__version__
    if any(m in CONFIGURED_ALIASES for m in selected):
        manifest["config_sha256"] = hashlib.sha256(CONFIG_SOURCE).hexdigest()
        manifest["configured_model_aliases"] = [m for m in selected if m in CONFIGURED_ALIASES]
        manifest["output_cap_overrides"] = {m: OUTPUT_CAP[m] for m in selected if m in CONFIGURED_ALIASES}
        (folder / "models.toml").write_bytes(CONFIG_SOURCE)
    (folder / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "runner_source.py").write_bytes(source)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [pool.submit(run_model, name, keys["claude" if name in CLAUDE else "hosted"], folder, args.local_url, args.thinking_on, args.answer_object)
                   for name in selected]
        outcomes = {name: future.result() for name, future in zip(selected, futures)}
    print(json.dumps({"completed": outcomes}), flush=True)
    if not all(outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
