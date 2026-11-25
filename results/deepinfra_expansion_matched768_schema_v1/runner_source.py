"""Single entry point for matched-768 sweeps, offline summaries and receipt audits.

python -B -m runner.matched_768 --models mistral scout gemma --out results/new_run
python -B -m runner.matched_768 --summarize

No smoke tests or automatic retries. Existing results are never overwritten.
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
          "gemma": "gemma4-12b-qat-q4_0-matched768"}
MODEL_OPTIONS = {alias: {"reasoning_effort": "none"}
                 for alias in ("glmflash", "gemma26", "nemotron", "qwen38")}
SYSTEM = "Respond with a single integer only. Do not explain or show working."
PROTOCOL = "matched768_integer_schema_v1"
FORMAT = {"type": "json_schema", "json_schema": {"name": "integer_answer", "strict": True,
                                                   "schema": {"type": "integer"}}}
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


def read_key(path, required=True):
    value = ""
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            name, sep, content = line.partition("=")
            if sep and name.strip() == "DEEPINFRA_API_KEY":
                value = content.strip().strip("\"'")
    else:
        value = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    if required and not value:
        raise ValueError("Missing DeepInfra credential in private key file or environment")
    return value


def run_model(alias, key, folder, local_url):
    model = MODELS[alias]
    endpoint = (local_url.rstrip("/") + "/v1/chat/completions" if alias == "gemma"
                else "https://api.deepinfra.com/v1/openai/chat/completions")
    with (folder / (alias + "_sweep.jsonl")).open("x", encoding="utf-8") as fh:
        for index, (task, n, labels, seq) in enumerate(CASES, 1):
            png, answer, meta, question = make_case(task, n, labels, seq)
            body = {"model": model, "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
                    {"type": "text", "text": question}]}],
                "temperature": 0, "max_tokens": 32, "stream": False, "response_format": FORMAT}
            if alias == "gemma":
                body["chat_template_kwargs"] = {"enable_thinking": False}
            body.update(MODEL_OPTIONS.get(alias, {}))
            rec = {"protocol": PROTOCOL, "response_format": FORMAT, "model": model,
                   "phase": "sweep", "task": task, "n": n, "labels": labels,
                   "seq_index": seq, "base_seed": 0, "condition": meta["condition"],
                   "source_image_wh": meta["image_wh"], "image_wh": [[768, 768]],
                   "image_sha256": hashlib.sha256(png).hexdigest(), "system_prompt": SYSTEM,
                   "prompt": question, "temperature": 0, "max_tokens": 32,
                   "time_utc": datetime.now(timezone.utc).isoformat()}
            rec["request_options"] = MODEL_OPTIONS.get(alias, {})
            headers = {"Content-Type": "application/json"}
            if alias != "gemma":
                headers["Authorization"] = "Bearer " + key
            req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
            start = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=90) as response:
                    result = json.load(response)
                rec["response"] = result
                rec["latency_s"] = round(time.perf_counter() - start, 3)
                choice = result["choices"][0]
                raw = choice["message"].get("content") or ""
                rec.update(raw_answer=raw, finish_reason=choice.get("finish_reason"), **score(task, answer, raw))
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
                rec["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
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
    key = read_key(key_file, required=False).encode()
    terms = [word.lower().encode() for word in forbidden]
    hits = []
    count = 0
    for path in ROOT.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            count += 1
            if any(t in raw.lower() for t in terms) or (key and key in raw):
                hits.append(path.relative_to(ROOT).as_posix())
    objects = subprocess.run(["git", "cat-file", "--batch", "--batch-all-objects"], cwd=ROOT,
                             input=b"", capture_output=True, check=True).stdout
    object_hit = any(t in objects.lower() for t in terms) or bool(key and key in objects)
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
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--summarize", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--forbid", nargs="*", default=[])
    args = parser.parse_args()
    if args.summarize:
        from analysis.matched768 import main as summarize
        summarize()
        return
    if args.audit:
        audit(args.key_file, args.forbid)
        return
    selected = list(dict.fromkeys(args.models))
    if args.dry_run:
        print(json.dumps({"models": selected, "calls_per_model": len(CASES),
                          "total_calls": len(CASES) * len(selected), "smoke_calls": 0,
                          "protocol": PROTOCOL, "input_size": [768, 768]}))
        return
    if args.out is None:
        parser.error("--out is required for a live sweep; choose a new directory")
    folder = args.out.resolve()
    if not folder.is_relative_to(ROOT / "results"):
        parser.error("--out must be a new directory under results/")
    if folder.exists():
        parser.error("output already exists; results will not be overwritten")
    key = read_key(args.key_file) if any(m != "gemma" for m in selected) else ""
    local = None
    if "gemma" in selected:
        local = json.loads(args.local_receipt.read_text())
        if local["alias"] != MODELS["gemma"]:
            parser.error("local server receipt alias differs from selected Gemma model")
        with urllib.request.urlopen(args.local_url.rstrip("/") + "/v1/models", timeout=10) as response:
            served = json.load(response)
        if MODELS["gemma"] not in [m["id"] for m in served["data"]]:
            parser.error("verified Gemma alias is not served at --local-url")
    folder.mkdir(parents=True)
    source = Path(__file__).read_bytes()
    catalog = {}
    if any(m != "gemma" for m in selected):
        with urllib.request.urlopen("https://api.deepinfra.com/v1/openai/models", timeout=20) as response:
            current = {item["id"]: item for item in json.load(response)["data"]}
        catalog = {name: current[MODELS[name]] for name in selected if name != "gemma"}
    manifest = {"protocol": PROTOCOL, "response_format": FORMAT,
                "models": {m: MODELS[m] for m in selected}, "system_prompt": SYSTEM,
                "request_options": {m: MODEL_OPTIONS.get(m, {}) for m in selected},
                "smoke_per_model": 0, "sweep_per_model": 150, "seed": 0,
                "tasks": ["moving_dot", "tile_probe", "count_events"], "n": [4, 9, 16, 36, 64],
                "seqs": list(range(5)), "labels": [False, True], "output_cap": 32,
                "temperature": 0, "input_size": [768, 768], "server": local, "provider_catalog": catalog,
                "runner_sha256": hashlib.sha256(source).hexdigest(),
                "versions": {"python": platform.python_version(), "pillow": PIL.__version__, "numpy": numpy.__version__}}
    (folder / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (folder / "runner_source.py").write_bytes(source)
    with ThreadPoolExecutor(max_workers=len(selected)) as pool:
        futures = [pool.submit(run_model, name, key, folder, args.local_url) for name in selected]
        outcomes = {name: future.result() for name, future in zip(selected, futures)}
    print(json.dumps({"completed": outcomes}), flush=True)
    if not all(outcomes.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
