"""Bounded DeepInfra image-token accounting probe; not an accuracy benchmark.

DEEPINFRA_API_KEY must be set for live calls. --dry-run makes no network calls.
Prompt-token deltas include image markers and are not exact vision-token counts.
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image

API = "https://api.deepinfra.com/v1/openai"
MODELS = ["mistralai/Mistral-Small-3.2-24B-Instruct-2506",
          "meta-llama/Llama-4-Scout-17B-16E-Instruct"]
SOURCE = "docs/mosaics/tile_probe_seq000_mosaic_n16_tot1536_ts1_bd1_row_al48.png"
PROMPT = "Briefly describe the image in one sentence."
CASES = [("text_before", None), ("square384", 384), ("square768", 768),
         ("square1536", 1536), ("square2048", 2048),
         ("square1536_repeat", 1536), ("text_after", None)]


def image_cases(root):
    source = root / SOURCE
    with Image.open(source) as im:
        image = im.convert("RGB")
    cases = []
    for name, side in CASES:
        if side is None:
            cases.append((name, None, None))
            continue
        resized = image if image.size == (side, side) else image.resize((side, side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        resized.save(buf, format="PNG")
        raw = buf.getvalue()
        info = {"width": side, "height": side, "png_bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest()}
        cases.append((name, info, "data:image/png;base64," + base64.b64encode(raw).decode("ascii")))
    return {"path": SOURCE, "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "width": image.width, "height": image.height}, cases


def request_json(path, key, body=None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(API + path, data=data,
                                 headers={"Authorization": "Bearer " + key,
                                          "Content-Type": "application/json"})
    # No retries: a timeout may already have incurred a charge.
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def summarize(rows):
    if len(rows) != len(CASES):
        return {"complete": False}
    before = rows[0]["response"]["usage"]["prompt_tokens"]
    after = rows[-1]["response"]["usage"]["prompt_tokens"]
    repeated = rows[3]["response"]["usage"]["prompt_tokens"] == rows[5]["response"]["usage"]["prompt_tokens"]
    counts = [{"case": row["case"], "added_prompt_tokens": row["response"]["usage"]["prompt_tokens"] - before}
              for row in rows if row["image"] is not None]
    valid = before == after and repeated and all(c["added_prompt_tokens"] > 0 for c in counts)
    return {"complete": True, "text_baseline_before": before, "text_baseline_after": after,
            "baseline_stable": before == after, "repeat_1536_stable": repeated,
            "accounting_check_passed": valid, "image_cases": counts,
            "exact_vision_token_count_established": False,
            "encoder_resolution_established": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source, cases = image_cases(root)
    print(json.dumps({"models": MODELS, "requests": len(cases) * len(MODELS),
                      "source": source, "cases": [{"name": n, "image": im} for n, im, _ in cases]}, indent=2), flush=True)
    if args.dry_run:
        return
    if args.out is None:
        parser.error("--out is required for live calls")
    # Keep receipts inside the public checkout and never overwrite an earlier run.
    out = args.out.resolve()
    if not out.is_relative_to(root):
        parser.error("--out must be inside this repository")
    if out.exists():
        parser.error("output already exists; choose a new receipt file")
    key = os.environ.get("DEEPINFRA_API_KEY", "").strip()
    if not key:
        parser.error("DEEPINFRA_API_KEY is not set")
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "endpoint": API,
              "source": source, "prompt": PROMPT, "temperature": 0, "max_tokens": 32,
              "image_detail": "omitted: provider documents it as unsupported",
              "interpretation": "Image-minus-text prompt-token deltas include image markers; they do not prove encoder resolution or equal visual information across models.",
              "planned_requests": 14, "attempted_requests": 0, "completed_requests": 0,
              "models": [], "status": "running"}
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    save()
    try:
        catalog = request_json("/models", key)["data"]
        for model in MODELS:
            entry = next((m for m in catalog if m["id"] == model), None)
            if entry is None:
                raise ValueError("selected model missing from current catalog")
            item = {"requested_model": model, "catalog": entry, "calls": []}
            report["models"].append(item)
            for name, info, data_url in cases:
                content = [] if data_url is None else [{"type": "image_url", "image_url": {"url": data_url}}]
                content.append({"type": "text", "text": PROMPT})
                body = {"model": model, "messages": [{"role": "user", "content": content}],
                        "temperature": 0, "max_tokens": 32, "stream": False}
                report["attempted_requests"] += 1
                save()
                start = time.perf_counter()
                response = request_json("/chat/completions", key, body)
                item["calls"].append({"case": name, "image": info,
                                      "latency_s": round(time.perf_counter() - start, 3),
                                      "response": response})
                report["completed_requests"] += 1
                save()
                usage = response.get("usage", {})
                if type(usage.get("prompt_tokens")) is not int:
                    raise ValueError("response did not expose integer prompt_tokens")
                if response.get("model") != model:
                    raise ValueError("response model identity differs from requested model")
                print(json.dumps({"model": model, "case": name, "usage": usage,
                                  "latency_s": item["calls"][-1]["latency_s"]}), flush=True)
            item["summary"] = summarize(item["calls"])
            save()
        report["status"] = "complete"
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        report["status"] = "failed"
        # Do not persist arbitrary provider error bodies or request headers.
        report["error"] = {"type": type(exc).__name__, "http_status": getattr(exc, "code", None)}
        save()
        print(json.dumps({"status": "failed", "error": report["error"]}), flush=True)
        raise SystemExit(1) from None
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save()
    for item in report["models"]:
        print(json.dumps({"model": item["requested_model"], "summary": item["summary"]}), flush=True)


if __name__ == "__main__":
    main()
