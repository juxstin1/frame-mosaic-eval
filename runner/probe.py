"""Server sanity probe: what is loaded, does an image request work, and what do images really cost in tokens.

python -m runner.probe --url http://127.0.0.1:8090 --profile gemma4uv [--max-tokens-setting 1024] --out logs/probe_x.json
Image tokens are measured as prompt_tokens(with images) - prompt_tokens(text only), and compared against
mosaic.profiles.vision_plan (port of llama.cpp smart_resize). Mismatch => our downscale model is wrong.
"""
import argparse
import json
from pathlib import Path

from PIL import Image

from gen.common import rng_for
from gen.tasks import counter
from mosaic.builder import Condition, build
from mosaic.profiles import vision_plan
from runner.client import chat, get

Q = "What number is shown in the image? Answer with the number only."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--profile", default="gemma4uv")
    ap.add_argument("--max-tokens-setting", type=int, default=None, help="the server's --image-max-tokens, if set")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rep = {"models": get(a.url, "/v1/models")}
    try:
        props = get(a.url, "/props")
        rep["props"] = {k: props.get(k) for k in ("model_path", "modalities", "build_info", "total_slots")}
        rep["props"]["n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
    except Exception as e:  # noqa: BLE001
        rep["props_error"] = str(e)

    frames, ans, _ = counter(4, rng_for("counter", 4, 0, 0), 512)
    base = chat(a.url, Q)["usage"]["prompt_tokens"]
    rep["text_only_prompt_tokens"] = base

    one = chat(a.url, Q, [frames[0]])
    rep["single_frame"] = {"truth": ans["values"][0], "answer": one["choices"][0]["message"].get("content"),
                           "reasoning": one["choices"][0]["message"].get("reasoning_content"),
                           "usage": one["usage"], "latency_s": one["latency_s"]}

    cases = [("frame512", [frames[0]])] + [
        (f"square{s}", [frames[0].resize((s, s))]) for s in (256, 768, 1024, 1536, 2048)]
    # 4 separate images in one message, and a 16-tile fixed-total mosaic
    cases.append(("4_separate_512", frames))
    f16, _, _ = counter(16, rng_for("counter", 16, 0, 0), 512)
    cases.append(("mosaic16_tot1536", build(f16, Condition(n=16, align=48 if a.profile == "gemma4uv" else 32))[0]))
    rows = []
    for name, imgs in cases:
        r = chat(a.url, Q, imgs, max_tokens=8)
        measured = r["usage"]["prompt_tokens"] - base
        pred = [vision_plan(*im.size, a.profile, a.max_tokens_setting) for im in imgs]
        rows.append({"case": name, "sizes": [list(im.size) for im in imgs], "measured_image_tokens": measured,
                     "predicted_vision_tokens": sum(p["tokens"] for p in pred),
                     "predicted_encoder_wh": [p["encoder_wh"] for p in pred],
                     "overhead_per_image": round((measured - sum(p["tokens"] for p in pred)) / len(imgs), 2),
                     "latency_s": r["latency_s"], "answer": r["choices"][0]["message"].get("content")})
    rep["token_cost"] = rows
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "token_cost"}, indent=1))
    for r in rows:
        print(f"{r['case']:18s} measured={r['measured_image_tokens']:5d} predicted={r['predicted_vision_tokens']:5d} "
              f"overhead/img={r['overhead_per_image']:5} enc={r['predicted_encoder_wh'][0]} t={r['latency_s']}s ans={r['answer']!r}")


if __name__ == "__main__":
    main()
