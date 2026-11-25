"""Minimal OpenAI-compatible client for llama-server with image inputs (stdlib only)."""
import base64
import io
import json
import time
import urllib.request
from collections.abc import Sequence

from PIL import Image


def png_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=30) as r:
        return json.loads(r.read())


def chat(url: str, prompt: str, images: Sequence[Image.Image] = (), max_tokens: int = 32, seed: int = 0,
         timeout: int = 600, thinking: bool = False) -> dict:
    """One user turn: images first, then the text. temperature 0, fixed seed, thinking off by default
    (Gemma 4 thinks by default under --jinja). Returns response json + latency_s."""
    content = [{"type": "image_url", "image_url": {"url": png_data_url(im)}} for im in images]
    content.append({"type": "text", "text": prompt})
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0, "seed": seed,
            "max_tokens": max_tokens, "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    out["latency_s"] = round(time.perf_counter() - t0, 3)
    return out
