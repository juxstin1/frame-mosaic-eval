"""Start/stop the study's llama-server and record exactly how it was launched.

python -m runner.server start --model gemma4 --label gemma4_default
python -m runner.server start --model gemma4 --label gemma4_max1024 --image-max-tokens 1024
python -m runner.server stop
Each start writes logs/server/<label>_<ts>.json (command, --version, file sha256s) and a .log with server output.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Override model storage with MODELS_DIR and build locations with the variables below.
BUILDS = {  # HIP builds crash on the RX 9070 XT (gfx1201): "device kernel image is invalid"
    "b9555-vulkan": (Path(os.environ.get("LLAMA_B9555_VULKAN_DIR", REPO_ROOT / "vendor" / "llama-b9555-vulkan")), "Vulkan0"),  # official release zip, sha256 59c0dbfa...
    "b8683-vulkan": (Path(os.environ.get("LLAMA_B8683_VULKAN_DIR", REPO_ROOT / "vendor" / "llama-b8683-vulkan")), "Vulkan0"),  # too old: no gemma4uv
    "b9555-hip": (Path(os.environ.get("LLAMA_B9555_HIP_DIR", REPO_ROOT / "vendor" / "llama-b9555-hip")), "ROCm1"),
}
LMS = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))
# name -> (model, mmproj, provenance). sha256 of both files is recorded in every launch log.
MODELS = {
    "gemma4": (LMS / "lmstudio-community/gemma-4-12B-it-QAT-GGUF/gemma-4-12B-it-QAT-Q4_0.gguf",
               LMS / "lmstudio-community/gemma-4-12B-it-QAT-GGUF/mmproj-gemma-4-12B-it-QAT-BF16.gguf",
               "Gemma 4 12B it QAT Q4_0 GGUF + BF16 mmproj, lmstudio-community"),
    # only vision-capable Qwen 3.5 on disk; a coder fine-tune of Qwopus3.5-9B-v3.5 (qwen3vl_merger projector)
    "qwen35": (LMS / "Jackrong/Qwopus3.5-9B-Coder-MTP-GGUF/Qwopus3.5-9B-Coder-MTP-Q4_K_M.gguf",
               LMS / "Jackrong/Qwopus3.5-9B-Coder-MTP-GGUF/mmproj-F32.gguf",
               "Qwopus3.5 9B Coder MTP Q4_K_M + F32 mmproj, Jackrong"),
}
LOG_DIR = Path("logs/server")
STATE = LOG_DIR / "current.json"
HASH_CACHE = LOG_DIR / "sha256_cache.json"


def sha256(path: str) -> str:
    cache = json.loads(HASH_CACHE.read_text()) if HASH_CACHE.exists() else {}
    st = Path(path).stat()
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        cache[key] = h.hexdigest()
        HASH_CACHE.write_text(json.dumps(cache, indent=1))
    return cache[key]


def healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start(a):
    if STATE.exists():
        sys.exit(f"server already recorded as running: {STATE.read_text()}  (run stop first)")
    llama_dir, device = BUILDS[a.build]
    exe = llama_dir / "llama-server.exe"
    model, mmproj, provenance = MODELS[a.model]
    model, mmproj = str(model), str(mmproj)
    missing = [f for f in (model, mmproj) if not Path(f).exists()]
    if missing:
        sys.exit(f"{a.model}: missing {missing}")
    got = (sha256(model), sha256(mmproj))
    cmd = [str(exe), "-m", model, "--mmproj", mmproj, "-c", str(a.ctx), "-ngl", "999", "-np", "1", "-b", "2048", "-ub", "2048", "--device", device,
           "--host", "127.0.0.1", "--port", str(a.port), "--jinja"]
    if a.image_max_tokens:
        cmd += ["--image-max-tokens", str(a.image_max_tokens)]
    version = subprocess.run([str(exe), "--version"], capture_output=True, text=True).stderr.strip().splitlines()
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = LOG_DIR / f"{a.label}_{ts}"
    rec = {"label": a.label, "build": a.build, "started": ts, "port": a.port, "cmd": cmd, "cmdline": subprocess.list2cmdline(cmd),
           "version": [l for l in version if "version" in l or "built" in l],
           "model": a.model, "provenance": provenance, "model_sha256": got[0], "mmproj_sha256": got[1],
           "log": str(stem) + ".log"}
    logf = open(str(stem) + ".log", "w")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(llama_dir), creationflags=flags)
    rec["pid"] = p.pid
    for _ in range(a.wait):
        if healthy(a.port):
            break
        if p.poll() is not None:
            sys.exit(f"llama-server exited with {p.returncode}; see {rec['log']}")
        time.sleep(1)
    else:
        sys.exit(f"not healthy after {a.wait}s; see {rec['log']}")
    Path(str(stem) + ".json").write_text(json.dumps(rec, indent=1))
    STATE.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec[k] for k in ("label", "pid", "port", "version", "cmdline")}, indent=1))


def stop(_a):
    if not STATE.exists():
        sys.exit("no recorded server")
    rec = json.loads(STATE.read_text())
    subprocess.run(["taskkill", "/PID", str(rec["pid"]), "/T", "/F"], capture_output=True)
    STATE.unlink()
    print(f"stopped pid {rec['pid']} ({rec['label']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--label", required=True)
    s.add_argument("--model", required=True, choices=list(MODELS))
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--ctx", type=int, default=32768)
    s.add_argument("--build", default="b9555-vulkan", choices=list(BUILDS))
    s.add_argument("--image-max-tokens", type=int, default=None)
    s.add_argument("--wait", type=int, default=240)
    sub.add_parser("stop")
    a = ap.parse_args()
    {"start": start, "stop": stop}[a.cmd](a)
