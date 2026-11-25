"""Dump GGUF key/value metadata (no tensors) with the stdlib only.

python tools/gguf_meta.py path/to/mmproj.gguf [--grep clip]
Long arrays are summarised. Used to record each mmproj's vision preprocessing params.
"""
import argparse
import json
import struct
import sys

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def rstr():
            return f.read(rd("Q")).decode("utf-8", "replace")

        def rval(t):
            if t in SCALARS:
                return rd(SCALARS[t])
            if t == 8:
                return rstr()
            if t == 9:
                et, n = rd("I"), rd("Q")
                vals = [rval(et) for _ in range(n)]
                return vals if n <= 32 else {"len": n, "head": vals[:8]}
            raise ValueError(f"unknown gguf type {t}")

        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, _n_tensors, n_kv = rd("I"), rd("Q"), rd("Q")
        meta = {"gguf.version": version}
        for _ in range(n_kv):
            k = rstr()
            meta[k] = rval(rd("I"))
        return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--grep", default="")
    a = ap.parse_args()
    m = {k: v for k, v in read_meta(a.path).items() if a.grep in k}
    json.dump(m, sys.stdout, indent=1, default=str)
