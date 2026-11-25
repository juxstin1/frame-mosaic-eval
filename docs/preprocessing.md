# How llama.cpp preprocesses images (verified on a running server)

## Server used
- **Build:** official llama.cpp release **b9555, Vulkan** (`llama-b9555-bin-win-vulkan-x64.zip`). The zip's sha256
  `59c0dbfa…290df5` matches GitHub's digest, and `--version` reports `9555 (daf6bc9f2)`. It is unpacked to `vendor/`,
  which is gitignored.
- **Builds that don't work on this PC:**
  - HIP builds crash on the RX 9070 XT (gfx1201) during warmup with `ROCm error: device kernel image is invalid`.
    The local `llamacpp-prebuilt/tuning/EVAL_HUB.md:62` already records this.
  - The b8683 Vulkan build in `llamacpp-prebuilt` is too old: `unknown projector type: gemma4uv`.
- **Launch:** `runner/server.py`. Exact command lines, `--version` and model sha256s are in `logs/server/*.json`.
  Fixed flags are `-c 32768 -ngl 999 -np 1 -b 2048 -ub 2048 --device Vulkan0 --jinja`, plus
  `--image-max-tokens 1024` for the no-downscale condition.
  - **`-ub 2048` is required.** Gemma 4 encodes an image with non-causal attention in a single ubatch. With the
    default `-ub 512`, a 1024-token image aborts the server with
    `GGML_ASSERT(... n_ubatch >= n_tokens) "non-causal attention requires n_ubatch >= n_tokens"`.
- **Thinking:** Gemma 4 thinks by default under `--jinja`; with `max_tokens` 32, `content` came back empty.
  The study sends `chat_template_kwargs: {"enable_thinking": false}` with every request, which gives `content='53'`
  in 3 completion tokens.

## Resize rule (b9555 source, confirmed by measurement)
Both Gemma 4 (`gemma4uv`) and Qwen3-VL (`qwen3vl_merger`) use `mtmd_image_preprocessor_dyn_size`
(`tools/mtmd/mtmd.cpp` L392, L545). That calls `calc_size_preserved_ratio` (`tools/mtmd/mtmd-image.cpp` L145-169),
which is transformers' "smart_resize":
1. Round each side to the nearest multiple of `align`, where align = patch_size × n_merge.
2. If the area is above `max_pixels`, scale by `sqrt(area/max_pixels)` and floor each side to a multiple of `align`.
3. If the area is below `min_pixels`, scale up and ceil each side to a multiple of `align`.

Here `max_pixels = max_tokens × align²` (`clip-model.h` L137-143), and `--image-max-tokens` overrides `max_tokens`.
Resize is bilinear. Each image also costs **2 marker tokens** (`<|image>` / `<image|>`).

`mosaic/profiles.py` is a port of this function. It matched the server exactly in all 16 probe cases
(`logs/probe_gemma4_default.json`, `logs/probe_gemma4_max1024.json`, and `tests/test_mosaic.py::test_resize_port_matches_server`).

| model | align | default tokens | default max image | source |
|---|---|---|---|---|
| Gemma 4 12B (`gemma4uv`) | 48 (16 × 3) | 40 – 280 | 645,120 px (~803²) | clip.cpp L1400-1415 |
| Qwen3-VL (`qwen3vl_merger`) | 32 (16 × 2) | 8 – 4096 | 4,194,304 px (2048²) | clip.cpp L1426-1442 |

## Measured image cost on Gemma 4 (image tokens = prompt tokens with the image minus text-only prompt tokens)
| input | default limits | `--image-max-tokens 1024` |
|---|---|---|
| one 512² frame | 123 (resized to 528²) | 123 |
| 256² | 51 (resized to 336²) | 51 |
| 768² | 258 | 258 |
| 1024² | 258 (**downscaled to 768²**) | 443 (resized to 1008²) |
| 1536² mosaic | 258 (**downscaled to 768², 2× per side**) | 1026 (native 1536²) |
| 2048² | 258 (**downscaled to 768²**) | 1026 (**downscaled to 1536²**) |
| 4 × 512² separate images in one message | 492 | 492 |

What this means for the study:
- With default limits, **every Gemma image of 768² or more is squashed to 768²**. A 64-tile fixed-total mosaic
  gives each frame 96 px. This is what real users get.
- With `--image-max-tokens 1024`, a 1536² mosaic is kept as-is. Fixed-tile mosaics larger than 1536² are still
  downscaled. `docs/mosaic_budget_table.md` lists the effective tile size and token cost for every condition.
- **Multiple images per message work:** 4 images cost 4 × (121 + 2) tokens.
- **Video:** b9555's server has no video input. `server-common.cpp` handles only `image_url` and `input_audio`.
  Video is out of scope; mosaic vs separate images is the only comparison.

## Model files
Gemma 4 12B it QAT: `<MODELS_DIR>/lmstudio-community/gemma-4-12B-it-QAT-GGUF/`
(`gemma-4-12B-it-QAT-Q4_0.gguf` + `mmproj-gemma-4-12B-it-QAT-BF16.gguf`); sha256 in every `logs/server/*.json`.
Model 2 (Qwen 3.5 9B, `Jackrong/Qwopus3.5-9B-Coder-MTP` coder fine-tune): a preliminary coarse sweep
exists (420 records), not token-matched to Gemma.
