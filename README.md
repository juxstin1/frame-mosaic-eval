# How much time fits in an image?

Capacity limits of frame mosaics for VLMs: how many video frames a vision-language model can reliably
read when packed into one mosaic image, and where it breaks. Runs locally against llama.cpp `llama-server`.

## Reproduce the results

Use Python 3.12. From the repository root, run:

```sh
python -m pip install -r requirements.txt
python -m analysis.make_all
python -m analysis.position
```

These commands read the included `results/*.jsonl` logs and regenerate `analysis/effects.md`,
`analysis/position.md`, and the six figures in `figures/`. They do not call a model.
Model weights, a GPU, a running server, `data/`, and `vendor/` are not needed for this analysis.

## Layout
- `gen/` — synthetic frame sequences with exact ground truth (`python -m gen.make`, `python -m gen.viewer`)
- `tests/` — invariants + seed reproducibility (`python -m pytest tests -q`)
- `data/` — generated frames (gitignored; regenerate from seed)
- `docs/sheets/` — contact sheets for eyeballing

## Tasks (frames are 512x512, frame numbers 1-based)
| task | what changes | ground truth |
|---|---|---|
| moving_dot | dot moves along a Bezier, turns blue->red at one frame and stays red | change frame, per-frame position + colour |
| counter | big two-digit number, ~30% of frames repeat the previous | value per frame |
| event_order | 3-5 distinct shapes, each in exactly one frame | order + frame of each |
| count_events | K yellow-star flashes, never in adjacent frames, K <= min(ceil(N/2), 10) | K + flash frames |
| tile_probe | static grey scene with +-3 px jitter; one frame adds a red star | probe frame |

Seeding: each sequence's RNG is `SeedSequence([base_seed, sha256(task)[:4], N, index])`, so any sequence
regenerates identically and independently of what else was generated.

## Mosaic knobs (`mosaic.builder.Condition`)
N in {4,6,9,12,16,20,25,36,49,64} (grids 2x2 ... 8x8); budget `fixed_total` (tile = floor(sqrt(1536^2/N)) to a
patch multiple, so 77-100% of the 1536^2 area) or `fixed_tile` (384 px); timestamps (frame number top-left; the
generators keep answer content out of the top-left 24%); borders; row-major or snake order; patch-aligned tiles or a
half-patch margin so every tile edge falls mid-patch; one mosaic or separate images (same tile pixels).

## Models
1. Gemma 4 12B it QAT (Q4_0 GGUF, lmstudio-community, in `~/.lmstudio/models`), run twice: default image-token
   limits and `--image-max-tokens 1024`.
2. Qwen 3.5 9B (`Jackrong/Qwopus3.5-9B-Coder-MTP`, a coder fine-tune): a preliminary coarse sweep
   exists (420 records), not token-matched to Gemma.

## Related work
- IG-VLM (arXiv 2403.18406): image grids of 4-20 frames, 6 best on 2024 models, aggregate scores only.
- Grid2Matrix (2026): exhaustive grid readout collapses early; errors depend on cell alignment with vision patches.
- Multi-image position bias (2025): position effects across separate images.
- DeepSeek-OCR (arXiv 2510.18234): accuracy vs vision-token compression for text.

Our contribution: the same kind of capacity curve for *time* in an image, with per-position heatmaps.
Comparing one mosaic with separate images is future work.

## Matched-768 answer-format comparison

See the [indexed results and token audit](analysis/matched768_results.md),
[HTML results page](docs/matched768_results.html), and [run index](analysis/run_index.json).
The page compares bare integer and `{"answer": N}` formats at a 32-token cap.
It includes the three comparison charts in `figures/`; the earlier `matched768.md`
and `wrapper768.md` summaries cover narrower historical subsets.

Regenerate this page and its charts without model calls:

```sh
python -B -m analysis.results_page
```

The generator validates selected result hashes and archived code/config snapshots,
checks identical cases and input hashes, and rechecks scores against raw answers.
Only `table_cells.selected_path` contributes to the main tables. Failed attempts,
retries, and the separate 4096-token thinking round are not pooled into them.

A row marked **thinking unconfirmed** has median completion tokens above 9,
positive reported reasoning tokens/text, or an unavailable reasoning counter.
The length threshold is a screening rule, not proof of hidden thinking. Repeated
digits and the wrapper itself also consume tokens. Explicit provider-reported zero
is distinguished from missing usage fields; no table claims universally disabled thinking.

### Historical run provenance

The Gemma 12B bare-integer result (91/150) needs **both**
[`runner_source.py`](results/gemma_matched768_schema_v1/runner_source.py) and
[`launch_source.py`](results/gemma_matched768_schema_v1/launch_source.py)
to reconstruct what ran: the launcher overrides settings before invoking the runner.
The runner hash is in the original manifest. The launcher hash was recorded later,
at indexing; it is not an original launch-time attestation. The original server launch
command was not saved, although settings and binary/model hashes were retained.

Two server logs had private absolute paths redacted **after** the wrapper run:
`logs/server/gemma_object_20261124-223126.json` and
`logs/server/gemma_object_20261124-223126.log.err`. Their current hashes and redaction
scope are recorded under `artifact_redactions` in the run index. They are redacted
artifacts, not untouched originals. No result JSONL records were changed.

The active sweep entry point is `runner/matched_768.py`; new DeepInfra aliases use
`runner/models.toml`. Existing built-in aliases retain the runner's archived defaults;
the config file does not override those built-ins. Per-run snapshots document the exact
code and settings used. Provider preprocessing and reasoning controls differ, so this
is an exploratory comparison of deployed configurations, not matched reasoning compute.
