# DeepInfra image-token preflight

This is an input-accounting probe, not an accuracy benchmark. Do not mix its
responses into the Gemma study's accuracy tables.

## Model choice

Start with Mistral Small 3.2 24B, then Llama 4 Scout, to add two model families
beyond the local Gemma/Qwen experiments. Mistral is the first candidate because
it has the lower published token price. Latency is unmeasured until a live run
succeeds; both run on the provider's hardware, with no local GPU requirement.

| Exact model ID | Model license | USD per million input/output tokens |
| --- | --- | --- |
| `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Apache 2.0 | 0.075 / 0.20 |
| `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Llama 4 Community License | 0.10 / 0.30 |

Prices checked against the live catalog and model pages on 2026-11-24.
Sources: [Mistral provider page](https://deepinfra.com/mistralai/Mistral-Small-3.2-24B-Instruct-2506),
[Mistral model card](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506),
[Scout provider page and license](https://deepinfra.com/meta-llama/Llama-4-Scout-17B-16E-Instruct).

## Run

Use the existing Python requirements. Set `DEEPINFRA_API_KEY` in the launching
process's environment, outside any repository file. Never put credentials in
command arguments, receipts, or source control.

```powershell
python -B -m runner.deepinfra_probe --dry-run
python -B -m runner.deepinfra_probe --out logs/deepinfra_image_tokens_probe_new.json
```

The dry run does not access the network. A live run makes one authenticated
catalog request and at most 14 inference requests. It stops on the first error,
does not retry, preserves completed responses, and refuses to overwrite receipts.

## Inputs and acceptance

The same existing 1536-square N=16 tile-probe mosaic is used for both models.
Each model receives a text-only baseline, the mosaic resized to 384, 768, 1536,
and 2048 pixels square, a repeat of the identical 1536-square PNG, then another
text-only baseline. The 2048 image is an upscale of the original, so it adds no
source detail. The prompt and generation settings stay fixed: temperature 0,
maximum 32 output tokens. SHA-256 hashes identify all input PNGs.

Receipts preserve the returned model ID, catalog metadata, complete response
usage fields, answers, and per-request latency. The accounting check requires
stable before/after text counts, stable repeated-image counts, and positive
image-minus-text prompt-token deltas. This establishes repeatable accounting for
the tested image only, not benchmark accuracy or matched visual information.

The delta includes image markers and any image-dependent prompt overhead. It
must be reported as **added prompt tokens**, not exact vision-encoder tokens.
Preserve any separately reported image-token field without assuming that it
reveals encoder resolution. A plateau across resolutions is evidence about
token accounting, not proof of a particular resize algorithm.

[DeepInfra's vision documentation](https://docs.deepinfra.com/chat/vision) says
images are automatically rescaled, tokens appear in `usage.prompt_tokens`, and
the image `detail` parameter is unsupported. The probe therefore omits it.
Matching deltas across architectures does not guarantee equivalent image detail.
Until preprocessing is established, any accuracy comparison must be described
as a comparison of hosted model configurations, not an isolated model effect.
Square-image results do not establish behavior for rectangular mosaics.

## Current receipt

`logs/deepinfra_image_tokens_probe_20261124.json` records an HTTP 401 during the
authenticated catalog request: 0 attempted and 0 completed inference calls.

After replacing the credential, `logs/deepinfra_image_tokens_probe_retry01.json`
completed all 14 inference calls at 2026-11-25T00:51:12.920076+00:00 (November 24
in America/Halifax). Both returned model IDs exactly match their requested IDs.

| Square input size | Mistral added prompt tokens | Scout added prompt tokens |
| --- | ---: | ---: |
| 384 | 210 | 727 |
| 768 | 812 | 1452 |
| 1536 | 3080 | 2467 |
| 2048 | 3080 | 2467 |
| 1536 repeat | 3080 | 2467 |

The text-only baseline was 13 before and after for Mistral, and 19 before and
after for Scout. Both models passed the accounting repeatability check. Neither
returned a separate image-token count: `prompt_tokens_details` was null in every
response. The observed plateau from 1536 to 2048 does not establish exact internal
image dimensions or matched visual information between the two models.

Provider-reported estimated cost summed to approximately USD 0.001842875 for
14 inference calls. Median image-request latency across five calls per model
was 2.287 seconds for Mistral and 1.091 seconds for Scout. These are descriptive
timings from one sequential probe, not controlled throughput measurements.
Mistral hit the 32-output-token cap in 6 of 7 requests (all five image calls);
these descriptions are not scored as benchmark answers. Scout finished normally
on all seven calls.

No full benchmark sweep has been started. The next experiment is a comparison
of hosted configurations using identical task inputs, with these accounting
differences reported alongside accuracy. Do not label it token-matched. Any
benchmark answer cap must be checked for truncation before collecting a sweep.
