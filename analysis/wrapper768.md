# Mistral answer-wrapper control

150 identical cases per run: three tasks, N=4/9/16/36/64, labels off/on, five sequences per cell.
Source hashes, all case IDs, image hashes, task/system prompts, ground truths and 32-token caps were verified.
Mistral retained temperature=0 and the same model ID and request options. Only the schema
changed from a bare integer to Haiku's exact {answer: integer} object; scoring extracts that field.
Haiku uses a different provider and omits temperature; its request explicitly disables thinking.

| Run | Overall | Labels off | Labels on | Cutoffs | Invalid answers | Mean output tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral bare integer | 88/150 (58.7%) | 31/75 (41.3%) | 57/75 (76.0%) | 0 | 0 | 2.39 |
| Mistral answer object | 21/150 (14.0%) | 8/75 (10.7%) | 13/75 (17.3%) | 114 | 114 | 28.26 |
| Haiku answer object | 124/150 (82.7%) | 56/75 (74.7%) | 68/75 (90.7%) | 0 | 0 | 9.24 |

## Paired Mistral changes

- Wrong to correct: 3
- Correct to wrong: 70
- Correct in both: 18
- Wrong in both: 59
- Parsed answer changed: 126
- Net accuracy change: -67/150 (-44.67 percentage points)

## By task and frame-number labels

| Task | Labels | Bare Mistral | Wrapped Mistral | Wrapped Haiku |
| --- | --- | ---: | ---: | ---: |
| moving_dot | off | 11/25 (44.0%) | 0/25 (0.0%) | 16/25 (64.0%) |
| moving_dot | on | 14/25 (56.0%) | 7/25 (28.0%) | 22/25 (88.0%) |
| tile_probe | off | 6/25 (24.0%) | 4/25 (16.0%) | 19/25 (76.0%) |
| tile_probe | on | 21/25 (84.0%) | 4/25 (16.0%) | 24/25 (96.0%) |
| count_events | off | 14/25 (56.0%) | 4/25 (16.0%) | 21/25 (84.0%) |
| count_events | on | 22/25 (88.0%) | 2/25 (8.0%) | 22/25 (88.0%) |

The wrapped Mistral run hit the cap 114/150 times; 114 of those outputs
contained only an opening brace and whitespace. These are retained as failures.
When output failures dominate, this control cannot isolate a wrapper benefit to task accuracy.

Provider-estimated cost of the new Mistral control: USD 0.01112880.

This is one exploratory run per configuration. It measures the observed wrapper difference
for this Mistral deployment; it cannot establish that format explains all differences between
models or isolate Haiku's visual perception. Sampling and provider changes are not controlled by
temperature=0 alone. Low output-token counts alone do not prove absence of internal reasoning.

## Reproduce

Offline summary (no model calls):

```powershell
python -B -m runner.matched_768 --summarize-wrapper
```

New paid control (use a fresh directory):

```powershell
python -B -u -m runner.matched_768 --models mistral --answer-object --out results/new_mistral_object_run
```

Inputs:

- `results/deepinfra_matched768_schema_v1/mistral_sweep.jsonl`
- `results/deepinfra_mistral_matched768_object_v1/mistral_sweep.jsonl`
- `results/claude_matched768_haiku_object_v1/haiku55_sweep.jsonl`
