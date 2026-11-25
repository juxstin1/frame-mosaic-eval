# Reasoning-trace audit (23 Oct 2026)

**Data:** 236 written answers from the N=16 position runs:
- reasoning without labels: 96
- reasoning with frame numbers: 96
- snake order at a 1024-token cap: 22
- snake order at a 256-token cap: 22

**How they were read:**
- Four blind LLM readers each got 59 traces, interleaved so every reader saw a mix of runs and tasks.
- With each trace, a reader got the true grid layout and the true content of every frame (dot colours, or which
  frame had the star). They were told nothing about earlier hypotheses.
- For every trace they hard-coded a judgment in a script, with no regex parsing: final answer, frame numbering,
  per-frame claims checked against the truth, first error type, and whether the final answer follows from the
  trace.
- The merged annotations are in `analysis/trace_annotations.jsonl`.

**Spot-check:** I read 10 traces myself, sampled across error types (tile_probe perception ×3, tile_probe none ×1,
moving_dot perception ×2, mapping ×1, moving_dot none ×2, cut off ×1). I agreed with the readers on all 10: final
answer, error type and wrong frames. That is enough to trust the counts, but 10 is a small check.

## What the readers found
| run | task | traces | correct | first error: none | perception | mapping | cut off | per-frame claims (wrong) | stated numbering wrong |
|---|---|---|---|---|---|---|---|---|---|
| reasoning, no labels | moving_dot | 48 | 47 | 46 | 0 | 2 | 0 | 665 (0) | 2/46 |
| reasoning, no labels | tile_probe | 48 | 34 | 33 | 15 | 0 | 0 | 236 (22) | 0/48 |
| reasoning, frame numbers | moving_dot | 48 | 43 | 39 | 9 | 0 | 0 | 641 (30) | 1/5 |
| reasoning, frame numbers | tile_probe | 48 | 38 | 38 | 10 | 0 | 0 | 685 (19) | 0/47 |
| snake, 1024 cap | moving_dot | 22 | 22 | 20 | 0 | 2 | 0 | 352 (0) | 2/22 |
| snake, 256 cap | moving_dot | 22 | 0 | 0 | 0 | 1 | 21 | 187 (0) | 1/22 |

- **The frame numbering in the traces is almost always right.** Of the 190 traces that state a frame-to-cell map, 6
  get one wrong, and none of the 6 leads to a wrong committed answer. Where Gemma writes the arithmetic, it is right
  (e.g. "(2 - 1) * 4 + 3 = 7").
- **Every final answer follows from its own trace.** No trace concludes something its own observations contradict.
- **The scorer agrees with the readers everywhere except the snake 1024-cap run.** That run used the strict prompt,
  the logged score takes the first number, and the readers confirm all 22 concluding numbers are correct (as
  REPORT.md already states).
- **Every wrong answer points later than the truth, never earlier.** Of the 26 wrong committed answers in the
  row-major runs, 20 are one row down (+4), 1 is two rows down, and 5 are one column right. None is up or left.
- **The errors are perception and placement errors:**
  - Gemma says the star or dot is in a different cell than it is, then numbers that cell correctly.
  - Example (framenumbers-013): "the red star is located in the cell between the labels '5' and '9'", then
    "Answer: 9". The truth is 5.
  - Example (framenumbers-016): "a red star located between the two frames" (6 and 7), after which it loops until
    cut off.
- **Frame-number labels introduce colour and visibility mistakes in moving_dot that do not occur without labels:**
  30 wrong per-frame claims with labels against 0 without. Most are "No dot is visible" for early frames (e.g.
  framenumbers-072, -087, -091). Sometimes Gemma still answers correctly.
- **Snake order:** the snake map is stated correctly in nearly every trace. All 21 cut-off answers at the 256-token
  cap had correct colours up to the cut.

## Errors are driven by where the object sits inside its tile
Error rate across all 48 targets per task (not only the errors), by the object's height inside its own tile:

| task | run | top third | middle third | bottom third |
|---|---|---|---|---|
| tile_probe | number only, no labels | 50% (8) | 64% (25) | **100% (15)** |
| tile_probe | reasoning, no labels | 0% (8) | 28% (25) | **47% (15)** |
| tile_probe | number only, frame numbers | 0% (8) | 0% (25) | **53% (15)** |
| tile_probe | reasoning, frame numbers | 0% (8) | 0% (25) | **67% (15)** |
| moving_dot | number only, frame numbers | 0% (11) | 3% (32) | 40% (5) |
| moving_dot | reasoning, frame numbers | 18% (11) | 0% (32) | 60% (5) |

- **With frame numbers on, tile_probe goes wrong only when the star is in the bottom third of its tile.** Every
  one-row-down error with labels has the star in the bottom third (5/5 number only, 7/7 reasoning).
- A star near the bottom of tile k sits just above the top-left label of tile k+4. The readers' quotes show Gemma
  reading position relative to those labels ("between the labels 5 and 9").
- **Without labels the effect is weaker but in the same direction.** The bottom third is still the worst band.
- **Caveats:**
  - Only 8 targets sit in the top third, because answer content is kept out of each tile's top-left corner.
  - moving_dot has only 5 targets in the bottom third.
  - The direction is clear; the exact rates are not.

## What this changes in REPORT.md
- "93 of 96 traces number correctly" came from a regex. The blind read gives 184 of 190 traces with correct stated
  numbering.
- "moving_dot with reasoning made no such errors" holds only without labels. With frame numbers, moving_dot
  reasoning had 5 wrong answers, from mislocated and missed dots.
- The tile-boundary account now rests on the full-sample position table above, not only on the errors.
- Prediction, not yet tested: borders, or putting labels where they do not sit next to the previous tile's bottom
  edge, should remove most of these errors. The existing borders + timestamps run (tile_probe 0.99) is consistent
  with this.
