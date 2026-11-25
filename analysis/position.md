# Controlled-position run (N=16, 4x4, default image limits, no borders) — 192 calls

## Error rate per target position (no timestamps, 5 sequences per position)

| task | pos 1 | pos 2 | pos 3 | pos 4 | pos 5 | pos 6 | pos 7 | pos 8 | pos 9 | pos 10 | pos 11 | pos 12 | pos 13 | pos 14 | pos 15 | pos 16 | overall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moving_dot | 0.8 | 0.0 | 0.0 | 0.0 | 0.8 | 0.0 | 0.8 | 1.0 | 0.0 | 0.8 | 0.4 | 1.0 | 0.0 | 1.0 | 1.0 | 0.0 | 0.47 (38/80 wrong) |
| tile_probe | 0.6 | 0.8 | 1.0 | 1.0 | 0.8 | 0.6 | 1.0 | 1.0 | 0.4 | 0.8 | 0.6 | 0.6 | 0.2 | 0.8 | 0.8 | 0.0 | 0.69 (55/80 wrong) |

## Control: timestamps on (1 sequence per position)

| task | accuracy | wrong positions |
|---|---|---|
| moving_dot | 0.81 (16 calls) | [2, 7, 9] |
| tile_probe | 0.88 (16 calls) | [9, 11] |

## Error vectors of wrong answers (no timestamps)

| task | wrong | row slip | column slip | ±1 frame across row break | diagonal | scattered | invalid |
|---|---|---|---|---|---|---|---|
| moving_dot | 38 | 10 (26%) | 7 (18%) | 5 (13%) | 0 (0%) | 13 (34%) | 3 (8%) |
| tile_probe | 55 | 13 (24%) | 7 (13%) | 3 (5%) | 8 (15%) | 24 (44%) | 0 (0%) |

## Signed frame difference of wrong answers (answered − true)

| task | differences |
|---|---|
| moving_dot | -4: 3, -2: 7, -1: 5, +1: 7, +2: 4, +4: 7, +5: 1, +9: 1 |
| tile_probe | -2: 4, -1: 4, +1: 6, +2: 6, +3: 9, +4: 13, +5: 3, +6: 1, +7: 1, +8: 6, +9: 2 |

## moving_dot confound check (no timestamps)

- Answers naming a red frame that is not the first red frame: **20/77 (26%)**; of these, the last frame (16): 0.
- Answers naming a blue frame (before the change): 15/77 (19%).
- Error rate, change in positions 1-8: 0.38; positions 9-16: 0.53.
- Position 1 has no blue frame (dot red from the start).

## Where wrong answers land: first cell of a row (frames 1, 5, 9, 13)

Chance share = 4 of the 15 other frames (27%) if wrong answers were spread evenly.

| task | wrong answers | on a row's first cell | same row as truth | top 3 answered frames (all answers) |
|---|---|---|---|---|
| moving_dot | 35 | 25 (71%) | 14 (40%) | 13 (18x), 9 (17x), 6 (6x) |
| tile_probe | 55 | 31 (56%) | 14 (25%) | 13 (24x), 11 (11x), 12 (10x) |

## Snake order vs row-major (N=16, no timestamps, same 3 sequences per position)

In snake order the reading-order row starts 5 and 13 sit on the RIGHT edge, and frames 8 and 16 sit on the left edge, so the two explanations for the row-start snap make different predictions.

| task | order | accuracy | wrong | reading-order row start AND left edge | reading-order row start (not left edge) | left-edge cell (not a row start) | row-major number of the true cell | other | invalid |
|---|---|---|---|---|---|---|---|---|---|
| moving_dot | row-major | 0.54 (48) | 22 | 13 (59%) | 0 (0%) | 0 (0%) | 0 (0%) | 7 (32%) | 2 (9%) |
| moving_dot | snake | 0.21 (48) | 38 | 0 (0%) | 10 (26%) | 0 (0%) | 2 (5%) | 4 (11%) | 22 (58%) |
| tile_probe | row-major | 0.27 (48) | 35 | 17 (49%) | 0 (0%) | 0 (0%) | 0 (0%) | 18 (51%) | 0 (0%) |
| tile_probe | snake | 0.17 (48) | 40 | 2 (5%) | 16 (40%) | 1 (2%) | 8 (20%) | 13 (32%) | 0 (0%) |

| task | most common answers, snake (all answers) | most common answers, row-major (same sequences) |
|---|---|---|
| moving_dot | None (22x), 13 (15x), 11 (6x), 10 (2x) | 9 (10x), 13 (9x), 6 (4x), 10 (4x) |
| tile_probe | 13 (18x), 11 (7x), 6 (5x), 12 (5x) | 13 (14x), 11 (8x), 12 (6x), 5 (5x) |

| task | pairs where both layouts answered wrong (parsed) | same number in both | snake answer = frame drawn where the row-major answer's cell is |
|---|---|---|---|
| moving_dot | 10 | 6 (60%) | 0 (0%) |
| tile_probe | 31 | 20 (65%) | 0 (0%) |

Snake answers cut off by the 16-token answer cap (model started explaining instead of answering): 22 (moving_dot, target positions [1, 2, 3, 4, 5, 6, 7, 8]). Counted as wrong. moving_dot snake accuracy among parsed answers: 10/26.

## Frame numbers x reasoning, and row/column-only labels (N=16, same 3 sequences per position)

Latency is inflated for the reasoning-without-labels run (a second agent shared the one-slot server for part of it); output tokens are unaffected.

| task | condition | calls | accuracy (95% CI) | unparsed | cut off | mean output tokens | median latency s |
|---|---|---|---|---|---|---|---|
| moving_dot | number only, no labels | 48 | 0.54 (0.40-0.67) | 2 | 2 | 3 | 0.52 |
| moving_dot | reasoning, no labels | 48 | 0.98 (0.89-1.00) | 1 | 1 | 336 | 5.11 |
| moving_dot | number only, frame numbers | 48 | 0.94 (0.83-0.98) | 1 | 1 | 3 | 0.54 |
| moving_dot | reasoning, frame numbers | 48 | 0.90 (0.78-0.95) | 1 | 1 | 273 | 4.58 |
| moving_dot | number only, column labels (c1-c4) | 48 | 0.42 (0.29-0.56) | 3 | 3 | 3 | 0.57 |
| moving_dot | number only, row labels (r1-r4) | 48 | 0.46 (0.33-0.60) | 3 | 3 | 3 | 0.56 |
| tile_probe | number only, no labels | 48 | 0.27 (0.17-0.41) | 0 | 0 | 3 | 0.57 |
| tile_probe | reasoning, no labels | 48 | 0.71 (0.57-0.82) | 1 | 1 | 349 | 5.55 |
| tile_probe | number only, frame numbers | 48 | 0.83 (0.70-0.91) | 0 | 0 | 2 | 0.54 |
| tile_probe | reasoning, frame numbers | 48 | 0.79 (0.66-0.88) | 1 | 1 | 266 | 3.69 |

Paired comparisons (exact McNemar; 'x vs y' = sequences only the second / only the first got right):

- moving_dot: **reasoning, no labels** vs number only, no labels: 21 vs 0, p=9.54e-07
- moving_dot: **number only, frame numbers** vs number only, no labels: 19 vs 0, p=3.81e-06
- moving_dot: **number only, frame numbers** vs reasoning, no labels: 1 vs 3, p=0.625
- moving_dot: **reasoning, frame numbers** vs number only, frame numbers: 1 vs 3, p=0.625
- moving_dot: **number only, column labels (c1-c4)** vs number only, no labels: 1 vs 7, p=0.0703
- moving_dot: **number only, row labels (r1-r4)** vs number only, no labels: 3 vs 7, p=0.344
- tile_probe: **reasoning, no labels** vs number only, no labels: 22 vs 1, p=5.72e-06
- tile_probe: **number only, frame numbers** vs number only, no labels: 27 vs 0, p=1.49e-08
- tile_probe: **number only, frame numbers** vs reasoning, no labels: 9 vs 3, p=0.146
- tile_probe: **reasoning, frame numbers** vs number only, frame numbers: 0 vs 2, p=0.5

| moving_dot condition | wrong (parsed) | right row, wrong column | right column, wrong row | both wrong | answer = first cell of the true row |
|---|---|---|---|---|---|
| number only, no labels | 20 | 6 | 8 | 6 | 5 |
| number only, column labels (c1-c4) | 25 | 9 | 5 | 11 | 5 |
| number only, row labels (r1-r4) | 23 | 8 | 6 | 9 | 7 |
| number only, frame numbers | 2 | 0 | 1 | 1 | 0 |

### Reasoning traces (no labels): numbering vs localisation

For each reasoning answer we read the trace's own row headings and the frame numbers listed under them, and check them against row-major numbering (row k holds frames 4k-3..4k).

| task | answers | trace numbering consistent | wrong answers | wrong with consistent numbering | wrong answer's offset from truth (rows, cols) |
|---|---|---|---|---|---|
| moving_dot | 48 | 46 | 0 | 0 |  |
| tile_probe | 48 | 47 | 13 | 13 | (+1,+0) x9, (+0,+1) x3, (+2,+0) x1 |
