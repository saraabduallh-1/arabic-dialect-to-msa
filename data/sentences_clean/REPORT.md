# Annotation errors in the original data (PLAN_v2 D5)

## train (2326 rows)

- glued يا: 1 — csv lines 1535
- malformed brackets (left for manual fix): 6 — csv lines 394, 770, 1217, 1624, 1888, 2149
- span equals its MSA: 2 — csv lines 419, 747
- span number on one side only: 1 — csv lines 1844

Corrected copy: `data/sentences_clean/train.csv` (4 rows changed, 0 problems left).

## dev (290 rows)

- span equals its MSA: 1 — csv lines 72

Not modified (user decides on a test_v2).

## test (291 rows)

- span equals its MSA: 2 — csv lines 179, 285
- span number on one side only: 1 — csv lines 255

Not modified (user decides on a test_v2).
