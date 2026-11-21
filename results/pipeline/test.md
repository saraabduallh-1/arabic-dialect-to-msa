# End-to-end pipeline on test (tagger + ALLaM +gen3 + dictionary / Qwen-agreement tier)

Sentences: 291; tokenisation mismatches: 0.

| step | value |
|---|---|
| dialect span detection P / R / F1 (exact boundaries) | 81.2 / 79.2 / 80.2 |
| spans that need a change, found by the tagger | 79.5% of 565 |
| end-to-end exact (found and replaced like the reference) | 37.0% |
| suggested tier: share of found / exact | 35.0% / 71.3% |
| review tier: exact | 33.2% |

Exact match against one reference underestimates by ~37 points (human review of 200 dev answers).
