# End-to-end pipeline on dev (tagger + ALLaM +gen3 + dictionary / Qwen-agreement tier)

Sentences: 290; tokenisation mismatches: 0.

| step | value |
|---|---|
| dialect span detection P / R / F1 (exact boundaries) | 80.1 / 80.6 / 80.4 |
| spans that need a change, found by the tagger | 81.4% of 555 |
| end-to-end exact (found and replaced like the reference) | 37.1% |
| suggested tier: share of found / exact | 38.9% / 70.5% |
| review tier: exact | 29.7% |

Exact match against one reference underestimates by ~37 points (human review of 200 dev answers).
