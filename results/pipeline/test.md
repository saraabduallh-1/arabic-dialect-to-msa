# End-to-end pipeline on test (tagger + ALLaM +gen3 + dictionary tier)

Sentences: 291; tokenisation mismatches: 22.

| step | value |
|---|---|
| dialect span detection P / R / F1 (exact boundaries) | 80.4 / 71.1 / 75.5 |
| spans that need a change, found by the tagger | 71.3% of 565 |
| end-to-end exact (found and replaced like the reference) | 33.5% |
| suggested tier: share of found / exact | 10.7% / 88.4% |
| review tier: exact | 41.9% |

Exact match against one reference underestimates by ~37 points (human review of 200 dev answers).
