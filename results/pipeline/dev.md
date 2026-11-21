# End-to-end pipeline on dev (tagger + ALLaM +gen3 + dictionary tier)

Sentences: 290; tokenisation mismatches: 22.

| step | value |
|---|---|
| dialect span detection P / R / F1 (exact boundaries) | 80.0 / 72.3 / 75.9 |
| spans that need a change, found by the tagger | 73.0% of 555 |
| end-to-end exact (found and replaced like the reference) | 33.2% |
| suggested tier: share of found / exact | 11.1% / 88.9% |
| review tier: exact | 40.0% |

Exact match against one reference underestimates by ~37 points (human review of 200 dev answers).
