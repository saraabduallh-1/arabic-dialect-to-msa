# Merged layer on test

Tagger: MARBERTv2, dialect + foreign (src/tag_merged.py). Foreign model: m2m100 fine-tune, unchanged. Dialect: dictionary + gemma4:12b (rag, k=8, +gen3 examples). Tier thresholds min_n=2, min_c=0.8.

## Dialect sentences (test, 291)

### variant: gemma

| step | value |
|---|---|
| dialect span detection P / R / F1 | 80.9 / 78.3 / 79.6 |
| spans that need a change, found | 78.6% of 565 |
| end-to-end exact | 35.4% |
| suggested tier: share of found / exact | 27.0% / 70.8% |
| review tier: exact | 35.5% |
| foreign spans wrongly found in dialect sentences | 1 |

### variant: m2m_first

| step | value |
|---|---|
| dialect span detection P / R / F1 | 80.9 / 78.3 / 79.6 |
| spans that need a change, found | 78.6% of 565 |
| end-to-end exact | 28.7% |
| suggested tier: share of found / exact | 27.0% / 70.8% |
| review tier: exact | 23.8% |
| foreign spans wrongly found in dialect sentences | 1 |

### The two sources on the found dialect spans that need a change

| source | changed the word | exact |
|---|---|---|
| foreign_model | 96.8% | 12.7% |
| gemma | 98.6% | 44.1% |

## Foreign sentences (test, 553)

| step | value |
|---|---|
| foreign span detection P / R / F1 | 77.6 / 77.5 / 77.6 |
| spans that need a change, found | 77.5% of 591 |
| end-to-end exact | 13.0% |
| suggested tier: share of found / exact | 3.1% / 50.0% |
| review tier: exact | 15.8% |
| dialect spans wrongly found in foreign sentences | 2 |
| words outside the foreign words changed: foreign model alone / merged layer | 2.2% / 2.5% of 2202 |
| sentence BLEU: copy input / foreign model alone / merged layer | 41.96 / 50.17 / 49.95 |

Exact match against one reference underestimates (a human review of 200 dialect dev answers found ~37 points more acceptable).
