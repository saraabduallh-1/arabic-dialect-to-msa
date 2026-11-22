# Pipeline on test, judged by meaning (qwen3.8:27b)

Judge check against the user's 200 dev labels: agreement correct-vs-not 58%, 3-way 51%; judge says correct 36% vs human 69%.

| | spans | share of all | correct | partial | wrong |
|---|---|---|---|---|---|
| suggested | 157 | 27.8% | 89.8% | 2.5% | 7.6% |
| review | 292 | 51.7% | 53.8% | 7.9% | 38.4% |
| missed by the tagger | 116 | 20.5% | | | |
| **all spans that need a change** | 565 | 100% | **52.7%** | 4.8% | |

Of the spans the tagger found: 66.4% correct by meaning.
