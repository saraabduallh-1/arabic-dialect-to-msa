# Merged layer on test, judged by meaning (qwen3.8:27b)

Judge agreement with the user's 200 dialect dev labels (earlier check, src/judge_pipeline.py): 78% correct-vs-not; the judge is stricter than the human.

| system | kind | tier | spans | share of all | correct | partial | wrong |
|---|---|---|---|---|---|---|---|
| layer | dialect | suggested | 120 | 21.2% | 89.2% | 5.0% | 5.8% |
| layer | dialect | review | 324 | 57.3% | 56.2% | 6.8% | 37.0% |
| layer | dialect | **all found** | 444 | 78.6% | **65.1%** | 6.3% | |
| layer | dialect | missed by the tagger | 121 | 21.4% | | | |
| layer | dialect | **end to end (all spans)** | 565 | 100% | **51.2%** | | |
| layer | foreign | suggested | 71 | 12.0% | 97.2% | 0.0% | 2.8% |
| layer | foreign | review | 387 | 65.5% | 73.1% | 8.0% | 18.9% |
| layer | foreign | **all found** | 458 | 77.5% | **76.9%** | 6.8% | |
| layer | foreign | missed by the tagger | 133 | 22.5% | | | |
| layer | foreign | **end to end (all spans)** | 591 | 100% | **59.6%** | | |
| foreign_model | foreign | suggested | 14 | 2.4% | 71.4% | 0.0% | 28.6% |
| foreign_model | foreign | review | 444 | 75.1% | 21.4% | 7.9% | 70.7% |
| foreign_model | foreign | **all found** | 458 | 77.5% | **22.9%** | 7.6% | |
| foreign_model | foreign | missed by the tagger | 133 | 22.5% | | | |
| foreign_model | foreign | **end to end (all spans)** | 591 | 100% | **17.8%** | | |
