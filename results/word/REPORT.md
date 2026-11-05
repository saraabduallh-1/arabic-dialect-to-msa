# Word-level task on the corrected data (sentences_v2)

The dialect words are marked in the input; the model returns only their MSA equivalents.
Test: 291 sentences, 565 dialect spans. Judge: **qwen3.8:27b** (agrees 87% with 60 Claude-written labels; human labels pending).

| # | model | mode | k | WordAcc% (exact) | WordLoose% | judged correct% | partial% | Copy% | Missing% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gemma3:27b | rag+gen2 | 8 | 42.3 | 45.31 | 63.4 | 6.9 | 0.88 | 0.18 |
| 2 | gemma3:27b | rag_lexf+gen2_k16 | 16 | 42.48 | 45.13 | 63.5 | 6.5 | 0.88 | 0.35 |
| 3 | gemma3:27b | rag_lexf+gen2 | 8 | 40.88 | 43.54 | 63.0 | 7.4 | 0.88 | 0.18 |
| 4 | gemma3:12b | rag+gen2 | 8 | 40.18 | 43.19 | 57.2 | 7.1 | 1.06 | 0.0 |
| 5 | gemma3:12b | rag_lexf+gen2 | 8 | 38.23 | 41.42 | 55.6 | 7.8 | 1.42 | 1.06 |
| 6 | qwen3.8:27b | rag+gen2 | 8 | 38.05 | 40.35 | 56.3 | 5.7 | 3.89 | 0.0 |
| 7 | qwen3.8:27b | rag_lexf+gen2 | 8 | 38.58 | 40.18 | — | — | 3.36 | 0.0 |
| 8 | command-r7b-arabic:7b | rag_lexf+gen2 | 8 | 34.51 | 38.76 | — | — | 3.54 | 1.95 |
| 9 | command-r7b-arabic:7b | rag+gen2 | 8 | 33.27 | 37.88 | — | — | 3.72 | 0.53 |
| 10 | gemma3:27b | zero | 8 | 32.57 | 35.75 | — | — | 1.06 | 0.0 |
| 11 | gemma3:12b | zero | 8 | 23.89 | 27.26 | — | — | 0.35 | 0.18 |
| 12 | qwen3.8:27b | zero | 8 | 21.59 | 23.19 | — | — | 13.81 | 0.0 |
| 13 | command-r7b-arabic:7b | zero | 8 | 17.35 | 23.19 | — | — | 4.07 | 0.0 |

## Sentence-level systems (rewrite the whole sentence) on the same corrected test set

| model | mode | SpanAcc% (gold word appears in the output) |
|---|---|---|
| gemma3:27b | rag_lexret+gen_k16 | 43.4 |
| gemma3:27b | rag_lexf+gen_k16 | 43.4 |
| gemma3:27b | rag_lexret+gen2_v2 | 42.7 |
| gemma3:27b | rag_lexf+gen | 42.2 |
| gemma3:27b | rag+gen | 42.3 |
| qwen3.8:27b | rag_lexf+gen | 37.0 |
