# Word-level task on the corrected data (sentences_v2)

The dialect words are marked in the input; the model returns only their MSA equivalents.
Test: 291 sentences, 565 dialect spans. Judge: **qwen3.8:27b** (agrees 87% with 60 Claude-written labels; human labels pending).

| # | model | mode | k | WordAcc% (exact) | WordLoose% | judged correct% | partial% | Copy% | Missing% |
|---|---|---|---|---|---|---|---|---|---|
| 1 | ALLaM-7B-Instruct-preview-qlora-word+gen3 | finetune | 0 | 46.9 | 48.85 | 65.1 | 4.6 | 1.59 | 0.18 |
| 2 | ALLaM-7B-Instruct-preview-qlora-word | finetune | 0 | 45.13 | 47.96 | 65.1 | 5.8 | 0.88 | 0.0 |
| 3 | ALLaM-7B-Instruct-preview-qlora-word+gen4 | finetune | 0 | 44.96 | 46.19 | 63.9 | 6.5 | 1.59 | 0.0 |
| 4 | gemma3:27b | rag+gen2 | 8 | 42.3 | 45.31 | 63.4 | 6.9 | 0.88 | 0.18 |
| 5 | gemma3:27b | rag_lexf+gen2_k16 | 16 | 42.48 | 45.13 | 63.5 | 6.5 | 0.88 | 0.35 |
| 6 | gemma3:27b | rag+gen3 | 8 | 41.59 | 43.72 | 61.9 | 7.8 | 0.88 | 0.35 |
| 7 | gemma3:27b | rag_lexf+gen2 | 8 | 40.88 | 43.54 | — | — | 0.88 | 0.18 |
| 8 | gemma3:12b | rag+gen2 | 8 | 40.18 | 43.19 | — | — | 1.06 | 0.0 |
| 9 | gemma3:12b | rag_lexf+gen2 | 8 | 38.23 | 41.42 | — | — | 1.42 | 1.06 |
| 10 | qwen3.8:27b | rag+gen2 | 8 | 38.05 | 40.35 | — | — | 3.89 | 0.0 |
| 11 | qwen3.8:27b | rag_lexf+gen2 | 8 | 38.58 | 40.18 | — | — | 3.36 | 0.0 |
| 12 | command-r7b-arabic:7b | rag_lexf+gen2 | 8 | 34.51 | 38.76 | — | — | 3.54 | 1.95 |
| 13 | command-r7b-arabic:7b | rag+gen2 | 8 | 33.27 | 37.88 | — | — | 3.72 | 0.53 |
| 14 | allam:7b | rag+gen2 | 8 | 32.39 | 36.99 | — | — | 2.83 | 0.18 |
| 15 | allam:7b | rag+gen3 | 8 | 33.81 | 36.99 | — | — | 3.01 | 0.18 |
| 16 | gemma3:27b | zero | 8 | 32.57 | 35.75 | — | — | 1.06 | 0.0 |
| 17 | Qwen2.5-7B-Instruct-qlora-word+gen3 | finetune | 0 | 29.91 | 30.97 | — | — | 2.3 | 0.0 |
| 18 | gemma3:12b | zero | 8 | 23.89 | 27.26 | — | — | 0.35 | 0.18 |
| 19 | qwen3.8:27b | zero | 8 | 21.59 | 23.19 | — | — | 13.81 | 0.0 |
| 20 | command-r7b-arabic:7b | zero | 8 | 17.35 | 23.19 | — | — | 4.07 | 0.0 |
| 21 | allam:7b | zero | 8 | 10.27 | 14.51 | — | — | 8.5 | 46.55 |

## Sentence-level systems (rewrite the whole sentence) on the same corrected test set

| model | mode | SpanAcc% (gold word appears in the output) |
|---|---|---|
| gemma3:27b | rag_lexret+gen_k16 | 43.4 |
| gemma3:27b | rag_lexf+gen_k16 | 43.4 |
| gemma3:27b | rag_lexret+gen2_v2 | 42.7 |
| gemma3:27b | rag_lexf+gen | 42.2 |
| gemma3:27b | rag+gen | 42.3 |
| qwen3.8:27b | rag_lexf+gen | 37.0 |
