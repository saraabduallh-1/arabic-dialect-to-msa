# LLM-judge evaluation of dialect spans

Judge: **qwen3.8:27b**. Spans already matching the gold MSA count as correct, spans left in dialect as wrong; the rest are judged.

## Agreement with the 60 hand-labelled spans

- Exact agreement (3 labels): 80%
- Agreement on correct vs not: 87%

| manual \ judge | correct | partial | wrong |
|---|---|---|---|
| correct | 25 | 1 | 5 |
| partial | 1 | 0 | 4 |
| wrong | 1 | 0 | 23 |

## Ranking by judged span accuracy

| # | System | chrF++ | auto SpanAcc% | judged correct% | partial% | wrong% |
|---|---|---|---|---|---|---|
| 1 | gemma3:12b / rag_lexf+gen | 56.79 | 41.3 | 61.3 | 5.1 | 33.5 |
| 2 | command-r7b-arabic:7b / rag_lexf+gen | 54.64 | 39.4 | 61.3 | 2.8 | 35.8 |
| 3 | gemma3:12b / rag+gen | 56.54 | 39.5 | 61.2 | 4.6 | 34.2 |
| 4 | gemma3:12b / rag | 55.53 | 37.4 | 60.8 | 6.9 | 32.3 |
| 5 | qwen3.8:27b / rag_lexf+gen | 59.07 | 39.7 | 60.5 | 3.7 | 35.8 |
| 6 | qwen3.8:27b / rag | 58.43 | 39.7 | 60.1 | 3.2 | 36.7 |
| 7 | qwen3.8:27b / rag_lex | 57.06 | 40.2 | 60.1 | 3.5 | 36.3 |
| 8 | qwen3.8:27b / rag+gen | 58.16 | 37.8 | 59.6 | 3.7 | 36.7 |
| 9 | gemma3:12b / rag_lexf | 55.34 | 36.9 | 59.6 | 6.4 | 34.0 |
| 10 | qwen3.8:27b / rag_lexf | 58.31 | 38.7 | 59.4 | 3.5 | 37.1 |
| 11 | command-r7b-arabic:7b / rag | 54.56 | 38.1 | 58.7 | 4.6 | 36.7 |
| 12 | command-r7b-arabic:7b / rag_lexf | 54.86 | 39.0 | 58.3 | 4.6 | 37.1 |
| 13 | command-r7b-arabic:7b / rag_lex | 54.24 | 39.9 | 58.0 | 3.9 | 38.1 |
| 14 | UBC-NLP/AraT5v2-base-1024 / finetune | 55.81 | 30.7 | 39.5 | 3.7 | 56.7 |
| 15 | UBC-NLP/AraT5v2-base-1024+gen / finetune | 55.09 | 29.1 | 38.3 | 3.5 | 58.2 |
