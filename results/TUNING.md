# I5 tuning (dev) and test check

Best two LLM setups on test before tuning: gemma3:27b / rag_lexf+gen (60.07), qwen3.8:27b / rag_lexf+gen (59.07)

## dev (290 sentences)

| model | setup | chrF++ | SpanAcc% | DialectLeft% | Δ vs rag_lexf k=8 |
|---|---|---|---|---|---|
| gemma3:27b+gen | rag_lexf k=4 | 60.23 | 42.16 | 3.69 | -1.77 |
| gemma3:27b+gen | rag_lexf k=8 | 62.0 | 44.11 | 3.82 | +0.00 |
| gemma3:27b+gen | rag_lexf k=16 | 62.57 | 45.02 | 4.22 | +0.57 |
| gemma3:27b+gen | rag_fix k=16 | 61.97 | 44.84 | 4.22 | -0.03 |
| gemma3:27b+gen | rag_lexret k=16 | 62.49 | 45.07 | 4.11 | +0.49 |
| qwen3.8:27b+gen | rag_lexf k=4 | 57.06 | 34.99 | 10.87 | -0.81 |
| qwen3.8:27b+gen | rag_lexf k=8 | 57.87 | 35.62 | 9.21 | +0.00 |
| qwen3.8:27b+gen | rag_lexf k=16 | 56.11 | 35.62 | 7.94 | -1.76 |
| qwen3.8:27b+gen | rag_fix k=8 | 57.84 | 35.74 | 8.86 | -0.03 |
| qwen3.8:27b+gen | rag_lexret k=8 | 58.07 | 36.54 | 8.63 | +0.20 |

## test: dev winners only

| model | setup | chrF++ | before (best test system) | Δ | p | SpanAcc% | DialectLeft% |
|---|---|---|---|---|---|---|---|
| gemma3:27b | rag_lexf+gen_k16 | 60.95 | rag_lexf+gen 60.07 | +0.88 | 0.196 | 43.25 | 4.47 |
| gemma3:27b | rag_lexret+gen_k16 | 61.04 | rag_lexf+gen 60.07 | +0.97 | 0.138 | 43.28 | 4.41 |
| qwen3.8:27b | rag_lexret+gen_k8 | 58.68 | rag_lexf+gen 59.07 | -0.39 | 0.484 | 37.02 | 8.73 |

Success criterion (PLAN_v2 I5): p < 0.05 on test after choosing on dev.

## Oracle over the top 5 test systems

Systems: gemma3:27b / rag_lexret+gen_k16, gemma3:27b / rag_lexf+gen_k16, gemma3:27b / rag_lexf+gen, gemma3:27b / rag+gen, qwen3.8:27b / rag_lexf+gen

Oracle chrF++ 68.50 vs best system 61.04 (gap +7.46). Gap ≥ 3: a combiner (e.g. selection by an LLM) may pay off.
