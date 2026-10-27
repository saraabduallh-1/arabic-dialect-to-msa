# Significance tests and error analysis

Test set: 291 sentences. Paired bootstrap with 1000 resamples (seed 42), same normalization as the main report. p < 0.05 means the difference is unlikely to be resampling noise.

## 1. Confidence intervals and comparison with the best system

p-value: two-sided paired bootstrap against **qwen3.8:27b / rag** on chrF++.

| # | System | chrF++ | 95% CI | BLEU | 95% CI | Δ chrF++ vs best | p |
|---|---|---|---|---|---|---|---|
| 1 | qwen3.8:27b / rag | 58.43 | 55.8–61.1 | 34.84 | 30.8–38.8 | +0.00 | — |
| 2 | qwen3.8:27b / rag_lex | 57.06 | 54.7–59.6 | 33.08 | 28.9–36.9 | -1.37 | 0.076 |
| 3 | UBC-NLP/AraT5v2-base-1024 / finetune | 55.81 | 53.4–58.3 | 32.7 | 28.7–36.4 | -2.62 | 0.042 |
| 4 | gemma3:12b / rag | 55.53 | 53.2–58.1 | 29.02 | 25.3–32.8 | -2.90 | 0.014 |
| 5 | command-r7b-arabic:7b / rag | 54.56 | 51.9–57.2 | 28.4 | 24.4–32.6 | -3.87 | 0.002 |
| 6 | command-r7b-arabic:7b / rag_lex | 54.24 | 51.5–56.9 | 27.64 | 23.7–31.7 | -4.19 | 0.000 |
| 7 | Qwen2.5-7B-Instruct-qlora / finetune | 52.85 | 50.2–55.6 | 28.3 | 23.8–32.0 | -5.58 | 0.000 |
| 8 | qwen3:14b / rag | 52.49 | 50.3–54.8 | 26.74 | 22.9–30.6 | -5.94 | 0.000 |
| 9 | gemma3:12b / rag_lex | 52.05 | 49.6–54.6 | 23.28 | 19.7–26.5 | -6.38 | 0.000 |
| 10 | qwen3:8b / rag | 51.5 | 49.2–53.8 | 26.27 | 22.3–30.1 | -6.93 | 0.000 |
| 11 | aya-expanse:8b / rag | 51.48 | 48.6–54.2 | 23.38 | 19.4–27.3 | -6.95 | 0.000 |
| 12 | qwen3.8:27b / zero | 51.31 | 48.6–53.9 | 26.68 | 22.5–30.6 | -7.12 | 0.000 |
| 13 | moussaKam/AraBART / finetune | 51.04 | 48.5–53.4 | 27.81 | 24.1–31.3 | -7.39 | 0.000 |
| 14 | qwen3:8b / rag_lex | 50.42 | 47.9–52.8 | 24.75 | 20.9–28.3 | -8.01 | 0.000 |
| 15 | qwen3:14b / rag_lex | 49.92 | 47.4–52.2 | 23.1 | 19.4–26.6 | -8.51 | 0.000 |
| 16 | google/mt5-base / finetune | 49.46 | 47.0–51.7 | 23.1 | 19.0–26.9 | -8.97 | 0.000 |
| 17 | gemma3:4b / rag | 48.75 | 46.4–51.3 | 22.32 | 18.8–26.1 | -9.68 | 0.000 |
| 18 | qwen2.5:7b / rag | 48.69 | 45.8–51.3 | 24.18 | 20.5–28.0 | -9.74 | 0.000 |
| 19 | qwen2.5:7b / rag_lex | 48.16 | 45.4–50.8 | 22.58 | 18.8–26.6 | -10.27 | 0.000 |
| 20 | gemma3:4b / rag_lex | 48.11 | 45.7–50.6 | 20.87 | 17.4–23.9 | -10.32 | 0.000 |
| 21 | aya-expanse:8b / rag_lex | 47.94 | 45.5–50.8 | 19.96 | 16.4–24.0 | -10.49 | 0.000 |
| 22 | qwen3.5:9b / rag | 47.2 | 44.5–49.9 | 19.59 | 15.9–23.3 | -11.23 | 0.000 |
| 23 | qwen3.5:9b / rag_lex | 46.95 | 44.3–49.5 | 19.05 | 15.7–22.7 | -11.48 | 0.000 |
| 24 | llama3.1:8b / rag_lex | 46.91 | 44.2–49.4 | 21.79 | 18.0–25.4 | -11.52 | 0.000 |
| 25 | llama3.1:8b / rag | 45.7 | 43.0–48.1 | 18.61 | 14.7–22.1 | -12.73 | 0.000 |
| 26 | gemma3:12b / zero | 45.09 | 42.6–47.6 | 18.39 | 15.0–21.5 | -13.34 | 0.000 |
| 27 | qwen3:14b / zero | 44.51 | 42.2–46.7 | 17.03 | 13.7–20.3 | -13.92 | 0.000 |
| 28 | aya-expanse:8b / zero | 43.7 | 41.1–46.1 | 16.8 | 13.4–19.8 | -14.73 | 0.000 |
| 29 | qwen3:8b / zero | 43.54 | 41.3–45.6 | 16.58 | 13.0–19.9 | -14.89 | 0.000 |
| 30 | command-r7b-arabic:7b / zero | 42.4 | 39.9–45.0 | 14.61 | 11.6–17.7 | -16.03 | 0.000 |
| 31 | qwen3.5:9b / zero | 40.04 | 37.9–42.1 | 11.32 | 8.6–14.1 | -18.39 | 0.000 |
| 32 | qwen2.5:7b / zero | 39.62 | 37.3–42.0 | 12.61 | 9.5–15.6 | -18.81 | 0.000 |
| 33 | llama3.1:8b / zero | 38.5 | 36.3–40.6 | 12.5 | 9.3–15.4 | -19.93 | 0.000 |
| 34 | gemma3:4b / zero | 36.73 | 34.6–39.0 | 8.67 | 6.3–10.9 | -21.70 | 0.000 |

Not significantly different from the best system (p ≥ 0.05): qwen3.8:27b / rag_lex.

## 2. Effect of the prompting modes (per model, chrF++)

| Model | zero | rag | Δ rag−zero | p | rag_lex | Δ rag_lex−rag | p |
|---|---|---|---|---|---|---|---|
| qwen3.8:27b | 51.31 | 58.43 | +7.12 | 0.000 | 57.06 | -1.37 | 0.076 |
| gemma3:12b | 45.09 | 55.53 | +10.44 | 0.000 | 52.05 | -3.48 | 0.000 |
| qwen3:14b | 44.51 | 52.49 | +7.98 | 0.000 | 49.92 | -2.57 | 0.002 |
| aya-expanse:8b | 43.7 | 51.48 | +7.78 | 0.000 | 47.94 | -3.54 | 0.004 |
| qwen3:8b | 43.54 | 51.5 | +7.96 | 0.000 | 50.42 | -1.08 | 0.168 |
| command-r7b-arabic:7b | 42.4 | 54.56 | +12.16 | 0.000 | 54.24 | -0.32 | 0.706 |
| qwen3.5:9b | 40.04 | 47.2 | +7.16 | 0.000 | 46.95 | -0.25 | 0.770 |
| qwen2.5:7b | 39.62 | 48.69 | +9.07 | 0.000 | 48.16 | -0.53 | 0.618 |
| llama3.1:8b | 38.5 | 45.7 | +7.20 | 0.000 | 46.91 | +1.21 | 0.162 |
| gemma3:4b | 36.73 | 48.75 | +12.02 | 0.000 | 48.11 | -0.64 | 0.470 |

## 3. Pairwise p-values among the top 8 (chrF++)

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| 1. qwen3.8:27b / rag | — | 0.08 | 0.04 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 2. qwen3.8:27b / rag_lex | 0.08 | — | 0.27 | 0.17 | 0.03 | 0.02 | 0.00 | 0.00 |
| 3. UBC-NLP/AraT5v2-base-1024 / finetune | 0.04 | 0.27 | — | 0.83 | 0.34 | 0.27 | 0.00 | 0.00 |
| 4. gemma3:12b / rag | 0.01 | 0.17 | 0.83 | — | 0.38 | 0.28 | 0.04 | 0.01 |
| 5. command-r7b-arabic:7b / rag | 0.00 | 0.03 | 0.34 | 0.38 | — | 0.71 | 0.18 | 0.06 |
| 6. command-r7b-arabic:7b / rag_lex | 0.00 | 0.02 | 0.27 | 0.28 | 0.71 | — | 0.29 | 0.16 |
| 7. Qwen2.5-7B-Instruct-qlora / finetune | 0.00 | 0.00 | 0.00 | 0.04 | 0.18 | 0.29 | — | 0.78 |
| 8. qwen3:14b / rag | 0.00 | 0.00 | 0.00 | 0.01 | 0.06 | 0.16 | 0.78 | — |

## 4. Span-level error analysis

564 gold dialect spans in the test set (spans identical to their MSA are skipped); 184 (33%) occur in the training lexicon, 64 (11%) are multi-word.

- **correct**: the gold MSA replacement appears in the output
- **kept**: the dialect word/phrase is still in the output
- **other**: changed to something else — a valid synonym or a wrong meaning (manual check of a sample: results/MANUAL_REVIEW.md)

| System | correct% | kept% | other% | correct% seen | correct% unseen | correct% 1-word | correct% multi |
|---|---|---|---|---|---|---|---|
| qwen3.8:27b / rag | 39.7 | 7.6 | 52.7 | 57.6 | 31.1 | 41.0 | 29.7 |
| qwen3.8:27b / rag_lex | 40.2 | 8.2 | 51.6 | 66.3 | 27.6 | 42.6 | 21.9 |
| UBC-NLP/AraT5v2-base-1024 / finetune | 30.7 | 17.4 | 52.0 | 53.8 | 19.5 | 33.0 | 12.5 |
| gemma3:12b / rag | 37.4 | 5.3 | 57.3 | 57.6 | 27.6 | 39.4 | 21.9 |
| command-r7b-arabic:7b / rag | 38.1 | 6.6 | 55.3 | 58.2 | 28.4 | 40.6 | 18.8 |
| command-r7b-arabic:7b / rag_lex | 39.9 | 7.6 | 52.5 | 65.2 | 27.6 | 42.2 | 21.9 |
| Qwen2.5-7B-Instruct-qlora / finetune | 27.7 | 10.3 | 62.1 | 50.0 | 16.8 | 29.8 | 10.9 |
| qwen3:14b / rag | 26.2 | 19.5 | 54.3 | 50.0 | 14.7 | 28.0 | 12.5 |
| gemma3:12b / rag_lex | 38.1 | 4.8 | 57.1 | 62.5 | 26.3 | 40.6 | 18.8 |
| qwen3:8b / rag | 20.6 | 33.2 | 46.3 | 40.8 | 10.8 | 21.8 | 10.9 |
| aya-expanse:8b / rag | 39.0 | 4.4 | 56.6 | 61.4 | 28.2 | 41.0 | 23.4 |
| qwen3.8:27b / zero | 26.8 | 14.4 | 58.9 | 35.9 | 22.4 | 28.0 | 17.2 |
| moussaKam/AraBART / finetune | 20.4 | 22.7 | 56.9 | 39.1 | 11.3 | 22.0 | 7.8 |
| qwen3:8b / rag_lex | 23.4 | 32.4 | 44.1 | 56.0 | 7.6 | 25.4 | 7.8 |
| qwen3:14b / rag_lex | 27.7 | 19.7 | 52.7 | 60.9 | 11.6 | 29.8 | 10.9 |
| google/mt5-base / finetune | 19.1 | 23.9 | 56.9 | 41.8 | 8.2 | 20.6 | 7.8 |
| gemma3:4b / rag | 29.6 | 9.9 | 60.5 | 53.8 | 17.9 | 31.6 | 14.1 |
| qwen2.5:7b / rag | 24.5 | 11.7 | 63.8 | 46.7 | 13.7 | 25.4 | 17.2 |
| qwen2.5:7b / rag_lex | 27.1 | 20.0 | 52.8 | 57.6 | 12.4 | 29.6 | 7.8 |
| gemma3:4b / rag_lex | 29.6 | 12.2 | 58.2 | 58.7 | 15.5 | 31.6 | 14.1 |
| aya-expanse:8b / rag_lex | 34.8 | 4.1 | 61.2 | 58.2 | 23.4 | 36.8 | 18.8 |
| qwen3.5:9b / rag | 31.9 | 3.9 | 64.2 | 56.0 | 20.3 | 32.8 | 25.0 |
| qwen3.5:9b / rag_lex | 32.8 | 5.3 | 61.9 | 60.3 | 19.5 | 34.4 | 20.3 |
| llama3.1:8b / rag_lex | 24.1 | 18.1 | 57.8 | 55.4 | 8.9 | 26.0 | 9.4 |
| llama3.1:8b / rag | 19.9 | 13.8 | 66.3 | 42.9 | 8.7 | 21.4 | 7.8 |
| gemma3:12b / zero | 29.8 | 5.1 | 65.1 | 44.6 | 22.6 | 32.0 | 12.5 |
| qwen3:14b / zero | 10.8 | 38.3 | 50.9 | 19.6 | 6.6 | 12.0 | 1.6 |
| aya-expanse:8b / zero | 26.6 | 8.0 | 65.4 | 39.7 | 20.3 | 28.2 | 14.1 |
| qwen3:8b / zero | 4.4 | 48.9 | 46.6 | 5.4 | 3.9 | 5.0 | 0.0 |
| command-r7b-arabic:7b / zero | 23.9 | 8.0 | 68.1 | 39.1 | 16.6 | 24.8 | 17.2 |
| qwen3.5:9b / zero | 21.8 | 6.9 | 71.3 | 40.8 | 12.6 | 23.2 | 10.9 |
| qwen2.5:7b / zero | 14.0 | 17.0 | 69.0 | 21.2 | 10.5 | 15.4 | 3.1 |
| llama3.1:8b / zero | 8.2 | 25.2 | 66.7 | 14.7 | 5.0 | 8.6 | 4.7 |
| gemma3:4b / zero | 15.8 | 6.9 | 77.3 | 25.5 | 11.1 | 16.6 | 9.4 |

## 5. Dictionary hints (rag_lex)

- Sentences that received at least one hint: 215 / 291 (74%)
- Hints given in total: 382; of these 179 (47%) match a gold dialect span, the other 203 point at words that should stay unchanged (common words that happen to appear in the lexicon) or at partial matches.

| Model | Δ correct% seen (rag_lex−rag) | Δ correct% unseen | Δ kept% | Δ chrF++ |
|---|---|---|---|---|
| qwen3.8:27b | +8.7 | -3.4 | +0.5 | -1.37 |
| command-r7b-arabic:7b | +7.1 | -0.8 | +1.1 | -0.32 |
| gemma3:12b | +4.9 | -1.3 | -0.5 | -3.48 |
| qwen3:8b | +15.2 | -3.2 | -0.7 | -1.08 |
| qwen3:14b | +10.9 | -3.2 | +0.2 | -2.57 |
| qwen2.5:7b | +10.9 | -1.3 | +8.3 | -0.53 |
| gemma3:4b | +4.9 | -2.4 | +2.3 | -0.64 |
| aya-expanse:8b | -3.3 | -4.7 | -0.4 | -3.54 |
| qwen3.5:9b | +4.3 | -0.8 | +1.4 | -0.25 |
| llama3.1:8b | +12.5 | +0.3 | +4.3 | +1.21 |

## 6. Hardest spans

209 of 564 spans (37%) are missed by all of the top 10 systems. Most frequent dialect words left unconverted by the best system:

ولدك (2), التوافه (1), العود (1), مصلحجي (1), عيوني (1), كرام (1), مسكتها (1), تلعب بذيلك (1), والا (1), بصرك (1), معها (1), جوهم (1), بنت (1), طيب (1), مرعوبه (1), الواحد (1), عند (1), وقفوا (1), كوي (1), البيت (1), اذا (1), كبرت (1), سيقان (1), طول (1), الغيبات (1)

## 7. Sentence-level behaviour

| System | copied input% | length ratio (pred/ref words) |
|---|---|---|
| qwen3.8:27b / rag | 1.4 | 0.97 |
| qwen3.8:27b / rag_lex | 1.0 | 0.97 |
| UBC-NLP/AraT5v2-base-1024 / finetune | 6.2 | 0.98 |
| gemma3:12b / rag | 0.7 | 0.99 |
| command-r7b-arabic:7b / rag | 1.0 | 1.01 |
| command-r7b-arabic:7b / rag_lex | 1.0 | 1.01 |
| Qwen2.5-7B-Instruct-qlora / finetune | 0.7 | 1.01 |
| qwen3:14b / rag | 5.8 | 0.98 |
| gemma3:12b / rag_lex | 0.7 | 0.98 |
| qwen3:8b / rag | 11.3 | 0.97 |
| aya-expanse:8b / rag | 0.0 | 1.05 |
| qwen3.8:27b / zero | 6.9 | 1.00 |
| moussaKam/AraBART / finetune | 8.9 | 0.96 |
| qwen3:8b / rag_lex | 5.8 | 0.97 |
| qwen3:14b / rag_lex | 2.1 | 0.98 |
| google/mt5-base / finetune | 7.2 | 0.98 |
| gemma3:4b / rag | 0.3 | 1.01 |
| qwen2.5:7b / rag | 1.4 | 1.01 |
| qwen2.5:7b / rag_lex | 1.4 | 1.01 |
| gemma3:4b / rag_lex | 1.0 | 0.98 |
| aya-expanse:8b / rag_lex | 0.0 | 1.07 |
| qwen3.5:9b / rag | 0.3 | 1.04 |
| qwen3.5:9b / rag_lex | 0.7 | 1.02 |
| llama3.1:8b / rag_lex | 3.1 | 0.98 |
| llama3.1:8b / rag | 1.4 | 1.00 |
| gemma3:12b / zero | 0.7 | 1.00 |
| qwen3:14b / zero | 13.4 | 0.98 |
| aya-expanse:8b / zero | 0.3 | 1.08 |
| qwen3:8b / zero | 19.6 | 0.96 |
| command-r7b-arabic:7b / zero | 0.7 | 1.03 |
| qwen3.5:9b / zero | 0.0 | 1.09 |
| qwen2.5:7b / zero | 1.0 | 1.02 |
| llama3.1:8b / zero | 3.8 | 1.00 |
| gemma3:4b / zero | 0.7 | 1.09 |
