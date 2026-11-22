# Smaller models, same QLoRA recipe (word task, sentences_v2 + generated/verified, 3 epochs), dev 555 spans

| model | epoch 1 | epoch 2 | epoch 3 (kept) | Wilson 95% |
|---|---|---|---|---|
| Qwen2.5-1.5B-Instruct | 17.66 | 20.90 | 22.52 | 19.2–26.2 |
| Qwen2.5-3B-Instruct | 22.52 | 25.95 | 27.39 | 23.8–31.2 |
| ALLaM-7B (+gen3, reference) | | | 48.1 | 44.0–52.3 |

WordAcc% (exact). Logs: small_models.log, small_models_1.5b.log. Not adopted (about half the accuracy).
