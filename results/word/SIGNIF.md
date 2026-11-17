# Paired bootstrap, word task (2,000 resamples of the 291 test sentences)

Reference: **ALLaM-7B-Instruct-preview-qlora-word+gen3 / finetune**. Δ = other − reference; p = share of resamples with Δ ≥ 0 (one-sided).

| system | judged correct% | Δ judged | 95% CI | p | WordAcc% | Δ exact | p |
|---|---|---|---|---|---|---|---|
| ALLaM-7B-Instruct-preview-qlora-word+gen3 / finetune | 65.1 | — | — | — | 46.90 | — | — |
| ALLaM-7B-Instruct-preview-qlora-word / finetune | 65.1 | +0.0 | [-3.5, +3.2] | 0.500 | 45.13 | -1.77 | 0.137 |
| ALLaM-7B-Instruct-preview-qlora-word+gen4 / finetune | 63.9 | -1.2 | [-4.6, +2.2] | 0.265 | 44.96 | -1.95 | 0.132 |
| ALLaM-7B-Instruct-preview-qlora-word+gen4+lex / finetune | 66.9 | +1.8 | [-1.7, +5.1] | 0.850 | 48.32 | +1.42 | 0.809 |
| gemma3:27b / rag+gen2 | 63.4 | -1.8 | [-6.4, +2.6] | 0.218 | 42.30 | -4.60 | 0.024 |
