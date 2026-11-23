# ALLaM-7B QLoRA + gen3, three seeds (same recipe; epoch chosen on dev)

| seed | dev exact (best epoch) | test exact |
|---|---|---|
| 42 (original) | 48.1 | 46.9 |
| 1 | 44.5 | 45.31 |
| 2 | 45.41 | 46.55 |
| mean ± sd | 46.0 ± 1.9 | 46.3 ± 0.8 |

Logs: seed_1.log, seed_2.log. dev 48.1 for seed 42 is slightly optimistic (configuration chosen on dev); test is stable.
