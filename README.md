# Arabic dialect → MSA: open-source model benchmark

Compares open-source models at converting Saudi/Gulf colloquial Arabic sentences into Modern Standard Arabic.

- **Data**: `data/sentences` (train 2326 / dev 290 / test 291 sentence pairs with marked dialect spans) and `data/vocab` (word-level pairs with context).
  The vocab splits share sentences with `sentences/test`, so all scores use `sentences/test` and only `sentences/train` is used for examples and training.
- **LLMs via Ollama** (`src/run_llm.py`): `zero` (instruction only), `rag` (8 most similar training pairs as few-shot examples), `rag_lex` (+ dictionary hints from the training lexicon).
- **Fine-tuned seq2seq** (`src/finetune.py`): AraT5v2, AraBART, mT5, model selection on dev.
- **Metrics** (`src/metrics.py`): BLEU, chrF++, SpanAcc (gold MSA replacements present), DialectLeft (dialect words left unconverted), exact match — all after orthographic normalization.

Results: [results/REPORT.md](results/REPORT.md). The whole pipeline runs unattended with `python src/autorun.py`; progress is in `results/autorun.log`.
