# Arabic dialect → MSA: open-source model benchmark

Compares open-source models at converting Saudi/Gulf colloquial Arabic sentences into Modern Standard Arabic.

- **Data**: `data/sentences` (train 2326 / dev 290 / test 291 sentence pairs with marked dialect spans) and `data/vocab` (word-level pairs with context).
  The vocab splits share sentences with `sentences/test`, so all scores use `sentences/test` and only `sentences/train` is used for examples and training.
- **LLMs via Ollama** (`src/run_llm.py`): `zero` (instruction only), `rag` (8 most similar training pairs as few-shot examples), `rag_lex` (+ dictionary hints from the training lexicon).
- **Fine-tuned seq2seq** (`src/finetune.py`): AraT5v2, AraBART, mT5, model selection on dev.
- **Metrics** (`src/metrics.py`): BLEU, chrF++, SpanAcc (gold MSA replacements present), DialectLeft (dialect words left unconverted), exact match — all after orthographic normalization.

Results: [results/REPORT.md](results/REPORT.md). The whole pipeline runs unattended with `python src/autorun.py`; progress is in `results/autorun.log`.

## Using the tool (end-to-end)

`src/pipeline.py` takes raw Saudi dialect sentences and returns MSA, each replaced word marked
**suggested** (applied automatically, ~90% correct by meaning on test) or **review** (show to a person).

```
python src/pipeline.py --text "وش تبي تسوي الحين"
```

Steps: MARBERTv2 tagger finds the dialect words → ALLaM-7B QLoRA (+gen3) replaces them → tier from the train
dictionary or agreement with a second QLoRA model (Qwen2.5-7B). Needs one GPU with 10 GB and the checkpoints in
`results/checkpoints/` (not in the repo). Scores: `results/pipeline/test.md`, `results/pipeline/judged_test.md`;
full report (Arabic): `docs/paper/report_ar.pdf`.
