# Data card

Saudi/Gulf colloquial Arabic → Modern Standard Arabic (MSA), sentence pairs with marked word alignments:
`(مدري)1 وش صار` ↔ `(لا أدري)1 ماذا حدث` (span number N links a dialect word to its MSA equivalent).

## Contents
| Folder | Rows | What it is | Use |
|---|---|---|---|
| `sentences/train.csv` | 2,326 | original sentence pairs | training, few-shot examples, lexicon |
| `sentences/dev.csv` | 290 | original sentence pairs | tuning and model selection only |
| `sentences/test.csv` | 291 | original sentence pairs | final measurement only |
| `vocab/{train,dev,test}.csv` | 3,609 / 451 / 451 | one row per dialect word, with its sentence | **not used** (see problems) |
| `generated/all.csv` | 2,268 | generated pairs from 20 batches (merged by `src/merge_generated.py`) | training only (`--extra generated/all`) |
| `generated/all_clean.csv` | 2,265 | `all.csv` after the D1 default decisions (`src/clean_generated.py`) | waiting for user approval |
| `sentences_clean/train.csv` | 2,326 | train with mechanical annotation fixes (`src/clean_train.py`) | optional |
| `real_test/` | — | template and instructions for a real, human-written test set (PLAN_v2 I4) | to be collected |

## Generated data
- **Per dialect:** Najdi 597, Hijazi 561, Eastern/Gulf 480, Southern 465, Northern (Hail) 162, general 3. 778 new dialect terms.
- **How it was made:** Claude agents collected new dialect terms, each documented by a real reference they opened
  (dictionary, paper, Wiktionary entry), then **wrote** 3 sentences per term with the MSA version. So the terms
  and meanings are referenced, but the sentences are model-written (`نوع الجملة = مؤلفة` for all rows).
- **References by rows:** English Wiktionary 945, Al-Abboudi 570, Al-Aqili (Jazan) 225, Al-Shahrani 2024 141,
  Al-Hanafi (Kuwaiti, 1964) 102, Al-Mufarreh 2024 69, Al-Anazi 2022 66, Saudipedia 54, AlAmmar 2017 39,
  Al-Shammari 2024 33, Al-Watan 21. Full citation and link on every row; licences in `generated/SOURCES.md`.
- **Leakage controls:** the generating agents never opened dev/test/vocab/results; `merge_generated.py` drops rows
  that match a dev/test sentence; the generated data is never used as a test set.
- **Checks:** `src/check_generated.py` (format, new terms, overlaps); `src/check_msa.py` (MSA equivalents in open
  lexicons: 84.4% found, rest flagged in `generated/msa_check.csv`).
- **Human review (PLAN_v2 D2):** sample of 100 rows + 20 links in `generated/review_sample.csv` — pending.
- **Known limits:** some terms are Kuwaiti but labelled Eastern; some are rare/old or close to MSA; 29 rows only
  differ from a train term by ال/ه (extra examples, not new vocabulary); see `autowork/QUESTIONS.md`.
- **Coverage of the test set:** only 34 of 564 test dialect spans are unseen in train but covered by the generated
  data, so gains on the current test set are expected to be small.

## Known problems in the original data
- **vocab/ overlaps the test set:** 249 of the 291 `sentences/test` sentences appear in `vocab/train`. All results
  use `sentences/test` and only `sentences/train` for training/examples; the vocab folder is not used.
- **Annotation errors** (`sentences_clean/REPORT.md`): train 4 auto-fixable rows + 6 rows with malformed brackets;
  dev 1 and test 3 problems, reported only (changing the test set would break comparability).
- **One reference per test sentence:** a correct synonym counts as an error. Multi-reference scoring is ready
  (`sentences/test_refs.csv`, PLAN_v2 I2) once the alternative references are approved.
