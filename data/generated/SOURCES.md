# Sources and licences

## Dialect terms (generated data)
Every generated row cites the reference that documents its new dialect term (columns `المصدر` and `رابط المصدر`).
Main references: Al-Abboudi's dictionaries (معجم الأصول الفصيحة للألفاظ الدارجة، معجم الكلمات الدخيلة), Al-Aqili's Jazan dictionary,
Al-Hanafi's Kuwaiti dictionary (1964), Arabic/English Wiktionary (Najdi, Hijazi, Gulf entries), peer-reviewed papers
(الشهراني 2024، العنزي 2022، الشمري 2024، المفرّح 2024، AlAmmar 2017 PhD thesis) and Saudipedia.
Only the words and their meanings were taken; no long passages were copied. The sentences were written by a language model.

## MSA-side checks (PLAN_v2 D7)
| Source | Licence | Use |
|---|---|---|
| Arabic WordNet (AWN v2, via Open Multilingual Wordnet `omw-arb:1.4`, Python package `wn`) | CC BY-SA 3.0 ([OMW table](https://omwn.org/omw1.html)) | lexicon lookup for MSA equivalents |
| Arabic Wiktionary page titles (`arwiktionary-latest-all-titles-in-ns0.gz` from dumps.wikimedia.org) | CC BY-SA 4.0 (Wikimedia content licence) | lexicon lookup for MSA equivalents |

Both are only used to *check* MSA equivalents (`src/check_msa.py`); no text from them is added to the data.
The downloaded files live in `data/lexicons/` (not committed; `check_msa.py` docstring says how to get them).

Result on the 20 batches (2026-11-01): 4,883 MSA spans, 4,122 (84.4%) found in the open lexicons;
761 flagged in `data/generated/msa_check.csv` for review (mostly inflected or compound forms the lexicons lack, e.g. طازجة، متعب).
Flagged rows are not deleted. The Riyadh dictionary is only used by hand through the link column (no automatic access).
