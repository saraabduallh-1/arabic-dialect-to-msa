# Error analysis: ALLaM QLoRA +gen3 (best word system; 565 test spans)

| dialect word in train? | spans | correct (exact or judged) |
|---|---|---|
| yes, with the same MSA | 123 | 94% |
| yes, other MSA | 62 | 56% |
| only in the verified generated data | 16 | 75% |
| never seen | 364 | 56% |

64% of test spans are words never seen in training; that is the main loss.
The 386 lahajatna/Darsouni words cover only 16 test spans, which is why +gen4 (770 rows) did not help.
Of 380 unseen spans, the exact form is in lahajatna for 55 and in Darsouni's text for 70 (103 either).

## Finding the dialect words (src/tag_dialect.py, token tagger on the bracket marks)

| model | span F1 | token F1 | recall: form seen as dialect / only as fusha / unseen |
|---|---|---|---|
| MARBERTv2 | 80.3 | 89.1 | 86.9 / 35.3 (n=17) / 93.4 |
| CAMeLBERT-DA | 79.7 | 87.2 | 83.9 / 29.4 / 90.8 |

Forms that train shows only as fusha but are dialect in context are mostly missed.
284 of 3,636 dialect-marked forms in train are also left unmarked elsewhere (e.g. ولدك 11/12, يوم 12/11,
فيه 15/13, ما 134/22), partly context, partly inconsistent marking.
