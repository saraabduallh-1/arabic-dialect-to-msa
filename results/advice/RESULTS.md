# Applying the KASS advice (docs/ADVICE_FROM_KASS.md)

All choices were made on dev; test was run once with the chosen settings.

## Dev predictions (new)

| system | dev WordAcc% | test WordAcc% |
|---|---|---|
| ALLaM QLoRA +gen3 | **48.11** | 46.90 |
| ALLaM QLoRA +gen4+lex | 46.67 | 48.32 |
| gemma3:27b rag+gen2 | 40.36 | 42.30 |

On dev the dictionary-hint model is worse, so its +1.4 on test (not significant) does not hold up; +gen3 is used below.

## Advice 1-3: two tiers + agreement + no-copy rule (`src/tiers.py`)

"suggested" = the form is in train >= 2 times with the same MSA in >= 80% of them (dict), or ALLaM and gemma give the same answer (agree).
Everything else is "review" (ALLaM's answer; a copied dialect word is replaced by the dict or gemma answer).
Thresholds chosen on dev (`tiers_sweep_dev.md`).

| split | tier | share of words | exact% | judged correct% (test) |
|---|---|---|---|---|
| dev | suggested | 44.3 | 74.4 | — |
| dev | review | 55.7 | 27.8 | — |
| test | **suggested** | **40.0** | 79.6 | **90.3** |
| test | — dict | 11.0 | 85.5 | 88.7 (lower bound: dict answers that differ from both systems were not judged) |
| test | — agree | 29.0 | 77.4 | 90.9 |
| test | review | 60.0 | 25.7 | 49.0 |
| test | all | 100 | 47.3 | 65.5 (ALLaM alone 65.1) |

T1 (85%) is met for the "suggested" tier: 40% of the dialect words are replaced automatically at ~90% correct (judge), the other 60% are flagged.
The no-copy rule touched only ~10 spans per split (small effect, as expected).

## Advice 5: choose words by tweet frequency (`src/tweet_freq.py`, 2.1 B tweet tokens from KASS)

| | dev | test |
|---|---|---|
| unseen single-word dialect spans | 294 | 320 |
| median tweet frequency of those words | 2,707 | 1,727 |
| in lahajatna at all | 33 | 45 |
| covered by the 770 verified rows | 1 | 0 |

The verified (gen4) words have median tweet frequency 96, against ~2,000 for the dialect words of dev/test.
So the missing words are everyday words, but lahajatna mostly does not list them: ranking lahajatna by frequency can cover
at most 33/294 of the unseen dev words. The source of new words must be real text (tweets), not the dictionary.

## Advice 6 (first try): corpus dialectness score (`src/dialectness.py`) — did not work

Log-odds of tweets vs fusha text (KASS Wikipedia sample 22 MB + the MSA side of train).
AUC on dev for dialect vs other tokens: **0.44** (below chance); for forms train shows only unmarked: 0.45.
The top "dialect" candidates are fusha words (الله، لكن، جدا، اليوم): the score separates conversational from
encyclopedic register, not dialect from fusha, and the fusha sample is too small. Not used further.
A usable version needs a large fusha reference of the same register (news or MSA tweets) or a labelled seed list.

## Advice 7: alternative correct answers

`dev_alt_answers_review.csv`: 559 dev answers (ALLaM or gemma) that differ from the reference, blind (no system name),
for human marking (correct / partial / wrong). A 200-row sample is enough for the kappa and the per-tier accuracy.

## Not done yet

- Advice 4 (short MSA equivalent per lexicon entry): needs model review per entry; postponed.
- Advice 6 (detector prior from train marking ratio): the cheap corpus version failed; the train-ratio feature remains to try.

## Human review of 200 dev answers that differ from the reference (one annotator, blind to the system)

| system | n | correct | partial | wrong |
|---|---|---|---|---|
| ALLaM +gen3 | 99 | 71 (71.7%) | 15 | 13 |
| gemma3:27b | 101 | 67 | 19 | 15 |
| all | 200 | 138 (69%) | 34 (17%) | 28 (14%) |

ALLaM dev: 48.1% exact + 51.9% non-exact × 71.7% judged correct by the human ≈ **85% correct by meaning** (sample of 99, rough 95% interval ±5 points).
Exact match against one reference therefore underestimates the system by ~37 points. Caveats: one annotator, who is also the data author; needs a second annotator and kappa.

## Tweet vocabulary (human-verified) as a lexicon for the tiers (dev only)

gemma3:27b proposed MSA equivalents for the 2,300 most frequent tweet-mined dialect forms; 2,074 it judged Saudi/Gulf dialect.
Human review of the 391 most frequent: 357 correct (91.3%; 10 of them with a corrected equivalent), 25 wrong equivalent, 9 not Saudi.

| use of the 357 verified words on dev | dev spans touched | effect |
|---|---|---|
| override (vocab answer replaces ALLaM) | 14 | exact 28.6% vs ALLaM 57.1% (8/14) on the same spans: worse |
| confirm only (vocab == ALLaM → suggested) | 1 | none (agreement with gemma already catches them) |
| same with all 2,074 unreviewed proposals | 1 | none |

A fixed lexicon entry ignores context and inflection (قاعد → يجلس, but the sentence needs جالس; حاط → يضع vs وضعت),
and ALLaM already translates these frequent words correctly in context. The verified vocabulary therefore does not raise
dev accuracy; it is kept as a resource (data/external/tweet_dialect/verified_vocab.json) and the override is not used.
The remaining errors are not missing everyday words: see the human review (exact match underestimates by ~37 points).

## End-to-end pipeline (`src/pipeline.py`): raw sentence → dialect tagger → ALLaM → tiers

The earlier pipeline numbers (73% found) were lowered by a tokenisation mismatch in the evaluation; fixed (gold tokenisation).
Second model for the agreement tier: Qwen2.5-7B QLoRA +gen3 instead of gemma3:27b (dev, gold spans: 38.4% suggested at 73.7% exact vs 44.3% / 74.4%).
Train-ratio / tweet-vocab boost of the tagger: +0.6 recall on dev, not used (`results/pipeline/tagger_boost_dev.md`).
Settings fixed on dev; test run once. See `results/pipeline/dev.md`, `test.md`.

### Pipeline on test judged by meaning (`src/judge_pipeline.py`, judge qwen3.8:27b)

| | share of spans that need a change | correct | partial | wrong |
|---|---|---|---|---|
| suggested | 27.8% | **89.8%** | 2.5% | 7.6% |
| review | 51.7% | 53.8% | 7.9% | 38.4% |
| missed by the tagger | 20.5% | | | |
| all | 100% | 52.7% | 4.8% | |

The judge is much stricter than the human: on the 200 dev answers the user labelled it calls 36% correct where she called 69%
(agreement correct-vs-not only 58%). These judged numbers are therefore a lower bound; the "suggested" tier stays near 90% even so.
