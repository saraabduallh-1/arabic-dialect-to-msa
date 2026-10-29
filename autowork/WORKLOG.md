# Work log

- 2026-10-29: plan created, caffeinate started (80h), hourly cron scheduled.
- 2026-10-29: batches 1 (Najdi, 120 rows; refs: Al-Abboudi dictionaries on archive.org, Al-Watan, Wiktionary) and 2 (Hijazi, 120 rows; ref: Wiktionary) done, checker OK, 0 dev/test overlap. References spot-checked: links resolve, book titles match.
- 2026-10-29: Mac training dropped (torch 2.15 crashes on macOS 14.2; 2.8 works but AdamW OOMs on 8GB). finetune.py gained --extra/--tag/--optim and no longer writes results on --limit smoke runs.
- 2026-10-29: run_llm/qlora gained --extra/--tag; merge_generated.py added; autorun.py phase 2 added (item D done). Batches 3 and 6 running.
- 2026-10-29: batch 6 done (Najdi feelings, 120 rows, ref: العبودي vols 1-13 on archive.org). Batch 7 started. Agents now get their own scratchpad subfolder (b06 reported a shared-scratchpad clash).
- 2026-10-29: batch 3 done (Eastern/Gulf home, 120 rows, ref: Wiktionary Gulf Arabic). REVIEW: some rows use Persian letters چ/پ (Gulf spelling, e.g. چبريت، سچين); Saudi data normally uses ك/ج — decide whether to normalize. Some terms are Kuwaiti-marked in Wiktionary but labelled شرقية. Batch 4 started.
- 2026-10-29: batch 7 done (Hijazi feelings, 120 rows, ref: Wiktionary Hijazi). Checked all 200 done terms: none in train lexicon, no dupes, no span equal to its MSA. REVIEW: some b07 terms are close to MSA forms (شقي، جيعان، سهران). Batch 8 started.
- 2026-10-29: batch 4 done (Southern, 120 rows; refs: الشهراني 2024 journal paper on لهجة شهران, العقيلي معجم اللهجة المحلية لمنطقة جازان). Links checked. Batch 5 started.
- 2026-10-29: batch 8 done (Eastern/Gulf feelings, 120 rows; refs: Wiktionary Gulf Arabic, جلال الحنفي معجم الألفاظ الكويتية 1964). REVIEW: Kuwaiti-source words labelled شرقية; مينون is a sound-change form of مجنون. Batch 9 started.
- 2026-10-29: batch 5 done (Northern/Hail, 120 rows; refs: العنزي 2022 HNSJ paper, AlAmmar 2017 Essex PhD thesis, Saudipedia Hail architecture). Only Hail found — no usable refs for Jouf/Tabuk. Batch 10 started.
- 2026-10-29 ~13:00: hit 5-hour usage limit; batches 9 and 10 interrupted. 14:17: limit reset, both resumed. Battery 25% and still discharging (charger not connected).
- 2026-10-29: batch 9 done (Southern feelings, 105 rows / 35 terms; refs: الشهراني 2024, العقيلي Jazan dictionary). Fell short of 40 rather than fill without references. Batch 11 started.
- 2026-10-29: batch 10 done but small (Northern feelings, 42 rows / 14 terms; refs: الشمري 2024 Azhar journal on Shammar idioms, العنزي 2022, العبودي). Northern refs exhausted → batches 15/20/25 replaced by Najdi 31/32/33. Batch 12 started.
- 2026-10-29: batch 11 done (Najdi people/speech, 120 rows; ref: العبودي vols 1-13). REVIEW: several are rare/old Najdi words (يفسكل، يتفنسق). Batch 13 started.
- 2026-10-29: batch 12 done (Hijazi people, 81 rows / 27 terms; ref: Wiktionary Hijazi). Hijazi references thinning: no usable newspaper/academic lists found. REVIEW: several b12 terms are pan-Arabic colloquial (هلا، سلامات، عقبال) rather than Hijazi-specific. Batch 14 started.
- 2026-10-29: batch 13 done (Eastern/Gulf people, 120 rows; refs: الحنفي 1964, Wiktionary Gulf). REVIEW: rare 1960s Kuwaiti words (درسعة، خماجير، ربربة). Batch 16 started.
