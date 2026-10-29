# Work log

- 2026-10-29: plan created, caffeinate started (80h), hourly cron scheduled.
- 2026-10-29: batches 1 (Najdi, 120 rows; refs: Al-Abboudi dictionaries on archive.org, Al-Watan, Wiktionary) and 2 (Hijazi, 120 rows; ref: Wiktionary) done, checker OK, 0 dev/test overlap. References spot-checked: links resolve, book titles match.
- 2026-10-29: Mac training dropped (torch 2.15 crashes on macOS 14.2; 2.8 works but AdamW OOMs on 8GB). finetune.py gained --extra/--tag/--optim and no longer writes results on --limit smoke runs.
- 2026-10-29: run_llm/qlora gained --extra/--tag; merge_generated.py added; autorun.py phase 2 added (item D done). Batches 3 and 6 running.
- 2026-10-29: batch 6 done (Najdi feelings, 120 rows, ref: العبودي vols 1-13 on archive.org). Batch 7 started. Agents now get their own scratchpad subfolder (b06 reported a shared-scratchpad clash).
- 2026-10-29: batch 3 done (Eastern/Gulf home, 120 rows, ref: Wiktionary Gulf Arabic). REVIEW: some rows use Persian letters چ/پ (Gulf spelling, e.g. چبريت، سچين); Saudi data normally uses ك/ج — decide whether to normalize. Some terms are Kuwaiti-marked in Wiktionary but labelled شرقية. Batch 4 started.
- 2026-10-29: batch 7 done (Hijazi feelings, 120 rows, ref: Wiktionary Hijazi). Checked all 200 done terms: none in train lexicon, no dupes, no span equal to its MSA. REVIEW: some b07 terms are close to MSA forms (شقي، جيعان، سهران). Batch 8 started.
