# Work log

- 2026-10-29: plan created, caffeinate started (80h), hourly cron scheduled.
- 2026-10-29: batches 1 (Najdi, 120 rows; refs: Al-Abboudi dictionaries on archive.org, Al-Watan, Wiktionary) and 2 (Hijazi, 120 rows; ref: Wiktionary) done, checker OK, 0 dev/test overlap. References spot-checked: links resolve, book titles match.
- 2026-10-29: Mac training dropped (torch 2.15 crashes on macOS 14.2; 2.8 works but AdamW OOMs on 8GB). finetune.py gained --extra/--tag/--optim and no longer writes results on --limit smoke runs.
