# Autonomous work plan (2026-10-29 → user back ~2026-11-01)

Hourly cron prompt re-enters this file. Each run: read this file, pick the first unfinished item,
work on it, update the status here and in WORKLOG.md, commit (and push) progress.

Rules
- Generation subagents must not read dev/test/vocab/results (see prompts/collect_data.md). The main
  session has seen test examples, so it never writes data rows itself; it only orchestrates and checks.
- At most 2 generation subagents at a time; each batch writes its own file.
- Never train on dev/test. Generated data is training-only.
- Commit after each finished item; push to origin/main.

## A. Data batches (subagents, prompts/collect_data.md + batch block)
40 new terms × 3 sentences per batch. Batch k → file data/generated/batches/b{k:02d}.csv, ids 100000+k*1000+1…

| k | dialect | domain | status |
|---|---|---|---|
| 1 | نجدية | الحياة اليومية والبيت | done (120 rows) |
| 2 | حجازية | الحياة اليومية والبيت | done (120 rows) |
| 3 | شرقية | الحياة اليومية والبيت | done (120 rows) |
| 4 | جنوبية | الحياة اليومية والبيت | done (120 rows) |
| 5 | شمالية | الحياة اليومية والبيت | done (120 rows) |
| 6 | نجدية | المشاعر والصفات | done (120 rows) |
| 7 | حجازية | المشاعر والصفات | done (120 rows) |
| 8 | شرقية | المشاعر والصفات | done (120 rows) |
| 9 | جنوبية | المشاعر والصفات | done (105 rows, 35 terms — refs ran out) |
| 10 | شمالية | المشاعر والصفات | done (42 rows, 14 terms — northern refs exhausted) |
| 11 | نجدية | الناس والعلاقات والكلام | done (120 rows) |
| 12 | حجازية | الناس والعلاقات والكلام | done (81 rows, 27 terms) |
| 13 | شرقية | الناس والعلاقات والكلام | done (120 rows) |
| 14 | جنوبية | الناس والعلاقات والكلام | done (120 rows) |
| 15 | شمالية | الناس والعلاقات والكلام | skipped (northern refs exhausted; see 31) |
| 16 | نجدية | الطعام واللباس والأدوات | done (120 rows) |
| 17 | حجازية | الطعام واللباس والأدوات | done (120 rows) |
| 18 | شرقية | الطعام واللباس والأدوات | done (120 rows) |
| 19 | جنوبية | الطعام واللباس والأدوات | done (120 rows) |
| 20 | شمالية | الطعام واللباس والأدوات | skipped (see 32) |
| 21 | نجدية | الأفعال والحركة والعمل | done (120 rows) |
| 22 | حجازية | الأفعال والحركة والعمل | done (120 rows) |
| 23 | شرقية | الأفعال والحركة والعمل | running |
| 24 | جنوبية | الأفعال والحركة والعمل | running |
| 25 | شمالية | الأفعال والحركة والعمل | skipped (see 33) |
| 26 | عامة | تعابير وأمثال قصيرة شائعة | todo |
| 27 | عامة | الزمن والمكان والأحوال | todo |
| 28 | عامة | الطقس والطبيعة والحيوان | todo |
| 29 | عامة | الأطفال والمدرسة والعائلة | todo |
| 30 | عامة | السوق والمال والسيارات | todo |
| 31 | نجدية | الزمن والطقس والطبيعة والحيوان (replaces 15) | todo |
| 32 | نجدية | التعابير والأمثال القصيرة (replaces 20) | todo |
| 33 | نجدية | البادية والإبل والرعي (replaces 25) | todo |

## B. Merge → data/generated/all.csv — script ready (src/merge_generated.py); rerun + commit all.csv after the last batch
## C. Mac experiment — dropped: Mac (8GB) needs Adafactor+bs8, ~10x slower and not comparable with the PC baseline; all training moves to the PC package (D)
## D. PC package — done: autorun.py phase 2 (merge, rag_lexf / rag+gen / rag_lexf+gen for 4 LLMs, 3 seq2seq +gen, QLoRA +gen, report, analysis, judge). User runs: git pull, then .venv\Scripts\python src\autorun.py
## E. Analysis: gain on terms covered by generated data vs not; update README, data card, REPORT — todo
## F. Final summary for the user in autowork/WORKLOG.md — todo
