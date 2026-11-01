# تسليم العمل للـ PC — 2026-11-01

## الملخص (للمستخدمة)

### وش توصلنا له على الماك
**البيانات الجديدة:** 20 دفعة جاهزة في `data/generated/batches/` ← **2,268 جملة** و**778 كلمة عامية جديدة**، كل كلمة بمرجع حقيقي.

| اللهجة | الجمل |
|---|---|
| نجدية | 597 |
| حجازية | 561 |
| شرقية/خليجية | 480 |
| جنوبية | 465 |
| شمالية (حائل) | 162 |

- **المراجع الرئيسية:** معجما العبودي («معجم الأصول الفصيحة للألفاظ الدارجة» 13 جزءًا، و«معجم الكلمات الدخيلة»)، معجم العقيلي لجازان، معجم الحنفي للألفاظ الكويتية (1964)، ويكاموس (أقسام النجدية والحجازية والخليجية)، أبحاث محكّمة (الشهراني 2024، العنزي 2022، الشمري 2024، المفرّح 2024، رسالة دكتوراه العمّار 2017)، وسعوديبيديا.
- **الفحوص:** لا تداخل مع جمل dev/test، لا تكرار، كل صف فيه كلمة جديدة على التدريب. الوكلاء اللي ولّدوا البيانات ما فتحوا ملفات الاختبار.

**تنبيه مهم:** 34 فقط من 564 كلمة عامية في الاختبار (6%) غير موجودة في التدريب وتغطيها البيانات الجديدة. يعني **التحسن على مجموعة الاختبار الحالية غالبًا بيكون صغير** حتى لو البيانات ممتازة؛ فائدتها الأكبر في تغطية مفردات جديدة خارج هذا الاختبار.

**الكود الجاهز:**
- `rag_lexf`: تلميحات قاموس مرشّحة (دقتها 77% بدل 47%)
- `--extra generated/all --tag +gen` في `run_llm.py` و`finetune.py` و`qlora.py`
- `src/judge.py` (محكّم بالمعنى)، `src/analysis.py` (فيه قسم جديد يقيس أثر البيانات المولّدة)
- `src/autorun.py`: المرحلة الثانية كاملة

### وش باقي (يكمله الـ PC)
1. (اختياري) **10 دفعات باقية** (23، 24، 26–33) ← قرابة 1,000 جملة إضافية
2. **التجارب** بأمر واحد: `autorun.py`
3. **مراجعتك أنتِ** لنقاط الجودة (القسم الأخير)

---

## Instructions for the Claude session on the PC

Read this file, then `autowork/PLAN.md` and `autowork/WORKLOG.md`. Python is `.venv\Scripts\python`.

### Step 1 — remaining data batches (optional; ask the user first whether to do them or go straight to Step 2)
- Remaining: 23, 24, 26–33 (see PLAN.md table).
- Generate **only through subagents** with `prompts/collect_data.md` plus a batch block (number, dialect, domain, 40 terms × 3 sentences, output `data/generated/batches/bNN.csv`, ids `1NN001…`, own helper-script folder). Copy the batch block style from the earlier batches' rows in WORKLOG.md.
- Add these quality rules to every batch block (learned from batches 21–22):
  - the **lemma itself** must be new — not a different inflection of a word already in train.csv;
  - only words in **common present-day use** (skip ones the reference marks old/rare/obsolete);
  - skip ordinary MSA words with the same meaning; standard Arabic letters only (no چ/پ/گ).
- Subagents must never open `data/sentences/dev.csv`, `data/sentences/test.csv`, `data/vocab/`, `results/`. The orchestrating session doesn't write data rows itself.
- At most 2 subagents at a time, on different dialect+domain. Usage limits may interrupt them — resume the same agent with SendMessage after the reset.
- After each batch: update PLAN.md status and WORKLOG.md, commit `bNN.csv` + `terms_done.txt` (the `.clean.csv` files are gitignored), push.

### Step 2 — merge + experiments
```
.venv\Scripts\python src\merge_generated.py
.venv\Scripts\python src\autorun.py
```
- `results/autorun_state.json` already marks every phase-1 step as done, so autorun goes straight to phase 2: merge → `rag_lexf`, `rag+gen` and `rag_lexf+gen` for qwen3.8:27b, gemma3:12b, command-r7b-arabic:7b and aya-expanse:8b → AraT5v2/AraBART/mT5 `+gen` → report → analysis → judge (qwen3.8:27b, top 15) → QLoRA `+gen` (slowest, last) → report 3.
- Each step commits and pushes, and is resumable (re-run the same command after a crash or reboot). Progress: `results/autorun.log`.
- If more batches are added **after** the `merge generated` step ran, delete that key (and the `+gen` steps you want redone) from `results/autorun_state.json`, and delete the matching `results/preds/*+gen*.jsonl` files (run_llm resumes from existing predictions).
- `sacrebleu` is needed by `analysis.py`/`metrics.py` (already installed for the benchmark; if not: `.venv\Scripts\python -m pip install sacrebleu`).

### Step 3 — write-up
- Section 8 of `results/ANALYSIS.md` is the key result: the change in chrF++ (with p-values) for each `+gen` system against its base, and the change in correct% on test spans covered by the generated data vs not covered.
- Check judge agreement with the manual labels at the top of `results/JUDGE_qwen3.8_27b.md`. If agreement is low, say so and don't rank by the judge.
- Update README.md (data section: the generated data, its references and limits) and write the final summary in WORKLOG.md (PLAN item F).

---

## نقاط تحتاج مراجعتك (من WORKLOG)
1. **حروف فارسية** في الدفعة 3 (چبريت، سچين — 12 صف): تحويلها لـ ج/ك أو تركها؟
2. **كلمات قريبة من الفصحى** في الدفعات 7 و12 (شقي، جيعان، سهران، هلا، سلامات، عقبال).
3. **كلمات نادرة أو قديمة** في الدفعتين 11 و21 (يفسكل، يتفنسق، هكف، نسع، كنهر)، وأفعال فصيحة عادية في 21 (مسح، همز).
4. **كلمات موجودة في التدريب بصيغة أخرى** في الدفعة 22 (نط، نقز، ساق) و20 كلمة تختلف عن كلمات التدريب بـ«ال» أو «ه» فقط ← هذي أمثلة إضافية وليست مفردات جديدة.
5. **ألفاظ كويتية** مصنّفة «شرقية» (الدفعتان 8 و13).
6. **مقابلات فصيحة وصفية** لأسماء أكلات وأدوات (القشد، الكليجا، الشنفاص).
7. **مراجعة عينة من الروابط**: افتحي 20 رابطًا عشوائيًا وتأكدي أن الكلمة موجودة فعلًا بالمعنى المذكور.
