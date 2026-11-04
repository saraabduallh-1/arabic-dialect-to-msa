# Corrected project scope — term-level dialect to MSA

**Goal:** Detect colloquial Arabic terms/expressions in a user-supplied sentence and propose their Modern Standard Arabic equivalents **only**. Do not rewrite, translate, or normalize the entire sentence.

## Expected behavior
Input: `الاجتماع كان خربوطي`
Output: `{"matches":[{"surface":"خربوطي","suggestion":"غير منظم"}]}`

Input: `مدري وين حطيت الملف`
Output: `{"matches":[{"surface":"مدري","suggestion":"لا أعلم"}]}`

If there is no supported colloquial term, return `{"matches":[]}`.

## Data schema
Each approved record should include:
- `term_dialect`: exact colloquial term, possibly multiword
- `term_msa`: approved MSA equivalent
- `context_dialect`: optional context to disambiguate sense
- `sense_id`, `region`, `variants`, `part_of_speech`
- `source_title`, `source_url`, `source_record_id`, `source_license`
- `provenance`: attested / dictionary / synthetic
- `review_status`, `reviewer_notes`
- `split`: train/dev/test assigned by term family to reduce leakage

## Evaluation
- term detection: span precision, recall, F1
- term substitution: exact/accepted-alternative accuracy, contextual sense accuracy
- false-positive rate on non-colloquial text
- multiword expressions and ambiguous senses
- preservation: no changes outside detected spans

## Do not use as training data
- Full sentence translation targets from `src/finetune.py` without adapting the task
- `data/augmentation/synthetic_candidates_v1.csv` (unreviewed and sentence-level)
- `experiments/build_large_candidate_dataset.py` outputs (quotation templates; not term-level gold data)
- Prior Colab baseline results as evidence of term-level accuracy

## Next implementation
1. Extract only aligned span pairs from **train** using `src/data.py`; keep source sentence as context, and retain original ID.
2. Validate pair meanings and correct annotations with dictionary evidence and review.
3. Add licensed attested dialect term inventories, variants, and multiword expressions.
4. Generate **term-level** examples with realistic contexts, not sentence rewrites.
5. Train a term detector plus contextual equivalent selector, and benchmark against the current baseline.
