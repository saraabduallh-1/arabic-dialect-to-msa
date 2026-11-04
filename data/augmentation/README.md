# Original synthetic Saudi/Gulf dialect-to-MSA candidate data

This folder contains **50 newly authored sentence pairs** (not extracted from a dictionary and not claimed to be attested speech). They are **candidate training data only**, not automatically added to the training set.

- File: `synthetic_candidates_v1.csv`
- Columns: id, dialect, msa, source_type, region_hint, review_status
- Provenance: original examples written for this project; `source_type=original_synthetic`
- Review status: `needs_human_review` for all records. The regional label is a broad heuristic, **not** verified dialect metadata.
- Important: some expressions are ambiguous or may be formalized in more than one valid way. A fluent Saudi Arabic reviewer should approve or correct pairs before training.
- Never mix with dev/test. Before approval, check exact and normalized overlap against all existing train/dev/test sources; also manually check near duplicates and paraphrases.
- Keep an immutable approved copy, split-independent record IDs, and a record of reviewer corrections.

## Next steps
1. Run automated schema/empty/duplicate/leakage checks.
2. Manually review dialect authenticity and meaning preservation.
3. Promote only approved rows to a separate additional-training file with the exact format expected by `src/finetune.py --extra`.
4. Compare against the unchanged baseline on unchanged dev/test, then evaluate a new held-out human-reviewed test set.

**No model training is started by these files.**
