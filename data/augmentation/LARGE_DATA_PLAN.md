# Large-scale candidate generation — NOT approved training data

Run `python experiments/build_large_candidate_dataset.py` from the repository root (after installing pandas).

The script extracts **only the annotated dialect→MSA span mappings from `data/sentences/train.csv`**. It creates six quotation/mention-style frames per unique mapping, filters exact normalized sentence overlaps against the project's dev and test sets, and writes a CSV and report.

**Scientific limitation:** This is a scale/provenance demonstration, not the requested natural, dictionary-grounded corpus. These are repetitive metalinguistic contexts, not ordinary Saudi conversations. Do **not** use them for fine-tuning as-is. No external dictionary has been imported, and no human review has occurred. Source mappings may contain errors.

## External source registry (review before importing)
- Riyadh Dictionary: https://dictionary.ksaa.gov.sa/ — reputable MSA reference; **license for bulk extraction and redistribution not established**.
- Saudi Dialect Corpus: https://github.com/TaghreedT/SDC — authentic Saudi dialect corpus, but **reuse/redistribution license not established**.
- fr3on/arabic-dialect-corpus: https://huggingface.co/datasets/fr3on/arabic-dialect-corpus — publisher declares MIT; contains Saudi-labeled records, **not parallel MSA translations**. Verify individual file license, provenance and labels before use.
- Source samples should remain in separate datasets with source URL, original ID, access date, license, transformation, quality score, and reviewer status.

## Proper production pipeline
1. Verify source rights and exact record-level provenance.
2. Extract Saudi source utterances and approved dialect↔MSA dictionary mappings.
3. Generate contextual MSA translations, retaining source text separately.
4. Apply human review, dialect/meaning checks, near-duplicate filtering and cross-split contamination controls.
5. Train only on approved pairs, keeping original dev/test unchanged and reporting synthetic vs attested results separately.
