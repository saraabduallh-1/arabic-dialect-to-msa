"""Data of the foreign -> MSA project (m2m100 fine-tune, used here unchanged), turned into word spans.

The files have whole sentence pairs (Foriegn, MSA, Source). The foreign words are the source words that the
MSA side changed; we find them with a word alignment (difflib), so no hand labels are needed.

Splits used by the merged layer:
  train : annotated train rows (minus the dev rows) + a sample of the transliteration rows  -> tagger training
  dev   : 300 annotated train rows held out from the tagger (the m2m model has seen them)  -> tuning
  test  : the project's own validation file (553 annotated rows)                            -> final numbers
The m2m model chose its best checkpoint on the validation file, so its numbers there are slightly optimistic.

usage: python src/foreign_data.py      # overlap check + span statistics + samples -> results/merge/foreign_data.md
"""
import csv
import difflib
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from data import ROOT, clean, normalize

# an ASCII path: sentencepiece on Windows cannot open paths with Arabic letters
FOREIGN_DIR = Path(os.environ.get("FOREIGN_DIR", Path.home() / "models" / "foreign_to_msa"))
MODEL_DIR = FOREIGN_DIR / "foreign_to_msa_model"
OUT = ROOT / "results" / "merge"
N_DEV, N_TRANSLIT = 300, 3000


def load_tokenizer():
    """The saved tokenizer config comes from transformers 5 (extra_special_tokens is a list there); 4.x wants a dict.
    The language codes are rebuilt by M2M100Tokenizer itself, so nothing is lost."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_DIR, extra_special_tokens={})


def read(name):
    with open(FOREIGN_DIR / name, encoding="utf-8-sig") as f:
        return [r for r in csv.DictReader(f) if r["Foriegn"].strip() and r["MSA"].strip()]


def diff_spans(src_words, tgt_words):
    """[(start, end, msa)] source word ranges that the target changed (replace / delete)."""
    out = []
    sm = difflib.SequenceMatcher(a=[normalize(w) for w in src_words], b=[normalize(w) for w in tgt_words],
                                 autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete") and i2 > i1:
            out.append((i1, i2, " ".join(tgt_words[j1:j2])))
    return out


def row(r, i):
    src, tgt = clean(r["Foriegn"]).split(), clean(r["MSA"]).split()
    return {"id": f"f{i}", "src": " ".join(src), "tgt": " ".join(tgt), "words": src,
            "spans": diff_spans(src, tgt), "source": r.get("Source", "")}


def load_splits(seed=0):
    tr, va = read("trained_data_foriegn_to_msa.csv"), read("val_data_foriegn_to_msa.csv")
    test = [row(r, f"v{i}") for i, r in enumerate(va)]
    val_src = {normalize(r["src"]) for r in test}
    ann = [row(r, f"a{i}") for i, r in enumerate(tr) if r["Source"] == "annotated"]
    ann = [r for r in ann if normalize(r["src"]) not in val_src]  # keep the test file clean of exact repeats
    tl = [row(r, f"t{i}") for i, r in enumerate(tr) if r["Source"] != "annotated"]
    rnd = random.Random(seed)
    rnd.shuffle(ann)
    rnd.shuffle(tl)
    dev = ann[:N_DEV]
    dev_src = {normalize(r["src"]) for r in dev}
    train = [r for r in ann[N_DEV:] + tl[:N_TRANSLIT] if normalize(r["src"]) not in dev_src]
    return {"train": train, "dev": dev, "test": test}


def foreign_dict(train):
    """normalized foreign span -> Counter(msa), from the annotated training rows only."""
    d = defaultdict(Counter)
    for r in train:
        if r["source"] != "annotated":
            continue
        for s, e, m in r["spans"]:
            f = " ".join(r["words"][s:e])
            if m and normalize(f) != normalize(m):
                d[normalize(f)][m] += 1
    return d


def main():
    tr, va = read("trained_data_foriegn_to_msa.csv"), read("val_data_foriegn_to_msa.csv")
    tr_src = Counter(normalize(clean(r["Foriegn"])) for r in tr)
    tr_pair = {(normalize(clean(r["Foriegn"])), normalize(clean(r["MSA"]))) for r in tr}
    rep_src = sum(normalize(clean(r["Foriegn"])) in tr_src for r in va)
    rep_pair = sum((normalize(clean(r["Foriegn"])), normalize(clean(r["MSA"]))) in tr_pair for r in va)
    sp = load_splits()
    lines = ["# Foreign -> MSA data (from the other project, used as is)", "",
             f"Train file: {len(tr)} rows ({Counter(r['Source'] for r in tr)}); validation file: {len(va)} rows.", "",
             f"Validation sentences also in train: {rep_src} ({100 * rep_src / len(va):.1f}%); "
             f"same sentence and same MSA: {rep_pair} ({100 * rep_pair / len(va):.1f}%).", "",
             "| split | sentences | foreign spans | sentences with no change |", "|---|---|---|---|"]
    for k, rows in sp.items():
        n = sum(len(r["spans"]) for r in rows)
        z = sum(not r["spans"] for r in rows)
        lines.append(f"| {k} | {len(rows)} | {n} | {z} |")
    span_len = Counter(e - s for r in sp["train"] for s, e, _ in r["spans"])
    lines += ["", f"Span length in words (train): {dict(sorted(span_len.items())[:6])}", "",
              "## Samples (derived foreign spans)", ""]
    for r in sp["test"][:15] + sp["train"][-5:]:
        sps = "; ".join(f"{' '.join(r['words'][s:e])} ← {m}" for s, e, m in r["spans"]) or "(none)"
        lines.append(f"- {r['src']}  →  {sps}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "foreign_data.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
