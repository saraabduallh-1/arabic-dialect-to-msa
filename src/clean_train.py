"""Fix mechanical annotation errors in the original data (PLAN_v2 D5).

train: writes a corrected copy to data/sentences_clean/train.csv (the original stays as is).
dev/test: never modified; only counts and row numbers go into the report, so the user can
decide on a test_v2 (changing the test set breaks comparison with all earlier results).

Fixes:
- a span whose MSA equivalent is the same word (apart from diacritics) -> unmarked on both sides
- a span number present on one side only -> unmarked on that side
- rows with malformed brackets are left unchanged and listed for a manual fix
- vocative "يا" glued to the next word on one side while the other side separates it
  (e.g. ياحبيبي vs يا حبيبي) -> separated

usage: python src/clean_train.py
"""
import re

import pandas as pd

from data import _DIACRITICS, DATA, DIALECT, MSA, normalize, spans

OUT = DATA / "sentences_clean"
SPAN = re.compile(r"\(([^()]*)\)\s*(\d+)")


def same(a, b):
    """Identical apart from diacritics/spaces. Stricter than normalize(): يآكل/يأكل or دايما/دائما are
    real dialect-vs-MSA spelling differences, not annotation errors."""
    strip = lambda x: re.sub(r"\s+", " ", _DIACRITICS.sub("", x)).strip()
    return strip(a) == strip(b)


def unmark(text, keep):
    """Remove the markers of every span whose number is not in keep."""
    return SPAN.sub(lambda m: m.group(0) if int(m.group(2)) in keep else m.group(1), text)


def split_ya(text, other):
    """'ياحبيبي' -> 'يا حبيبي' when the other side writes 'يا حبيبي' (or 'يا' + the same word)."""
    other_words = set(normalize(re.sub(r"[()\d]", " ", other)).split())

    def fix(m):
        rest = m.group(2)
        return f"{m.group(1)}يا {rest}" if normalize(rest) in other_words and "يا" in other_words else m.group(0)
    return re.sub(r"(^|[\s(])يا([^\s()]{2,})", fix, text)


def malformed(text):
    """Stray brackets left after removing the valid (word)N spans, e.g. ')حساس)1' or '(مين(1'."""
    return bool(re.search(r"[()]", SPAN.sub("", text)))


def check(df):
    """Per-row problems: list of (kind, csv line number)."""
    out = []
    for i, r in df.iterrows():
        s, t = spans(r[DIALECT]), spans(r[MSA])
        line = i + 2
        if malformed(r[DIALECT]) or malformed(r[MSA]):
            out.append(("malformed brackets (left for manual fix)", line))
            continue
        if any(same(s[k], t[k]) for k in s if k in t):
            out.append(("span equals its MSA", line))
        if set(s) ^ set(t):
            out.append(("span number on one side only", line))
        if split_ya(r[DIALECT], r[MSA]) != r[DIALECT] or split_ya(r[MSA], r[DIALECT]) != r[MSA]:
            out.append(("glued يا", line))
    return out


def fix_row(r):
    if malformed(r[DIALECT]) or malformed(r[MSA]):  # numbering is shifted; unmarking would lose real pairs
        return r[DIALECT], r[MSA]
    d, m = split_ya(r[DIALECT], r[MSA]), split_ya(r[MSA], r[DIALECT])
    s, t = spans(d), spans(m)
    keep = {k for k in set(s) & set(t) if not same(s[k], t[k])}
    return unmark(d, keep), unmark(m, keep)


def main():
    OUT.mkdir(exist_ok=True)
    report = ["# Annotation errors in the original data (PLAN_v2 D5)", ""]
    for split in ("train", "dev", "test"):
        df = pd.read_csv(DATA / "sentences" / f"{split}.csv", dtype=str)
        probs = check(df)
        kinds = sorted({k for k, _ in probs})
        report += [f"## {split} ({len(df)} rows)", ""]
        report += [f"- {k}: {sum(x == k for x, _ in probs)} — csv lines {', '.join(str(l) for x, l in probs if x == k)}"
                   for k in kinds] or ["- none found"]
        if split == "train":
            fixed = df.copy()
            fixed[[DIALECT, MSA]] = [fix_row(r) for _, r in df.iterrows()]
            fixed.to_csv(OUT / "train.csv", index=False, encoding="utf-8")
            left = [p for p in check(fixed) if not p[0].startswith("malformed")]
            report += ["", f"Corrected copy: `data/sentences_clean/train.csv` "
                           f"({int((fixed[[DIALECT, MSA]] != df[[DIALECT, MSA]]).any(axis=1).sum())} rows changed, "
                           f"{len(left)} problems left)."]
        else:
            report += ["", "Not modified (user decides on a test_v2)."]
        report.append("")
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
