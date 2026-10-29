"""Check collected dialect -> MSA data before it is reviewed or used.

Validates the format (span markers, columns, sources), flags entries already in train, and
removes rows whose sentence matches a dev/test sentence (compared on normalized text, since
ids differ between files). Only counts are printed for dev/test, never their content.

usage: python src/check_generated.py data/generated/pilot.csv
       writes data/generated/pilot.clean.csv (rows without dev/test overlap)
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from data import DIALECT, MSA, clean, lexicon, load, normalize, spans

COLS = [DIALECT, MSA, "Feature 1", "اللهجة", "مقابلات بديلة", "نوع الجملة", "المصدر", "رابط المصدر"]
DIALECTS = {"نجدية", "حجازية", "شرقية", "جنوبية", "شمالية", "عامة"}
KINDS = {"منقولة", "مؤلفة"}


def main(path):
    path = Path(path)
    df = pd.read_csv(path, dtype=str).fillna("")
    errors, warnings = [], []

    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"missing columns: {missing}")

    for i, r in df.iterrows():
        row = f"row {i + 2}"  # line number in the csv
        s, t = spans(r[DIALECT]), spans(r[MSA])
        if not s:
            errors.append(f"{row}: no marked dialect span like (word)1")
        if set(s) != set(t):
            errors.append(f"{row}: span numbers differ between dialect {sorted(s)} and MSA {sorted(t)}")
        if any(normalize(s[k]) == normalize(t[k]) for k in s if k in t):
            warnings.append(f"{row}: a dialect span equals its MSA equivalent")
        if not r["المصدر"].strip():
            errors.append(f"{row}: no source")
        if r["نوع الجملة"] not in KINDS:
            errors.append(f"{row}: نوع الجملة must be one of {sorted(KINDS)}")
        if r["اللهجة"] not in DIALECTS:
            errors.append(f"{row}: اللهجة must be one of {sorted(DIALECTS)}")
        if r["رابط المصدر"] and not r["رابط المصدر"].startswith("http"):
            errors.append(f"{row}: رابط المصدر is not a URL")

    ids = pd.to_numeric(df["Feature 1"], errors="coerce")
    if ids.isna().any() or ids.duplicated().any() or (ids < 100001).any():
        errors.append("Feature 1 must be unique integers starting at 100001")

    src = df[DIALECT].map(lambda x: normalize(clean(x)))
    if src.duplicated().any():
        warnings.append(f"{src.duplicated().sum()} duplicate sentences inside the file")

    # overlap with the existing splits
    train = load("train")
    lex = lexicon(train)
    train_src = {normalize(r["src"]) for r in train}
    held = {normalize(r[k]) for s in ("dev", "test") for r in load(s) for k in ("src", "tgt")}
    held_terms = {normalize(d) for s in ("dev", "test") for r in load(s) for d, _ in r["pairs"]}
    terms = [normalize(v) for x in df[DIALECT] for v in spans(x).values()]

    no_new = sum(all(normalize(v) in lex for v in spans(x).values()) for x in df[DIALECT])
    if no_new:
        warnings.append(f"{no_new} rows have no dialect term that is new to the training lexicon")
    # same word as a train term apart from the article or a final ه (e.g. استكان / استكانة)
    base = lambda t: (t[2:] if t.startswith("ال") else t).rstrip("ه")
    lex_base = {base(k) for k in lex}
    near = sorted({normalize(v) for x in df[DIALECT] for v in spans(x).values()
                   if normalize(v) not in lex and base(normalize(v)) in lex_base})
    if near:
        warnings.append(f"{len(near)} terms differ from a train term only by ال or ه — swap them if they are "
                        f"the new term of a row: {', '.join(near)}")
    dup_train = src.isin(train_src).sum()
    if dup_train:
        warnings.append(f"{dup_train} sentences already exist in train")
    leak = src.isin(held) | df[MSA].map(lambda x: normalize(clean(x))).isin(held)

    print(f"{len(df)} rows, {len(terms)} dialect terms "
          f"({100 * sum(len(t.split()) > 1 for t in terms) / max(len(terms), 1):.0f}% multi-word)")
    print("dialects:", dict(Counter(df["اللهجة"]).most_common()))
    top = Counter(df["اللهجة"]).most_common(1)
    if top and top[0][1] > 0.4 * len(df):
        warnings.append(f"more than 40% of rows are {top[0][0]}")
    print("sentence kinds:", dict(Counter(df["نوع الجملة"])))
    print("sources:", df["المصدر"].nunique())
    print(f"dev/test: {leak.sum()} sentences overlap (removed in the clean file); "
          f"{sum(t in held_terms for t in set(terms))} distinct terms also occur in dev/test (allowed, reported)")

    for w in warnings:
        print("WARNING", w)
    for e in errors:
        print("ERROR", e)
    out = path.with_suffix(".clean.csv")
    df[~leak].to_csv(out, index=False, encoding="utf-8")
    print(f"{'FAILED' if errors else 'OK'}: {len(errors)} errors, {len(warnings)} warnings; wrote {out}")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main(sys.argv[1])
