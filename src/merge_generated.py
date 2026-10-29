"""Merge the generated batches into data/generated/all.csv (training-only extra data).

Drops duplicate sentences and any row whose dialect or MSA sentence matches a dev/test
sentence (normalized text). Prints counts only, never dev/test content.

usage: python src/merge_generated.py
"""
from collections import Counter

import pandas as pd

from data import DATA, DIALECT, MSA, clean, lexicon, load, normalize, spans

GEN = DATA / "generated"


def main():
    files = sorted(p for p in (GEN / "batches").glob("b*.csv") if not p.name.endswith(".clean.csv"))
    df = pd.concat([pd.read_csv(p, dtype=str).fillna("").assign(batch=p.stem) for p in files], ignore_index=True)
    n0 = len(df)
    key = lambda x: normalize(clean(x))

    held = {normalize(r[k]) for s in ("dev", "test") for r in load(s) for k in ("src", "tgt")}
    leak = df[DIALECT].map(key).isin(held) | df[MSA].map(key).isin(held)
    df = df[~leak]
    dup = df[DIALECT].map(key).duplicated()
    df = df[~dup]
    bad = df.apply(lambda r: not spans(r[DIALECT]) or set(spans(r[DIALECT])) != set(spans(r[MSA])), axis=1)
    df = df[~bad]
    df.drop(columns="batch").to_csv(GEN / "all.csv", index=False, encoding="utf-8")

    lex = lexicon(load("train"))
    terms = {normalize(v) for x in df[DIALECT] for v in spans(x).values()}
    new = {t for t in terms if t not in lex}
    print(f"{len(files)} batches, {n0} rows -> {len(df)} "
          f"(dropped {leak.sum()} dev/test overlaps, {dup.sum()} duplicates, {bad.sum()} malformed)")
    print(f"{len(terms)} distinct dialect terms, {len(new)} new to the training lexicon")
    print("dialects:", dict(Counter(df["اللهجة"]).most_common()))
    print("rows per batch:", dict(Counter(df["batch"])))
    print("references:", df["المصدر"].nunique())


if __name__ == "__main__":
    main()
