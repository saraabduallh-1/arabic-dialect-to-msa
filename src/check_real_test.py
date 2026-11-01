"""Check a collected real test file (PLAN_v2 I4): format, duplicates, overlap with training data.

Writes <file>.checked.csv with the rows that pass. Rows that also appear in train or the
generated data are dropped (they could not measure generalization).

usage: python src/check_real_test.py data/real_test/round1.csv
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from data import DATA, DIALECT, clean, load, normalize

COLS = ["الجملة العامية", "اللهجة", "المنطقة", "الكلمات العامية", "المرجع الفصيح 1", "المرجع الفصيح 2", "المساهم"]
DIALECTS = {"نجدية", "حجازية", "شرقية", "جنوبية", "شمالية"}


def main(path):
    path = Path(path)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in COLS if c not in df.columns]
    if missing:
        sys.exit(f"missing columns: {missing}")

    problems = []
    for i, r in df.iterrows():
        n = len(r["الجملة العامية"].split())
        if not 4 <= n <= 20:
            problems.append((i, f"length {n} words (4-20)"))
        if r["اللهجة"] not in DIALECTS:
            problems.append((i, f"اللهجة must be one of {sorted(DIALECTS)}"))
        if not r["المرجع الفصيح 1"].strip():
            problems.append((i, "no MSA reference"))
        words = [w.strip() for w in r["الكلمات العامية"].split("/") if w.strip()]
        if not words:
            problems.append((i, "no dialect words listed"))
        elif any(normalize(w) not in normalize(r["الجملة العامية"]) for w in words):
            problems.append((i, "a listed dialect word is not in the sentence"))
    for i, msg in problems:
        print(f"row {i + 2}: {msg}")

    key = df["الجملة العامية"].map(normalize)
    seen = {normalize(r["src"]) for r in load("train")}
    gen = DATA / "generated" / "all.csv"
    if gen.exists():
        seen |= {normalize(clean(x)) for x in pd.read_csv(gen, dtype=str)[DIALECT]}
    dup, overlap = key.duplicated(), key.isin(seen)
    bad = df.index.isin([i for i, _ in problems])
    ok = df[~(dup | overlap | bad)]
    ok.to_csv(path.with_suffix(".checked.csv"), index=False, encoding="utf-8-sig")
    print(f"{len(df)} rows: {len(set(i for i, _ in problems))} with problems, {dup.sum()} duplicates, "
          f"{overlap.sum()} already in train/generated -> {len(ok)} kept")
    print("per dialect:", dict(Counter(ok["اللهجة"])))
    print("contributors:", ok["المساهم"].nunique())


if __name__ == "__main__":
    main(sys.argv[1])
