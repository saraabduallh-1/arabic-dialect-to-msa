"""Apply the D1 review decisions to the merged generated data (PLAN_v2 D1).

Never edits the batch files: reads data/generated/all.csv and writes
data/generated/all_clean.csv plus a report in data/generated/CLEANING.md.
all_clean.csv is NOT used for training until the user approves the decisions
(autowork/QUESTIONS.md); until then experiments keep using all.csv.

Default decisions (PLAN_v2 D1, used while the user has not decided):
- Persian letters: چ -> تش or ك per word (table below); unclear words -> drop the row. پ -> ب, گ -> ق, ڤ -> ف.
- Rare/old words: kept.
- Terms that differ from a train term only by ال or a final ه: kept as extra examples, but
  marked new_vocab=0 so statistics don't count them as new vocabulary.

usage: python src/clean_generated.py
"""
import re
from collections import Counter

import pandas as pd

from data import DATA, DIALECT, MSA, lexicon, load, normalize, spans

GEN = DATA / "generated"

# چ words found in the data: replacement, or None = unclear -> drop the row
CHE = {"سويچ": "سويتش", "سچين": "سكين", "چبريت": "كبريت", "چولة": None}
SIMPLE = str.maketrans({"پ": "ب", "گ": "ق", "ڤ": "ف"})


def fix_che(text):
    """Returns the text with known چ words replaced, or None if an unclear چ word remains."""
    for word, rep in CHE.items():
        if rep:
            text = text.replace(word, rep)
    return None if "چ" in text else text


def main():
    df = pd.read_csv(GEN / "all.csv", dtype=str).fillna("")
    n0 = len(df)
    log = Counter()

    for col in (DIALECT, MSA):
        before = df[col].copy()
        df[col] = df[col].str.translate(SIMPLE)
        log[f"پ/گ/ڤ replaced ({col})"] += int((before != df[col]).sum())
        before = df[col].copy()
        df[col] = df[col].map(fix_che)
        log[f"چ replaced ({col})"] += int((before.notna() & df[col].notna() & (before != df[col])).sum())
    unclear = df[DIALECT].isna() | df[MSA].isna()
    log["dropped: unclear چ word"] = int(unclear.sum())
    df = df[~unclear].copy()

    lex = lexicon(load("train"))
    base = lambda t: (t[2:] if t.startswith("ال") else t).rstrip("ه")
    lex_base = {base(k) for k in lex}

    def new_vocab(x):
        terms = [normalize(v) for v in spans(x).values()]
        return int(any(t not in lex and base(t) not in lex_base for t in terms))

    df["new_vocab"] = df[DIALECT].map(new_vocab)
    log["rows kept but not new vocabulary (ال/ه variant of a train term)"] = int((df["new_vocab"] == 0).sum())
    df.to_csv(GEN / "all_clean.csv", index=False, encoding="utf-8")

    report = ["# Cleaning report (PLAN_v2 D1, default decisions)", "",
              f"Input `all.csv`: {n0} rows -> `all_clean.csv`: {len(df)} rows.", "",
              "| change | rows |", "|---|---|"]
    report += [f"| {k} | {v} |" for k, v in log.items()]
    report += ["", "چ word table:", ""] + [f"- {w} -> {r or 'unclear, row dropped'}" for w, r in CHE.items()]
    report += ["", "Status: waiting for the user's review (autowork/QUESTIONS.md). Not used for training yet."]
    (GEN / "CLEANING.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
