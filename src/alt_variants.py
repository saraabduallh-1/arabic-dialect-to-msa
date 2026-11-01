"""PLAN_v2 D4.2: a second copy of each generated row using an alternative MSA equivalent.

For every row of data/generated/all2.csv whose `مقابلات بديلة` lists alternatives, the MSA span of
the row's new term (the span whose dialect word is in terms_done.txt) is replaced by an alternative.
Conservative, because a plain swap can break MSA grammar: only one-word span -> one-word alternative,
not already in the sentence, and with the same ending (tanween alif / taa marbuta / other).
Teaches that a dialect word can have more than one correct MSA equivalent. Training only, and only
after a reviewed sample (autowork/QUESTIONS.md): a swap can still change number or verb form.
Writes data/generated/alt_variants.csv (ids: original id * 10 + 1; column `أصل` = original id).

usage: python src/alt_variants.py
"""
import re

import pandas as pd

from data import DATA, DIALECT, MSA, SPAN_RE, normalize, spans

GEN = DATA / "generated"


def ending(word):
    w = re.sub(r"[ً-ْ]", "", word)
    # final alif usually marks the accusative (صموتا), so it must match too
    return "alif" if w.endswith("ا") else "a" if w.endswith("ة") else "other"


def main():
    df = pd.read_csv(GEN / "all2.csv", dtype=str).fillna("")
    done = {normalize(t) for t in (GEN / "terms_done.txt").read_text(encoding="utf-8").splitlines() if t.strip()}
    out = []
    for _, r in df.iterrows():
        alts = [a.strip() for a in r["مقابلات بديلة"].split("/") if a.strip()]
        s, t = spans(r[DIALECT]), spans(r[MSA])
        new = [k for k in sorted(s) if normalize(s[k]) in done and k in t]
        if not alts or len(new) != 1 or len(t[new[0]].split()) != 1:
            continue
        k = new[0]
        sent = normalize(re.sub(r"[()\d]", " ", r[MSA])).split()
        alt = next((a for a in alts if len(a.split()) == 1 and normalize(a) != normalize(t[k])
                    and normalize(a) not in sent and ending(a) == ending(t[k])), None)
        if alt is None:
            continue
        msa = SPAN_RE.sub(lambda m: f"({alt}){k}" if int(m.group(2)) == k else m.group(0), r[MSA])
        out.append({**r.to_dict(), MSA: msa, "Feature 1": str(int(r["Feature 1"]) * 10 + 1), "أصل": r["Feature 1"]})
    pd.DataFrame(out).to_csv(GEN / "alt_variants.csv", index=False, encoding="utf-8")
    print(f"{len(df)} rows -> {len(out)} variants written to {GEN / 'alt_variants.csv'}")


if __name__ == "__main__":
    main()
