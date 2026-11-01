"""Build the human review sample for the generated data (PLAN_v2 D2).

Writes data/generated/review_sample.csv: 5 random rows per batch, plus 20 random distinct
source links (rows marked نوع=رابط). The user fills الحكم with one of:
صحيح / المعنى خطأ / الجملة غير طبيعية / الرابط ما فيه الكلمة

After the review: python src/review_sample.py --score  (error rate per batch and per source)

usage: python src/review_sample.py
"""
import argparse
import random
from urllib.parse import quote

import pandas as pd

from data import DATA, DIALECT, MSA, clean, lexicon, load, normalize, spans

GEN = DATA / "generated"
OUT = GEN / "review_sample.csv"
VERDICTS = ["صحيح", "المعنى خطأ", "الجملة غير طبيعية", "الرابط ما فيه الكلمة"]


def new_term(row, lex):
    """(dialect term, MSA equivalent) of the row's term that is new to the train lexicon."""
    s, t = spans(row[DIALECT]), spans(row[MSA])
    for k in sorted(s):
        if normalize(s[k]) not in lex:
            return s[k], t.get(k, "")
    k = min(s)
    return s[k], t.get(k, "")


def build():
    random.seed(7)
    lex = lexicon(load("train"))
    files = sorted(p for p in (GEN / "batches").glob("b*.csv") if not p.name.endswith(".clean.csv"))
    out = []
    for p in files:
        df = pd.read_csv(p, dtype=str).fillna("")
        for _, r in df.sample(n=min(5, len(df)), random_state=random.randrange(10**6)).iterrows():
            term, msa = new_term(r, lex)
            out.append({"النوع": "جملة", "الدفعة": p.stem, "Feature 1": r["Feature 1"], "اللهجة": r["اللهجة"],
                        "الكلمة": term, "المعنى المذكور": msa, "مقابلات بديلة": r["مقابلات بديلة"],
                        "الجملة العامية": clean(r[DIALECT]), "الجملة الفصحى": clean(r[MSA]),
                        "المصدر": r["المصدر"], "الرابط": r["رابط المصدر"],
                        "رابط معجم الرياض": "https://dictionary.ksaa.gov.sa/result/" + quote(msa),
                        "الحكم": "", "ملاحظة": ""})
    # 20 random distinct links: does the page really contain the word with this meaning?
    alldf = pd.concat([pd.read_csv(p, dtype=str).fillna("").assign(batch=p.stem) for p in files])
    alldf = alldf[alldf["رابط المصدر"].str.startswith("http")]
    links = alldf.drop_duplicates("رابط المصدر").sample(n=20, random_state=11)
    for _, r in links.iterrows():
        term, msa = new_term(r, lex)
        out.append({"النوع": "رابط", "الدفعة": r["batch"], "Feature 1": r["Feature 1"], "اللهجة": r["اللهجة"],
                    "الكلمة": term, "المعنى المذكور": msa, "مقابلات بديلة": r["مقابلات بديلة"],
                    "الجملة العامية": clean(r[DIALECT]), "الجملة الفصحى": clean(r[MSA]),
                    "المصدر": r["المصدر"], "الرابط": r["رابط المصدر"],
                    "رابط معجم الرياض": "https://dictionary.ksaa.gov.sa/result/" + quote(msa),
                    "الحكم": "", "ملاحظة": ""})
    pd.DataFrame(out).to_csv(OUT, index=False, encoding="utf-8-sig")  # BOM so Excel shows Arabic
    print(f"wrote {OUT} ({len(out)} rows); الحكم: {' / '.join(VERDICTS)}")


def score():
    df = pd.read_csv(OUT, dtype=str).fillna("")
    done = df[df["الحكم"].isin(VERDICTS)]
    print(f"{len(done)}/{len(df)} rows judged")
    done = done.assign(error=(done["الحكم"] != "صحيح").astype(int))
    for col in ("الدفعة", "المصدر"):
        t = done.groupby(col)["error"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        t["mean"] = (100 * t["mean"]).round(1)
        print(f"\nerror % by {col} (>20% -> exclude or fully review):\n{t.to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    score() if ap.parse_args().score else build()
