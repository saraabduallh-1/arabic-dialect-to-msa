"""Merge the verified generated batches into data/generated/verified.csv (used as --extra generated/verified).

Kept rows:
  - every row of data/generated/vbatches/vNN.clean.csv (prompts/collect_verified.md, no dev/test overlap)
    that the independent reviewer (prompts/verify_batch.md) marked اعتمد in vNN.verdict.csv;
    a batch without a verdict file is skipped;
  - the older generated rows (data/generated/all2.csv) the user approved in the review workbook
    (data/generated/review_all.csv, الحكم = اعتمد).

usage: python src/merge_verified.py
"""
import pandas as pd

from data import DATA, DIALECT, MSA

GEN = DATA / "generated"
BATCHES = GEN / "vbatches"
OUT = GEN / "verified.csv"


def main():
    parts, report = [], []
    for p in sorted(BATCHES.glob("v[0-9][0-9].clean.csv")):
        verdict = p.with_name(p.name.replace(".clean.csv", ".verdict.csv"))
        if not verdict.exists():
            report.append(f"{p.stem}: no verdict file, skipped")
            continue
        df = pd.read_csv(p, dtype=str).fillna("")
        v = pd.read_csv(verdict, dtype=str).fillna("")
        keep = set(v.loc[v["القرار"].str.strip() == "اعتمد", "Feature 1"])
        # rows the reviewer did not list are dropped too
        kept = df[df["Feature 1"].isin(keep)].assign(الدفعة=p.name.split(".")[0])
        report.append(f"{p.name.split('.')[0]}: {len(kept)}/{len(df)} rows kept")
        parts.append(kept)

    review = GEN / "review_all.csv"
    if review.exists():
        ok = pd.read_csv(review, dtype=str).fillna("")
        ok = set(ok.loc[ok["الحكم"].str.strip() == "اعتمد", "Feature 1"])
        old = pd.read_csv(GEN / "all2.csv", dtype=str).fillna("")
        old = old[old["Feature 1"].isin(ok)].assign(الدفعة="all2 (اعتمدتها صاحبة المشروع)")
        report.append(f"all2 rows approved by the user: {len(old)}")
        parts.append(old)

    out = pd.concat(parts, ignore_index=True)
    assert not out["Feature 1"].duplicated().any()
    out = out[[DIALECT, MSA, "Feature 1"] + [c for c in out.columns if c not in (DIALECT, MSA, "Feature 1")]]
    out.to_csv(OUT, index=False, encoding="utf-8")
    print("\n".join(report) + f"\nwrote {OUT}: {len(out)} rows")


if __name__ == "__main__":
    main()
