"""Coverage of the unseen dev dialect words by external resources (dev only; test is not used).

  saudial : SauDial (Mendeley 10.17632/mzdwkb2t6d, CC BY 4.0), 1,000 MSA/dialect sentence pairs, 4 Saudi dialects
  radd    : "قاموس رد العامي للفصيح" (Ahmad Rida) from mysamai/m3ajim (CC BY-NC 4.0), 1,068 entries
  tweets  : tweet-mined candidates (src/mine_tweets.py), for comparison
Writes results/advice/coverage_external_dev.md
"""
import csv
import json
import re

import pandas as pd

from data import DATA, ROOT, load
from lexicon import norm

DIAC = re.compile(r"[ً-ْٰـ]")
WORD = re.compile(r"[ء-ي]+")


def main():
    known = {norm(d) for r in load("train", "sentences_v2") for d, _ in r["pairs"]}
    known |= {norm(r["الكلمة"]) for r in csv.DictReader((DATA / "generated" / "verified.csv").open(encoding="utf-8"))}
    unseen = [norm(d) for r in load("dev", "sentences_v2") for d, m in r["pairs"] if norm(d) != norm(m) and norm(d) not in known]
    single = [u for u in unseen if " " not in u]

    s = pd.read_excel(DATA / "external" / "saudial" / "saudial.xlsx")
    sau = {norm(w) for t in s["Dialect Translation"].astype(str) for w in WORD.findall(t)}
    msa = {norm(w) for t in s["Modern Standard Arabic (MSA) Translation"].astype(str) for w in WORD.findall(t)}
    sau_only = sau - msa  # words of the dialect side that the MSA side never uses
    r = pd.read_parquet(DATA / "external" / "m3ajim" / "roots.parquet")
    radd = {norm(w) for h in r[r.dictionary_id == 78]["root"] for w in WORD.findall(DIAC.sub("", h))}
    tw = {json.loads(l)["form"] for l in (DATA / "external" / "tweet_dialect" / "candidates.jsonl").open(encoding="utf-8")}

    rows = [("SauDial dialect side (any word)", sau), ("SauDial dialect-only words", sau_only),
            ("رد العامي للفصيح headwords", radd), ("tweet-mined candidates (all)", tw)]
    out = ["# External resources: coverage of the unseen dev dialect words", "",
           f"Unseen dev dialect spans: {len(unseen)} (single words: {len(single)}).", "",
           "| resource | size (forms) | unseen spans covered | single words covered | examples |", "|---|---|---|---|---|"]
    for name, S in rows:
        hit = [u for u in unseen if u in S]
        out.append(f"| {name} | {len(S):,} | {len(hit)} ({100*len(hit)/len(unseen):.1f}%) | "
                   f"{sum(u in S for u in single)} | {'، '.join(sorted(set(hit))[:15])} |")
    both = [u for u in unseen if u in sau_only and u not in tw]
    out += ["", f"SauDial dialect-only words not already in the tweet candidates: {len(both)} ({'، '.join(sorted(set(both))[:30])})"]
    (ROOT / "results" / "advice" / "coverage_external_dev.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
