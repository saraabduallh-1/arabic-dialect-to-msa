"""Dialect dictionary hints for a marked word (dictionary-augmented prompting, cf. DiPMT, Ghazvininejad et al. 2023).

Sources (external, never the train/test pairs themselves):
  lahajatna.com entries (machine-checked meanings, data/external/lahajatna/entries.jsonl)
  the verified generated data's meaning column (data/generated/verified.csv)
Lookup: the exact form first, then light stemming (clitics, the article, present-tense and plural affixes),
keeping stems of at least 3 letters.

usage: from lexicon import Lexicon; Lexicon().hint("نبتلش")  -> ("بلش", "تورّط ...") or None
"""
import csv
import json
import re

from data import DATA

DIAC = re.compile(r"[ً-ْٰـ]")
PRE = ["وال", "بال", "فال", "لل", "ال", "و", "ف", "ب", "ل", "ي", "ت", "ن", "ا"]
SUF = ["ونه", "ينه", "تها", "تهم", "وها", "وهم", "كم", "هم", "ها", "نا", "ني", "ين", "ون", "وا", "ات", "ه", "ك", "ي", "ج", "ة", "ت"]


def norm(s):
    s = DIAC.sub("", s).strip()
    return re.sub("[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")


def variants(w):
    """The word, then stems: suffix stripped, prefix stripped, both (longest stem first)."""
    w = norm(w)
    out = [w]
    sufs = [w] + [w[: -len(s)] for s in map(norm, SUF) if w.endswith(s) and len(w) - len(s) >= 3]
    for x in sufs:
        out.append(x)
        out += [x[len(p):] for p in map(norm, PRE) if x.startswith(p) and len(x) - len(p) >= 3]
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res


class Lexicon:
    def __init__(self, max_len=90):
        self.d = {}
        for l in (DATA / "external" / "lahajatna" / "entries.jsonl").open(encoding="utf-8"):
            e = json.loads(l)
            if e["meanings"]:
                m = e["meanings"][0].split("،")[0] if len(e["meanings"][0]) > max_len else e["meanings"][0]
                self.d.setdefault(norm(e["plain"]), (e["plain"], m[:max_len].rstrip("…")))
        p = DATA / "generated" / "verified.csv"
        if p.exists():
            for r in csv.DictReader(p.open(encoding="utf-8")):
                if r.get("الكلمة") and r.get("المعنى"):
                    self.d.setdefault(norm(r["الكلمة"]), (r["الكلمة"], r["المعنى"][:max_len]))

    def hint(self, span):
        """(dictionary headword, meaning) for a marked span, or None. Multi-word spans: the whole, then each word."""
        for cand in [span] + (span.split() if len(span.split()) > 1 else []):
            for v in variants(cand):
                if v in self.d:
                    return self.d[v]
        return None
