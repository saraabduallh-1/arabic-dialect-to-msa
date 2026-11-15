"""Candidate dialect words from lahajatna for verified generation.

For every lahajatna entry:
  fusha_form: the word itself (diacritics removed, also without ال) appears in the maajim.com definitions
              (lisan al-arab, wasit, mu'asira, ...) -> likely fusha, dropped;
  dars: the word appears in Darsouni's dictionary of Saudi spoken dialects (second, independent source);
        the surrounding text is kept so the meaning can be compared.
Multi-word entries are kept only when at least one token is missing from the fusha dictionaries' text
(rough: otherwise they are usually fusha idioms).
Output: data/external/lahajatna/candidates.jsonl, sorted: in Darsouni first, then by meaning confidence.

usage: python src/filter_lahajatna.py
"""
import json
import re

from data import DATA

DIR = DATA / "external" / "lahajatna"
DIAC = re.compile(r"[\u064B-\u0652\u0670\u0640\u200e\u200f\u202a-\u202e]")


def norm(s):
    s = DIAC.sub("", s)
    return re.sub("[أإآ]", "ا", s).replace("ة", "ه").replace("ى", "ي")


def main():
    maajim = {}
    for l in (DIR / "maajim.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        maajim[r["word"]] = set(norm(r["text"]).split())
    fusha_vocab = set().union(*maajim.values())
    dars = re.sub(r"\s+", " ", norm((DATA / "external" / "darsouni" / "dars.txt").read_text(encoding="utf-8")))

    out, n_fusha = [], 0
    for l in (DIR / "entries.jsonl").open(encoding="utf-8"):
        e = json.loads(l)
        w = norm(e["plain"])
        toks = w.split()
        forms = {w, w[2:] if w.startswith("ال") else w}
        if forms & maajim.get(e["plain"], set()) or (len(toks) > 1 and all(t in fusha_vocab for t in toks)):
            n_fusha += 1
            continue
        hits, i = [], dars.find(f" {w} ")
        while i >= 0 and len(hits) < 3:
            hits.append(i + 1)
            i = dars.find(f" {w} ", i + 1)
        e["dars"] = [dars[max(0, i - 40): i + 260] for i in hits]
        out.append(e)
    out.sort(key=lambda e: (not e["dars"], -(e["conf"] or 0)))
    with (DIR / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"dropped as fusha form: {n_fusha}; candidates: {len(out)}; also in Darsouni: {sum(bool(e['dars']) for e in out)}")


if __name__ == "__main__":
    main()
