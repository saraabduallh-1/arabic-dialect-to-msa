"""How often each lahajatna headword (and each dialect form of our data) occurs in a large Arabic tweet corpus.

Advice 5 (docs/ADVICE_FROM_KASS.md): pick words for generation by everyday use, not by dictionary presence.
A token counts for a headword if it equals it after normalization, or after removing one common proclitic
(و ف ب ل ال وال بال فال لل). No model is used.

Writes data/external/tweet_freq.json:
  {"lahajatna": {headword: count}, "train_forms": {...}, "dev_forms": {...}, "test_forms": {...}, "tokens": N}
The test counts are only for reporting coverage, never for choosing words.

usage: python src/tweet_freq.py [--corpus PATH] [--limit_lines N]
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from data import DATA, load
from lexicon import norm

CORPUS = Path.home() / "Documents" / "kass-layer" / "data" / "corpus" / "tweets" / "lm_twitter.txt"
PRE = ["وال", "بال", "فال", "لل", "ال", "و", "ف", "ب", "ل"]
WORD = re.compile(r"[ء-ي٠-٩ً-ْ]+")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--limit_lines", type=int, default=0)
    a = ap.parse_args()

    targets = {}
    for l in (DATA / "external" / "lahajatna" / "entries.jsonl").open(encoding="utf-8"):
        e = json.loads(l)
        targets.setdefault(norm(e["plain"]), set()).add(("lahajatna", e["plain"]))
    for split in ("train", "dev", "test"):
        for r in load(split, "sentences_v2"):
            for d, _ in r["pairs"]:
                if len(d.split()) == 1:
                    targets.setdefault(norm(d), set()).add((f"{split}_forms", d))

    hits, n = Counter(), 0
    with open(a.corpus, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if a.limit_lines and i >= a.limit_lines:
                break
            if i % 5_000_000 == 0:
                print(f"{i:,} lines, {n:,} tokens", flush=True)
            for w in WORD.findall(line):
                n += 1
                w = norm(w)
                if w in targets:
                    hits[w] += 1
                    continue
                for p in PRE:
                    if w.startswith(p) and len(w) - len(p) >= 2 and w[len(p):] in targets:
                        hits[w[len(p):]] += 1
                        break

    out = {"tokens": n}
    for w, refs in targets.items():
        for kind, form in refs:
            out.setdefault(kind, {})[form] = hits[w]
    (DATA / "external" / "tweet_freq.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("tokens", n)


if __name__ == "__main__":
    main()
