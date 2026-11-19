"""Mine everyday Saudi dialect words from tweets with the trained dialect tagger (src/tag_dialect.py --save).

1. Keep tweets that contain at least one common Saudi/Gulf marker word (وش، ابي، الحين، ...), 3-40 words long.
2. Tag each tweet with the MARBERT tagger; collect every predicted dialect span (B + following I).
3. Per normalized span: how often it was tagged dialect, how often the same form occurred untagged, example tweets.
4. Candidates: tagged dialect >= MIN times and in >= 50% of its occurrences, not a dialect form of train or of the
   verified generated data, not a single letter / number. Each gets its lahajatna entry if one exists.

Writes data/external/tweet_dialect/candidates.jsonl (sorted by frequency) and results/advice/mine_tweets_dev.md
(coverage of the unseen dev dialect words; test is not used to choose anything).

usage: python src/mine_tweets.py --tagger results/checkpoints/tagger_marbert --lines 2000000
"""
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import DATA, ROOT, load
from lexicon import Lexicon, norm

CORPUS = Path.home() / "Documents" / "kass-layer" / "data" / "corpus" / "tweets" / "lm_twitter.txt"
MARKERS = {norm(w) for w in "وش ابي ابغى ابغي الحين عشان ليش يبي يبغى زين كذا مره وشو ايش اللي مب مو هالحين بعدين توني".split()}
WORD = re.compile(r"[ء-ي]+")
OUT = DATA / "external" / "tweet_dialect"


def tweets(n):
    k = 0
    with open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = WORD.findall(line)
            if 3 <= len(w) <= 40 and any(norm(x) in MARKERS for x in w):
                yield w
                k += 1
                if k >= n:
                    return


def spans(words, labels):
    out, cur = [], []
    for w, l in zip(words, labels):
        if l == 1 or (l == 2 and not cur):
            if cur:
                out.append(" ".join(cur))
            cur = [w]
        elif l == 2:
            cur.append(w)
        else:
            if cur:
                out.append(" ".join(cur))
            cur = []
    if cur:
        out.append(" ".join(cur))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tagger", default=str(ROOT / "results" / "checkpoints" / "tagger_marbert"))
    ap.add_argument("--lines", type=int, default=2_000_000)
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--bs", type=int, default=128)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.tagger)
    model = AutoModelForTokenClassification.from_pretrained(a.tagger).cuda().eval().half()

    dia, occ, ex = Counter(), Counter(), defaultdict(list)
    batch = []

    def run(batch):
        enc = tok(batch, is_split_into_words=True, truncation=True, max_length=96, padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = model(**{k: v.cuda() for k, v in enc.items()}).logits.argmax(-1).cpu()
        for b, words in enumerate(batch):
            lab, prev = [0] * len(words), None
            for i, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid != prev:
                    lab[wid] = int(pred[b, i])
                prev = wid
            for w in words:
                occ[norm(w)] += 1
            for s in spans(words, lab):
                k = norm(s)
                dia[k] += 1
                if len(ex[k]) < 3:
                    ex[k].append(" ".join(words))

    for i, w in enumerate(tweets(a.lines)):
        batch.append(w)
        if len(batch) == a.bs:
            run(batch)
            batch = []
        if i % 200_000 == 0:
            print(f"{i:,} tweets, {len(dia):,} dialect forms", flush=True)
    if batch:
        run(batch)

    known = {norm(d) for r in load("train", "sentences_v2") for d, _ in r["pairs"]}
    known |= {norm(r["الكلمة"]) for r in csv.DictReader((DATA / "generated" / "verified.csv").open(encoding="utf-8"))}
    lex = Lexicon()
    cands = []
    for k, n in dia.most_common():
        if n < a.min:
            break
        total = occ[k] if " " not in k else n  # phrases: no untagged count
        if len(k) < 2 or k in known or n / max(total, 1) < 0.5:
            continue
        h = lex.hint(k)
        cands.append({"form": k, "tagged": n, "occurrences": total, "ratio": round(n / max(total, 1), 2),
                      "lahajatna": h[1] if h and norm(h[0]) == k else None,
                      "lahajatna_stem": [h[0], h[1]] if h and norm(h[0]) != k else None, "examples": ex[k]})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "candidates.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # coverage on dev only
    unseen = [norm(d) for r in load("dev", "sentences_v2") for d, m in r["pairs"]
              if norm(d) != norm(m) and norm(d) not in known]
    forms = {c["form"] for c in cands}
    lines = ["# Dialect words mined from tweets (MARBERT tagger), coverage on dev", "",
             f"Tweets tagged: {a.lines:,} (with a Saudi/Gulf marker word). Candidates (tagged >= {a.min}, ratio >= 0.5, new): {len(cands):,}.",
             f"With an exact lahajatna entry: {sum(1 for c in cands if c['lahajatna'])}; via stem: {sum(1 for c in cands if c['lahajatna_stem'])}.", "",
             "| top-N candidates | unseen dev dialect spans covered (of %d) |" % len(unseen), "|---|---|"]
    for n in (500, 1000, 2000, 5000, len(cands)):
        top = {c["form"] for c in cands[:n]}
        lines.append(f"| {n:,} | {sum(u in top for u in unseen)} |")
    lines += ["", "Top 80: " + "، ".join(c["form"] for c in cands[:80])]
    (ROOT / "results" / "advice" / "mine_tweets_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
