"""Corpus dialectness of a word form: how much more often it occurs in Arabic tweets than in fusha text.

Advice 5 and 6 (docs/ADVICE_FROM_KASS.md). Two uses:
  - candidate list for data generation: frequent tweet words that fusha text rarely uses, ranked by frequency;
  - a context-free prior for the dialect tagger (forms that look fusha but behave as dialect).
Score = log-odds ratio with an informative Dirichlet prior (Monroe et al., 2008), tweets vs fusha.
Fusha side: the KASS Arabic Wikipedia sample + the MSA side of our train sentences.

step 1 (slow, ~1 h, CPU): python src/dialectness.py count     -> data/external/tweet_vocab.tsv (form, count >= 20)
step 2 (fast)           : python src/dialectness.py eval      -> results/advice/dialectness_dev.md
  eval on dev only: AUC of the score for marked (dialect) vs unmarked tokens, overall and for forms that train
  only shows unmarked; and how many unseen dev dialect words fall in the top-N candidate list.
"""
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

from data import DATA, ROOT, load
from lexicon import norm

KASS = Path.home() / "Documents" / "kass-layer" / "data" / "corpus"
WORD = re.compile(r"[ء-ي]+")
VOCAB = DATA / "external" / "tweet_vocab.tsv"
OUT = ROOT / "results" / "advice"


def toks(s):
    return [norm(w) for w in WORD.findall(s)]


def count():
    c = Counter()
    with open(KASS / "tweets" / "lm_twitter.txt", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            c.update(toks(line))
            if i % 10_000_000 == 0:
                print(f"{i:,} lines, {len(c):,} types", flush=True)
            if i % 20_000_000 == 0 and i:  # keep memory bounded: drop hapaxes
                c = Counter({w: n for w, n in c.items() if n > 1})
    with VOCAB.open("w", encoding="utf-8") as f:
        for w, n in c.most_common():
            if n < 20:
                break
            f.write(f"{w}\t{n}\n")


def fusha_counts():
    c = Counter()
    for p in (KASS / "wikipedia").glob("*.jsonl"):
        for l in p.open(encoding="utf-8"):
            c.update(toks(json.loads(l)["text"]))
    for r in load("train", "sentences_v2"):
        c.update(toks(r["tgt"]))
    return c


def scores():
    tw = Counter({w: int(n) for w, n in (l.rstrip("\n").split("\t") for l in VOCAB.open(encoding="utf-8"))})
    fu = fusha_counts()
    nt, nf = sum(tw.values()), sum(fu.values())
    prior = tw + fu
    a0 = 1000.0  # prior strength
    np_ = sum(prior.values())
    out = {}
    for w in set(tw) | set(fu):
        a = a0 * prior[w] / np_ + 0.01
        lt = math.log((tw[w] + a) / (nt + a0 - tw[w] - a))
        lf = math.log((fu[w] + a) / (nf + a0 - fu[w] - a))
        var = 1 / (tw[w] + a) + 1 / (fu[w] + a)
        out[w] = (lt - lf) / math.sqrt(var)
    return out, tw


def auc(pos, neg):
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    r, rank_sum = 0, 0.0
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + j + 1) / 2
        rank_sum += avg * sum(1 for k in range(i, j) if allv[k][1])
        i = j
    return round((rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 3)


def evaluate():
    from tag_dialect import tokens
    import pandas as pd
    from data import DIALECT
    sc, tw = scores()
    as_dia, as_fus = set(), set()
    for s in (tokens(str(x)) for x in pd.read_csv(DATA / "sentences_v2" / "train.csv")[DIALECT]):
        for w, l in s:
            (as_dia if l else as_fus).add(norm(w))
    pos, neg, pos_f, neg_f = [], [], [], []
    for s in (tokens(str(x)) for x in pd.read_csv(DATA / "sentences_v2" / "dev.csv")[DIALECT]):
        for w, l in s:
            w = norm(w)
            v = sc.get(w, 0.0)
            (pos if l else neg).append(v)
            if w in as_fus and w not in as_dia:  # train shows it only unmarked
                (pos_f if l else neg_f).append(v)
    lines = ["# Corpus dialectness (tweets vs fusha), evaluated on dev", "",
             f"Tweet vocabulary: {len(tw):,} forms (count >= 20).", "",
             "| tokens | dialect n | other n | AUC |", "|---|---|---|---|",
             f"| all dev tokens | {len(pos)} | {len(neg)} | {auc(pos, neg)} |",
             f"| forms train shows only unmarked | {len(pos_f)} | {len(neg_f)} | {auc(pos_f, neg_f)} |", ""]
    # candidate list: frequent, dialect-leaning, not already a train dialect form
    train_forms = {norm(d) for r in load("train", "sentences_v2") for d, _ in r["pairs"]}
    unseen_dev = {norm(d) for r in load("dev", "sentences_v2") for d, m in r["pairs"]
                  if len(d.split()) == 1 and norm(d) != norm(m) and norm(d) not in train_forms}
    lines += ["| candidate list (tweet count >= 200, score > t, not a train dialect form) | size | unseen dev dialect words covered (of %d) |" % len(unseen_dev), "|---|---|---|"]
    for t in (2, 4, 8):
        cand = {w for w, v in sc.items() if v > t and tw[w] >= 200 and w not in train_forms}
        lines.append(f"| score > {t} | {len(cand):,} | {len(unseen_dev & cand)} |")
    top = sorted(((w, v) for w, v in sc.items() if tw[w] >= 200 and w not in train_forms), key=lambda x: -x[1])[:60]
    lines += ["", "Top candidates: " + "، ".join(w for w, _ in top)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dialectness_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    json.dump({w: round(v, 2) for w, v in sc.items() if tw[w] >= 20}, (DATA / "external" / "dialectness.json").open("w", encoding="utf-8"), ensure_ascii=False)
    print("\n".join(lines))


if __name__ == "__main__":
    {"count": count, "eval": evaluate}[sys.argv[1]]()
