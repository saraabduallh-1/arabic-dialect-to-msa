"""Raise the recall of the dialect tagger with what train says about each word form (dev only, to choose).

After the MARBERT tagger, an untagged single word becomes a dialect span when
  - train marks the same normalized form as dialect in >= RATIO of its occurrences (and it occurs >= MIN times), or
  - it is in the human-verified tweet vocabulary (optional).
Grid over RATIO x MIN on dev; prints span P/R/F1 (exact boundaries). Writes results/pipeline/tagger_boost_dev.md

usage: python src/tagger_boost.py
"""
import json
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import DATA, ROOT, load, normalize
from tag_dialect import spans_of, tokens

TAGGER = ROOT / "results" / "checkpoints" / "tagger_marbert"


def form_stats():
    dia, tot = Counter(), Counter()
    for r in load("train", "sentences_v2"):
        for w, l in tokens(r["src_marked"]):
            tot[normalize(w)] += 1
            dia[normalize(w)] += l == 1
    return dia, tot


def boost(lab, words, dia, tot, ratio, mn, voc):
    lab = list(lab)
    for i, w in enumerate(words):
        if lab[i]:
            continue
        f = normalize(w)
        if (tot[f] >= mn and dia[f] / tot[f] >= ratio) or f in voc:
            lab[i] = 1
            if i + 1 < len(lab) and lab[i + 1] == 2:  # do not glue onto the next span
                lab[i + 1] = 1
    return lab


def main():
    tok = AutoTokenizer.from_pretrained(TAGGER)
    model = AutoModelForTokenClassification.from_pretrained(TAGGER).cuda().eval().half()
    dev = [tokens(r["src_marked"]) for r in load("dev", "sentences_v2")]
    preds = []
    for s in dev:
        words = [w for w, _ in s]
        enc = tok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad():
            lg = model(**enc).logits[0]
        lab, prev = [0] * len(words), None
        for i, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                lab[wid] = int(lg[i].argmax())
            prev = wid
        preds.append(lab)
    dia, tot = form_stats()
    voc = {normalize(v["form"]) for v in json.load(open(DATA / "external" / "tweet_dialect" / "verified_vocab.json", encoding="utf-8"))}

    def prf(ratio, mn, use_voc):
        tp = fp = fn = 0
        for s, lab in zip(dev, preds):
            words, gold = [w for w, _ in s], spans_of([l for _, l in s])
            p = spans_of(boost(lab, words, dia, tot, ratio, mn, voc if use_voc else set()) if ratio is not None else lab)
            tp += len(gold & p); fp += len(p - gold); fn += len(gold - p)
        P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        return f"{100*P:.1f} | {100*R:.1f} | {200*P*R/max(P+R,1e-9):.1f}"

    lines = ["# Tagger + train-ratio boost, dev span P / R / F1", "", "| ratio | min count | tweet vocab | P | R | F1 |", "|---|---|---|---|---|---|",
             f"| tagger only | | | {prf(None, 0, False)} |"]
    for ratio in (0.5, 0.7, 0.9):
        for mn in (1, 2, 3):
            lines.append(f"| {ratio} | {mn} | no | {prf(ratio, mn, False)} |")
    for ratio, mn in ((0.5, 2), (0.7, 2)):
        lines.append(f"| {ratio} | {mn} | yes | {prf(ratio, mn, True)} |")
    lines.append(f"| vocab only | | yes | {prf(2, 10**9, True)} |")
    (ROOT / "results" / "pipeline" / "tagger_boost_dev.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
