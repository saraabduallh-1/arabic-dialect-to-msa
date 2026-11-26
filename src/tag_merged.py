"""One MARBERTv2 tagger for both kinds of words: dialect (B-D / I-D) and foreign (B-F / I-F).

Dialect labels: the "(...)N" marks of data/sentences_v2 (as src/tag_dialect.py).
Foreign labels: the words the MSA side changed in the foreign -> MSA project's data (src/foreign_data.py).
Scored on the two dev sets (dialect dev, foreign dev); test is scored once by src/merged.py --final.

Punctuation (--punct p): the dialect training sentences have no marks, so a word with a glued mark ("بكرة،") was
missed. With p > 0 a second copy of the dialect sentences is added with a mark glued to a share p of the words (and
a final period half the time), and a copy of the foreign sentences without the final period, so "." stops being a
sign of foreign data. Labels stay the same. Chosen on the glued dev copies (25% of words, seed 0), as in
merged.Layer.tag_punct.

usage: python src/tag_merged.py [--epochs 5] [--dia_x 1] [--punct 0.15] [--save results/checkpoints/tagger_merged]
"""
import argparse
import json
import random
import re
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import DATA, ROOT
from foreign_data import OUT, load_splits
from tag_dialect import encode, predict, prf, read, spans_of

LABELS = ["O", "B-D", "I-D", "B-F", "I-F"]


def foreign_tokens(r):
    lab = [0] * len(r["words"])
    for s, e, _ in r["spans"]:
        lab[s] = 3
        for i in range(s + 1, e):
            lab[i] = 4
    return list(zip(r["words"], lab))


MARKS = ["،", "؟", ".", "!"]
END = ".,،؛;:!?؟…"
LATIN = re.compile("[A-Za-z]")


def glue(sents, share, seed, final=0.0):
    """A copy where each word gets a random mark with probability `share` (and the last word a period with
    probability `final`); labels are unchanged."""
    rng = random.Random(seed)
    out = []
    for s in sents:
        t = [(w + rng.choice(MARKS) if rng.random() < share else w, l) for w, l in s]
        if t and rng.random() < final and t[-1][0][-1] not in "".join(MARKS):
            t[-1] = (t[-1][0] + ".", t[-1][1])
        out.append(t)
    return out


def unperiod(sents):
    return [s[:-1] + [(s[-1][0].rstrip("."), s[-1][1])] if s and s[-1][0].endswith(".") and len(s[-1][0]) > 1 else s
            for s in sents]


def typed_spans(labels):
    """({dialect spans}, {foreign spans}) from 5-way labels."""
    d = spans_of([l if l in (1, 2) else 0 for l in labels])
    f = spans_of([l - 2 if l in (3, 4) else 0 for l in labels])
    return d, f


def score(model, tok, sents, dev):
    c = Counter()
    for s in sents:
        gd, gf = typed_spans([l for _, l in s])
        pd_, pf = typed_spans(predict(model, tok, s, dev))
        for k, g, p in (("D", gd, pd_), ("F", gf, pf)):
            c[k + "tp"] += len(g & p); c[k + "fp"] += len(p - g); c[k + "fn"] += len(g - p)
    return {"dialect": prf(c["Dtp"], c["Dfp"], c["Dfn"]), "foreign": prf(c["Ftp"], c["Ffp"], c["Ffn"]),
            "false_dialect_spans": c["Dfp"], "false_foreign_spans": c["Ffp"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="UBC-NLP/MARBERTv2")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--dia_x", type=int, default=1, help="repeat the dialect training sentences this many times")
    ap.add_argument("--punct", type=float, default=0.0, help="share of words given a glued mark in the extra dialect copy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mask_dia", action="store_true",
                    help="no loss on foreign-data words that are dialect words in the dialect training data")
    ap.add_argument("--save", default=str(ROOT / "results" / "checkpoints" / "tagger_merged"))
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    fsp = load_splits()
    dia_train = read(DATA / "sentences_v2" / "train.csv")
    for_train = [foreign_tokens(r) for r in fsp["train"]]
    if a.mask_dia:
        # the foreign data labels the dialect words in it O or F (لازم: F 25 times, بكرة: O 5 / F 7), which
        # teaches the tagger that common Gulf words are not dialect
        dvocab = {w.strip(END) for s in dia_train for w, l in s if l in (1, 2)}
        for_train = [[(w, -100 if w.strip(END) in dvocab and not LATIN.search(w) else l) for w, l in s] for s in for_train]
    train = dia_train * a.dia_x + for_train
    if a.punct > 0:
        train += glue(dia_train, a.punct, seed=1, final=0.5) + unperiod(for_train)
    dd, fd = read(DATA / "sentences_v2" / "dev.csv"), [foreign_tokens(r) for r in fsp["dev"]]
    dev_sets = {"dialect_dev": dd, "foreign_dev": fd,
                "dialect_dev_glued": glue(dd, 0.25, seed=0), "foreign_dev_glued": glue(fd, 0.25, seed=0)}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(
        a.model, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    data = [encode(tok, s) for s in train]
    steps = a.epochs * ((len(data) + 15) // 16)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, i / (0.1 * steps)) * max(0.0, 1 - i / steps))
    model.train()
    for ep in range(a.epochs):
        random.shuffle(data)
        tot = 0.0
        for b in range(0, len(data), 16):
            batch = tok.pad([{k: d[k] for k in ("input_ids", "attention_mask")} for d in data[b:b + 16]],
                            return_tensors="pt")
            n = batch["input_ids"].shape[1]
            labels = torch.tensor([d["labels"] + [-100] * (n - len(d["labels"])) for d in data[b:b + 16]])
            loss = model(**batch.to(dev), labels=labels.to(dev)).loss
            loss.backward()
            opt.step(); sched.step(); opt.zero_grad()
            tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot / (len(data) / 16):.4f}", flush=True)
    model.eval()
    model.save_pretrained(a.save)
    tok.save_pretrained(a.save)
    res = {"epochs": a.epochs, "dia_x": a.dia_x, "punct": a.punct, "seed": a.seed, "mask_dia": a.mask_dia, "save": a.save, "train_sentences": {"dialect": len(dia_train), "foreign": len(fsp["train"])},
           **{k: score(model, tok, v, dev) for k, v in dev_sets.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tagger_dev.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
