"""One MARBERTv2 tagger for both kinds of words: dialect (B-D / I-D) and foreign (B-F / I-F).

Dialect labels: the "(...)N" marks of data/sentences_v2 (as src/tag_dialect.py).
Foreign labels: the words the MSA side changed in the foreign -> MSA project's data (src/foreign_data.py).
Scored on the two dev sets (dialect dev, foreign dev); test is scored once by src/merged.py --final.

usage: python src/tag_merged.py [--epochs 5] [--dia_x 1] [--save results/checkpoints/tagger_merged]
"""
import argparse
import json
import random
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
    ap.add_argument("--save", default=str(ROOT / "results" / "checkpoints" / "tagger_merged"))
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    fsp = load_splits()
    dia_train = read(DATA / "sentences_v2" / "train.csv")
    train = dia_train * a.dia_x + [foreign_tokens(r) for r in fsp["train"]]
    dev_sets = {"dialect_dev": read(DATA / "sentences_v2" / "dev.csv"),
                "foreign_dev": [foreign_tokens(r) for r in fsp["dev"]]}

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
    res = {"epochs": a.epochs, "dia_x": a.dia_x, "train_sentences": {"dialect": len(dia_train), "foreign": len(fsp["train"])},
           **{k: score(model, tok, v, dev) for k, v in dev_sets.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "tagger_dev.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
