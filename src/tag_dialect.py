"""Which words of a sentence are dialect? Token tagger (BIO) trained on the bracket marks of the data.

The word-level benchmark gives the dialect words to the model; a real system has to find them first.
The labels come for free: every "(...)N" span of the dialect sentence is dialect, everything else is not.

Reports on the test set:
  span P/R/F1 (exact span boundaries) and token P/R/F1 for the dialect class;
  recall of dialect tokens split by what train says about the same form:
    seen_dialect  - marked as dialect somewhere in train
    seen_fusha    - occurs in train only unmarked (looks fusha / ambiguous: the context decides)
    unseen        - never in train;
  false positives split the same way.
Writes results/tagger/<model>.json and the test predictions.

usage: python src/tag_dialect.py --model UBC-NLP/MARBERTv2 [--extra generated/verified] [--epochs 5]
"""
import argparse
import json
import random
from collections import Counter

import pandas as pd
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import DATA, DIALECT, ROOT, SPAN_RE, clean, normalize

OUT = ROOT / "results" / "tagger"


def tokens(marked):
    """[(token, 0=O / 1=B / 2=I)] from a marked dialect sentence."""
    out, pos = [], 0
    for m in SPAN_RE.finditer(marked):
        out += [(t, 0) for t in clean(marked[pos:m.start()]).split()]
        out += [(t, 1 if i == 0 else 2) for i, t in enumerate(clean(m.group(1)).split())]
        pos = m.end()
    return out + [(t, 0) for t in clean(marked[pos:]).split()]


def read(path):
    return [tokens(str(s)) for s in pd.read_csv(path)[DIALECT]]


def encode(tok, sent):
    words = [w for w, _ in sent]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=128)
    labels, prev = [], None
    for wid in enc.word_ids():
        labels.append(-100 if wid is None or wid == prev else sent[wid][1])
        prev = wid
    enc["labels"] = labels
    return enc


def predict(model, tok, sent, dev):
    words = [w for w, _ in sent]
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
    with torch.no_grad():
        logits = model(**enc).logits[0]
    out, prev = [0] * len(words), None
    for i, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != prev:
            out[wid] = int(logits[i].argmax())
        prev = wid
    return out


def spans_of(labels):
    s, cur = set(), None
    for i, l in enumerate(labels + [0]):
        if cur is not None and l != 2:
            s.add((cur, i))
            cur = None
        if l == 1 or (l == 2 and cur is None):
            cur = i
    return s


def prf(tp, fp, fn):
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"P": round(100 * p, 1), "R": round(100 * r, 1), "F1": round(200 * p * r / max(p + r, 1e-9), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="UBC-NLP/MARBERTv2")
    ap.add_argument("--data", default="sentences_v2")
    ap.add_argument("--extra", default="", help="extra training csv under data/, e.g. generated/verified")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=3e-5)
    a = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    train = read(DATA / a.data / "train.csv")
    if a.extra:
        train += read(DATA / f"{a.extra}.csv")
    test = read(DATA / a.data / "test.csv")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForTokenClassification.from_pretrained(a.model, num_labels=3).to(dev)
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
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item()
        print(f"epoch {ep + 1}: loss {tot / (len(data) / 16):.4f}", flush=True)
    model.eval()

    # what train says about each form
    as_dia, as_fus = set(), set()
    for s in train:
        for w, l in s:
            (as_dia if l else as_fus).add(normalize(w))

    def kind(w):
        w = normalize(w)
        return "seen_dialect" if w in as_dia else "seen_fusha" if w in as_fus else "unseen"

    sp, tk = Counter(), Counter()
    rec, fp_kind, examples = Counter(), Counter(), {"missed": [], "false_alarm": []}
    preds = []
    for s in test:
        gold = [l for _, l in s]
        pred = predict(model, tok, s, dev)
        g, p = spans_of(gold), spans_of(pred)
        sp["tp"] += len(g & p); sp["fp"] += len(p - g); sp["fn"] += len(g - p)
        sent = " ".join(w for w, _ in s)
        for (w, gl), pl in zip(s, pred):
            k = kind(w)
            if gl and pl:
                tk["tp"] += 1; rec[(k, True)] += 1
            elif gl:
                tk["fn"] += 1; rec[(k, False)] += 1
                examples["missed"].append(f"{w} [{k}] | {sent}")
            elif pl:
                tk["fp"] += 1; fp_kind[k] += 1
                examples["false_alarm"].append(f"{w} [{k}] | {sent}")
        preds.append({"src": sent, "gold": gold, "pred": pred})

    res = {
        "model": a.model, "extra": a.extra, "epochs": a.epochs,
        "span": prf(sp["tp"], sp["fp"], sp["fn"]), "token": prf(tk["tp"], tk["fp"], tk["fn"]),
        "token_recall_by_kind": {k: {"n": rec[(k, True)] + rec[(k, False)],
                                     "recall": round(100 * rec[(k, True)] / max(rec[(k, True)] + rec[(k, False)], 1), 1)}
                                 for k in ("seen_dialect", "seen_fusha", "unseen")},
        "false_alarms_by_kind": dict(fp_kind),
        "examples": {k: v[:40] for k, v in examples.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    name = a.model.replace("/", "_") + (f"+{a.extra.split('/')[-1]}" if a.extra else "")
    (OUT / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    with (OUT / f"{name}.preds.jsonl").open("w", encoding="utf-8") as f:
        for d in preds:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(json.dumps({k: res[k] for k in ("span", "token", "token_recall_by_kind", "false_alarms_by_kind")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
