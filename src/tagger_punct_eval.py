"""Compare taggers on clean and glued copies (25% of words with a mark, seed 0): plain tag(), and the old tagger with
merged.Layer.tag_punct (dialect spans from words without end marks, foreign spans as written, foreign wins overlaps).
usage: python src/tagger_punct_eval.py --split dev|test ckpt [ckpt ...]"""
import argparse
import json
from collections import Counter

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from data import DATA
from foreign_data import OUT, load_splits
from tag_dialect import predict, prf, read
from tag_merged import foreign_tokens, glue, typed_spans

END = ".,،؛;:!?؟…"


def spans(model, tok, s, dev, punct):
    pd_, pf = typed_spans(predict(model, tok, s, dev))
    if punct:
        stripped = [(w.rstrip(END) or w, l) for w, l in s]
        pd_, _ = typed_spans(predict(model, tok, stripped, dev))
        used = {i for a, b in pf for i in range(a, b)}
        pd_ = {(a, b) for a, b in pd_ if not used & set(range(a, b))}
    return pd_, pf


def score(model, tok, sents, dev, punct):
    c = Counter()
    for s in sents:
        gd, gf = typed_spans([l for _, l in s])
        pd_, pf = spans(model, tok, s, dev, punct)
        for k, g, p in (("D", gd, pd_), ("F", gf, pf)):
            c[k + "tp"] += len(g & p); c[k + "fp"] += len(p - g); c[k + "fn"] += len(g - p)
    return prf(c["Dtp"], c["Dfp"], c["Dfn"]), prf(c["Ftp"], c["Ffp"], c["Ffn"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("ckpts", nargs="+")
    a = ap.parse_args()
    fsp = load_splits()
    d = read(DATA / "sentences_v2" / f"{a.split}.csv")
    f = [foreign_tokens(r) for r in fsp[a.split]]
    sets = {"dialect": d, "dialect_glued": glue(d, 0.25, 0), "foreign": f, "foreign_glued": glue(f, 0.25, 0)}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ck in a.ckpts:
        tok = AutoTokenizer.from_pretrained(ck)
        model = AutoModelForTokenClassification.from_pretrained(ck).to(dev).eval()
        for punct in ([False, True] if ck.endswith("tagger_merged") else [False]):
            row = {"split": a.split, "tagger": ck.split("/")[-1] + ("+tag_punct" if punct else "")}
            for k, v in sets.items():
                D, F = score(model, tok, v, dev, punct)
                row[k] = D if k.startswith("dialect") else F
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with (OUT / "tagger_punct_eval.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
