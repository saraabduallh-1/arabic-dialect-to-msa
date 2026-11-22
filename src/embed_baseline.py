"""Light baselines without a generative LLM, on gold dev spans (user question: a word list + embeddings instead of LLMs?).

  dict       : train dictionary (most frequent MSA for the normalized form), no answer for unseen forms
  char-kNN   : dictionary, else nearest train dialect form by char 2-4-gram TF-IDF (catches inflection) -> its MSA
  ctx-kNN    : nearest train span by contextual MARBERT embedding (span mean-pooled inside its sentence) -> its MSA
  dict+ctx   : dictionary, else ctx-kNN
  ALLaM      : the current fine-tuned model (results/word/preds), for reference
Metric: exact / loose as in src/tiers.py. CPU only (the GPU is busy with the demo).

usage: python src/embed_baseline.py   (dev only)
"""
import json
from collections import Counter

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from data import ROOT, load, normalize
from tag_dialect import tokens
from tiers import ALLAM, preds, train_dict

TAGGER = ROOT / "results" / "checkpoints" / "tagger_marbert"


def spans(rows):
    """[(row, words, (start, end), dialect, msa, span_id)] for spans whose MSA differs from the dialect."""
    out = []
    for r in rows:
        t = tokens(r["src_marked"])
        words, bounds = [w for w, _ in t], []
        for i, (_, b) in enumerate(t):
            if b == 1:
                bounds.append([i, i + 1])
            elif b == 2:
                bounds[-1][1] = i + 1
        if len(bounds) != len(r["pairs"]):
            continue
        for se, k, (d, m) in zip(bounds, r["span_ids"], r["pairs"]):
            if normalize(d) != normalize(m):
                out.append((r, words, se, d, m, k))
    return out


@torch.no_grad()
def embed(items, tok, enc):
    vecs = []
    for _, words, (s, e), *_ in items:
        x = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        h = enc(**x).last_hidden_state[0]
        idx = [i for i, w in enumerate(x.word_ids()) if w is not None and s <= w < e]
        v = h[idx].mean(0) if idx else h[0]
        vecs.append((v / v.norm()).numpy())
    return np.stack(vecs)


def main():
    tr, dv = spans(load("train", "sentences_v2")), spans(load("dev", "sentences_v2"))
    dic = train_dict()
    top = {k: c.most_common(1)[0][0] for k, c in dic.items()}
    A = preds(ALLAM, "dev")

    forms = sorted(top)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)).fit(forms)
    F = vec.transform(forms)

    tok = AutoTokenizer.from_pretrained(TAGGER)
    enc = AutoModel.from_pretrained(TAGGER).eval()
    Etr, Edv = embed(tr, tok, enc), embed(dv, tok, enc)
    tr_msa = [m for *_, m, _ in tr]
    nn_ctx = (Edv @ Etr.T).argmax(1)

    st = Counter()
    for i, (r, _, _, d, m, k) in enumerate(dv):
        nd, y = normalize(d), normalize(m)
        seen = nd in top
        char = top[nd] if seen else forms[int((vec.transform([nd]) @ F.T).toarray().argmax())]
        char = top.get(nd) or top[char]
        ctx = tr_msa[nn_ctx[i]]
        ans = {"dict": top.get(nd, ""), "char-kNN": char, "ctx-kNN": ctx,
               "dict+ctx": top.get(nd) or ctx, "ALLaM": A[r["id"]].get(str(k), "")}
        for name, a in ans.items():
            x = normalize(a)
            ok = bool(x) and x == y
            loose = bool(x) and (ok or f" {y} " in f" {x} " or f" {x} " in f" {y} ")
            for g in ("all", "seen" if seen else "unseen"):
                st[(name, g, "n")] += 1
                st[(name, g, "ans")] += bool(x)
                st[(name, g, "ok")] += ok
                st[(name, g, "loose")] += loose

    pc = lambda a, b: round(100 * a / b, 1) if b else 0
    n, ns = st[("dict", "all", "n")], st[("dict", "seen", "n")]
    lines = [f"Dev spans: {n} (form seen in train: {ns} = {pc(ns, n)}%, unseen: {n - ns})", "",
             "| method | answered% | exact% all | loose% all | exact% seen | exact% unseen |", "|---|---|---|---|---|---|"]
    for name in ("dict", "char-kNN", "ctx-kNN", "dict+ctx", "ALLaM"):
        g = lambda grp, key: st[(name, grp, key)]
        lines.append(f"| {name} | {pc(g('all', 'ans'), n)} | {pc(g('all', 'ok'), n)} | {pc(g('all', 'loose'), n)} | "
                     f"{pc(g('seen', 'ok'), g('seen', 'n'))} | {pc(g('unseen', 'ok'), g('unseen', 'n'))} |")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "results" / "advice" / "embed_baseline_dev.md").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
