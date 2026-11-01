"""Evaluate open-source LLMs served by Ollama on dialect -> MSA conversion.

Modes:
  zero     instruction only
  rag      + the k most similar training pairs as few-shot examples
  rag_lex  + dictionary hints for dialect words found in the training lexicon
  rag_lexf     like rag_lex, but drops hints for words used more often as MSA than as dialect
  rag_lexret   rag_lexf, but half of the examples are training pairs that contain the same dialect terms
  rag_fix      rag_lexf, then one follow-up turn if dialect terms from the lexicon are left in the answer

usage: python src/run_llm.py --models qwen2.5:7b gemma3:12b --modes zero rag rag_lex
"""
import argparse
import json
import re
from collections import Counter

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from tqdm import tqdm

from data import ROOT, lexicon, load, normalize
from metrics import score

OLLAMA = "http://localhost:11434/api/chat"
PRED_DIR = ROOT / "results" / "preds"

SYSTEM = (
    "أنت خبير في اللغة العربية ولهجاتها (خاصة اللهجة السعودية/الخليجية). "
    "مهمتك تحويل الجملة العامية إلى عربية فصحى سليمة مع الحفاظ على المعنى وبنية الجملة قدر الإمكان، "
    "واستبدال الكلمات العامية فقط بمقابلها الفصيح. "
    "أجب بالجملة الفصحى فقط في سطر واحد بدون أي شرح أو علامات تنصيص."
)


class Retriever:
    def __init__(self, train):
        self.train = train
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
        self.mat = self.vec.fit_transform([normalize(r["src"]) for r in train])
        self.lex = lexicon(train)
        # how often each term is annotated as dialect vs. occurs as a plain word on the MSA side
        self.as_dialect = Counter(normalize(d) for r in train for d, _ in r["pairs"])
        self.as_msa = Counter(w for r in train for w in normalize(r["tgt"]).split())

        self.by_term = {}
        for i, r in enumerate(train):
            for d, _ in r["pairs"]:
                self.by_term.setdefault(normalize(d), []).append(i)

    def examples(self, src, k, by_terms=False):
        sims = linear_kernel(self.vec.transform([normalize(src)]), self.mat)[0]
        picked = []
        if by_terms:  # up to k/2 examples that contain the sentence's dialect terms, most similar first
            for term, _ in self.hints(src, filtered=True):
                cand = sorted(self.by_term.get(term, []), key=lambda i: -sims[i])
                picked += [i for i in cand[:2] if i not in picked]
            picked = picked[: k // 2]
        picked += [i for i in sims.argsort()[::-1] if i not in picked][: k - len(picked)]
        return [self.train[i] for i in sorted(picked, key=lambda i: sims[i])]  # most similar last

    def left(self, src, pred):
        """Dialect terms of src (filtered lexicon hits) still present in pred, with their MSA equivalents."""
        p = f" {normalize(pred)} "
        return [(d, m) for d, m in self.hints(src, filtered=True)
                if f" {d} " in p and not any(f" {normalize(x)} " in p for x in m)]

    def hints(self, src, max_n=3, filtered=False):
        words = normalize(src).split()
        found = []
        for n in range(max_n, 0, -1):
            for i in range(len(words) - n + 1):
                g = " ".join(words[i:i + n])
                if g in self.lex and g not in [f[0] for f in found]:
                    if filtered and self.as_msa[g] > self.as_dialect[g]:  # e.g. common words like ما / عند
                        continue
                    found.append((g, self.lex[g]))
        return found


def build_messages(src, mode, ret, k):
    msgs = [{"role": "system", "content": SYSTEM}]
    if mode in ("rag", "rag_lex", "rag_lexf", "rag_lexret", "rag_fix"):
        for ex in ret.examples(src, k, by_terms=mode == "rag_lexret"):
            msgs.append({"role": "user", "content": f"العامية: {ex['src']}"})
            msgs.append({"role": "assistant", "content": ex["tgt"]})
    user = f"العامية: {src}"
    if mode in ("rag_lex", "rag_lexf", "rag_lexret", "rag_fix"):
        h = ret.hints(src, filtered=mode != "rag_lex")
        if h:
            user += "\nتلميحات من القاموس (قد لا تنطبق كلها على السياق):\n" + "\n".join(
                f"- {d} ← {' / '.join(m[:3])}" for d, m in h)
    msgs.append({"role": "user", "content": user})
    return msgs


def postprocess(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    line = next((l for l in text.splitlines() if l.strip()), "")
    line = re.sub(r"^\s*(الفصحى|الفصيح|الجملة الفصحى)\s*[:：]\s*", "", line)
    return line.strip().strip('"«»“”\'')


def chat(model, messages):
    body = {"model": model, "messages": messages, "stream": False, "think": False,
            "keep_alive": "10m", "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 128}}
    r = requests.post(OLLAMA, json=body, timeout=600)
    if r.status_code == 400 and "think" in r.text:
        body.pop("think")
        r = requests.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def pred_path(model, mode, tag="", split="test"):
    # test files keep their old names; tuning runs on dev get an @dev suffix
    return PRED_DIR / f"{model.replace(':', '_').replace('/', '_')}__{mode}{tag}{'' if split == 'test' else '@' + split}.jsonl"


def fix_turn(model, msgs, first, left):
    """rag_fix: ask once more when dialect terms are left in the first answer."""
    note = "\n".join(f"- {d} ← {' / '.join(m[:3])}" for d, m in left)
    msgs = msgs + [{"role": "assistant", "content": first},
                   {"role": "user", "content": "ما زالت في إجابتك كلمات عامية، استبدلها بمقابلها الفصيح المناسب للسياق:\n"
                                               f"{note}\nأعد كتابة الجملة الفصحى كاملة في سطر واحد فقط."}]
    return chat(model, msgs)


def run(model, mode, rows, ret, k, tag="", split="test"):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = pred_path(model, mode, tag, split)
    base = {}
    if mode == "rag_fix" and pred_path(model, "rag_lexf", tag, split).exists():  # reuse the first answers
        base = {d["id"]: d for d in map(json.loads, pred_path(model, "rag_lexf", tag, split).open(encoding="utf-8"))}
    done = {}
    if path.exists():
        done = {d["id"]: d for d in map(json.loads, path.open(encoding="utf-8"))}
    with path.open("a", encoding="utf-8") as f:
        for r in tqdm(rows, desc=f"{model} {mode}"):
            if r["id"] in done:
                continue
            msgs = build_messages(r["src"], mode, ret, k)
            if mode == "rag_fix":
                first = base[r["id"]]["raw"] if r["id"] in base else chat(model, msgs)
                left = ret.left(r["src"], postprocess(first))
                raw = fix_turn(model, msgs, first, left) if left else first
                d = {"id": r["id"], "src": r["src"], "tgt": r["tgt"], "first": first, "raw": raw,
                     "fixed": bool(left), "pred": postprocess(raw)}
            else:
                raw = chat(model, msgs)
                d = {"id": r["id"], "src": r["src"], "tgt": r["tgt"], "raw": raw, "pred": postprocess(raw)}
            done[r["id"]] = d
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
    return [done[r["id"]]["pred"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["zero", "rag", "rag_lex"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--extra", nargs="*", default=[], help="extra example/lexicon data as dataset/split, e.g. generated/all")
    ap.add_argument("--tag", default="", help="suffix for the mode name, e.g. +gen")
    a = ap.parse_args()

    train = load("train")
    for e in a.extra:
        ds, split = e.split("/")
        train += load(split, ds)
    rows = load(a.split)[: a.limit]
    ret = Retriever(train)
    for model in a.models:
        for mode in a.modes:
            preds = run(model, mode, rows, ret, a.k, a.tag, a.split)
            res = {"model": model, "mode": mode + a.tag, **score(rows, preds)}
            print(json.dumps(res, ensure_ascii=False))
            if a.limit:  # smoke runs don't write scores
                continue
            out = "scores.jsonl" if a.split == "test" else f"scores_{a.split}.jsonl"
            with (ROOT / "results" / out).open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
