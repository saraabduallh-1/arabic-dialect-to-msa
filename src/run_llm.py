"""Evaluate open-source LLMs served by Ollama on dialect -> MSA conversion.

Modes:
  zero     instruction only
  rag      + the k most similar training pairs as few-shot examples
  rag_lex  + dictionary hints for dialect words found in the training lexicon
  rag_lexf like rag_lex, but drops hints for words used more often as MSA than as dialect

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

    def examples(self, src, k):
        sims = linear_kernel(self.vec.transform([normalize(src)]), self.mat)[0]
        return [self.train[i] for i in sims.argsort()[::-1][:k]][::-1]  # most similar last

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
    if mode in ("rag", "rag_lex", "rag_lexf"):
        for ex in ret.examples(src, k):
            msgs.append({"role": "user", "content": f"العامية: {ex['src']}"})
            msgs.append({"role": "assistant", "content": ex["tgt"]})
    user = f"العامية: {src}"
    if mode in ("rag_lex", "rag_lexf"):
        h = ret.hints(src, filtered=mode == "rag_lexf")
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


def run(model, mode, rows, ret, k):
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    path = PRED_DIR / f"{model.replace(':', '_').replace('/', '_')}__{mode}.jsonl"
    done = {}
    if path.exists():
        done = {d["id"]: d for d in map(json.loads, path.open(encoding="utf-8"))}
    with path.open("a", encoding="utf-8") as f:
        for r in tqdm(rows, desc=f"{model} {mode}"):
            if r["id"] in done:
                continue
            raw = chat(model, build_messages(r["src"], mode, ret, k))
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
    a = ap.parse_args()

    train = load("train")
    rows = load(a.split)[: a.limit]
    ret = Retriever(train)
    for model in a.models:
        for mode in a.modes:
            res = {"model": model, "mode": mode, **score(rows, run(model, mode, rows, ret, a.k))}
            print(json.dumps(res, ensure_ascii=False))
            with (ROOT / "results" / "scores.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
