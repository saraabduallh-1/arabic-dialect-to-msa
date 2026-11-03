"""Word-level task: the dialect words are marked, the model returns only their MSA equivalents.

Input  : the dialect sentence with its marked spans, e.g. "(مدري)1 وش صار"
Output : one line per span number, e.g. "1: لا أدري"
This matches the project's actual task (replace a known dialect word in context), unlike
run_llm.py, which rewrites the whole sentence and must find the dialect words itself.

Modes: zero | rag (k most similar training sentences as worked examples) | rag_lexf (+ filtered
dictionary hints for the marked words).

Metrics per span (spans whose dialect and MSA differ):
  WordAcc%    prediction equals the gold MSA after normalization
  WordLoose%  gold contained in the prediction or vice versa (tolerates ال/و/ب and extra words)
  Copy%       the dialect word was returned unchanged
  Missing%    no answer for that span number

usage: python src/run_word.py --models gemma3:27b --modes zero rag rag_lexf --data sentences_v2 \
                              --extra generated/all2 --tag +gen2
"""
import argparse
import json
import re

from tqdm import tqdm

from data import ROOT, load, normalize
from run_llm import Retriever, chat

RES = ROOT / "results" / "word"
SYSTEM = (
    "أنت خبير في اللغة العربية ولهجاتها (خاصة اللهجات السعودية). "
    "ستُعطى جملة عامية فيها كلمات أو عبارات عامية بين قوسين وبعد كل منها رقم. "
    "اكتب لكل رقم مقابله في العربية الفصحى المناسب لمعنى الجملة وسياقها، بنفس الصيغة النحوية قدر الإمكان. "
    "أجب بسطر واحد لكل رقم بالشكل «الرقم: المقابل الفصيح» فقط، بدون شرح."
)
LINE = re.compile(r"^\s*[\(\[]?(\d+)[\)\]]?\s*[:：\-–.]\s*(.+?)\s*$")


def answer(r):
    return "\n".join(f"{k}: {m}" for k, (_, m) in zip(r["span_ids"], r["pairs"]))


def build(r, mode, ret, k):
    msgs = [{"role": "system", "content": SYSTEM}]
    if mode in ("rag", "rag_lexf"):
        for ex in ret.examples(r["src"], k):
            if ex["pairs"] and "src_marked" in ex:
                msgs += [{"role": "user", "content": f"الجملة: {ex['src_marked']}"},
                         {"role": "assistant", "content": answer(ex)}]
    user = f"الجملة: {r['src_marked']}"
    if mode == "rag_lexf":
        h = [(d, m) for d, m in ret.hints(r["src"], filtered=True)
             if any(normalize(d) == normalize(x) for x, _ in r["pairs"])]
        if h:
            user += "\nتلميحات من القاموس (قد لا تنطبق على السياق):\n" + "\n".join(
                f"- {d} ← {' / '.join(m[:3])}" for d, m in h)
    msgs.append({"role": "user", "content": user})
    return msgs


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = {}
    for line in text.splitlines():
        m = LINE.match(line)
        if m and int(m.group(1)) not in out:
            out[int(m.group(1))] = m.group(2).strip().strip('"«»“”')
    return out


def score(rows, preds):
    n = acc = loose = copy = miss = 0
    for r, p in zip(rows, preds):
        for k, (d, m) in zip(r["span_ids"], r["pairs"]):
            if normalize(d) == normalize(m):
                continue
            n += 1
            g, x = normalize(m), normalize(p.get(str(k), p.get(k, "")))
            if not x:
                miss += 1
                continue
            acc += x == g
            loose += x == g or f" {g} " in f" {x} " or f" {x} " in f" {g} "
            copy += x == normalize(d)
    pct = lambda v: round(100 * v / n, 2)
    return {"WordAcc%": pct(acc), "WordLoose%": pct(loose), "Copy%": pct(copy), "Missing%": pct(miss), "spans": n}


def run(model, mode, rows, ret, k, tag, split):
    path = RES / "preds" / f"{model.replace(':', '_').replace('/', '_')}__{mode}{tag}{'' if split == 'test' else '@' + split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {d["id"]: d for d in map(json.loads, path.open(encoding="utf-8"))} if path.exists() else {}
    with path.open("a", encoding="utf-8") as f:
        for r in tqdm(rows, desc=f"{model} {mode}{tag}"):
            if r["id"] in done:
                continue
            raw = chat(model, build(r, mode, ret, k))
            d = {"id": r["id"], "src": r["src_marked"], "gold": dict(zip(map(str, r["span_ids"]), [m for _, m in r["pairs"]])),
                 "raw": raw, "pred": {str(a): b for a, b in parse(raw).items()}}
            done[r["id"]] = d
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            f.flush()
    return [done[r["id"]]["pred"] for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--modes", nargs="+", default=["zero", "rag", "rag_lexf"])
    ap.add_argument("--data", default="sentences_v2")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    train = load("train", a.data)
    for e in a.extra:
        ds, sp = e.split("/")
        train += load(sp, ds)
    rows = load(a.split, a.data)[: a.limit]
    ret = Retriever(train)
    for model in a.models:
        for mode in a.modes:
            res = {"model": model, "mode": mode + a.tag, "data": a.data, "k": a.k,
                   **score(rows, run(model, mode, rows, ret, a.k, a.tag, a.split))}
            print(json.dumps(res, ensure_ascii=False))
            if a.limit:
                continue
            with (RES / ("scores.jsonl" if a.split == "test" else f"scores_{a.split}.jsonl")).open("a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
