"""Meaning-level score of the end-to-end pipeline (src/pipeline.py) with the local LLM judge of src/judge.py.

1. Calibrate: the judge labels the 200 dev answers the user reviewed (results/advice/human_review_dev.json);
   agreement on correct-vs-not is reported.
2. Score: every gold dialect span of the split that needs a change is
     missed   - the tagger did not find it with the exact boundaries (dialect left in the output)
     correct  - found and equal to the reference, or judged correct
     partial / wrong - judged
   per tier (suggested / review) and overall.
Writes results/pipeline/judged_<split>.md

usage: python src/judge_pipeline.py --split test [--judge qwen3.8:27b]
"""
import argparse
import json
from collections import Counter

from data import ROOT, load, normalize
from judge import judge_all
from tag_dialect import spans_of, tokens

OUT = ROOT / "results" / "pipeline"
HL = {"صحيح": "correct", "جزئي": "partial", "خطأ": "wrong"}


def replaced(words, s, e, ans):
    return " ".join(words[:s] + [ans] + words[e:])


def calibrate(judge, tag):
    rows = {str(r["id"]): r for r in load("dev", "sentences_v2")}
    hum = json.load(open(ROOT / "results" / "advice" / "human_review_dev.json", encoding="utf-8"))
    items = []
    for h in hum:
        r = rows[h["id"]]
        items.append({"id": f"h{h['id']}_{h['رقم الكلمة']}", "src": r["src"], "d": h["الكلمة العامية"], "m": h["المرجع"],
                      "pred": r["src"].replace(h["الكلمة العامية"], h["الإجابة المقترحة"], 1)})
    got = judge_all(judge, items, OUT / f"judge_{tag}__human200.jsonl")
    gold = [HL[h["label"]] for h in hum]
    binary = sum((g == "correct") == (m == "correct") for g, m in zip(got, gold)) / len(gold)
    exact = sum(g == m for g, m in zip(got, gold)) / len(gold)
    return binary, exact, sum(g == "correct" for g in got) / len(got), sum(m == "correct" for m in gold) / len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--judge", default="qwen3.8:27b")
    a = ap.parse_args()
    tag = a.judge.replace(":", "_").replace("/", "_")
    binary, exact, j_rate, h_rate = calibrate(a.judge, tag)

    rows = {r["id"]: r for r in load(a.split, "sentences_v2")}
    preds = [json.loads(l) for l in (OUT / f"{a.split}.jsonl").open(encoding="utf-8")]
    st, todo = Counter(), []
    for o in preds:
        r = rows[o["id"]]
        gt = tokens(r["src_marked"])
        words = [t for t, _ in gt]
        pred = {tuple(x["span"]): x for x in o["replacements"]}
        for (s, e), (d, m) in zip(sorted(spans_of([l for _, l in gt])), r["pairs"]):
            if normalize(d) == normalize(m):
                continue
            x = pred.get((s, e))
            if not x:
                st["missed"] += 1
                continue
            t = x["tier"]
            st[t] += 1
            if normalize(x["msa"]) == normalize(m):
                st[t + "_correct"] += 1
            elif not x["msa"] or normalize(x["msa"]) == normalize(d):
                st[t + "_wrong"] += 1
            else:
                todo.append((t, {"id": f"{o['id']}_{s}", "src": r["src"], "d": d, "m": m,
                                 "pred": replaced(words, s, e, x["msa"])}))
    labels = judge_all(a.judge, [x for _, x in todo], OUT / f"judge_{tag}__{a.split}.jsonl")
    for (t, _), l in zip(todo, labels):
        st[f"{t}_{l}"] += 1

    need = st["missed"] + st["suggested"] + st["review"]
    pct = lambda x, n: f"{100 * x / max(n, 1):.1f}"
    lines = [f"# Pipeline on {a.split}, judged by meaning ({a.judge})", "",
             f"Judge check against the user's 200 dev labels: agreement correct-vs-not {100*binary:.0f}%, "
             f"3-way {100*exact:.0f}%; judge says correct {100*j_rate:.0f}% vs human {100*h_rate:.0f}%.", "",
             "| | spans | share of all | correct | partial | wrong |", "|---|---|---|---|---|---|"]
    for t in ("suggested", "review"):
        n = st[t]
        lines.append(f"| {t} | {n} | {pct(n, need)}% | {pct(st[t+'_correct'], n)}% | {pct(st[t+'_partial'], n)}% | {pct(st[t+'_wrong'], n)}% |")
    lines.append(f"| missed by the tagger | {st['missed']} | {pct(st['missed'], need)}% | | | |")
    c = st["suggested_correct"] + st["review_correct"]
    lines += [f"| **all spans that need a change** | {need} | 100% | **{pct(c, need)}%** | "
              f"{pct(st['suggested_partial'] + st['review_partial'], need)}% | |", "",
              f"Of the spans the tagger found: {pct(c, st['suggested'] + st['review'])}% correct by meaning."]
    (OUT / f"judged_{a.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
