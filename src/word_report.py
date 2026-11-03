"""Word-level results on the corrected test set (data/sentences_v2) -> results/word/REPORT.md.

Judges every span the strict match could not decide (same judge and prompt as judge.py) for the
given systems, and compares with the sentence-level systems re-scored on the corrected test set.

usage: python src/word_report.py --judge qwen3.8:27b
"""
import argparse
import json
import re

from data import ROOT, load, normalize
from judge import judge_all
from metrics import span_accuracy

RES = ROOT / "results"
W = RES / "word"


def latest(path):
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            d = json.loads(line)
            rows[(d["model"], d["mode"])] = d
    return rows


def pred_path(model, mode):
    return W / "preds" / f"{model.replace(':', '_').replace('/', '_')}__{mode}.jsonl"


def judged(judge, model, mode, test):
    """Judged correct% over all spans: exact matches count as correct, the rest are judged."""
    preds = {d["id"]: d["pred"] for d in map(json.loads, pred_path(model, mode).open(encoding="utf-8"))}
    items, exact, n = [], 0, 0
    for r in test:
        p = preds[r["id"]]
        for k, (d, m) in zip(r["span_ids"], r["pairs"]):
            if normalize(d) == normalize(m):
                continue
            n += 1
            x = p.get(str(k), "")
            if normalize(x) == normalize(m):
                exact += 1
                continue
            # the judge rates a translation: put the predicted word into the dialect sentence
            sent = re.sub(r"\(([^()]*)\)\s*(\d+)", lambda g: x if int(g.group(2)) == k else g.group(1), r["src_marked"])
            items.append({"id": r["id"], "src": r["src"], "d": d, "m": m, "pred": sent})
    name = f"word__{model}__{mode}".replace(":", "_").replace("/", "_")
    labels = judge_all(judge, items, RES / "judge" / f"{judge.replace(':', '_')}__{name}.jsonl")
    return {"correct": 100 * (exact + labels.count("correct")) / n, "partial": 100 * labels.count("partial") / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3.8:27b")
    ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args()
    test = load("test", "sentences_v2")
    rows = sorted(latest(W / "scores.jsonl").values(), key=lambda d: -d["WordLoose%"])
    md = ["# Word-level task on the corrected data (sentences_v2)", "",
          "The dialect words are marked in the input; the model returns only their MSA equivalents.",
          f"Test: {len(test)} sentences, {rows[0]['spans'] if rows else '?'} dialect spans. Judge: **{a.judge}** "
          "(agrees 87% with 60 Claude-written labels; human labels pending).", "",
          "| # | model | mode | k | WordAcc% (exact) | WordLoose% | judged correct% | partial% | Copy% | Missing% |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for i, d in enumerate(rows, 1):
        j = judged(a.judge, d["model"], d["mode"], test) if i <= a.top else None
        cor, par = (f"{j['correct']:.1f}", f"{j['partial']:.1f}") if j else ("—", "—")
        md.append(f"| {i} | {d['model']} | {d['mode']} | {d['k']} | {d['WordAcc%']} | {d['WordLoose%']} | "
                  f"{cor} | {par} | {d['Copy%']} | {d['Missing%']} |")

    # sentence-level systems re-scored on the corrected test references, for comparison
    md += ["", "## Sentence-level systems (rewrite the whole sentence) on the same corrected test set", "",
           "| model | mode | SpanAcc% (gold word appears in the output) |", "|---|---|---|"]
    sent = latest(RES / "scores.jsonl")
    best = sorted(sent.values(), key=lambda d: -d["chrF++"])[:6]
    for d in best:
        f = RES / "preds" / f"{d['model'].replace(':', '_').replace('/', '_')}__{d['mode']}.jsonl"
        if not f.exists():
            continue
        p = {x["id"]: x["pred"] for x in map(json.loads, f.open(encoding="utf-8"))}
        sp = [s for r in test if r["id"] in p and (s := span_accuracy(p[r["id"]], r["pairs"])) is not None]
        md.append(f"| {d['model']} | {d['mode']} | {100 * sum(sp) / len(sp):.1f} |")
    (W / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
