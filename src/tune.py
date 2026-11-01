"""PLAN_v2 I5: tune the best LLM setups on dev, then measure only the winners on test.

Serves T1/T2. Steps (each resumable, predictions are cached by run_llm.py):
1. Pick the 2 best LLMs on test so far (rag-type modes), keeping their +gen choice.
2. dev: rag_lexf with k = 4, 8, 16 -> best k per model.
3. dev: rag_fix and rag_lexret with the best k.
4. test: every dev winner (dev chrF++ above that model's dev rag_lexf k=8), compared with the
   model's best test system by paired bootstrap (p < 0.05 = success criterion).
5. Oracle over the top 5 test systems (per-sentence best chrF++): if the gap to the best
   system is < 3 points, combining systems is not worth it.
Writes results/TUNING.md.

usage: python src/tune.py
"""
import json
import subprocess
import sys

import numpy as np
from sacrebleu.metrics import CHRF

from analysis import Boot, p_value, systems
from data import ROOT, load, normalize
from report import pred_file

RES = ROOT / "results"
PY = sys.executable
RAG_MODES = ("rag", "rag_lex", "rag_lexf")


def latest(path):
    rows = {}
    if path.exists():
        for line in path.open(encoding="utf-8"):
            d = json.loads(line)
            rows[(d["model"], d["mode"])] = d
    return rows


def run_llm(model, mode, tag, split, k):
    extra = ["--extra", "generated/all"] if tag.startswith("+gen") else []
    subprocess.run([PY, "src/run_llm.py", "--models", model, "--modes", mode, "--split", split,
                    "--k", str(k), "--tag", tag] + extra, cwd=ROOT, check=True)
    return latest(RES / ("scores.jsonl" if split == "test" else f"scores_{split}.jsonl"))[(model, mode + tag)]


def main():
    test_scores = latest(RES / "scores.jsonl")
    llm = [d for d in test_scores.values() if d["mode"].split("+")[0] in RAG_MODES]
    best = {}
    for d in sorted(llm, key=lambda d: -d["chrF++"]):
        best.setdefault(d["model"], d)
    top2 = list(best.values())[:2]
    lines = ["# I5 tuning (dev) and test check", "",
             "Best two LLM setups on test before tuning: " + ", ".join(f"{d['model']} / {d['mode']} ({d['chrF++']})" for d in top2), ""]

    test = load("test")
    boot = Boot(test, len(test))
    dev_rows, test_rows = [], []
    for d in top2:
        model, gen = d["model"], "+gen" if "+gen" in d["mode"] else ""
        dev = {k: run_llm(model, "rag_lexf", f"{gen}_k{k}", "dev", k) for k in (4, 8, 16)}
        k_best = max(dev, key=lambda k: dev[k]["chrF++"])
        tag = f"{gen}_k{k_best}"
        cand = {f"rag_lexf k={k}": (("rag_lexf", f"{gen}_k{k}", k), dev[k]) for k in dev}
        for mode in ("rag_fix", "rag_lexret"):
            cand[f"{mode} k={k_best}"] = ((mode, tag, k_best), run_llm(model, mode, tag, "dev", k_best))
        base_dev = dev[8]["chrF++"]
        for name, (_, s) in cand.items():
            dev_rows.append(f"| {model}{gen} | {name} | {s['chrF++']} | {s['SpanAcc%']} | {s['DialectLeft%']} | {s['chrF++'] - base_dev:+.2f} |")

        # test: only the dev winners, against this model's best test system
        base_boot = boot.scores(_preds(d, test))["chrF++"]
        for name, ((mode, t, k), s) in cand.items():
            if name == "rag_lexf k=8" or s["chrF++"] <= base_dev:
                continue
            res = run_llm(model, mode, t, "test", k)
            p = p_value(boot.scores(_preds(res, test))["chrF++"], base_boot)
            test_rows.append(f"| {model} | {mode}{t} | {res['chrF++']} | {d['mode']} {d['chrF++']} | "
                             f"{res['chrF++'] - d['chrF++']:+.2f} | {p:.3f} | {res['SpanAcc%']} | {res['DialectLeft%']} |")

    lines += ["## dev (290 sentences)", "", "| model | setup | chrF++ | SpanAcc% | DialectLeft% | Δ vs rag_lexf k=8 |",
              "|---|---|---|---|---|---|"] + dev_rows
    lines += ["", "## test: dev winners only", "",
              "| model | setup | chrF++ | before (best test system) | Δ | p | SpanAcc% | DialectLeft% |",
              "|---|---|---|---|---|---|---|---|"] + (test_rows or ["| — | no setup beat rag_lexf k=8 on dev | | | | | | |"])
    lines += ["", "Success criterion (PLAN_v2 I5): p < 0.05 on test after choosing on dev.", ""] + oracle(test)
    (RES / "TUNING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _preds(d, test):
    p = {x["id"]: x["pred"] for x in map(json.loads, pred_file(d).open(encoding="utf-8"))}
    return [p[r["id"]] for r in test]


def oracle(test):
    """Per-sentence best of the top 5 test systems (upper bound for combining systems)."""
    top = systems(test)[:5]
    chrf = CHRF(word_order=2)
    refs = [normalize(r["tgt"]) for r in test]
    pick = []
    for i, ref in enumerate(refs):
        cands = [normalize(s["preds"][i]) for s in top]
        pick.append(max(cands, key=lambda h: chrf.sentence_score(h, [ref]).score))
    o = chrf.corpus_score(pick, [refs]).score
    gap = o - top[0]["chrF++"]
    return ["## Oracle over the top 5 test systems", "",
            f"Systems: {', '.join(s['name'] for s in top)}", "",
            f"Oracle chrF++ {o:.2f} vs best system {top[0]['chrF++']} (gap {gap:+.2f}). "
            + ("Gap < 3: combining systems is not worth it." if gap < 3 else "Gap ≥ 3: a combiner (e.g. selection by an LLM) may pay off.")]


if __name__ == "__main__":
    main()
