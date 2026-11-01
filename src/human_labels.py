"""Human labels for judge validation (PLAN_v2 I3).

Build: results/human_labels_todo.csv with 150 random "other" spans (neither matching the gold MSA
nor left in dialect) from the 5 best systems. The user fills الحكم with صحيح / حدّي / خطأ.
Import: converts the filled file to results/human_labels.jsonl, which judge.py then uses instead
of the Claude-written manual_labels.jsonl.

usage: python src/human_labels.py            # build the todo file
       python src/human_labels.py --import   # after the user filled it
"""
import argparse
import json
import random

import pandas as pd

from analysis import span_outcomes, systems
from data import ROOT, lexicon, load

RES = ROOT / "results"
TODO = RES / "human_labels_todo.csv"
LABELS = {"صحيح": "correct", "حدّي": "partial", "حدي": "partial", "خطأ": "wrong"}


def build(n=150):
    random.seed(13)
    test, lex = load("test"), lexicon(load("train"))
    pool = []
    for s in systems(test)[:5]:
        for x in span_outcomes(test, s["preds"], lex):
            if x["outcome"] == "other":
                pool.append({**x, "system": s["name"]})
    rows = random.sample(pool, min(n, len(pool)))
    pd.DataFrame([{"النظام": x["system"], "Feature 1": x["id"], "الجملة العامية": x["src"], "الكلمة العامية": x["d"],
                   "المقابل في المرجع": x["m"], "الجملة المرجعية": x["tgt"], "مخرج النظام": x["pred"],
                   "الحكم": ""} for x in rows]).to_csv(TODO, index=False, encoding="utf-8-sig")
    print(f"wrote {TODO}: {len(rows)} of {len(pool)} undecided spans; الحكم = صحيح / حدّي / خطأ")


def import_labels():
    df = pd.read_csv(TODO, dtype=str).fillna("")
    done = df[df["الحكم"].str.strip().isin(LABELS)]
    out = RES / "human_labels.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for _, r in done.iterrows():
            f.write(json.dumps({"id": int(r["Feature 1"]), "src": r["الجملة العامية"], "d": r["الكلمة العامية"],
                                "m": r["المقابل في المرجع"], "pred": r["مخرج النظام"],
                                "label": LABELS[r["الحكم"].strip()]}, ensure_ascii=False) + "\n")
    print(f"{len(done)}/{len(df)} labelled -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", action="store_true")
    import_labels() if ap.parse_args().imp else build()
