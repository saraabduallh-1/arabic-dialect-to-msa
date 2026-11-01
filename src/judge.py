"""LLM-as-judge for the dialect spans the automatic metric cannot decide -> results/JUDGE.md.

SpanAcc only counts the single gold MSA word, so a valid synonym ("رأيت" for "شاهدت") is scored
as a miss. Spans whose gold MSA is already in the output stay 'correct' and spans whose dialect
word was left untouched stay 'wrong'; every other span is labelled correct / partial / wrong by
a judge model served by Ollama.

Before judging, the judge is checked against the 60 hand-labelled spans in
results/manual_labels.jsonl; agreement is written to the report.

usage: python src/judge.py --judge qwen3.8:27b [--top 10]
"""
import argparse
import json
from collections import Counter

import requests
from tqdm import tqdm

from analysis import span_outcomes, systems
from data import ROOT, lexicon, load

OLLAMA = "http://localhost:11434/api/chat"
RES = ROOT / "results"
OUT = RES / "judge"
LABELS = ["correct", "partial", "wrong"]

SYSTEM = (
    "أنت محكّم خبير في العربية الفصحى واللهجات السعودية والخليجية. "
    "ستُعطى جملة عامية، وكلمة أو عبارة عامية منها، ومقابلها الفصيح المرجعي، وترجمة نموذج للجملة إلى الفصحى. "
    "احكم على طريقة نقل الترجمة لتلك الكلمة العامية فقط:\n"
    "correct: نقلت معناها في السياق بلفظ فصيح سليم (المرادفات المقبولة صحيحة وإن خالفت المرجع)\n"
    "partial: المعنى قريب لكن فيه انزياح واضح أو خلل في الصياغة\n"
    "wrong: معنى مختلف، أو حُذفت، أو بقيت عامية، أو الصياغة غير سليمة\n"
    'أجب بصيغة JSON فقط: {"label": "correct" | "partial" | "wrong"}'
)


def ask(model, x):
    user = (f"الجملة العامية: {x['src']}\nالكلمة العامية: {x['d']}\nالمقابل المرجعي: {x['m']}\n"
            f"ترجمة النموذج: {x['pred']}")
    body = {"model": model, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 32}}
    r = requests.post(OLLAMA, json=body, timeout=600)
    if r.status_code == 400 and "think" in r.text:
        body.pop("think")
        r = requests.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    try:
        label = json.loads(r.json()["message"]["content"]).get("label")
    except (json.JSONDecodeError, AttributeError):
        label = None
    return label if label in LABELS else "wrong"


def judge_all(model, items, path):
    """Label items (resumable); key = sentence id + dialect span."""
    done = {}
    if path.exists():
        done = {(d["id"], d["d"]): d["label"] for d in map(json.loads, path.open(encoding="utf-8"))}
    with path.open("a", encoding="utf-8") as f:
        for x in tqdm(items, desc=path.stem, leave=False):
            k = (x["id"], x["d"])
            if k not in done:
                done[k] = ask(model, x)
                f.write(json.dumps({"id": x["id"], "d": x["d"], "label": done[k]}, ensure_ascii=False) + "\n")
                f.flush()
    return [done[(x["id"], x["d"])] for x in items]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, help="Ollama model used as judge")
    ap.add_argument("--top", type=int, help="only the N best systems by chrF++")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.judge.replace(":", "_").replace("/", "_")

    # 1. agreement with the hand labels
    # human labels (PLAN_v2 I3) replace the Claude-written ones once the user has filled them in
    human = RES / "human_labels.jsonl"
    src = human if human.exists() and human.stat().st_size else RES / "manual_labels.jsonl"
    manual = [json.loads(l) for l in src.open(encoding="utf-8")]
    got = judge_all(a.judge, manual, OUT / f"{tag}__{'human' if src == human else 'manual'}.jsonl")
    exact = sum(g == m["label"] for g, m in zip(got, manual)) / len(manual)
    binary = sum((g == "correct") == (m["label"] == "correct") for g, m in zip(got, manual)) / len(manual)
    conf = Counter((m["label"], g) for g, m in zip(got, manual))

    # 2. judge every undecided span of every system
    train, test = load("train"), load("test")
    lex = lexicon(train)
    sy = systems(test)[: a.top]
    rows = []
    for s in sy:
        sp = span_outcomes(test, s["preds"], lex)
        other = [x for x in sp if x["outcome"] == "other"]
        name = f"{s['model']}__{s['mode']}".replace(":", "_").replace("/", "_")
        lab = Counter(judge_all(a.judge, other, OUT / f"{tag}__{name}.jsonl"))
        auto = sum(x["outcome"] == "correct" for x in sp)
        n = len(sp)
        rows.append({"name": s["name"], "chrF++": s["chrF++"], "auto": 100 * auto / n,
                     "correct": 100 * (auto + lab["correct"]) / n, "partial": 100 * lab["partial"] / n,
                     "wrong": 100 * (n - auto - lab["correct"] - lab["partial"]) / n})

    rows.sort(key=lambda r: -r["correct"])
    md = ["# LLM-judge evaluation of dialect spans", "",
          f"Judge: **{a.judge}**. Spans already matching the gold MSA count as correct, spans left in dialect "
          "as wrong; the rest are judged.", "",
          f"## Agreement with the {len(manual)} labelled spans ({'human' if src == human else 'Claude-written'} labels, {src.name})", "",
          f"- Exact agreement (3 labels): {100 * exact:.0f}%",
          f"- Agreement on correct vs not: {100 * binary:.0f}%", "",
          "| manual \\ judge | " + " | ".join(LABELS) + " |", "|---" * 4 + "|"]
    md += [f"| {m} | " + " | ".join(str(conf[(m, g)]) for g in LABELS) + " |" for m in LABELS]
    md += ["", "## Ranking by judged span accuracy", "",
           "| # | System | chrF++ | auto SpanAcc% | judged correct% | partial% | wrong% |", "|---" * 7 + "|"]
    md += [f"| {i} | {r['name']} | {r['chrF++']} | {r['auto']:.1f} | {r['correct']:.1f} | {r['partial']:.1f} | "
           f"{r['wrong']:.1f} |" for i, r in enumerate(rows, 1)]
    text = "\n".join(md) + "\n"
    (RES / f"JUDGE_{tag}.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
