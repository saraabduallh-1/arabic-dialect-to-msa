"""Meaning-level score of the merged layer (src/merged.py) on test, with the local LLM judge of src/judge.py.

Every gold span that needs a change (dialect test, foreign test) is
  missed   - the tagger did not find it with the exact boundaries
  correct  - found and equal to the reference, or judged correct
  partial / wrong - judged (a copy of the word is wrong without asking)
Foreign spans are scored for both answers: Gemma (final layer) and the foreign model (first version).
The judge was checked against the user's 200 dialect dev labels (src/judge_pipeline.py); the human review page
of the merged layer (results/merge/human_review_items.json) gives a second check, for foreign words too.

usage: python src/judge_merged.py [--judge qwen3.8:27b]
"""
import argparse
import json
from collections import Counter

import requests
from tqdm import tqdm

from data import load, normalize
from foreign_data import OUT, load_splits
from judge import LABELS, OLLAMA, SYSTEM as DSYSTEM
from tag_dialect import tokens
from tag_merged import typed_spans

FSYSTEM = (
    "أنت محكّم خبير في العربية الفصحى. "
    "ستُعطى جملة فيها كلمة أو عبارة أعجمية (منقولة من الإنجليزية أو غيرها)، ومقابلها الفصيح المرجعي، وترجمة نموذج للجملة إلى الفصحى. "
    "احكم على طريقة نقل الترجمة لتلك الكلمة الأعجمية فقط:\n"
    "correct: نقلت معناها في السياق بلفظ عربي فصيح سليم (المرادفات المقبولة صحيحة وإن خالفت المرجع)\n"
    "partial: المعنى قريب لكن فيه انزياح واضح أو خلل في الصياغة\n"
    "wrong: معنى مختلف، أو حُذفت، أو بقيت أعجمية، أو الصياغة غير سليمة\n"
    'أجب بصيغة JSON فقط: {"label": "correct" | "partial" | "wrong"}'
)


def ask(model, x):
    sysm, lab = (DSYSTEM, "العامية") if x["kind"] == "dialect" else (FSYSTEM, "الأعجمية")
    user = (f"الجملة: {x['src']}\nالكلمة {lab}: {x['d']}\nالمقابل المرجعي: {x['m']}\nترجمة النموذج: {x['pred']}")
    body = {"model": model, "stream": False, "think": False, "format": "json", "keep_alive": "10m",
            "messages": [{"role": "system", "content": sysm}, {"role": "user", "content": user}],
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 32}}
    r = requests.post(OLLAMA, json=body, timeout=900)
    if r.status_code == 400 and "think" in r.text:
        body.pop("think")
        r = requests.post(OLLAMA, json=body, timeout=900)
    r.raise_for_status()
    try:
        label = json.loads(r.json()["message"]["content"]).get("label")
    except (json.JSONDecodeError, AttributeError):
        label = None
    return label if label in LABELS else "wrong"


def judge_all(model, items, path):
    """Resumable; key = sentence id + span + answer, so two systems' answers never share a label by mistake."""
    key = lambda x: f"{x['id']}|{x['span']}|{normalize(x['ans'])}"
    done = {}
    if path.exists():
        done = {d["key"]: d["label"] for d in map(json.loads, path.open(encoding="utf-8"))}
    with path.open("a", encoding="utf-8") as f:
        for x in tqdm(items, desc=path.stem):
            k = key(x)
            if k not in done:
                done[k] = ask(model, x)
                f.write(json.dumps({"key": k, "kind": x["kind"], "label": done[k]}, ensure_ascii=False) + "\n")
                f.flush()
    return [done[key(x)] for x in items]


def spans_to_score():
    """(system, kind, gold spans that need a change, predictions) for dialect test and foreign test."""
    out = []
    rows = load("test", "sentences_v2")
    preds = [json.loads(l) for l in (OUT / "dialect_test_gemma.jsonl").open(encoding="utf-8")]
    items = []
    for r, o in zip(rows, preds):
        gsp = sorted(typed_spans([l for _, l in tokens(r["src_marked"])])[0])
        gold = {sp: m for sp, (_, m) in zip(gsp, r["pairs"])}
        items.append((r["id"], o, gold))
    out.append(("layer", "dialect", items))
    frows = load_splits()["test"]
    for name, file in (("layer", "foreign_test_gemma.jsonl"), ("foreign_model", "foreign_test_m2m.jsonl")):
        preds = [json.loads(l) for l in (OUT / file).open(encoding="utf-8")]
        out.append((name, "foreign", [(r["id"], o, {(s, e): m for s, e, m in r["spans"]}) for r, o in zip(frows, preds)]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3.8:27b")
    a = ap.parse_args()
    tag = a.judge.replace(":", "_").replace("/", "_")
    pct = lambda x, n: f"{100 * x / max(n, 1):.1f}"
    L = [f"# Merged layer on test, judged by meaning ({a.judge})", "",
         "Judge agreement with the user's 200 dialect dev labels (earlier check, src/judge_pipeline.py): 78% correct-vs-not; "
         "the judge is stricter than the human.", "",
         "| system | kind | tier | spans | share of all | correct | partial | wrong |", "|---|---|---|---|---|---|---|---|"]
    for system, kind, items in spans_to_score():
        st, todo = Counter(), []
        for sid, o, gold in items:
            words = o["src"].split()
            pred = {tuple(x["span"]): x for x in o["replacements"] if x["type"] == kind}
            for sp, m in gold.items():
                d = " ".join(words[sp[0]:sp[1]])
                if normalize(d) == normalize(m):
                    continue
                x = pred.get(sp)
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
                    todo.append((t, {"kind": kind, "id": sid, "span": list(sp), "ans": x["msa"], "src": o["src"], "d": d, "m": m,
                                     "pred": " ".join(words[:sp[0]] + [x["msa"]] + words[sp[1]:])}))
        labels = judge_all(a.judge, [x for _, x in todo], OUT / f"judge_{tag}__test.jsonl")
        for (t, _), l in zip(todo, labels):
            st[f"{t}_{l}"] += 1
        need = st["missed"] + st["suggested"] + st["review"]
        for t in ("suggested", "review"):
            n = st[t]
            L.append(f"| {system} | {kind} | {t} | {n} | {pct(n, need)}% | {pct(st[t+'_correct'], n)}% | "
                     f"{pct(st[t+'_partial'], n)}% | {pct(st[t+'_wrong'], n)}% |")
        c = st["suggested_correct"] + st["review_correct"]
        found = st["suggested"] + st["review"]
        L.append(f"| {system} | {kind} | **all found** | {found} | {pct(found, need)}% | **{pct(c, found)}%** | "
                 f"{pct(st['suggested_partial'] + st['review_partial'], found)}% | |")
        L.append(f"| {system} | {kind} | missed by the tagger | {st['missed']} | {pct(st['missed'], need)}% | | | |")
        L.append(f"| {system} | {kind} | **end to end (all spans)** | {need} | 100% | **{pct(c, need)}%** | | |")
    (OUT / "judged_test.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
