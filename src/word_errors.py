"""Error analysis of a fine-tuned ALLaM word system: accuracy by whether the dialect form was seen in train.
usage: python src/word_errors.py +gen3 25   (tag, examples per group)"""
import json, sys, csv, collections, re
sys.path.insert(0, "src")
from data import load, normalize, DATA
sys.stdout.reconfigure(encoding="utf-8")
test = load("test", "sentences_v2")
train = load("train", "sentences_v2")
def pairs(rows):
    s = collections.defaultdict(set)
    for r in rows:
        for d, m in r["pairs"]: s[normalize(d)].add(normalize(m))
    return s
tr = pairs(train)
ver = collections.defaultdict(set)
for r in csv.DictReader(open(DATA/"generated/verified.csv", encoding="utf-8")):
    for d, m in zip(re.findall(r"\(([^()]*)\)", r["السياق العامي"]), re.findall(r"\(([^()]*)\)", r["السياق الفصيح"])):
        ver[normalize(d)].add(normalize(m))
fv = set()
for l in open(DATA/"external/lahajatna/maajim.jsonl", encoding="utf-8"):
    fv |= set(normalize(json.loads(l)["text"]).split())
tag = sys.argv[1]
preds = {d["id"]: d["pred"] for d in map(json.loads, open(f"results/word/preds/ALLaM-7B-Instruct-preview-qlora-word{tag}__finetune.jsonl", encoding="utf-8"))}
lab = {(d["id"], d["d"]): d["label"] for d in map(json.loads, open(f"results/judge/qwen3.8_27b__word__ALLaM-7B-Instruct-preview-qlora-word{tag}__finetune.jsonl", encoding="utf-8"))}
C = collections.Counter(); ex = collections.defaultdict(list)
for r in test:
    p = preds[r["id"]]
    for k, (d, m) in zip(r["span_ids"], r["pairs"]):
        nd, nm = normalize(d), normalize(m)
        if nd == nm: continue
        x = p.get(str(k), "")
        res = "exact" if normalize(x) == nm else lab.get((r["id"], d), "?")
        ok = res in ("exact", "correct")
        seen = "train" if nd in tr else ("verified" if nd in ver else "unseen")
        gseen = seen == "train" and nm in tr[nd]
        multi = len(nd.split()) > 1
        fus = (not multi) and nd in fv
        for key in [("seen", seen), ("seen+gold", seen, gseen), ("multiword", multi), ("form_in_fusha_dict", fus), ("copy", normalize(x) == nd)]:
            C[key + (ok,)] += 1
        if not ok: ex[(seen, fus)].append(f"{d} → gold:{m} | pred:{x} | {r['src']}")
def show(prefix):
    keys = sorted({k[:-1] for k in C if k[:len(prefix)] == prefix})
    for k in keys:
        t, f = C[k + (True,)], C[k + (False,)]
        print(k, f"n={t+f} correct={100*t/(t+f):.0f}%")
for p in [("seen",), ("seen+gold",), ("multiword",), ("form_in_fusha_dict",), ("copy",)]: show(p)
for k, v in ex.items():
    print("\n##", k, len(v))
    for s in v[:int(sys.argv[2])]: print(" ", s)
