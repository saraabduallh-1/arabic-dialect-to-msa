"""Two confidence tiers for the word task (advice 1-3 in docs/ADVICE_FROM_KASS.md).

Per marked span:
  1. train dictionary: the dialect form (normalized) was seen in train at least MIN_N times and its MSA equivalent
     agrees in at least MIN_C of them -> tier "suggested", answer = that equivalent.
  2. two systems agree: ALLaM (QLoRA) and gemma3:27b give the same answer after normalization -> "suggested".
  3. otherwise -> "review" with the ALLaM answer.
  No-copy rule (always): an answer equal to the dialect form is replaced by the dictionary equivalent, else by the
  other system's answer if that is not a copy too; otherwise the span is left "review" with no answer.

Thresholds are chosen on dev (--split dev prints a sweep); test is run once with the chosen values.
Metric: exact and loose match (run_word.py definitions). Judged accuracy needs the judge on these outputs.

usage: python src/tiers.py --split dev
       python src/tiers.py --split test --min_n 2 --min_c 0.8
"""
import argparse
import json
from collections import Counter, defaultdict

from data import ROOT, load, normalize

P = ROOT / "results" / "word" / "preds"
ALLAM = "ALLaM-7B-Instruct-preview-qlora-word+gen3__finetune"  # chosen on dev (48.1 vs 46.7 exact for +gen4+lex)
GEMMA = "gemma3_27b__rag+gen2"


def preds(name, split):
    f = P / f"{name}{'' if split == 'test' else '@' + split}.jsonl"
    return {d["id"]: d["pred"] for d in map(json.loads, f.open(encoding="utf-8"))}


def train_dict():
    d = defaultdict(Counter)
    for r in load("train", "sentences_v2"):
        for x, m in r["pairs"]:
            if normalize(x) != normalize(m):
                d[normalize(x)][m] += 1
    return d


def vocab(path):
    """Human-verified tweet vocabulary {normalized form: msa} (src/propose_equivalents.py + review page)."""
    if not path:
        return {}
    if path.endswith(".jsonl"):  # unreviewed gemma proposals (saudi and dialect)
        rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        return {normalize(v["form"]): v["msa"] for v in rows if v["saudi"] and v["dialect"] and v["msa"]}
    return {normalize(v["form"]): v["msa"] for v in json.load(open(path, encoding="utf-8"))}


def decide(d, m, a, g, dic, min_n, min_c, voc=None):
    """(tier, answer, source) for one span."""
    nd = normalize(d)
    c = dic.get(nd)
    top = None
    if c:
        top, k = c.most_common(1)[0]
        if sum(c.values()) >= min_n and k / sum(c.values()) >= min_c:
            return "suggested", top, "dict"
    a_copy, g_copy = normalize(a) in ("", nd), normalize(g) in ("", nd)
    if not a_copy and normalize(a) == normalize(g):
        return "suggested", a, "agree"
    # a fixed lexicon entry ignores context and inflection (قاعد: يجلس/جالس), so it only confirms ALLaM
    if voc and not a_copy and normalize(a) == normalize(voc.get(nd, "")):
        return "suggested", a, "agree_vocab"
    if a_copy:  # no-copy rule
        if voc and nd in voc:
            return "review", voc[nd], "nocopy:vocab"
        if top:
            return "review", top, "nocopy:dict"
        if not g_copy:
            return "review", g, "nocopy:gemma"
        return "review", "", "nocopy:none"
    return "review", a, "allam"


def run(split, min_n, min_c, verbose=True, voc=None, second=GEMMA):
    rows = load(split, "sentences_v2")
    A, G, dic = preds(ALLAM, split), preds(second, split), train_dict()
    st = Counter()
    for r in rows:
        for k, (d, m) in zip(r["span_ids"], r["pairs"]):
            if normalize(d) == normalize(m):
                continue
            a, g = A[r["id"]].get(str(k), ""), G[r["id"]].get(str(k), "")
            tier, ans, src = decide(d, m, a, g, dic, min_n, min_c, voc)
            x, y = normalize(ans), normalize(m)
            ok = bool(x) and x == y
            loose = bool(x) and (ok or f" {y} " in f" {x} " or f" {x} " in f" {y} ")
            for key in (("all",), (tier,), (tier, src)):
                st[key + ("n",)] += 1
                st[key + ("ok",)] += ok
                st[key + ("loose",)] += loose
            st[("base", "ok")] += bool(normalize(a)) and normalize(a) == y
    n = st[("all", "n")]
    res = {"split": split, "min_n": min_n, "min_c": min_c, "spans": n,
           "ALLaM alone exact%": round(100 * st[("base", "ok")] / n, 1)}
    for key in sorted({k[:-1] for k in st if k[0] != "base"}, key=lambda k: (len(k), k)):
        m = st[key + ("n",)]
        res[" / ".join(key)] = {"n": m, "share%": round(100 * m / n, 1),
                                "exact%": round(100 * st[key + ("ok",)] / m, 1),
                                "loose%": round(100 * st[key + ("loose",)] / m, 1)}
    if verbose:
        print(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--min_n", type=int, default=2)
    ap.add_argument("--min_c", type=float, default=0.8)
    ap.add_argument("--vocab", default="", help="verified tweet vocabulary json (dev to choose)")
    ap.add_argument("--second", default=GEMMA, help="second system for the agreement tier (preds name)")
    ap.add_argument("--sweep", action="store_true", help="dev only: grid over min_n x min_c")
    a = ap.parse_args()
    out = ROOT / "results" / "advice"
    out.mkdir(parents=True, exist_ok=True)
    if a.sweep:
        assert a.split != "test", "choose thresholds on dev"
        lines = ["| min_n | min_c | suggested share% | suggested exact% | review exact% | all exact% |", "|---|---|---|---|---|---|"]
        for mn in (1, 2, 3):
            for mc in (0.6, 0.8, 1.0):
                r = run(a.split, mn, mc, verbose=False)
                s, v = r.get("suggested", {}), r.get("review", {})
                lines.append(f"| {mn} | {mc} | {s.get('share%')} | {s.get('exact%')} | {v.get('exact%')} | {r['all']['exact%']} |")
        print("\n".join(lines))
        (out / f"tiers_sweep_{a.split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    r = run(a.split, a.min_n, a.min_c, voc=vocab(a.vocab), second=a.second)
    (out / f"tiers_{a.split}{'_vocab' if a.vocab else ''}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
