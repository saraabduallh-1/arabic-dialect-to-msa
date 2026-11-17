"""Paired bootstrap over test sentences for the judged word systems (exact or judged-correct per span).

For each pair (reference system, other system): resample the 291 sentences with replacement 2,000 times,
recompute correct% for both, and report the difference, its 95% interval and p (share of resamples where
the other system is not worse). Also reports exact WordAcc differences.
Writes results/word/SIGNIF.md.

usage: python src/word_signif.py
"""
import json
import random

from data import ROOT, load, normalize

W = ROOT / "results" / "word"
J = ROOT / "results" / "judge"
SYSTEMS = [
    ("ALLaM-7B-Instruct-preview-qlora-word+gen3", "finetune"),
    ("ALLaM-7B-Instruct-preview-qlora-word", "finetune"),
    ("ALLaM-7B-Instruct-preview-qlora-word+gen4", "finetune"),
    ("ALLaM-7B-Instruct-preview-qlora-word+gen4+lex", "finetune"),
    ("gemma3:27b", "rag+gen2"),
]


def per_sentence(test, model, mode):
    name = f"{model.replace(':', '_').replace('/', '_')}__{mode}"
    pp = W / "preds" / f"{name}.jsonl"
    jp = J / f"qwen3.8_27b__word__{name}.jsonl"
    if not pp.exists() or not jp.exists():
        return None
    preds = {d["id"]: d["pred"] for d in map(json.loads, pp.open(encoding="utf-8"))}
    lab = {(d["id"], d["d"]): d["label"] for d in map(json.loads, jp.open(encoding="utf-8"))}
    out = []
    for r in test:
        c = e = n = 0
        for k, (d, m) in zip(r["span_ids"], r["pairs"]):
            if normalize(d) == normalize(m):
                continue
            n += 1
            x = preds[r["id"]].get(str(k), "")
            ex = normalize(x) == normalize(m)
            e += ex
            c += ex or lab.get((r["id"], d)) == "correct"
        out.append((c, e, n))
    return out


def main():
    test = load("test", "sentences_v2")
    data = {s: per_sentence(test, *s) for s in SYSTEMS}
    data = {s: v for s, v in data.items() if v}
    ref = SYSTEMS[0]
    rng = random.Random(0)
    idx = [[rng.randrange(len(test)) for _ in test] for _ in range(2000)]

    def pct(v, ids, j):
        return 100 * sum(v[i][j] for i in ids) / max(sum(v[i][2] for i in ids), 1)

    lines = ["# Paired bootstrap, word task (2,000 resamples of the 291 test sentences)", "",
             f"Reference: **{ref[0]} / {ref[1]}**. Δ = other − reference; p = share of resamples with Δ ≥ 0 (one-sided).", "",
             "| system | judged correct% | Δ judged | 95% CI | p | WordAcc% | Δ exact | p |", "|---|---|---|---|---|---|---|---|"]
    for s, v in data.items():
        full = list(range(len(test)))
        row = [f"{s[0]} / {s[1]}", f"{pct(v, full, 0):.1f}"]
        if s == ref:
            lines.append("| " + " | ".join(row + ["—", "—", "—", f"{pct(v, full, 1):.2f}", "—", "—"]) + " |")
            continue
        for j in (0, 1):
            d = sorted(pct(v, ids, j) - pct(data[ref], ids, j) for ids in idx)
            delta = pct(v, full, j) - pct(data[ref], full, j)
            p = sum(x >= 0 for x in d) / len(d)
            if j == 0:
                row += [f"{delta:+.1f}", f"[{d[50]:+.1f}, {d[1949]:+.1f}]", f"{p:.3f}"]
            else:
                row += [f"{pct(v, full, 1):.2f}", f"{delta:+.2f}", f"{p:.3f}"]
        lines.append("| " + " | ".join(row) + " |")
    (W / "SIGNIF.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
