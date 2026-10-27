"""Significance tests and error analysis on the saved test predictions -> results/ANALYSIS.md.

1. Paired bootstrap (1000 resamples of the 291 test sentences): 95% CIs for chrF++/BLEU,
   p-values for every system vs the best one, and for rag vs zero / rag_lex vs rag per model.
2. Span-level error analysis: each gold dialect span is 'correct' (gold MSA present),
   'kept' (dialect left as is) or 'other' (changed to something else), split by whether
   the span occurs in the training lexicon and by span length.

usage: python src/analysis.py
"""
import json
from collections import Counter

import numpy as np
from sacrebleu.metrics import BLEU, CHRF

from data import ROOT, lexicon, load, normalize
from metrics import dialect_left, span_accuracy
from report import pred_file
from run_llm import Retriever

RES = ROOT / "results"
B, SEED = 1000, 42


def systems(test):
    rows = {}
    for line in (RES / "scores.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        rows[(d["model"], d["mode"])] = d
    out = []
    for d in sorted(rows.values(), key=lambda d: -d["chrF++"]):
        p = {x["id"]: x["pred"] for x in map(json.loads, pred_file(d).open(encoding="utf-8"))}
        out.append({**d, "name": f"{d['model']} / {d['mode']}", "preds": [p[r["id"]] for r in test]})
    return out


class Boot:
    """Per-sentence sufficient statistics so each resample is just a sum."""

    def __init__(self, test, n):
        self.refs = [normalize(r["tgt"]) for r in test]
        self.metrics = {"chrF++": CHRF(word_order=2), "BLEU": BLEU()}
        self.idx = np.random.default_rng(SEED).integers(0, n, (B, n))

    def scores(self, preds):
        """{metric: array of B resampled corpus scores}."""
        hyp = [normalize(p) for p in preds]
        out = {}
        for name, m in self.metrics.items():
            st = np.array(m._extract_corpus_statistics(hyp, [self.refs]))
            out[name] = np.array([m._compute_score_from_stats(st[i].sum(0).tolist()).score for i in self.idx])
        return out


def p_value(a, b):
    """Two-sided paired bootstrap p-value for 'a and b score the same'."""
    d = a - b
    return min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return f"{lo:.1f}–{hi:.1f}"


def span_outcomes(test, preds, lex):
    """One record per gold span: outcome + span properties."""
    out = []
    for r, p in zip(test, preds):
        for d, m in r["pairs"]:
            if normalize(d) == normalize(m):
                continue
            if span_accuracy(p, [(d, m)]):
                o = "correct"
            elif dialect_left(p, [(d, m)]):
                o = "kept"
            else:
                o = "other"
            out.append({"id": r["id"], "d": d, "m": m, "pred": p, "src": r["src"], "tgt": r["tgt"], "outcome": o,
                        "seen": normalize(d) in lex, "multi": len(d.split()) > 1})
    return out


def pct(recs, o):
    return 100 * sum(x["outcome"] == o for x in recs) / len(recs) if recs else 0


def main():
    train, test = load("train"), load("test")
    lex = lexicon(train)
    sy = systems(test)
    boot = Boot(test, len(test))
    for s in sy:
        s["boot"] = boot.scores(s["preds"])
    best = sy[0]

    md = ["# Significance tests and error analysis", "",
          f"Test set: {len(test)} sentences. Paired bootstrap with {B} resamples (seed {SEED}), "
          "same normalization as the main report. p < 0.05 means the difference is unlikely to be resampling noise.", "",
          "## 1. Confidence intervals and comparison with the best system", "",
          f"p-value: two-sided paired bootstrap against **{best['name']}** on chrF++.", "",
          "| # | System | chrF++ | 95% CI | BLEU | 95% CI | Δ chrF++ vs best | p |", "|---" * 8 + "|"]
    for i, s in enumerate(sy, 1):
        delta = s["chrF++"] - best["chrF++"]
        p = "—" if s is best else f"{p_value(best['boot']['chrF++'], s['boot']['chrF++']):.3f}"
        md.append(f"| {i} | {s['name']} | {s['chrF++']} | {ci(s['boot']['chrF++'])} | {s['BLEU']} | "
                  f"{ci(s['boot']['BLEU'])} | {delta:+.2f} | {p} |")

    # systems statistically tied with the best one
    tied = [s["name"] for s in sy[1:] if p_value(best["boot"]["chrF++"], s["boot"]["chrF++"]) >= 0.05]
    md += ["", f"Not significantly different from the best system (p ≥ 0.05): {', '.join(tied) or 'none'}."]

    by = {(s["model"], s["mode"]): s for s in sy}
    md += ["", "## 2. Effect of the prompting modes (per model, chrF++)", "",
           "| Model | zero | rag | Δ rag−zero | p | rag_lex | Δ rag_lex−rag | p |", "|---" * 8 + "|"]
    for model in dict.fromkeys(s["model"] for s in sy if s["mode"] == "zero"):
        z, r, l = (by[(model, m)] for m in ("zero", "rag", "rag_lex"))
        md.append(f"| {model} | {z['chrF++']} | {r['chrF++']} | {r['chrF++'] - z['chrF++']:+.2f} | "
                  f"{p_value(r['boot']['chrF++'], z['boot']['chrF++']):.3f} | {l['chrF++']} | "
                  f"{l['chrF++'] - r['chrF++']:+.2f} | {p_value(l['boot']['chrF++'], r['boot']['chrF++']):.3f} |")

    # pairwise among the top systems
    top = sy[:8]
    md += ["", "## 3. Pairwise p-values among the top 8 (chrF++)", "",
           "| | " + " | ".join(str(i) for i in range(1, len(top) + 1)) + " |", "|---" * (len(top) + 1) + "|"]
    for i, a in enumerate(top, 1):
        cells = ["—" if a is b else f"{p_value(a['boot']['chrF++'], b['boot']['chrF++']):.2f}" for b in top]
        md.append(f"| {i}. {a['name']} | " + " | ".join(cells) + " |")

    # span-level error analysis
    for s in sy:
        s["spans"] = span_outcomes(test, s["preds"], lex)
    n_sp = len(best["spans"])
    n_seen = sum(x["seen"] for x in best["spans"])
    n_multi = sum(x["multi"] for x in best["spans"])
    md += ["", "## 4. Span-level error analysis", "",
           f"{n_sp} gold dialect spans in the test set (spans identical to their MSA are skipped); "
           f"{n_seen} ({100 * n_seen / n_sp:.0f}%) occur in the training lexicon, "
           f"{n_multi} ({100 * n_multi / n_sp:.0f}%) are multi-word.", "",
           "- **correct**: the gold MSA replacement appears in the output",
           "- **kept**: the dialect word/phrase is still in the output",
           "- **other**: changed to something else — a valid synonym or a wrong meaning (manual check of a sample: results/MANUAL_REVIEW.md)", "",
           "| System | correct% | kept% | other% | correct% seen | correct% unseen | correct% 1-word | correct% multi |",
           "|---" * 8 + "|"]
    for s in sy:
        sp = s["spans"]
        seen, unseen = [x for x in sp if x["seen"]], [x for x in sp if not x["seen"]]
        one, multi = [x for x in sp if not x["multi"]], [x for x in sp if x["multi"]]
        md.append(f"| {s['name']} | {pct(sp, 'correct'):.1f} | {pct(sp, 'kept'):.1f} | {pct(sp, 'other'):.1f} | "
                  f"{pct(seen, 'correct'):.1f} | {pct(unseen, 'correct'):.1f} | {pct(one, 'correct'):.1f} | "
                  f"{pct(multi, 'correct'):.1f} |")

    # why rag_lex does not help: hints that do not belong to a gold dialect span
    ret = Retriever(train)
    n_h = n_gold = n_sent = 0
    for r in test:
        h = ret.hints(r["src"])
        gold = {normalize(d) for d, _ in r["pairs"]}
        n_h += len(h)
        n_gold += sum(g in gold for g, _ in h)
        n_sent += bool(h)
    md += ["", "## 5. Dictionary hints (rag_lex)", "",
           f"- Sentences that received at least one hint: {n_sent} / {len(test)} ({100 * n_sent / len(test):.0f}%)",
           f"- Hints given in total: {n_h}; of these {n_gold} ({100 * n_gold / max(n_h, 1):.0f}%) match a gold "
           f"dialect span, the other {n_h - n_gold} point at words that should stay unchanged "
           "(common words that happen to appear in the lexicon) or at partial matches.", "",
           "| Model | Δ correct% seen (rag_lex−rag) | Δ correct% unseen | Δ kept% | Δ chrF++ |", "|---" * 5 + "|"]
    for model in dict.fromkeys(s["model"] for s in sy if s["mode"] == "rag_lex"):
        r, l = by[(model, "rag")], by[(model, "rag_lex")]
        f = lambda s, o, k=None: pct([x for x in s["spans"] if k is None or x["seen"] == k], o)
        md.append(f"| {model} | {f(l, 'correct', True) - f(r, 'correct', True):+.1f} | "
                  f"{f(l, 'correct', False) - f(r, 'correct', False):+.1f} | {f(l, 'kept') - f(r, 'kept'):+.1f} | "
                  f"{l['chrF++'] - r['chrF++']:+.2f} |")

    # hardest spans: missed by most of the top 10 systems
    top10 = sy[:10]
    miss = Counter()
    for s in top10:
        for x in s["spans"]:
            if x["outcome"] != "correct":
                miss[(x["id"], x["d"], x["m"])] += 1
    always = [k for k, v in miss.items() if v == len(top10)]
    md += ["", "## 6. Hardest spans", "",
           f"{len(always)} of {n_sp} spans ({100 * len(always) / n_sp:.0f}%) are missed by all of the top 10 systems. "
           "Most frequent dialect words left unconverted by the best system:", ""]
    kept = Counter(normalize(x["d"]) for x in best["spans"] if x["outcome"] == "kept")
    md.append(", ".join(f"{w} ({c})" for w, c in kept.most_common(25)))

    # sentence-level behaviour
    md += ["", "## 7. Sentence-level behaviour", "",
           "| System | copied input% | length ratio (pred/ref words) |", "|---|---|---|"]
    for s in sy:
        copy = sum(normalize(p) == normalize(r["src"]) for p, r in zip(s["preds"], test))
        ratio = sum(len(p.split()) for p in s["preds"]) / sum(len(r["tgt"].split()) for r in test)
        md.append(f"| {s['name']} | {100 * copy / len(test):.1f} | {ratio:.2f} |")

    text = "\n".join(md) + "\n"
    (RES / "ANALYSIS.md").write_text(text, encoding="utf-8")
    # 'other' spans of the best system, for manual judgement
    with (RES / "best_other_spans.jsonl").open("w", encoding="utf-8") as f:
        for x in best["spans"]:
            if x["outcome"] == "other":
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(text)


if __name__ == "__main__":
    main()
