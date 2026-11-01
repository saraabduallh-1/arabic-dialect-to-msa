import sacrebleu

from data import normalize


def span_accuracy(pred, pairs, alts=None):
    """Share of gold MSA replacements that appear in the prediction.
    With alts (one list of accepted forms per pair), any accepted form counts."""
    if not pairs:
        return None
    p = normalize(pred)  # substring match tolerates attached prefixes such as و/ب/ل
    alts = alts or [[m] for _, m in pairs]
    return sum(any(normalize(a) in p for a in forms) for forms in alts) / len(pairs)


def dialect_left(pred, pairs):
    """Share of dialect words that were left untouched (and differ from the MSA)."""
    p = f" {normalize(pred)} "
    cand = [(d, m) for d, m in pairs if normalize(d) != normalize(m)]
    if not cand:
        return None
    return sum(f" {normalize(d)} " in p for d, _ in cand) / len(cand)


def multi_ref(rows, hyp):
    """Scores against the main reference plus the approved extra ones (PLAN_v2 I2)."""
    n = max(len(r.get("refs", [])) for r in rows)
    # sacrebleu needs the same number of reference streams for every sentence: pad with the main reference
    streams = [[normalize(([r["tgt"]] + r.get("refs", []) + [r["tgt"]] * n)[i]) for r in rows] for i in range(n + 1)]
    span = [s for r, h in zip(rows, hyp) if (s := span_accuracy(h, r["pairs"], r.get("alts"))) is not None]
    return {
        "BLEU_multi": round(sacrebleu.corpus_bleu(hyp, streams).score, 2),
        "chrF++_multi": round(sacrebleu.corpus_chrf(hyp, streams, word_order=2).score, 2),
        "SpanAcc%_multi": round(100 * sum(span) / len(span), 2),
    }


def score(rows, preds):
    hyp = [normalize(p) for p in preds]
    ref = [normalize(r["tgt"]) for r in rows]
    span = [s for r, p in zip(rows, preds) if (s := span_accuracy(p, r["pairs"])) is not None]
    left = [s for r, p in zip(rows, preds) if (s := dialect_left(p, r["pairs"])) is not None]
    res = {
        "BLEU": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
        "chrF++": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
        "ExactMatch%": round(100 * sum(h == r for h, r in zip(hyp, ref)) / len(ref), 2),
        "SpanAcc%": round(100 * sum(span) / len(span), 2),
        "DialectLeft%": round(100 * sum(left) / len(left), 2),
        "n": len(rows),
    }
    # single-reference numbers above stay comparable with earlier results; multi-reference ones are extra
    if any(r.get("refs") or any(len(a) > 1 for a in r.get("alts", [])) for r in rows):
        res.update(multi_ref(rows, hyp))
    return res
