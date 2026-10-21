import sacrebleu

from data import normalize


def span_accuracy(pred, pairs):
    """Share of gold MSA replacements that appear in the prediction."""
    if not pairs:
        return None
    p = normalize(pred)  # substring match tolerates attached prefixes such as و/ب/ل
    return sum(normalize(m) in p for _, m in pairs) / len(pairs)


def dialect_left(pred, pairs):
    """Share of dialect words that were left untouched (and differ from the MSA)."""
    p = f" {normalize(pred)} "
    cand = [(d, m) for d, m in pairs if normalize(d) != normalize(m)]
    if not cand:
        return None
    return sum(f" {normalize(d)} " in p for d, _ in cand) / len(cand)


def score(rows, preds):
    hyp = [normalize(p) for p in preds]
    ref = [normalize(r["tgt"]) for r in rows]
    span = [s for r, p in zip(rows, preds) if (s := span_accuracy(p, r["pairs"])) is not None]
    left = [s for r, p in zip(rows, preds) if (s := dialect_left(p, r["pairs"])) is not None]
    return {
        "BLEU": round(sacrebleu.corpus_bleu(hyp, [ref]).score, 2),
        "chrF++": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
        "ExactMatch%": round(100 * sum(h == r for h, r in zip(hyp, ref)) / len(ref), 2),
        "SpanAcc%": round(100 * sum(span) / len(span), 2),
        "DialectLeft%": round(100 * sum(left) / len(left), 2),
        "n": len(rows),
    }
