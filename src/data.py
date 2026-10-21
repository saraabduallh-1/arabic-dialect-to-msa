"""Load the dialect -> MSA datasets and extract the annotated spans.

Sentences mark the dialect words like "(word)1"; the MSA side marks the
matching replacement with the same number.
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

DIALECT, MSA = "السياق العامي", "السياق الفصيح"
SPAN_RE = re.compile(r"\(([^()]*)\)\s*(\d+)")


def clean(text):
    """Drop the span markers: '(مدري)1 أهو' -> 'مدري أهو'."""
    text = SPAN_RE.sub(r"\1", str(text))
    text = text.replace("(", "").replace(")", "")
    return re.sub(r"\s+", " ", text).strip()


def spans(text):
    """{index: span_text} for the marked spans of a sentence."""
    return {int(n): s.strip() for s, n in SPAN_RE.findall(str(text))}


def load(split, dataset="sentences"):
    df = pd.read_csv(DATA / dataset / f"{split}.csv")
    rows = []
    for _, r in df.iterrows():
        src_sp, tgt_sp = spans(r[DIALECT]), spans(r[MSA])
        rows.append({
            "id": int(r["Feature 1"]),
            "src": clean(r[DIALECT]),
            "tgt": clean(r[MSA]),
            # aligned (dialect, msa) word pairs
            "pairs": [(src_sp[k], tgt_sp[k]) for k in sorted(src_sp) if k in tgt_sp],
        })
    return rows


def lexicon(train_rows):
    """Dialect word/phrase -> list of MSA equivalents, from the training split only."""
    lex = {}
    for r in train_rows:
        for d, m in r["pairs"]:
            lex.setdefault(normalize(d), [])
            if m not in lex[normalize(d)]:
                lex[normalize(d)].append(m)
    return lex


_DIACRITICS = re.compile(r"[ؗ-ًؚ-ْٰـ]")


def normalize(text):
    """Orthographic normalization used for matching / scoring."""
    text = _DIACRITICS.sub("", str(text))
    text = re.sub("[إأآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()
