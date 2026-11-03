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
            "span_ids": [k for k in sorted(src_sp) if k in tgt_sp],
            "src_marked": str(r[DIALECT]),  # with the (word)N markers, for the word-level task
        })
    refs = DATA / dataset / f"{split}_refs.csv"
    if refs.exists():
        add_refs(rows, refs)
    return rows


def add_refs(rows, path):
    """Attach approved extra references (PLAN_v2 I2) from e.g. data/sentences/test_refs.csv.

    Columns: Feature 1, مرجع بديل 1..3 (alternative MSA sentences),
    بدائل الكلمات ("1: بديل / بديل ; 2: بديل" = accepted alternatives per span number).
    Adds r["refs"] (extra sentences) and r["alts"] (one list of accepted MSA forms per pair).
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    by_id = {int(r["Feature 1"]): r for _, r in df.iterrows()}
    for r in rows:
        x = by_id.get(r["id"])
        r["refs"] = [] if x is None else [clean(x[c]) for c in df.columns if c.startswith("مرجع بديل") and x[c].strip()]
        alts = {}
        if x is not None:
            for part in x.get("بدائل الكلمات", "").split(";"):
                if ":" in part:
                    n, words = part.split(":", 1)
                    alts[int(n.strip())] = [w.strip() for w in words.split("/") if w.strip()]
        r["alts"] = [[m] + alts.get(k, []) for k, (_, m) in zip(r["span_ids"], r["pairs"])]


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
