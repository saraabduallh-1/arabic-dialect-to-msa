"""Saleem's specialization layer glossary (Siwar Data & AI dictionary, saleem-specialization-layer-).

A foreign word that sounds like one of its English terms is flagged as a possible glossary term (field "glossary" of
each replacement), so the specialization layer can apply its official term; when Gemma's own answer already equals
that term, the two layers agree and the answer is "suggested".
Tried and dropped: showing the term to Gemma as a hint. Sound alone gives false hits (تشيز/cheese, شيفت/safety) and
Gemma used the hinted term out of context 9 times on the foreign test (exact 34.5% -> 34.2%).

A transliterated word is matched to an English term by a consonant skeleton: vowels and weak letters dropped,
letters with the same sound in Arabic transliteration merged (p/b, v/f, g/j/ق/غ/ج, s/z/ص/ث/ش, ...).
A 2-letter skeleton matches only a single-word term that is the only one with that skeleton (e.g. داتا -> Data);
Latin words also match by exact spelling.

usage: python src/domain_terms.py مودل داتا "ماشين ليرنينق" model
"""
import json
import os
import re
import sys
from pathlib import Path

# Saleem's specialization glossary. A copy ships in data/ (from saleem-specialization-layer-,
# data/processed/data_ai/terminology_base.json); SPEC_TERMS points elsewhere.
_SPEC_COPY = Path(__file__).resolve().parents[1] / "data" / "terminology_base.json"
_SPEC_LOCAL = Path.home() / "Documents" / "saleem-specialization-layer-" / "data" / "processed" / "data_ai" / "terminology_base.json"
SPEC_TERMS = Path(os.environ.get("SPEC_TERMS", _SPEC_COPY if _SPEC_COPY.exists() else _SPEC_LOCAL))

EN = [("ph", "f"), ("ck", "k"), ("th", "t"), ("sh", "s"), ("ch", "s"), ("x", "ks"), ("qu", "k")]
EN_MAP = {"b": "B", "p": "B", "f": "F", "v": "F", "t": "T", "d": "D", "s": "S", "z": "S", "c": "K", "k": "K", "q": "K",
          "g": "G", "j": "G", "h": "H", "l": "L", "m": "M", "n": "N", "r": "R"}
AR_MAP = {"ب": "B", "پ": "B", "ف": "F", "ڤ": "F", "ت": "T", "ط": "T", "د": "D", "ض": "D", "ذ": "D", "س": "S", "ص": "S",
          "ز": "S", "ث": "T", "ش": "S", "ك": "K", "خ": "K", "ق": "G", "غ": "G", "ج": "G", "گ": "G", "ه": "H", "ح": "H",
          "ل": "L", "م": "M", "ن": "N", "ر": "R"}


def skeleton(text):
    out = []
    for w in str(text).split():
        if re.search(r"[A-Za-z]", w):
            w = w.lower()
            for a, b in EN:
                w = w.replace(a, b)
            out += [EN_MAP[c] for c in w if c in EN_MAP]
        else:
            w = re.sub(r"[^ء-يپڤگ]", "", w)
            if w.startswith("ال") and len(w) > 4:
                w = w[2:]
            out += [AR_MAP[c] for c in w if c in AR_MAP]
    return re.sub(r"(.)\1+", r"\1", "".join(out))


class DomainTerms:
    def __init__(self, path=SPEC_TERMS, max_words=3):
        self.by_skel, self.by_latin, short = {}, {}, {}
        if not Path(path).exists():
            return
        for x in json.load(open(path, encoding="utf-8")):
            ar = x.get("canonical_term") or x.get("term")
            en = (x.get("english_term") or "").strip()
            for e in list(x.get("abbreviations") or []):  # abbreviations: exact Latin spelling only
                self.by_latin.setdefault(e.strip().lower(), (en, ar))
            for e in [en]:
                e = re.sub(r"\(.*?\)", "", e).strip()
                if not e or len(e.split()) > max_words:
                    continue
                self.by_latin.setdefault(e.lower(), (en, ar))
                k = skeleton(e)
                if len(k) >= 3:
                    self.by_skel.setdefault(k, (en, ar))
                elif len(k) == 2 and len(e.split()) == 1:
                    short.setdefault(k, set()).add((en, ar))
        for k, v in short.items():
            if len(v) == 1 and k not in self.by_skel:
                self.by_skel[k] = next(iter(v))

    def __len__(self):
        return len(self.by_latin)

    def lookup(self, word):
        """(English term, official Arabic term) for a foreign word that may be a glossary term, or None."""
        w = re.sub(r"[^\w\s-]", "", word).strip()
        if re.search(r"[A-Za-z]", w) and w.lower() in self.by_latin:
            return self.by_latin[w.lower()]
        k = skeleton(w)
        return self.by_skel.get(k) if len(k) >= 2 else None


if __name__ == "__main__":
    t = DomainTerms()
    print(len(t), "glossary entries")
    for w in sys.argv[1:]:
        print(w, "->", t.lookup(w), f"[{skeleton(w)}]")
