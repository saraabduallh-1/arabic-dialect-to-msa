"""Check the MSA side of the generated data against open lexicons (PLAN_v2 D7).

Lexicons (licences in data/generated/SOURCES.md):
- Arabic WordNet (OMW omw-arb 1.4, CC BY-SA 3.0), via the `wn` package
- Arabic Wiktionary page titles (CC BY-SA 4.0), data/lexicons/arwiktionary-titles.gz

For every generated row, the MSA equivalent of each marked span (and its listed alternatives)
counts as found when every content word matches a lexicon entry after stripping common
clitics/affixes. Rows not found are flagged for review, never deleted.
Writes data/generated/msa_check.csv (flagged rows, with a Riyadh dictionary link for manual checks).

Get the lexicons first:
  .venv\Scripts\python -m pip install wn
  .venv\Scripts\python -c "import wn; wn.download('omw-arb:1.4')"
  curl -L -o data/lexicons/arwiktionary-titles.gz https://dumps.wikimedia.org/arwiktionary/latest/arwiktionary-latest-all-titles-in-ns0.gz

usage: python src/check_msa.py
"""
import gzip
from urllib.parse import quote

import pandas as pd

from data import DATA, DIALECT, MSA, clean, normalize, spans

LEX = DATA / "lexicons"
GEN = DATA / "generated"
PREFIXES = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك", "س",
            "سي", "ست", "سن", "سا", "وي", "وت", "ي", "ت", "ن", "ا", "لي", "لت", "است", "ات"]  # incl. verb prefixes
SUFFIXES = ["تما", "تم", "تن", "وها", "وه", "وا", "ات", "ون", "ين", "ان", "ها", "هم", "هن", "كم", "نا", "ني", "ه", "ك", "ي", "ا", "و", "ت", "ن"]
STOP = {"من", "في", "علي", "الي", "عن", "مع", "ان", "لا", "لم", "لن", "ما", "قد", "او", "ثم", "هذا", "هذه", "ذلك",
        "التي", "الذي", "كان", "غير", "ليس", "يا", "به", "بها", "له", "لها", "كل", "بعد", "قبل", "عند", "حتي"}


def load_lexicon():
    import wn
    words = {normalize(f) for w in wn.Wordnet("omw-arb:1.4").words() for f in w.forms()}
    with gzip.open(LEX / "arwiktionary-titles.gz", "rt", encoding="utf-8") as f:
        words |= {normalize(t.replace("_", " ")) for t in f}
    return {w for w in words if w}


def forms(word):
    """The word plus versions with common clitics/affixes stripped (keeps at least 2 letters)."""
    out = {word}
    for p in [""] + PREFIXES:
        if not word.startswith(p):
            continue
        w = word[len(p):]
        out.add(w)
        for s in SUFFIXES:
            if w.endswith(s) and len(w) - len(s) >= 2:
                out.add(w[: -len(s)])
                out.add(w[: -len(s)] + "ه")  # taa marbuta (normalized to ه) before a suffix
    return {f for f in out if len(f) >= 2}


def found(phrase, lex):
    words = [w for w in normalize(phrase).split() if w not in STOP and len(w) > 1]
    return all(forms(w) & lex for w in words)  # only function words -> nothing to check


def main():
    lex = load_lexicon()
    df = pd.read_csv(GEN / "all.csv", dtype=str).fillna("")
    flagged, total, ok = [], 0, 0
    for _, r in df.iterrows():
        for k, m in spans(r[MSA]).items():
            total += 1
            if found(m, lex):
                ok += 1
                continue
            alts = [a.strip() for a in r["مقابلات بديلة"].split("/") if a.strip()]
            flagged.append({"Feature 1": r["Feature 1"], "الدفعة": str(r["Feature 1"])[1:3], "رقم الكلمة": k,
                            "العامية": spans(r[DIALECT]).get(k, ""), "المقابل الفصيح": m,
                            "بديل موجود في المعاجم": " / ".join(a for a in alts if found(a, lex)),
                            "الجملة الفصحى": clean(r[MSA]),
                            "رابط معجم الرياض": "https://dictionary.ksaa.gov.sa/result/" + quote(m), "الحكم": ""})
    pd.DataFrame(flagged).to_csv(GEN / "msa_check.csv", index=False, encoding="utf-8-sig")
    print(f"lexicon entries: {len(lex)}")
    print(f"MSA spans: {total}, found in open lexicons: {ok} ({100 * ok / total:.1f}%), "
          f"flagged for review: {len(flagged)} -> data/generated/msa_check.csv")


if __name__ == "__main__":
    main()
