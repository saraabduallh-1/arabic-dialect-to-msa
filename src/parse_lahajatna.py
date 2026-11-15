"""Parse data/external/lahajatna/pages.jsonl into data/external/lahajatna/entries.jsonl.

One entry per page: word (vocalized headword), plain (no diacritics), dialect, root, meaning confidence,
meanings (list), examples (list of {"ex", "expl"}: dialect sentence and the site's MSA explanation).

usage: python src/parse_lahajatna.py
"""
import json
import re

from data import DATA

DIR = DATA / "external" / "lahajatna"
DIAC = re.compile(r"[ً-ْٰـ]")


def parse(url, t):
    m = re.match(r"(.+?) — معناها في (.+?) — لهجاتنا", t)
    if not m:
        return None
    word, dialect = m.group(1).strip(), m.group(2).strip()
    root = re.search(r"الجذر (.+?) · ", t)
    conf = re.search(r"ثقة المعنى (\d+)", t)
    meanings = re.findall(r"المعنى: «المعنى \d+: (.+?)»", t) or re.findall(r"المعنى: «(?!المعاني عمومًا)(.+?)»", t)
    examples = []
    for seg in t.split("بلاغ عن خطأ في المثال المثال: «")[1:]:
        q, _, rest = seg.partition("» ما الخطأ؟")
        examples.append({"ex": q.strip(), "expl": ""})
    # explanations precede each report form: "<ex> ... معنى المثال + أي: <expl> بلاغ عن خطأ في المثال"
    expls = re.findall(r"معنى المثال \+ أي: (.{1,400}?) بلاغ عن خطأ في المثال", t)
    if len(expls) == len(examples):
        for e, x in zip(examples, expls):
            e["expl"] = x.strip()
    return {"url": url, "word": word, "plain": DIAC.sub("", word), "dialect": dialect,
            "root": root.group(1).replace(" ", "") if root else "", "conf": int(conf.group(1)) if conf else None,
            "meanings": meanings, "examples": examples}


def main():
    out, bad = [], 0
    for l in (DIR / "pages.jsonl").open(encoding="utf-8"):
        p = json.loads(l)
        e = parse(p["url"], p["text"])
        if e:
            out.append(e)
        else:
            bad += 1
    with (DIR / "entries.jsonl").open("w", encoding="utf-8") as f:
        for e in out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"{len(out)} entries, {bad} unparsed, {sum(bool(e['meanings']) for e in out)} with meanings, "
          f"{sum(len(e['examples']) for e in out)} examples")


if __name__ == "__main__":
    main()
