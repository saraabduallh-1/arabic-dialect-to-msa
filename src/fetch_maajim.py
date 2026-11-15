"""Look up every lahajatna headword in the classical/MSA dictionaries of maajim.com.

maajim falls back to the root when the exact word is missing, so the page text is kept and
the decision (is the dialect word itself fusha with the same meaning?) is made later (filter_lahajatna.py).
One request per second, resumable.
Output: data/external/lahajatna/maajim.jsonl  {"word", "text"}  (text = the definitions part of the page)

usage: python src/fetch_maajim.py
"""
import json
import time
import urllib.parse

from data import DATA
from fetch_lahajatna import get, text

DIR = DATA / "external" / "lahajatna"
OUT = DIR / "maajim.jsonl"


def main():
    words = sorted({json.loads(l)["plain"] for l in (DIR / "entries.jsonl").open(encoding="utf-8")})
    done = {json.loads(l)["word"] for l in OUT.open(encoding="utf-8")} if OUT.exists() else set()
    print(f"{len(words)} words, {len(done)} done", flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        for i, w in enumerate(w for w in words if w not in done):
            for t in range(3):
                try:
                    s = text(get("https://www.maajim.com/dictionary/" + urllib.parse.quote(w)))
                    s = s.split("اسأل المساعد الذكي", 1)[-1][:6000]
                    f.write(json.dumps({"word": w, "text": s}, ensure_ascii=False) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    print("retry", w, e, flush=True)
                    time.sleep(15 * (t + 1))
            if i % 500 == 0:
                print(i, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
