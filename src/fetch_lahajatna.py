"""Download the word pages of lahajatna.com (Saudi dialect atlas) listed in its sitemap, as plain text.

robots.txt allows /word/ pages; one request per second. Resumable: pages already saved are skipped.
Output: data/external/lahajatna/pages.jsonl  {"url", "text"}

usage: python src/fetch_lahajatna.py
"""
import html
import json
import re
import time
import urllib.request

from data import DATA

OUT = DATA / "external" / "lahajatna" / "pages.jsonl"
UA = {"User-Agent": "Mozilla/5.0 (research dataset builder)"}


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return r.read().decode("utf-8", "replace")


def text(page):
    page = re.sub(r"<script.*?</script>|<style.*?</style>", "", page, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", page))).strip()


def main():
    urls = [u for u in re.findall(r"<loc>([^<]+)</loc>", get("https://www.lahajatna.com/sitemap.xml")) if "/word/" in u]
    done = set()
    if OUT.exists():
        done = {json.loads(l)["url"] for l in OUT.open(encoding="utf-8")}
    print(f"{len(urls)} word pages, {len(done)} already saved", flush=True)
    with OUT.open("a", encoding="utf-8") as f:
        for i, u in enumerate(u for u in urls if u not in done):
            for t in range(3):
                try:
                    f.write(json.dumps({"url": u, "text": text(get(u))}, ensure_ascii=False) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    print("retry", u, e, flush=True)
                    time.sleep(10 * (t + 1))
            if i % 500 == 0:
                print(i, flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
