"""Transcripts of SADA (Saudi Audio Dataset for Arabic, SDAIA + Saudi Broadcasting Authority, ~667h of TV speech,
mostly Saudi dialects; CC BY-NC-SA 4.0) from the Hugging Face mirror m6011/sada2022.
Only the text columns are read (parquet column projection), not the audio.
Note: this mirror holds only 6,193 segments (~81k words), not the full corpus.
Output: data/external/sada/text.jsonl {"text", "dialect", "show", "category"}; resumable per file.

usage: python src/fetch_sada.py
"""
import json

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

from data import DATA

OUT = DATA / "external" / "sada"
COLS = ["ProcessedText", "SpeakerDialect", "ShowName", "Category"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done_p = OUT / "done.txt"
    done = set(done_p.read_text().split()) if done_p.exists() else set()
    fs = HfFileSystem()
    files = sorted(fs.glob("datasets/m6011/sada2022/data/*.parquet"))
    for i, path in enumerate(files):
        name = path.split("/")[-1]
        if name in done:
            continue
        for attempt in range(3):
            try:
                t = pq.ParquetFile(fs.open(path)).read(columns=COLS).to_pylist()
                break
            except Exception as e:
                print(name, e, flush=True)
        else:
            continue
        with (OUT / "text.jsonl").open("a", encoding="utf-8") as f:
            for r in t:
                f.write(json.dumps({"text": r["ProcessedText"], "dialect": r["SpeakerDialect"], "show": r["ShowName"],
                                    "category": r["Category"]}, ensure_ascii=False) + "\n")
        with done_p.open("a") as f:
            f.write(name + "\n")
        print(f"{i + 1}/{len(files)} {name}: {len(t)}", flush=True)


if __name__ == "__main__":
    main()
