"""For the most frequent tweet-mined dialect words, ask a local model whether the word is Saudi/Gulf dialect and
for a short MSA equivalent, using two real tweets as context. Output is a proposal for human verification only.

Input : data/external/tweet_dialect/candidates.jsonl (src/mine_tweets.py)
Output: data/external/tweet_dialect/proposals.jsonl  {form, saudi, dialect, msa, meaning, lahajatna, examples}
Resumable. Local model only (Ollama), no paid API.

usage: python src/propose_equivalents.py --model gemma3:27b --single 2000 --phrases 300
"""
import argparse
import json
import re

from data import DATA
from run_llm import chat

D = DATA / "external" / "tweet_dialect"
SYSTEM = (
    "أنت خبير في اللهجات السعودية والخليجية وفي العربية الفصحى. ستُعطى كلمة أو عبارة مأخوذة من تغريدات، مع أمثلة من سياقها. "
    "أجب بكائن JSON فقط بالمفاتيح: "
    '"saudi": هل تُستعمل في اللهجات السعودية أو الخليجية بهذا المعنى (true/false)، '
    '"dialect": هل هي عامية وليست فصحى بالمعنى المستعمل في الأمثلة (true/false)، '
    '"msa": أقصر مقابل فصيح يصلح أن يحل محلها في الأمثلة (كلمة أو كلمتان، بلا شرح)، '
    '"meaning": معناها في سطر قصير. '
    "إذا كانت مصرية أو شامية فقط فاجعل saudi=false. إذا كانت فصحى أصلًا أو اسم علم أو خطأ إملائي فاجعل dialect=false."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:27b")
    ap.add_argument("--single", type=int, default=2000)
    ap.add_argument("--phrases", type=int, default=300)
    a = ap.parse_args()
    C = [json.loads(l) for l in (D / "candidates.jsonl").open(encoding="utf-8")]
    pick = [c for c in C if " " not in c["form"]][: a.single] + [c for c in C if " " in c["form"]][: a.phrases]
    out = D / "proposals.jsonl"
    done = {json.loads(l)["form"] for l in out.open(encoding="utf-8")} if out.exists() else set()
    with out.open("a", encoding="utf-8") as f:
        for i, c in enumerate(pick):
            if c["form"] in done:
                continue
            user = f"الكلمة: {c['form']}\nأمثلة:\n" + "\n".join(f"- {e}" for e in c["examples"][:2])
            raw = chat(a.model, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            m = re.search(r"\{.*\}", raw, re.S)
            try:
                p = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:  # unescaped quotes inside "meaning": read the keys one by one
                p = {k: re.search(rf'"{k}"\s*:\s*(true|false)', raw).group(1) == "true"
                     for k in ("saudi", "dialect") if re.search(rf'"{k}"\s*:\s*(true|false)', raw)}
                ms = re.search(r'"msa"\s*:\s*"([^"]*)"', raw)
                mn = re.search(r'"meaning"\s*:\s*"(.*)"\s*\}', raw, re.S)
                if ms:
                    p["msa"] = ms.group(1)
                if mn:
                    p["meaning"] = mn.group(1).strip()
            f.write(json.dumps({"form": c["form"], "tagged": c["tagged"], "saudi": p.get("saudi"), "dialect": p.get("dialect"),
                                "msa": p.get("msa"), "meaning": p.get("meaning"), "lahajatna": c["lahajatna"],
                                "examples": c["examples"][:2], "raw": raw if not p else None}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(i, flush=True)


if __name__ == "__main__":
    main()
