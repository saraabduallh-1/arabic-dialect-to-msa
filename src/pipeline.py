"""End-to-end tool: a raw Saudi dialect sentence in, an MSA sentence out, every replaced word marked
"suggested" (safe to apply automatically) or "review" (show to a person).

1. find the dialect words: MARBERT BIO tagger (src/tag_dialect.py --save)
2. replace them: ALLaM-7B QLoRA word model (+gen3, chosen on dev), prompt of src/run_word.py
3. tier: suggested = train dictionary form seen >= 2 times with the same MSA in >= 80%  (src/tiers.py thresholds,
   chosen on dev); everything else = review. (The gemma agreement of tiers.py is left out: a second 27B model
   is too heavy for a tool.)

usage:
  python src/pipeline.py --text "وش تبي تسوي الحين"
  python src/pipeline.py --eval dev        # end-to-end score against the gold marks (settings fixed in advance)
Writes results/pipeline/<split>.jsonl and results/pipeline/<split>.md for --eval.
"""
import argparse
import json
import re
from collections import Counter

import torch
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import (AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer,
                          BitsAndBytesConfig)

import run_word
from data import ROOT, clean, load, normalize
from tag_dialect import spans_of, tokens
from tiers import train_dict

BASE = "ALLaM-AI/ALLaM-7B-Instruct-preview"
ADAPTER = ROOT / "results" / "checkpoints" / "ALLaM-7B-Instruct-preview-qlora-word+gen3"
TAGGER = ROOT / "results" / "checkpoints" / "tagger_marbert"
OUT = ROOT / "results" / "pipeline"


class Pipeline:
    def __init__(self, min_n=2, min_c=0.8):
        self.ttok = AutoTokenizer.from_pretrained(TAGGER)
        self.tagger = AutoModelForTokenClassification.from_pretrained(TAGGER).cuda().eval().half()
        self.tok = AutoTokenizer.from_pretrained(BASE)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        base = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                                    dtype=torch.bfloat16)
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.llm = get_peft_model(base, lora)
        set_peft_model_state_dict(self.llm, load_file(ADAPTER / "adapter_model.safetensors"))
        self.llm.eval()
        self.dic, self.min_n, self.min_c = train_dict(), min_n, min_c

    def tag(self, words):
        enc = self.ttok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = self.tagger(**enc).logits[0]
        lab, prev = [0] * len(words), None
        for i, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                lab[wid] = int(logits[i].argmax())
            prev = wid
        return sorted(spans_of(lab))

    def replace(self, items):
        """items: [(words, spans)] -> [{k: msa}] with one batched generation."""
        prompts, idx = [], []
        for j, (w, sp) in enumerate(items):
            if not sp:
                continue
            marked, pos = [], 0
            for k, (s, e) in enumerate(sp, 1):
                marked += w[pos:s] + [f"({' '.join(w[s:e])}){k}"]
                pos = e
            marked += w[pos:]
            msgs = [{"role": "system", "content": run_word.SYSTEM}, {"role": "user", "content": "الجملة: " + " ".join(marked)}]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            idx.append(j)
        out = [{} for _ in items]
        for b in range(0, len(prompts), 16):
            enc = self.tok(prompts[b:b + 16], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with torch.no_grad():
                gen = self.llm.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=self.tok.pad_token_id)
            for j, t in zip(idx[b:b + 16], self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
                out[j] = run_word.parse(t)
        return out

    def tier(self, d, a):
        c = self.dic.get(normalize(d))
        if c:
            top, k = c.most_common(1)[0]
            if sum(c.values()) >= self.min_n and k / sum(c.values()) >= self.min_c:
                return "suggested", top
        if not a or normalize(a) == normalize(d):  # no-copy rule
            return "review", c.most_common(1)[0][0] if c else ""
        return "review", a

    def __call__(self, sentences):
        items = [(clean(s).split(), None) for s in sentences]
        items = [(w, self.tag(w)) for w, _ in items]
        answers = self.replace(items)
        res = []
        for (w, sp), ans in zip(items, answers):
            words, reps, pos = [], [], 0
            for k, (s, e) in enumerate(sp, 1):
                d = " ".join(w[s:e])
                t, m = self.tier(d, ans.get(k, ""))
                reps.append({"span": [s, e], "dialect": d, "msa": m, "tier": t})
                words += w[pos:s] + [m or d]
                pos = e
            res.append({"src": " ".join(w), "msa": " ".join(words + w[pos:]), "replacements": reps})
        return res


def evaluate(p, split):
    rows = load(split, "sentences_v2")
    out = p([r["src"] for r in rows])
    st = Counter()
    for r, o in zip(rows, out):
        gold_tok = tokens(r["src_marked"])
        gold = spans_of([l for _, l in gold_tok])
        words = [t for t, _ in gold_tok]
        pred = {tuple(x["span"]): x for x in o["replacements"]}
        if words != o["src"].split():  # tokenisation differs: count as missed
            st["tok_mismatch"] += 1
            pred = {}
        st["gold"] += len(gold); st["pred"] += len(pred); st["hit"] += len(gold & set(pred))
        g2m = {}
        for (s, e), (d, m) in zip(sorted(gold), r["pairs"]):
            g2m[(s, e)] = m
        for sp, m in g2m.items():
            d = " ".join(words[sp[0]:sp[1]])
            if normalize(d) == normalize(m):
                continue
            st["need"] += 1
            x = pred.get(sp)
            if x:
                ok = normalize(x["msa"]) == normalize(m)
                st[x["tier"]] += 1; st[x["tier"] + "_ok"] += ok; st["e2e_ok"] += ok
        o["id"] = r["id"]; o["ref"] = r["tgt"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    P, R = st["hit"] / max(st["pred"], 1), st["hit"] / max(st["gold"], 1)
    pct = lambda a, b: f"{100 * a / max(b, 1):.1f}"
    lines = [f"# End-to-end pipeline on {split} (tagger + ALLaM +gen3 + dictionary tier)", "",
             f"Sentences: {len(rows)}; tokenisation mismatches: {st['tok_mismatch']}.", "",
             "| step | value |", "|---|---|",
             f"| dialect span detection P / R / F1 (exact boundaries) | {100*P:.1f} / {100*R:.1f} / {200*P*R/max(P+R,1e-9):.1f} |",
             f"| spans that need a change, found by the tagger | {pct(st['suggested'] + st['review'], st['need'])}% of {st['need']} |",
             f"| end-to-end exact (found and replaced like the reference) | {pct(st['e2e_ok'], st['need'])}% |",
             f"| suggested tier: share of found / exact | {pct(st['suggested'], st['suggested'] + st['review'])}% / {pct(st['suggested_ok'], st['suggested'])}% |",
             f"| review tier: exact | {pct(st['review_ok'], st['review'])}% |", "",
             "Exact match against one reference underestimates by ~37 points (human review of 200 dev answers)."]
    (OUT / f"{split}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*")
    ap.add_argument("--eval", nargs="*", help="splits, e.g. dev test")
    a = ap.parse_args()
    p = Pipeline()
    for s in a.eval or []:
        evaluate(p, s)
    if a.text:
        for o in p(a.text):
            print(json.dumps(o, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
