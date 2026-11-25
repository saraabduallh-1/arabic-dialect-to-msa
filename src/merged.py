"""The merged layer: foreign words and Saudi dialect words -> MSA in one pass, the rest of the sentence untouched.

1. find the words: one MARBERTv2 tagger, two kinds of spans (dialect / foreign) (src/tag_merged.py);
   any word written in Latin letters is a foreign span.
2. the foreign model (m2m100 fine-tune of the other project, used unchanged) rewrites the whole sentence once;
   difflib aligns its output to the input and only the output words aligned to a tagged span are used.
3. foreign span  -> the foreign model's words.
   dialect span  -> the dictionary, then gemma4:12b (word prompt of src/run_word.py with retrieved examples);
                    the foreign model's words for the same span are a second opinion.
4. tiers: "suggested" = a confident dictionary entry, or two independent sources give the same answer;
   everything else = "review".  Words the foreign model changed outside any span are listed, never applied.

Variants for dialect spans (chosen on dev, then test once with --final):
   gemma      Gemma's answer; the foreign model only raises the tier when it agrees
   m2m_first  the foreign model's answer when it changed the span, Gemma's otherwise (the "only what it missed" design)

usage:
  python src/merged.py --text "ابي اموديفاي العقد بكرة"
  python src/merged.py --eval dev                 # dialect dev + foreign dev
  python src/merged.py --eval test --final        # once, settings fixed on dev
"""
import argparse
import difflib
import gc
import hashlib
import json
import re
from collections import Counter

import requests
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoTokenizer

import run_word
from data import ROOT, clean, load, normalize
from foreign_data import MODEL_DIR, OUT, foreign_dict, load_splits, load_tokenizer
from run_llm import Retriever, chat
from tag_dialect import tokens
from tag_merged import typed_spans
from tiers import train_dict

TAGGER = ROOT / "results" / "checkpoints" / "tagger_merged"
GEMMA = "gemma4:12b"
LATIN = re.compile(r"[A-Za-z]")
CACHE = OUT / "gemma_cache.jsonl"


def aligned(src, hyp, a, b):
    """Output words aligned to source words a..b (None when the model left them as they were)."""
    sm = difflib.SequenceMatcher(a=[normalize(w) for w in src], b=[normalize(w) for w in hyp], autojunk=False)
    got, changed = [], False
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if i2 <= a or i1 >= b:
            continue
        if op == "equal":
            lo, hi = max(i1, a), min(i2, b)
            got += hyp[j1 + lo - i1:j1 + hi - i1]
        else:
            got += hyp[j1:j2]
            changed = True
    return " ".join(got) if changed else None


def outside_changes(src, hyp, spans):
    """Source word ranges the model changed that touch no tagged span (listed for review, not applied)."""
    sm = difflib.SequenceMatcher(a=[normalize(w) for w in src], b=[normalize(w) for w in hyp], autojunk=False)
    inside = {i for s, e in spans for i in range(s, e)}
    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op in ("replace", "delete") and not inside & set(range(i1, i2)):
            out.append({"span": [i1, i2], "word": " ".join(src[i1:i2]), "model": " ".join(hyp[j1:j2])})
    return out


class Layer:
    def __init__(self, mode="rag", k=8, min_n=2, min_c=0.8, variant="gemma", device="cpu", gemma=GEMMA):
        self.mode, self.k, self.min_n, self.min_c, self.variant, self.dev, self.gemma = mode, k, min_n, min_c, variant, device, gemma
        self.ttok = AutoTokenizer.from_pretrained(TAGGER)
        self.tagger = AutoModelForTokenClassification.from_pretrained(TAGGER).to(device).eval()
        self.mtok = load_tokenizer()
        self.m2m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device).eval()
        self.ddic = train_dict()
        self.fdic = foreign_dict(load_splits()["train"])
        train = load("train", "sentences_v2") + load("verified", "generated")  # +gen3, as the best word systems
        self.ret = Retriever(train)
        self.cache = {}
        if CACHE.exists():
            for l in CACHE.open(encoding="utf-8"):
                d = json.loads(l)
                self.cache[d["key"]] = d["raw"]

    def to_gpu(self):
        """Unload Gemma from Ollama and put the small models back on the GPU."""
        if not torch.cuda.is_available():
            return
        try:
            requests.post("http://localhost:11434/api/generate", json={"model": self.gemma, "keep_alive": 0}, timeout=60)
        except requests.RequestException:
            pass
        self.tagger.to("cuda"); self.m2m.to("cuda"); self.dev = "cuda"

    def free_gpu(self):
        """Move the small models off the GPU so Ollama has the whole card for Gemma."""
        self.tagger.to("cpu"); self.m2m.to("cpu"); self.dev = "cpu"
        gc.collect(); torch.cuda.empty_cache()

    # --- step 1: tag
    def tag(self, words):
        enc = self.ttok(words, is_split_into_words=True, truncation=True, max_length=128, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            logits = self.tagger(**enc).logits[0]
        lab, prev = [0] * len(words), None
        for i, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                lab[wid] = int(logits[i].argmax())
            prev = wid
        for i, w in enumerate(words):  # Latin letters -> foreign
            if LATIN.search(w):
                lab[i] = 4 if i and lab[i - 1] in (3, 4) else 3
        d, f = typed_spans(lab)
        return sorted([(s, e, "dialect") for s, e in d] + [(s, e, "foreign") for s, e in f])

    # --- step 2: the foreign model, unchanged, on the whole sentence
    def translate(self, sents, bs=16):
        out = []
        for b in range(0, len(sents), bs):
            enc = self.mtok(sents[b:b + bs], return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.dev)
            with torch.no_grad():
                gen = self.m2m.generate(**enc, max_new_tokens=256)
            # the language tag (__ar__) is not a special token in transformers 4.x: drop it by hand
            out += [re.sub(r"__[a-z]{2,3}__", " ", t).strip() for t in self.mtok.batch_decode(gen, skip_special_tokens=True)]
        return out

    # --- step 3: Gemma for the dialect spans
    def ask_gemma(self, words, dspans):
        marked, pos = [], 0
        for n, (s, e) in enumerate(dspans, 1):
            marked += words[pos:s] + [f"({' '.join(words[s:e])}){n}"]
            pos = e
        marked += words[pos:]
        r = {"src": " ".join(words), "src_marked": " ".join(marked), "pairs": [(" ".join(words[s:e]), "") for s, e in dspans]}
        msgs = run_word.build(r, self.mode, self.ret, self.k)
        key = hashlib.sha1(json.dumps([self.gemma, msgs], ensure_ascii=False).encode()).hexdigest()
        if key not in self.cache:
            self.cache[key] = chat(self.gemma, msgs)
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "raw": self.cache[key]}, ensure_ascii=False) + "\n")
        p = run_word.parse(self.cache[key])
        return [p.get(n, "") for n in range(1, len(dspans) + 1)]

    def confident(self, dic, w):
        c = dic.get(normalize(w))
        if c:
            top, k = c.most_common(1)[0]
            if sum(c.values()) >= self.min_n and k / sum(c.values()) >= self.min_c:
                return top
        return None

    # --- step 4: choose and tier
    def decide(self, kind, w, m, g):
        """w: the span; m: the foreign model's words (None = unchanged); g: Gemma's answer (dialect only)."""
        copy = lambda x: not x or normalize(x) == normalize(w)
        dic = self.ddic if kind == "dialect" else self.fdic
        top = self.confident(dic, w)
        if top:
            return top, "suggested", "dictionary"
        if kind == "foreign":
            c = dic.get(normalize(w))
            if not copy(m) and c and normalize(m) == normalize(c.most_common(1)[0][0]):
                return m, "suggested", "foreign_model+dictionary"
            if not copy(m):
                return m, "review", "foreign_model"
            return (c.most_common(1)[0][0], "review", "dictionary") if c else (w, "review", "none")
        if not copy(m) and not copy(g) and normalize(m) == normalize(g):
            return g, "suggested", "gemma+foreign_model"
        c = dic.get(normalize(w))
        if not copy(g) and c and normalize(g) == normalize(c.most_common(1)[0][0]):  # chosen on dev (28.5% at 76.9%)
            return g, "suggested", "gemma+dictionary"
        order = [(m, "foreign_model"), (g, "gemma")] if self.variant == "m2m_first" else [(g, "gemma"), (m, "foreign_model")]
        for x, src in order:
            if not copy(x):
                return x, "review", src
        c = dic.get(normalize(w))
        return (c.most_common(1)[0][0], "review", "dictionary") if c else (w, "review", "none")

    def run(self, sentences, free=False):
        if free:
            self.to_gpu()
        words = [clean(s).split() for s in sentences]
        spans = [self.tag(w) for w in words]
        hyps = [h.split() for h in self.translate([" ".join(w) for w in words])]
        if free:
            self.free_gpu()
        res = []
        for w, sp, h in zip(words, spans, hyps):
            dsp = [(s, e) for s, e, t in sp if t == "dialect"]
            gans = dict(zip(dsp, self.ask_gemma(w, dsp))) if dsp else {}
            reps, out, pos = [], [], 0
            for s, e, t in sp:
                span = " ".join(w[s:e])
                m = aligned(w, h, s, e)
                g = gans.get((s, e))
                msa, tier, src = self.decide(t, span, m, g)
                reps.append({"span": [s, e], "type": t, "word": span, "msa": msa, "tier": tier, "source": src,
                             "foreign_model": m, "gemma": g})
                out += w[pos:s] + [msa]
                pos = e
            res.append({"src": " ".join(w), "msa": " ".join(out + w[pos:]), "replacements": reps,
                        "foreign_model_raw": " ".join(h),
                        "not_applied": outside_changes(w, h, [(s, e) for s, e, _ in sp])})
        return res


# ---------------- evaluation
def span_stats(st, kind, gold, pred, need):
    """gold: {(s,e): msa}; pred: {(s,e): replacement dict}; need: spans whose msa differs from the word."""
    st[kind + "_gold"] += len(gold)
    st[kind + "_pred"] += len(pred)
    st[kind + "_hit"] += len(set(gold) & set(pred))
    for sp in need:
        st[kind + "_need"] += 1
        x = pred.get(sp)
        if not x:
            continue
        ok = normalize(x["msa"]) == normalize(gold[sp])
        st[f"{kind}_{x['tier']}"] += 1; st[f"{kind}_{x['tier']}_ok"] += ok; st[kind + "_ok"] += ok
        for alt, val in (("foreign_model", x["foreign_model"]), ("gemma", x["gemma"])):
            if val is not None and kind == "dialect":
                st[f"{kind}_{alt}_given"] += 1
                st[f"{kind}_{alt}_changed"] += bool(val) and normalize(val) != normalize(x["word"])
                st[f"{kind}_{alt}_ok"] += normalize(val) == normalize(gold[sp])


def eval_dialect(layer, split, variants):
    rows = load(split, "sentences_v2")
    sents = [" ".join(t for t, _ in tokens(r["src_marked"])) for r in rows]
    outs = layer.run(sents, free=True)
    table = {}
    for v in variants:
        layer.variant = v
        st = Counter()
        for r, o in zip(rows, outs):
            words = o["src"].split()
            for x in o["replacements"]:  # re-decide for this variant from the stored candidates
                x["msa"], x["tier"], x["source"] = layer.decide(x["type"], x["word"], x["foreign_model"], x["gemma"])
            gold_sp = sorted(s for s in typed_spans([l for _, l in tokens(r["src_marked"])])[0])
            gold = {sp: m for sp, (_, m) in zip(gold_sp, r["pairs"])}
            need = [sp for sp, m in gold.items() if normalize(" ".join(words[sp[0]:sp[1]])) != normalize(m)]
            pred = {tuple(x["span"]): x for x in o["replacements"] if x["type"] == "dialect"}
            span_stats(st, "dialect", gold, pred, need)
            st["false_foreign"] += sum(x["type"] == "foreign" for x in o["replacements"])
            st["not_applied"] += len(o["not_applied"])
        table[v] = st
        dump(OUT / f"dialect_{split}_{v}.jsonl", [dict(o, id=r["id"], ref=r["tgt"]) for r, o in zip(rows, outs)])
    return table, len(rows)


def eval_foreign(layer, split):
    rows = load_splits()[split]
    outs = layer.run([r["src"] for r in rows], free=True)
    st = Counter()
    for r, o in zip(rows, outs):
        gold = {(s, e): m for s, e, m in r["spans"]}
        need = [sp for sp, m in gold.items() if normalize(" ".join(r["words"][sp[0]:sp[1]])) != normalize(m)]
        pred = {tuple(x["span"]): x for x in o["replacements"] if x["type"] == "foreign"}
        span_stats(st, "foreign", gold, pred, need)
        st["false_dialect"] += sum(x["type"] == "dialect" for x in o["replacements"])
        # words outside the gold foreign spans: how many does each system change?
        keep = [i for i in range(len(r["words"])) if not any(s <= i < e for s, e in gold)]
        st["keep_words"] += len(keep)
        raw_changed = set()
        sm = difflib.SequenceMatcher(a=[normalize(x) for x in r["words"]], b=[normalize(x) for x in o["foreign_model_raw"].split()],
                                     autojunk=False)
        for op, i1, i2, _, _ in sm.get_opcodes():
            if op != "equal":
                raw_changed |= set(range(i1, i2))
        st["raw_changed_outside"] += len(raw_changed & set(keep))
        st["layer_changed_outside"] += sum(any(x["span"][0] <= i < x["span"][1] and normalize(x["msa"]) != normalize(x["word"])
                                               for x in o["replacements"]) for i in keep)
    bleu_raw = sacrebleu.corpus_bleu([o["foreign_model_raw"] for o in outs], [[r["tgt"] for r in rows]]).score
    bleu_layer = sacrebleu.corpus_bleu([o["msa"] for o in outs], [[r["tgt"] for r in rows]]).score
    bleu_copy = sacrebleu.corpus_bleu([r["src"] for r in rows], [[r["tgt"] for r in rows]]).score
    dump(OUT / f"foreign_{split}.jsonl", [dict(o, id=r["id"], ref=r["tgt"]) for r, o in zip(rows, outs)])
    return st, len(rows), {"copy": bleu_copy, "raw": bleu_raw, "layer": bleu_layer}


def dump(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")


def pct(a, b):
    return f"{100 * a / max(b, 1):.1f}"


def prf_line(st, k):
    P, R = st[k + "_hit"] / max(st[k + "_pred"], 1), st[k + "_hit"] / max(st[k + "_gold"], 1)
    return f"{100 * P:.1f} / {100 * R:.1f} / {200 * P * R / max(P + R, 1e-9):.1f}"


def tier_lines(st, k):
    found = st[k + "_suggested"] + st[k + "_review"]
    return [f"| spans that need a change, found | {pct(found, st[k + '_need'])}% of {st[k + '_need']} |",
            f"| end-to-end exact | {pct(st[k + '_ok'], st[k + '_need'])}% |",
            f"| suggested tier: share of found / exact | {pct(st[k + '_suggested'], found)}% / {pct(st[k + '_suggested_ok'], st[k + '_suggested'])}% |",
            f"| review tier: exact | {pct(st[k + '_review_ok'], st[k + '_review'])}% |"]


def report(split, dia, n_dia, fst, n_for, bleu, layer):
    L = [f"# Merged layer on {split}", "",
         f"Tagger: MARBERTv2, dialect + foreign (src/tag_merged.py). Foreign model: m2m100 fine-tune, unchanged. "
         f"Dialect: dictionary + {layer.gemma} ({layer.mode}, k={layer.k}, +gen3 examples). Tier thresholds min_n={layer.min_n}, min_c={layer.min_c}.", "",
         f"## Dialect sentences ({split}, {n_dia})", ""]
    for v, st in dia.items():
        L += [f"### variant: {v}", "", "| step | value |", "|---|---|",
              f"| dialect span detection P / R / F1 | {prf_line(st, 'dialect')} |", *tier_lines(st, "dialect"),
              f"| foreign spans wrongly found in dialect sentences | {st['false_foreign']} |", ""]
    st = next(iter(dia.values()))
    L += ["### The two sources on the found dialect spans that need a change", "", "| source | changed the word | exact |", "|---|---|---|"]
    for alt in ("foreign_model", "gemma"):
        L.append(f"| {alt} | {pct(st[f'dialect_{alt}_changed'], st[f'dialect_{alt}_given'])}% | "
                 f"{pct(st[f'dialect_{alt}_ok'], st[f'dialect_{alt}_given'])}% |")
    L += ["", f"## Foreign sentences ({split}, {n_for})", "", "| step | value |", "|---|---|",
          f"| foreign span detection P / R / F1 | {prf_line(fst, 'foreign')} |", *tier_lines(fst, "foreign"),
          f"| dialect spans wrongly found in foreign sentences | {fst['false_dialect']} |",
          f"| words outside the foreign words changed: foreign model alone / merged layer | "
          f"{pct(fst['raw_changed_outside'], fst['keep_words'])}% / {pct(fst['layer_changed_outside'], fst['keep_words'])}% of {fst['keep_words']} |",
          f"| sentence BLEU: copy input / foreign model alone / merged layer | {bleu['copy']:.2f} / {bleu['raw']:.2f} / {bleu['layer']:.2f} |",
          "", "Exact match against one reference underestimates (a human review of 200 dialect dev answers found ~37 points more acceptable)."]
    (OUT / f"{split}.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", nargs="*")
    ap.add_argument("--eval", nargs="*")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--mode", default="rag")
    ap.add_argument("--variant", default="gemma")
    ap.add_argument("--variants", nargs="*", default=["gemma", "m2m_first"])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.eval or []:
        assert s == "dev" or a.final, "settings are chosen on dev; pass --final to run test once"
        layer = Layer(mode=a.mode, variant=a.variant)
        dia, n_dia = eval_dialect(layer, s, a.variants)
        layer.variant = a.variant
        fst, n_for, bleu = eval_foreign(layer, s)
        report(s, dia, n_dia, fst, n_for, bleu, layer)
    if a.text:
        layer = Layer(mode=a.mode, variant=a.variant)
        for o in layer.run(a.text, free=True):
            print(json.dumps(o, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
