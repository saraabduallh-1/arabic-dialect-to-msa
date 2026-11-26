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
import os
import re
from collections import Counter
from pathlib import Path

import requests
import sacrebleu
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForTokenClassification, AutoTokenizer

import run_word
from data import ROOT, clean, load, normalize
from domain_terms import DomainTerms
from foreign_data import MODEL_DIR, OUT, foreign_dict, load_splits, load_tokenizer
from run_llm import LLM_URL, Retriever, chat
from tag_dialect import tokens
from tag_merged import typed_spans
from tiers import train_dict

TAGGER = Path(os.environ.get("FUSHA_TAGGER_DIR", ROOT / "results" / "checkpoints" / "tagger_merged"))
GEMMA = os.environ.get("FUSHA_LLM_MODEL", "gemma4:12b")
LATIN = re.compile(r"[A-Za-z]")
END_MARKS = ".,،؛;:!?؟…"  # sentence marks glued to a word; brackets, quotes and hyphens are left alone
CACHE = OUT / "gemma_cache.jsonl"
FSYSTEM = (
    "أنت خبير في اللغة العربية الفصحى. "
    "ستُعطى جملة عربية فيها كلمات أو عبارات أعجمية (منقولة من الإنجليزية أو غيرها بحروف عربية أو لاتينية) بين قوسين وبعد كل منها رقم. "
    "اكتب لكل رقم مقابله في العربية الفصحى المناسب لمعنى الجملة وسياقها، بنفس الصيغة النحوية قدر الإمكان. "
    "أجب بسطر واحد لكل رقم بالشكل «الرقم: المقابل الفصيح» فقط، بدون شرح."
)


def foreign_rows(rows):
    """Foreign rows in the shape of data.load (marked sentence + pairs), for retrieval of worked examples."""
    out = []
    for r in rows:
        if r["source"] != "annotated" or not r["spans"]:
            continue
        w, marked, pos = r["words"], [], 0
        for n, (s, e, _) in enumerate(r["spans"], 1):
            marked += w[pos:s] + [f"({' '.join(w[s:e])}){n}"]
            pos = e
        marked += w[pos:]
        out.append({"src": r["src"], "tgt": r["tgt"], "src_marked": " ".join(marked),
                    "pairs": [(" ".join(w[s:e]), m) for s, e, m in r["spans"]], "span_ids": list(range(1, len(r["spans"]) + 1))})
    return out


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
    def __init__(self, mode="rag", k=8, min_n=2, min_c=0.8, variant="gemma", device="cpu", gemma=GEMMA, foreign="gemma"):
        self.mode, self.k, self.min_n, self.min_c, self.variant, self.dev, self.gemma = mode, k, min_n, min_c, variant, device, gemma
        self.foreign = foreign  # "gemma": Gemma answers foreign spans, the foreign model is a witness; "m2m": the model answers
        self.ttok = AutoTokenizer.from_pretrained(TAGGER)
        self.tagger = AutoModelForTokenClassification.from_pretrained(TAGGER).to(device).eval()
        self.mtok = load_tokenizer()
        self.m2m = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(device).eval()
        self.ddic = train_dict()
        ftrain = load_splits()["train"]
        self.fdic = foreign_dict(ftrain)
        self.fret = Retriever(foreign_rows(ftrain))
        self.domain = DomainTerms()  # Saleem's specialization glossary: its terms win (src/domain_terms.py)
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
            requests.post(f"{LLM_URL}/api/generate", json={"model": self.gemma, "keep_alive": 0}, timeout=60)
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

    def tag_punct(self, words):
        """tag() for text as people write it. The dialect training sentences have no punctuation, so a mark glued to a
        word («بكرة،») hides it from the tagger: dialect spans come from the words without their end marks, foreign
        spans from the words as written (the foreign data has its marks), and a foreign span wins an overlap.
        Dev, 25% of words with a glued mark: dialect recall 0.645 -> 0.710; clean dev and foreign dev unchanged."""
        bare = [w.rstrip(END_MARKS) or w for w in words]
        if bare == words:
            return self.tag(words)
        fore = [x for x in self.tag(words) if x[2] == "foreign"]
        dia = [x for x in self.tag(bare) if x[2] == "dialect"]
        return sorted(fore + [x for x in dia if not any(x[0] < e and s < x[1] for s, e, _ in fore)])

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
    def ask_gemma(self, words, dspans, kind="dialect"):
        marked, pos = [], 0
        for n, (s, e) in enumerate(dspans, 1):
            marked += words[pos:s] + [f"({' '.join(words[s:e])}){n}"]
            pos = e
        marked += words[pos:]
        r = {"src": " ".join(words), "src_marked": " ".join(marked), "pairs": [(" ".join(words[s:e]), "") for s, e in dspans]}
        if kind == "dialect":
            msgs = run_word.build(r, self.mode, self.ret, self.k)
        else:  # same word prompt, foreign instruction, examples retrieved from the foreign project's annotated pairs
            msgs = [{"role": "system", "content": FSYSTEM}]
            for ex in self.fret.examples(r["src"], self.k):
                msgs += [{"role": "user", "content": f"الجملة: {ex['src_marked']}"},
                         {"role": "assistant", "content": run_word.answer(ex)}]
            user = f"الجملة: {r['src_marked']}"
            msgs.append({"role": "user", "content": user})
        key = hashlib.sha1(json.dumps([self.gemma, msgs], ensure_ascii=False).encode()).hexdigest()
        if key not in self.cache:
            raw = chat(self.gemma, msgs)
            if not raw.strip():  # an empty answer (server trouble) is not kept: the next call asks again
                return [""] * len(dspans)
            self.cache[key] = raw
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            with CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
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
        if kind == "foreign" and self.foreign == "gemma":
            c = dic.get(normalize(w))
            h = self.domain.lookup(w)
            if h and not copy(g) and normalize(g) == normalize(h[1]):  # same as the specialization layer's term
                return g, "suggested", "gemma+domain_glossary"
            if not copy(g) and not copy(m) and normalize(g) == normalize(m):
                return g, "suggested", "gemma+foreign_model"
            if not copy(g) and c and normalize(g) == normalize(c.most_common(1)[0][0]):
                return g, "suggested", "gemma+dictionary"
            for x, src in ((g, "gemma"), (m, "foreign_model")):
                if not copy(x):
                    return x, "review", src
            return (c.most_common(1)[0][0], "review", "dictionary") if c else (w, "review", "none")
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

    def run(self, sentences, free=False, ask_foreign=None):
        ask_foreign = self.foreign == "gemma" if ask_foreign is None else ask_foreign
        if free:
            self.to_gpu()
        words = [clean(s).split() for s in sentences]
        spans = [self.tag_punct(w) for w in words]
        hyps = [h.split() for h in self.translate([" ".join(w) for w in words])]
        if free:
            self.free_gpu()
        res = []
        for w, sp, h in zip(words, spans, hyps):
            dsp = [(s, e) for s, e, t in sp if t == "dialect"]
            gans = dict(zip(dsp, self.ask_gemma(w, dsp))) if dsp else {}
            fsp = [(s, e) for s, e, t in sp if t == "foreign"]
            if fsp and ask_foreign:
                gans.update(zip(fsp, self.ask_gemma(w, fsp, "foreign")))
            reps, out, pos = [], [], 0
            for s, e, t in sp:
                span = " ".join(w[s:e])
                m = aligned(w, h, s, e)
                g = gans.get((s, e))
                msa, tier, src = self.decide(t, span, m, g)
                gl = self.domain.lookup(span) if t == "foreign" else None
                reps.append({"span": [s, e], "type": t, "word": span, "msa": msa, "tier": tier, "source": src,
                             "foreign_model": m, "gemma": g,
                             # possible term of Saleem's specialization glossary: that layer decides, not this one
                             "glossary": {"english": gl[0], "arabic": gl[1]} if gl else None})
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
            if val is not None:
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
                if x["source"] == "domain_glossary":
                    continue
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


def eval_foreign_variants(layer, split):
    """Foreign spans: the foreign model's answer (m2m) vs Gemma's with the model as witness (gemma).
    Nearly all foreign data was in the foreign model's training, so there is no clean dev set: the test file is cut in
    two halves (A = even rows, B = odd rows); the choice is made on A and reported on B."""
    rows = load_splits()[split]
    outs = layer.run([r["src"] for r in rows], free=True, ask_foreign=True)
    subsets = {"all": range(len(rows)), "A": range(0, len(rows), 2), "B": range(1, len(rows), 2)}
    L = [f"# Foreign spans: foreign model vs Gemma ({split})", "",
         "| variant | subset | found | end-to-end exact | suggested share / exact | review exact | Gemma exact / model exact on found | BLEU |",
         "|---|---|---|---|---|---|---|---|"]
    for v in ("m2m", "gemma"):
        layer.foreign = v
        for o in outs:
            for x in o["replacements"]:
                if x["source"] == "domain_glossary":
                    continue
                x["msa"], x["tier"], x["source"] = layer.decide(x["type"], x["word"], x["foreign_model"], x["gemma"])
            w, out, pos = o["src"].split(), [], 0
            for x in o["replacements"]:
                out += w[pos:x["span"][0]] + [x["msa"]]
                pos = x["span"][1]
            o["msa"] = " ".join(out + w[pos:])
        dump(OUT / f"foreign_{split}_{v}.jsonl", [dict(o, id=r["id"], ref=r["tgt"]) for r, o in zip(rows, outs)])
        for name, idx in subsets.items():
            st = Counter()
            for i in idx:
                r, o = rows[i], outs[i]
                gold = {(s, e): m for s, e, m in r["spans"]}
                need = [sp for sp, m in gold.items() if normalize(" ".join(r["words"][sp[0]:sp[1]])) != normalize(m)]
                span_stats(st, "foreign", gold, {tuple(x["span"]): x for x in o["replacements"] if x["type"] == "foreign"}, need)
            found = st["foreign_suggested"] + st["foreign_review"]
            bleu = sacrebleu.corpus_bleu([outs[i]["msa"] for i in idx], [[rows[i]["tgt"] for i in idx]]).score
            L.append(f"| {v} | {name} ({len(idx)}) | {pct(found, st['foreign_need'])}% | {pct(st['foreign_ok'], st['foreign_need'])}% | "
                     f"{pct(st['foreign_suggested'], found)}% / {pct(st['foreign_suggested_ok'], st['foreign_suggested'])}% | "
                     f"{pct(st['foreign_review_ok'], st['foreign_review'])}% | "
                     f"{pct(st['foreign_gemma_ok'], st['foreign_gemma_given'])}% / {pct(st['foreign_foreign_model_ok'], st['foreign_foreign_model_given'])}% | {bleu:.2f} |")
    (OUT / f"foreign_gemma_{split}.md").write_text(chr(10).join(L) + chr(10), encoding="utf-8")
    print(chr(10).join(L))


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
    ap.add_argument("--foreign_eval", nargs="*", help="compare foreign model vs Gemma on foreign spans, e.g. dev test")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.eval or []:
        assert s == "dev" or a.final, "settings are chosen on dev; pass --final to run test once"
        layer = Layer(mode=a.mode, variant=a.variant)
        dia, n_dia = eval_dialect(layer, s, a.variants)
        layer.variant = a.variant
        fst, n_for, bleu = eval_foreign(layer, s)
        report(s, dia, n_dia, fst, n_for, bleu, layer)
    for s in a.foreign_eval or []:
        eval_foreign_variants(Layer(mode=a.mode), s)
    if a.text:
        layer = Layer(mode=a.mode, variant=a.variant)
        for o in layer.run(a.text, free=True):
            print(json.dumps(o, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
