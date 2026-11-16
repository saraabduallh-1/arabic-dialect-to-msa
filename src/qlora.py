"""QLoRA fine-tuning of an instruction LLM (4-bit base + LoRA adapters) on dialect -> MSA.

--task sentence: rewrite the whole sentence (run_llm.py prompt).
--task word: the dialect words are marked, the answer is "N: MSA" per word (run_word.py prompt and metrics).

usage: python src/qlora.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3
       python src/qlora.py --model ALLaM-AI/ALLaM-7B-Instruct-preview --task word --data sentences_v2
"""
import argparse
import json
import random

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_cosine_schedule_with_warmup

from data import ROOT, load
from metrics import score
import run_word
from run_llm import SYSTEM, postprocess

TASK = "sentence"  # set from --task
LEX = None  # lexicon.Lexicon when --lex: dictionary meanings of the marked words go into the word prompt


def user_word(r):
    u = f"الجملة: {r['src_marked']}"
    if LEX:
        h = [(k, LEX.hint(d)) for k, (d, _) in zip(r["span_ids"], r["pairs"])]
        h = [f"{k}: {x[0]} = {x[1]}" for k, x in h if x]
        if h:
            u += "\nمعاني من معجم اللهجات (آلية، قد تخطئ أو لا تناسب السياق):\n" + "\n".join(h)
    return u


def prompt(tok, r):
    if TASK == "word":
        msgs = [{"role": "system", "content": run_word.SYSTEM}, {"role": "user", "content": user_word(r)}]
    else:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"العامية: {r['src']}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def answer(r):
    return run_word.answer(r) if TASK == "word" else r["tgt"]


def evaluate(rows, outs):
    if TASK == "word":
        s = run_word.score(rows, [{str(k): v for k, v in run_word.parse(o).items()} for o in outs])
        return s, s["WordAcc%"]
    s = score(rows, [postprocess(o) for o in outs])
    return s, s["chrF++"]


def encode(tok, r):
    """Prompt + answer; loss only on the answer tokens."""
    p = tok(prompt(tok, r), add_special_tokens=False)["input_ids"]
    a = tok(answer(r) + tok.eos_token, add_special_tokens=False)["input_ids"]
    return p + a, [-100] * len(p) + a


def batches(tok, rows, bs):
    rows = rows[:]
    random.shuffle(rows)
    for i in range(0, len(rows), bs):
        enc = [encode(tok, r) for r in rows[i:i + bs]]
        n = max(len(x) for x, _ in enc)
        ids = torch.tensor([x + [tok.pad_token_id] * (n - len(x)) for x, _ in enc])
        labels = torch.tensor([y + [-100] * (n - len(y)) for _, y in enc])
        yield ids, labels, (ids != tok.pad_token_id) | (labels != -100)


@torch.no_grad()
def generate(model, tok, rows, bs=16):
    model.eval()
    tok.padding_side = "left"
    out = []
    for i in tqdm(range(0, len(rows), bs), desc="generate", leave=False):
        enc = tok([prompt(tok, r) for r in rows[i:i + bs]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [t for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--limit", type=int, help="use only N rows per split (smoke test)")
    ap.add_argument("--extra", nargs="*", default=[], help="extra training data as dataset/split, e.g. generated/all")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. +gen")
    ap.add_argument("--task", choices=["sentence", "word"], default="sentence")
    ap.add_argument("--data", default="sentences", help="sentences (original) or sentences_v2 (corrected)")
    ap.add_argument("--lex", action="store_true", help="word task: add dictionary meanings (src/lexicon.py)")
    a = ap.parse_args()
    global TASK, LEX
    TASK = a.task
    if a.lex:
        from lexicon import Lexicon
        LEX = Lexicon()
    random.seed(42)
    torch.manual_seed(42)

    train, dev, test = (load(s, a.data)[: a.limit] for s in ("train", "dev", "test"))
    for e in a.extra:
        ds, split = e.split("/")
        train += load(split, ds)[: a.limit]
    if TASK == "word":  # rows without aligned dialect words teach nothing here
        train = [r for r in train if r["pairs"]]
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map={"": 0},
                                                dtype=torch.bfloat16)
    base = prepare_model_for_kbit_training(
        base, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    lora = LoraConfig(r=a.r, lora_alpha=2 * a.r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(base, lora)
    model.print_trainable_parameters()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    name = a.model.split("/")[-1] + "-qlora" + ("-word" if TASK == "word" else "") + a.tag
    ckpt = ROOT / "results" / "checkpoints" / name
    best = -1
    model.train()
    for ep in range(a.epochs):
        tok.padding_side = "right"
        for ids, labels, mask in tqdm(batches(tok, train, a.bs), desc=f"epoch {ep + 1}", leave=False):
            loss = model(input_ids=ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda()).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        s, key = evaluate(dev, generate(model, tok, dev))
        print(f"epoch {ep + 1} loss {loss.item():.3f} dev {s}", flush=True)
        if key > best:  # model selection on dev only
            best = key
            model.save_pretrained(ckpt)

    set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))  # best epoch
    outs = generate(model, tok, test)
    res_dir = ROOT / "results" / ("word" if TASK == "word" else "")
    (res_dir / "preds").mkdir(parents=True, exist_ok=True)
    with (res_dir / "preds" / f"{name}__finetune.jsonl").open("w", encoding="utf-8") as f:
        for r, o in zip(test, outs):
            pred = {str(k): v for k, v in run_word.parse(o).items()} if TASK == "word" else postprocess(o)
            f.write(json.dumps({"id": r["id"], "src": r["src_marked"] if TASK == "word" else r["src"], "tgt": r["tgt"],
                                "raw": o, "pred": pred}, ensure_ascii=False) + "\n")
    s, _ = evaluate(test, outs)
    res = {"model": name, "mode": "finetune", **({"data": a.data, "k": 0} if TASK == "word" else {}), **s}
    print(json.dumps(res, ensure_ascii=False))
    if a.limit:  # smoke runs don't write scores
        return
    with (res_dir / "scores.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
