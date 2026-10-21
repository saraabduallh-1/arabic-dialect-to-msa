"""QLoRA fine-tuning of an instruction LLM (4-bit base + LoRA adapters) on dialect -> MSA.

usage: python src/qlora.py --model Qwen/Qwen2.5-7B-Instruct --epochs 3
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
from run_llm import SYSTEM, postprocess


def prompt(tok, src):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"العامية: {src}"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode(tok, r):
    """Prompt + answer; loss only on the answer tokens."""
    p = tok(prompt(tok, r["src"]), add_special_tokens=False)["input_ids"]
    a = tok(r["tgt"] + tok.eos_token, add_special_tokens=False)["input_ids"]
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
        enc = tok([prompt(tok, r["src"]) for r in rows[i:i + bs]], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=96, do_sample=False, pad_token_id=tok.pad_token_id)
        out += [postprocess(t) for t in tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
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
    a = ap.parse_args()
    random.seed(42)
    torch.manual_seed(42)

    train, dev, test = (load(s)[: a.limit] for s in ("train", "dev", "test"))
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

    name = a.model.split("/")[-1] + "-qlora"
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
        s = score(dev, generate(model, tok, dev))
        print(f"epoch {ep + 1} loss {loss.item():.3f} dev {s}", flush=True)
        if s["chrF++"] > best:  # model selection on dev only
            best = s["chrF++"]
            model.save_pretrained(ckpt)

    set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))  # best epoch
    preds = generate(model, tok, test)
    out = ROOT / "results" / "preds" / f"{name}__finetune.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r, p in zip(test, preds):
            f.write(json.dumps({"id": r["id"], "src": r["src"], "tgt": r["tgt"], "pred": p}, ensure_ascii=False) + "\n")
    res = {"model": name, "mode": "finetune", **score(test, preds)}
    print(json.dumps(res, ensure_ascii=False))
    with (ROOT / "results" / "scores.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
