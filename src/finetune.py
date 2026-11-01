"""Fine-tune an Arabic seq2seq model (AraT5 / AraBART / mT5) on dialect -> MSA.

usage: python src/finetune.py --model UBC-NLP/AraT5v2-base-1024 --epochs 15
"""
import argparse
import json
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor

from data import ROOT, load
from metrics import score


def batches(rows, tok, bs, shuffle):
    rows = rows[:]
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), bs):
        b = rows[i:i + bs]
        enc = tok([r["src"] for r in b], text_target=[r["tgt"] for r in b],
                  padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc["labels"][enc["labels"] == tok.pad_token_id] = -100
        yield enc


@torch.no_grad()
def generate(model, tok, rows, bs=32, beams=4):
    model.eval()
    out = []
    for i in range(0, len(rows), bs):
        enc = tok([r["src"] for r in rows[i:i + bs]], padding=True, truncation=True,
                  max_length=128, return_tensors="pt").to(model.device)
        gen = model.generate(**enc, num_beams=beams, max_new_tokens=96)
        out += tok.batch_decode(gen, skip_special_tokens=True)
    return [o.strip() for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--optim", choices=["adamw", "adafactor"], default="adamw", help="adafactor needs far less memory")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, help="use only N rows per split (smoke test)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--extra", nargs="*", default=[], help="extra training data as dataset/split, e.g. generated/all")
    ap.add_argument("--tag", default="", help="suffix for output names, e.g. +gen")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)

    train, dev, test = (load(s)[: a.limit] for s in ("train", "dev", "test"))
    for e in a.extra:
        ds, split = e.split("/")
        train += load(split, ds)[: a.limit]
    print(f"train {len(train)} dev {len(dev)} test {len(test)}")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model).to(a.device)
    if a.optim == "adafactor":
        opt = Adafactor(model.parameters(), lr=a.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    steps = a.epochs * ((len(train) + a.bs - 1) // a.bs)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    name = a.model.split("/")[-1] + a.tag
    ckpt = ROOT / "results" / "checkpoints" / name
    best = -1
    for ep in range(a.epochs):
        model.train()
        for enc in tqdm(batches(train, tok, a.bs, True), desc=f"epoch {ep + 1}", leave=False):
            loss = model(**enc.to(model.device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(), sched.step(), opt.zero_grad()
        s = score(dev, generate(model, tok, dev, beams=1))
        print(f"epoch {ep + 1} loss {loss.item():.3f} dev {s}")
        if s["chrF++"] > best:  # model selection on dev only
            best, best_dev = s["chrF++"], {**s, "epoch": ep + 1}
            model.save_pretrained(ckpt), tok.save_pretrained(ckpt)
    if not a.limit:  # best dev score, for tuning decisions on dev (PLAN_v2 I5)
        with (ROOT / "results" / "scores_dev.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"model": a.model + a.tag, "mode": "finetune", **best_dev}, ensure_ascii=False) + "\n")

    model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(a.device)
    preds = generate(model, tok, test)
    if a.limit:  # smoke test: don't touch the results
        print(json.dumps(score(test, preds), ensure_ascii=False))
        return
    out = ROOT / "results" / "preds" / f"{name}__finetune.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r, p in zip(test, preds):
            f.write(json.dumps({"id": r["id"], "src": r["src"], "tgt": r["tgt"], "pred": p}, ensure_ascii=False) + "\n")
    res = {"model": a.model + a.tag, "mode": "finetune", **score(test, preds)}
    print(json.dumps(res, ensure_ascii=False))
    with (ROOT / "results" / "scores.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
