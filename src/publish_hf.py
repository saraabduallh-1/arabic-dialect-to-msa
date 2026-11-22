"""Publish the two word-level LoRA adapters to the Hugging Face Hub with a model card (user approved public release).

The MARBERTv2 tagger is not published: its base model declares no license (see autowork/QUESTIONS.md).

usage: python src/publish_hf.py
"""
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from data import ROOT

GH = "https://github.com/saraabduallh-1/arabic-dialect-to-msa"
MODELS = [
    ("ALLaM-7B-Instruct-preview-qlora-word+gen3", "ALLaM-AI/ALLaM-7B-Instruct-preview", "allam-7b-saudi-dialect-to-msa-lora",
     "Main replacement model of the tool. Test (dialect words given): 46.9% exact, 65.1% correct by meaning (LLM judge); "
     "dev ~85% correct by meaning from a human review of 99 answers."),
    ("Qwen2.5-7B-Instruct-qlora-word+gen3", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-saudi-dialect-to-msa-lora",
     "Second model of the tool: when it agrees with the ALLaM adapter, the answer is marked 'suggested'. "
     "Dev (dialect words given): 32.3% exact alone; agreement tier covers 38% of words at 73.7% exact."),
]
SYSTEM = ("أنت خبير في اللغة العربية ولهجاتها (خاصة اللهجات السعودية). ستُعطى جملة عامية فيها كلمات أو عبارات عامية بين قوسين "
          "وبعد كل منها رقم. اكتب لكل رقم مقابله في العربية الفصحى المناسب لمعنى الجملة وسياقها، بنفس الصيغة النحوية قدر الإمكان. "
          "أجب بسطر واحد لكل رقم بالشكل «الرقم: المقابل الفصيح» فقط، بدون شرح.")


def card(base, note):
    return f"""---
base_model: {base}
library_name: peft
license: apache-2.0
language: [ar]
tags: [arabic, saudi-dialect, dialect-to-msa, lora, qlora]
---

# Saudi dialect word → MSA (LoRA adapter for {base})

Replaces marked Saudi dialect words in a sentence with their Modern Standard Arabic equivalent, in context.
QLoRA (4-bit NF4, r=16, alpha=32, all linear projections), trained on ~2.3k marked sentence pairs plus verified
generated sentences. Part of the project {GH} (code, data, Arabic report: `docs/paper/report_ar.pdf`).

**Status: research release, work in progress.** {note}

End-to-end tool (MARBERT tagger → this adapter → suggested/review tiers): on test the "suggested" tier covers 28% of
dialect words at ~90% correct by meaning; the tagger misses ~20% of dialect words. Use it as an assistant for an
editor, not as an unattended converter.

## Prompt

System:
```
{SYSTEM}
```
User: `الجملة: كيف أحلق شعري ورأسي (صقعة)1` → Assistant: `1: أصلع` (one line `N: MSA` per marked span).

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
tok = AutoTokenizer.from_pretrained("{base}")
base = AutoModelForCausalLM.from_pretrained("{base}", device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(base, "<this repo>")
```
Greedy decoding, max 96 new tokens. See `src/pipeline.py` in the GitHub repo.

## Limitations

Small data (one annotator, who is also the data author); single seed; exact-match scores against one reference
underestimate quality by ~37 points (human review); weaker on rare words absent from training; Saudi dialects only,
not separated by region.
"""


def main():
    api = HfApi()
    user = api.whoami()["name"]
    for ck, base, repo, note in MODELS:
        rid = f"{user}/{repo}"
        api.create_repo(rid, repo_type="model", exist_ok=True, private=False)
        with tempfile.TemporaryDirectory() as d:
            for f in ("adapter_config.json", "adapter_model.safetensors"):
                shutil.copy(ROOT / "results" / "checkpoints" / ck / f, Path(d) / f)
            (Path(d) / "README.md").write_text(card(base, note), encoding="utf-8")
            api.upload_folder(folder_path=d, repo_id=rid, commit_message="Adapter and model card")
        print("https://huggingface.co/" + rid)


if __name__ == "__main__":
    main()
