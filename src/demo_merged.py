"""Web demo of the merged layer (src/merged.py): dialect + foreign words -> MSA.

The tagger and the foreign model run on the CPU, Gemma through Ollama on the GPU.
usage: python src/demo_merged.py [--share] [--password PW] [--foreign gemma|m2m]
--share gives a temporary public gradio.live link (72 h).
"""
import argparse
import html

import gradio as gr

from merged import Layer

KIND = {"dialect": "عامي", "foreign": "أعجمي"}
SOURCE = {"dictionary": "القاموس", "gemma": "Gemma", "foreign_model": "نموذج الأعجمي",
          "gemma+foreign_model": "Gemma + نموذج الأعجمي", "gemma+dictionary": "Gemma + القاموس",
          "foreign_model+dictionary": "نموذج الأعجمي + القاموس", "none": "—"}


def render(o):
    words = o["src"].split()
    out, pos = [], 0
    for r in o["replacements"]:
        s, e = r["span"]
        out += [html.escape(w) for w in words[pos:s]]
        color = "#2e7d4f" if r["tier"] == "suggested" else "#b7791f"
        bg = "#eef6ff" if r["type"] == "foreign" else "transparent"
        out.append(f'<span style="border-bottom:3px solid {color};background:{bg}" title="{html.escape(r["word"])}">'
                   f'{html.escape(r["msa"] or r["word"])}</span>')
        pos = e
    out += [html.escape(w) for w in words[pos:]]
    if not o["replacements"]:
        return f'<div dir="rtl" style="font-size:20px">لم تُكتشف كلمات عامية أو أعجمية: {html.escape(o["src"])}</div>'
    rows = "".join(
        f'<tr><td>{html.escape(r["word"])}</td><td>{KIND[r["type"]]}</td><td>{html.escape(r["msa"] or "—")}</td>'
        f'<td>{"مقترح ✅" if r["tier"] == "suggested" else "للمراجعة ⚠️"}</td><td>{SOURCE.get(r["source"], r["source"])}</td></tr>'
        for r in o["replacements"])
    return (f'<div dir="rtl" style="font-size:22px;line-height:2">{" ".join(out)}</div>'
            f'<table dir="rtl" style="margin-top:12px;font-size:16px"><tr><th>الكلمة</th><th>النوع</th><th>الفصحى</th>'
            f'<th>المستوى</th><th>المصدر</th></tr>{rows}</table>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--password")
    ap.add_argument("--foreign", default="gemma", choices=["gemma", "m2m"])
    ap.add_argument("--port", type=int, default=7861)
    a = ap.parse_args()
    layer = Layer(device="cpu", foreign=a.foreign)

    def run(text):
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()][:10]
        return "<hr>".join(render(o) for o in layer.run(lines)) if lines else ""

    with gr.Blocks(title="الأعجمي والعامي إلى الفصحى") as ui:
        gr.Markdown("<div dir='rtl'><h2>تحويل الكلمات العامية والأعجمية إلى الفصحى (تجريبي)</h2>"
                    "اكتب جملة (أو حتى 10 جمل، كل جملة في سطر). تُستبدل الكلمات العامية والأعجمية فقط، وتبقى بقية الجملة كما هي.<br>"
                    "الخلفية الزرقاء: كلمة أعجمية. بلا خلفية: كلمة عامية.<br>"
                    "الخط الأخضر: <b>مقترح</b> (اتفق عليه مصدران أو ثابت في القاموس). البرتقالي: <b>للمراجعة</b>.<br>"
                    "قد تفوت الأداة نحو 20% من الكلمات.</div>")
        box = gr.Textbox(lines=3, rtl=True, label="الجملة", placeholder="ابي اموديفاي العقد بكرة قبل الميتنق")
        btn = gr.Button("حوّل")
        out = gr.HTML()
        gr.Examples(["ابي اموديفاي العقد بكرة قبل الميتنق", "وش رايك نروح الكوفي شوب الحين",
                     "ارتدى بليزر في الاجتماع الرسمي", "مدري وش صار بالبزنس بلان"], box)
        btn.click(run, box, out)
        box.submit(run, box, out)
    ui.queue(default_concurrency_limit=1).launch(share=a.share, server_port=a.port,
                                                 auth=("demo", a.password) if a.password else None)


if __name__ == "__main__":
    main()
