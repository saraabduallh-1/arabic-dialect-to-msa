"""Web demo of the end-to-end tool (src/pipeline.py). Both word models stay loaded; one GPU with 10 GB.

usage: python src/demo.py [--share] [--password PW]
--share gives a temporary public gradio.live link (72 h); protect it with --password (user "demo").
"""
import argparse
import html

import gradio as gr

from pipeline import ADAPTER, BASE, SECOND, Pipeline


class Demo(Pipeline):
    def __init__(self, single=False):
        super().__init__()
        self.load_llm(BASE, ADAPTER)
        self.m1 = (self.tok, self.llm)
        if single:  # ALLaM only (half the GPU memory): no agreement tier, "suggested" comes from the dictionary only
            self.m2 = (self.tok, None)
        else:
            self.load_llm(*SECOND)
            self.m2 = (self.tok, self.llm)

    def replace(self, items):
        return [{} for _ in items] if self.llm is None else super().replace(items)

    def load_llm(self, base_id, adapter):
        """At start-up: really load. While serving: switch to the model already in memory."""
        loaded = {BASE: getattr(self, "m1", None), SECOND[0]: getattr(self, "m2", None)}.get(base_id)
        if loaded:
            self.tok, self.llm = loaded
        else:
            super().load_llm(base_id, adapter)

    def unload_llm(self):  # keep both models in memory
        pass


def render(o):
    words = o["src"].split()
    out, pos = [], 0
    for r in o["replacements"]:
        s, e = r["span"]
        out += [html.escape(w) for w in words[pos:s]]
        color = "#2e7d4f" if r["tier"] == "suggested" else "#b7791f"
        out.append(f'<span style="border-bottom:3px solid {color}" title="{html.escape(r["dialect"])}">'
                   f'{html.escape(r["msa"] or r["dialect"])}</span>')
        pos = e
    out += [html.escape(w) for w in words[pos:]]
    rows = "".join(f'<tr><td>{html.escape(r["dialect"])}</td><td>{html.escape(r["msa"] or "—")}</td>'
                   f'<td>{"مقترح ✅" if r["tier"] == "suggested" else "للمراجعة ⚠️"}</td></tr>' for r in o["replacements"])
    return (f'<div dir="rtl" style="font-size:22px;line-height:2">{" ".join(out)}</div>'
            f'<table dir="rtl" style="margin-top:12px;font-size:16px"><tr><th>العامية</th><th>الفصحى</th><th>المستوى</th></tr>{rows}</table>'
            if o["replacements"] else f'<div dir="rtl" style="font-size:20px">لم تُكتشف كلمات عامية: {html.escape(o["src"])}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--password")
    ap.add_argument("--single", action="store_true", help="load ALLaM only (frees the GPU for other work)")
    a = ap.parse_args()
    p = Demo(a.single)

    def run(text):
        lines = [l.strip() for l in (text or "").splitlines() if l.strip()][:10]
        return "<hr>".join(render(o) for o in p(lines)) if lines else ""

    with gr.Blocks(title="العامية السعودية إلى الفصحى") as ui:
        gr.Markdown("<div dir='rtl'><h2>تحويل العامية السعودية إلى الفصحى (تجريبي، قيد العمل)</h2>"
                    "اكتب جملة عامية (أو حتى 10 جمل، كل جملة في سطر). الخط الأخضر: <b>مقترح</b> (صحيح في نحو 90% من الحالات)، "
                    "البرتقالي: <b>للمراجعة</b>. قد تفوت الأداة نحو 20% من الكلمات العامية.</div>")
        box = gr.Textbox(lines=3, rtl=True, label="الجملة العامية", placeholder="وش تبي تسوي الحين")
        btn = gr.Button("حوّل")
        out = gr.HTML()
        btn.click(run, box, out)
        box.submit(run, box, out)
    ui.queue(default_concurrency_limit=1).launch(share=a.share, auth=("demo", a.password) if a.password else None)


if __name__ == "__main__":
    main()
