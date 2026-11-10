"""One Excel workbook for the user's review: judge validation labels + generated data.

Sheets:
  1 تحكيم الكلمات   the 150 spans of results/human_labels_todo.csv (system name hidden, to avoid bias)
  2 عينة المولدة     5 random rows per generated batch (40 batches -> 200 rows): the error rate per batch
                     decides which batches can be used for training
  3 باقي المولدة     every other generated row (optional)

Build: python src/review_excel.py            -> review/review.xlsx
Import: python src/review_excel.py --import  (after the user sends the filled file back)
  - sheet 1 -> results/human_labels_todo.csv + results/human_labels.jsonl (used by judge.py)
  - sheets 2-3 -> data/generated/review_all.csv, and error % per batch on the sample
"""
import argparse
from urllib.parse import quote

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from data import DATA, DIALECT, MSA, ROOT, lexicon, load
from human_labels import TODO, import_labels
from review_sample import new_term

OUT = ROOT / "review" / "review.xlsx"
GEN = DATA / "generated" / "all2.csv"
WORD_VERDICTS = ["صحيح", "حدّي", "خطأ"]
GEN_VERDICTS = ["اعتمد", "فيها خطأ", "احذف"]
ERRORS = ["المعنى خطأ", "الجملة غير طبيعية", "خطأ نحوي أو إملائي", "الكلمة ليست عامية", "الرابط ما فيه الكلمة", "أخرى"]

HELP = [
    ("المطلوب", "ثلاث أوراق. الورقة 1 والورقة 2 هي المهمة، والورقة 3 اختيارية."),
    ("", ""),
    ("الورقة 1: تحكيم الكلمات (150 صفًا)",
     "النموذج حوّل جملة عامية إلى فصحى، واختار مقابلًا للكلمة العامية يختلف عن المرجع. "
     "احكمي: هل معنى مخرج النظام صحيح للكلمة العامية في هذا السياق؟"),
    ("   صحيح", "المعنى صحيح حتى لو الكلمة غير كلمة المرجع (مرادف مقبول)"),
    ("   حدّي", "قريب من المعنى لكن ناقص أو غير دقيق"),
    ("   خطأ", "المعنى خطأ، أو الكلمة بقيت عامية"),
    ("   الهدف", "نعرف هل المحكّم الآلي يحكم مثلك؛ لو اتفق معك نعتمد أرقامه"),
    ("", ""),
    ("الورقة 2: عينة البيانات المولّدة (200 صف)",
     "5 جمل عشوائية من كل دفعة من الدفعات الأربعين. الكلمة الجديدة في عمود «الكلمة العامية»، "
     "وفي الجملتين بين قوسين مع رقمها."),
    ("   اعتمد", "الكلمة عامية فعلًا، والمعنى صحيح، والجملتان طبيعيتان"),
    ("   فيها خطأ", "تُصلح: اختاري نوع الخطأ، واكتبي التصحيح إن عرفتيه"),
    ("   احذف", "الجملة لا تصلح أصلًا"),
    ("   الهدف", "نحسب نسبة الخطأ لكل دفعة: الدفعة اللي خطؤها أكثر من 20% ما تدخل التدريب، أو تُراجع كاملة"),
    ("", ""),
    ("الورقة 3: باقي البيانات المولّدة (اختيارية)", "نفس طريقة الورقة 2. أي صف تحكمين عليه يُستخدم، والفاضي يبقى حسب نتيجة دفعته."),
    ("", ""),
    ("روابط", "عمود «المصدر» يفتح صفحة المعجم اللي أُخذت منها الكلمة، و«معجم الرياض» يبحث عن المقابل الفصيح."),
    ("بعد ما تخلصين", "احفظي الملف بنفس الاسم وأرسليه، أو ارفعيه في مجلد review في المشروع."),
]

HEAD = PatternFill("solid", fgColor="DDE7F0")
MINE = PatternFill("solid", fgColor="FFF4CC")  # the columns the user fills
GREY = Font(color="888888", size=9)


def sheet(wb, title, cols, rows, widths, mine, choices):
    ws = wb.create_sheet(title)
    ws.sheet_view.rightToLeft = True
    ws.append(cols)
    for r in rows:
        ws.append([r.get(c, "") for c in cols])
    for i, c in enumerate(cols, 1):
        cell = ws.cell(1, i)
        cell.font, cell.fill = Font(bold=True), MINE if c in mine else HEAD
        ws.column_dimensions[cell.column_letter].width = widths.get(c, 14)
    wrap = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = wrap
            c = cols[cell.column - 1]
            if c in mine:
                cell.fill = MINE
            elif str(cell.value).startswith("http"):
                cell.hyperlink, cell.value, cell.font = cell.value, "افتح", Font(color="0563C1", underline="single")
            elif c in ("Feature 1", "النظام", "الدفعة"):
                cell.font = GREY
    for c, opts in choices.items():
        dv = DataValidation(type="list", formula1='"' + ",".join(opts) + '"', allow_blank=True)
        col = ws.cell(1, cols.index(c) + 1).column_letter
        dv.add(f"{col}2:{col}{len(rows) + 1}")
        ws.add_data_validation(dv)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    return ws


def build():
    wb = Workbook()
    ws = wb.active
    ws.title = "التعليمات"
    ws.sheet_view.rightToLeft = True
    for a, b in HELP:
        ws.append([a, b])
    ws.column_dimensions["A"].width, ws.column_dimensions["B"].width = 38, 110
    for row in ws.iter_rows():
        row[0].font = Font(bold=True)
        row[1].alignment = Alignment(wrap_text=True)

    # 1: judge validation, same rows as results/human_labels_todo.csv
    todo = pd.read_csv(TODO, dtype=str).fillna("")
    rows = [{"#": i, "الجملة العامية": r["الجملة العامية"], "الكلمة العامية": r["الكلمة العامية"],
             "المقابل في المرجع": r["المقابل في المرجع"], "الجملة المرجعية": r["الجملة المرجعية"],
             "مخرج النظام": r["مخرج النظام"], "الحكم": r["الحكم"], "ملاحظة": "",
             "Feature 1": r["Feature 1"], "النظام": r["النظام"]} for i, r in enumerate(todo.to_dict("records"), 1)]
    ws = sheet(wb, "1 تحكيم الكلمات", list(rows[0]), rows,
               {"#": 5, "الجملة العامية": 40, "الكلمة العامية": 16, "المقابل في المرجع": 18, "الجملة المرجعية": 40,
                "مخرج النظام": 40, "الحكم": 10, "ملاحظة": 30},
               {"الحكم", "ملاحظة"}, {"الحكم": WORD_VERDICTS})
    ws.column_dimensions["J"].hidden = True  # system name: hidden so it can't bias the label

    # 2-3: generated data, 5 random rows per batch first
    lex = lexicon(load("train", "sentences_v2"))
    gen = pd.read_csv(GEN, dtype=str).fillna("")
    gen["الدفعة"] = gen["Feature 1"].map(lambda x: f"b{int(x) // 1000 - 100:02d}")
    sample = gen.groupby("الدفعة").sample(n=5, random_state=7)  # every batch has >= 40 rows
    rest = gen.drop(sample.index)

    def gen_rows(df):
        out = []
        for i, (_, r) in enumerate(df.iterrows(), 1):
            term, msa = new_term(r, lex)
            out.append({"#": i, "الكلمة العامية": term, "المقابل الفصيح": msa, "مقابلات بديلة": r["مقابلات بديلة"],
                        "اللهجة": r["اللهجة"], "الجملة العامية": r[DIALECT], "الجملة الفصحى": r[MSA],
                        "الحكم": "", "نوع الخطأ": "", "التصحيح": "", "ملاحظة": "",
                        "المصدر": r["رابط المصدر"], "معجم الرياض": "https://dictionary.ksaa.gov.sa/result/" + quote(msa),
                        "اسم المصدر": r["المصدر"], "الدفعة": r["الدفعة"], "Feature 1": r["Feature 1"]})
        return out

    widths = {"#": 5, "الكلمة العامية": 14, "المقابل الفصيح": 16, "مقابلات بديلة": 16, "اللهجة": 8,
              "الجملة العامية": 38, "الجملة الفصحى": 38, "الحكم": 10, "نوع الخطأ": 16, "التصحيح": 28,
              "ملاحظة": 22, "المصدر": 7, "معجم الرياض": 9, "اسم المصدر": 30, "الدفعة": 6}
    mine = {"الحكم", "نوع الخطأ", "التصحيح", "ملاحظة"}
    choices = {"الحكم": GEN_VERDICTS, "نوع الخطأ": ERRORS}
    for title, df in (("2 عينة المولدة", sample), ("3 باقي المولدة", rest)):
        rows = gen_rows(df)
        sheet(wb, title, list(rows[0]), rows, widths, mine, choices)
    OUT.parent.mkdir(exist_ok=True)
    wb.save(OUT)
    print(f"wrote {OUT}: 150 word labels, {len(sample)} sample rows, {len(rest)} other generated rows")


def read(ws):
    vals = list(ws.values)
    return pd.DataFrame(vals[1:], columns=vals[0]).fillna("").astype(str)


def do_import(path):
    wb = load_workbook(path)
    words = read(wb["1 تحكيم الكلمات"])
    todo = pd.read_csv(TODO, dtype=str).fillna("")
    todo["الحكم"] = words["الحكم"].str.strip().values
    todo["ملاحظة"] = words["ملاحظة"].values
    todo.to_csv(TODO, index=False, encoding="utf-8-sig")
    import_labels()

    gen = pd.concat([read(wb["2 عينة المولدة"]).assign(عينة="نعم"), read(wb["3 باقي المولدة"]).assign(عينة="")])
    gen = gen[gen["الحكم"].str.strip() != ""]
    gen.to_csv(DATA / "generated" / "review_all.csv", index=False, encoding="utf-8-sig")
    s = gen[gen["عينة"] == "نعم"]
    print(f"generated: {len(gen)} rows judged ({len(s)} in the sample)")
    if len(s):
        t = s.assign(error=(s["الحكم"].str.strip() != "اعتمد").astype(int)).groupby("الدفعة")["error"].agg(["mean", "count"])
        t["mean"] = (100 * t["mean"]).round(1)
        print("error % per batch on the sample (>20% -> exclude or review fully):")
        print(t.sort_values("mean", ascending=False).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--import", dest="imp", nargs="?", const=str(OUT))
    a = ap.parse_args()
    do_import(a.imp) if a.imp else build()
