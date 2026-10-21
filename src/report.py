"""Write results/REPORT.md: leaderboard (latest score per model+mode) and sample outputs."""
import json

from data import ROOT

RES = ROOT / "results"
COLS = ["BLEU", "chrF++", "SpanAcc%", "DialectLeft%", "ExactMatch%"]


def pred_file(d):
    name = d["model"].split("/")[-1] if d["mode"] == "finetune" else d["model"].replace(":", "_").replace("/", "_")
    return RES / "preds" / f"{name}__{d['mode']}.jsonl"


def main():
    rows = {}
    for line in (RES / "scores.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        rows[(d["model"], d["mode"])] = d
    table = sorted(rows.values(), key=lambda d: -d["chrF++"])

    md = ["# Dialect → MSA benchmark", "",
          f"Test set: data/sentences/test.csv ({table[0]['n']} sentences). Ranked by chrF++.", "",
          "- **SpanAcc%**: gold MSA replacements found in the output (higher is better)",
          "- **DialectLeft%**: dialect words left unconverted (lower is better)", "",
          "| # | Model | Mode | " + " | ".join(COLS) + " |", "|---" * (len(COLS) + 3) + "|"]
    for i, d in enumerate(table, 1):
        md.append(f"| {i} | {d['model']} | {d['mode']} | " + " | ".join(str(d[c]) for c in COLS) + " |")

    best = table[0]
    preds = [json.loads(l) for l in pred_file(best).open(encoding="utf-8")]
    md += ["", f"## Sample outputs of the best system ({best['model']}, {best['mode']})", "",
           "| العامية | المرجع الفصيح | مخرج النموذج |", "|---|---|---|"]
    md += [f"| {p['src']} | {p['tgt']} | {p['pred']} |" for p in preds[:20]]

    text = "\n".join(md) + "\n"
    (RES / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
