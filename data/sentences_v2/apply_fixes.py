"""Apply grammar_fixes.FIXES to data/sentences_v2/train.csv, with checks.

Refuses to write anything unless every old text occurs exactly once in its row, and afterwards
every fixed row has well-formed brackets and the same span numbers on both sides.

Already applied once (2026-11-03); re-running it on the fixed file stops with "old text found 0 times".
The 3 follow-up entries at the end of FIXES and FIXES_2 were applied in a second pass.

usage: .venv\\Scripts\\python data\\sentences_v2\\apply_fixes.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "src"))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from data import DIALECT, MSA, spans  # noqa: E402
from grammar_fixes import FIXES  # noqa: E402

COL = {"M": MSA, "D": DIALECT}
path = HERE / "train.csv"
tr = pd.read_csv(path, dtype=str)

errors = []
for fid, c, old, new, _ in FIXES:
    idx = tr.index[tr["Feature 1"] == fid]
    if len(idx) != 1:
        errors.append((fid, f"{len(idx)} rows"))
        continue
    text = tr.loc[idx[0], COL[c]]
    if text.count(old) != 1:
        errors.append((fid, f"old text found {text.count(old)} times: {old}"))
        continue
    tr.loc[idx[0], COL[c]] = text.replace(old, new)

for fid in {f[0] for f in FIXES}:
    r = tr[tr["Feature 1"] == fid].iloc[0]
    for x in (r[DIALECT], r[MSA]):
        if re.search(r"[()]", re.sub(r"\(([^()]*)\)\s*(\d+)", "", x)):
            errors.append((fid, f"stray bracket: {x}"))
    if set(spans(r[DIALECT])) != set(spans(r[MSA])):
        errors.append((fid, f"span numbers differ: {sorted(spans(r[DIALECT]))} vs {sorted(spans(r[MSA]))}"))

if errors:
    print("NOT applied, problems:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
tr.to_csv(path, index=False, encoding="utf-8")
print(f"applied {len(FIXES)} fixes")
