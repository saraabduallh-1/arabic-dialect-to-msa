# Cleaning report (PLAN_v2 D1, default decisions)

Input `all.csv`: 2268 rows -> `all_clean.csv`: 2265 rows.

| change | rows |
|---|---|
| پ/گ/ڤ replaced (السياق العامي) | 0 |
| چ replaced (السياق العامي) | 9 |
| پ/گ/ڤ replaced (السياق الفصيح) | 0 |
| چ replaced (السياق الفصيح) | 0 |
| dropped: unclear چ word | 3 |
| rows kept but not new vocabulary (ال/ه variant of a train term) | 29 |

چ word table:

- سويچ -> سويتش
- سچين -> سكين
- چبريت -> كبريت
- چولة -> unclear, row dropped

Status: waiting for the user's review (autowork/QUESTIONS.md). Not used for training yet.
