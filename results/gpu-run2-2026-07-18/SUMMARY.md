# GPU runs 2 — signed coupling and n=1024 scale (2026-07-18)

## Signed coupling vs last night's unsigned winner (both n=256, judge loss)

| epoch | signed | unsigned (baseline) |
|---|---|---|
| 10 | 0.0191 | 0.0175 |
| 25 | 0.0159 | 0.0156 |
| 50 | 0.0151 | 0.0150 |
| 75 | 0.0150 | 0.0148 |
| 100 | 0.0148 | 0.0146 |

Final (last-5 mean): signed 0.0147 vs unsigned 0.0145 (unsigned better by 1.3%)
Repulsive pairs: 49.3%; |J| median 0.0122; gain 0.1513

## n=1024 scale run (device+anneal, judge loss, batch 128)

No completed epochs parsed — check n1024.log.
