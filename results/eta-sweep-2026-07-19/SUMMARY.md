# eta sweep — does memristor drift affect results? (n=256, 50 epochs)

Faithful Knowm W+SDC physics (Vp 0.26, Vn 0.11, Ap=An 4000, lambda 0,
v_osc 0.35), device-mode storage, no annealing. Only `eta` differs, so
any gap is attributable to in-run device drift alone.

| eta | final dino loss | device mean | device spread | drift vs frozen |
|---|---|---|---|---|
| 0 | 0.0150 | 0.009 | 0.0126 | 0.0000 |
| 3e-9 | 0.0150 | 0.011 | 0.0125 | 0.0062 |
| 3e-8 | 0.0150 | 0.015 | 0.0122 | 0.0085 |
| 3e-7 | 0.0151 | 0.044 | 0.0052 | 0.0345 |

Best arm: eta=3e-8 (loss 0.0150).
Frozen-device control (eta=0): 0.0150.
Best vs frozen: -0.3% — negative means drift helped.

Epochs completed per arm: 0=53, 3e-9=53, 3e-8=53, 3e-7=53
