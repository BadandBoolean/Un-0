# GPU A/B result — judge loss, n=256, 100 epochs (2026-07-18)

Run A: memristor device-stored weights + disturb annealing.
Run B: learned stored table (reflash baseline). Identical otherwise.

| epoch | A (device+anneal) dino loss | B (reflash) dino loss |
|---|---|---|
| 10 | 0.0175 | 0.0197 |
| 25 | 0.0156 | 0.0181 |
| 50 | 0.0150 | 0.0155 |
| 75 | 0.0148 | 0.0152 |
| 100 | 0.0146 | 0.0149 |

Final (last-5-epoch mean): A = 0.0145, B = 0.0149 (A better by 2.6%)

Device weights (A): mean 0.229, spread 0.0206
Learned table (B):  mean 0.500, spread 0.0069
Gain: A 0.0235, B 0.0053

Sample grids: samples-device-anneal/ and samples-reflash/ (epoch_0100.png is final).
