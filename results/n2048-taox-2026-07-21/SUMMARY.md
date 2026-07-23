# n=2048 scale run — TaOx preset, device mode, judge loss (2026-07-21)

Config: n=2048, batch 64 (A100-80GB), 50 epochs, bf16, device-mode
crossbar, TaOx preset (Vp 0.91, Vn 1.425, Ap/An 72.5/21, v_osc 1.5,
eta 1e-4, lambda 0), drive-scale 10, no annealing. ~7.5 h, ~$11.

Note: epoch indices below are approximate (+~7%) — the log parser
counts progress-bar refresh lines, which overcount steps slightly.

| ~epoch | dino loss |
|---|---|
| 5 | 0.0125 |
| 10 | 0.0125 |
| 25 | 0.0125 |
| 40 | 0.0125 |
| 50 | 0.0125 |
| 54 (last parsed) | 0.0124 |

Final (last-3 mean): 0.01246

Scale series (all device-mode, judge loss, 50-100 epochs; batch and
preset differ across rows — a trend, not a controlled comparison):
  n=256 (knowm-w, batch 256): 0.0150
  n=1024 (tuned params, batch 128): 0.0133
  n=2048 (taox, batch 64): 0.0125

## Device-state finding: full crossbar saturation

All 2096128 device pairs pinned at ON: mean 0.995, spread 0.0002, 100% above 0.95.
Learned gain compensated down to 0.0135.

Mechanism: the TaOx extraction is strongly SET-dominant under symmetric
drive at v_osc=1.5 (SET outrates RESET ~21:1), so every run potentiated
the devices and nothing reset them; the stored weights degenerated to a
uniform all-ON matrix by ~epoch 10, where the loss also plateaued.

Read-in: best loss of the series was achieved with an effectively
uniform coupling matrix — the stored pattern was absent. Confounded
with scale and batch; the clean control (frozen-uniform vs structured
coupling at n=2048) has not been run.

Constructive note: TaOx thresholds (0.91 V) mean any v_osc below 0.91
gives zero device disturbance — clean nonvolatile weight storage with
margin. Operated above threshold (as here), the same device erases
stored structure. Operating point, not device choice alone, decides
whether the crossbar stores or destroys.

Artifacts: n2048.log, checkpoints/ (epochs 25, 50, final, latest),
checkpoints/samples/ (grids at 1, 5, 25, 50).
