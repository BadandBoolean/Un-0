# Dynamics ablation study

Measures how much of CIFAR-10 generation quality comes from the Kuramoto
dynamics versus the decoder alone. Eight experiments span the dynamics axis,
each swept over learning rate (8-point sweep, one LR per GPU) under the single
release recipe (AdamW, dense coupling).

| experiment | flags | dynamics |
| --- | --- | --- |
| `decoder_only_raw` | `--num-steps 0 --encoding raw` | none; decoder reads raw phases (no readout transform) |
| `decoder_only` | `--num-steps 0` | none; decoder reads the sin/cos readout of random phases |
| `reservoir_euler1` | `--num-steps 1 --solver euler --freeze-dynamics` | frozen random |
| `reservoir_euler10` | `--num-steps 10 --solver euler --freeze-dynamics` | frozen random |
| `trained_euler1/2/5/10` | `--solver euler --num-steps 1/2/5/10` | trained |

The two `decoder_only` rows isolate the readout transform: `decoder_only` feeds
the sin/cos readout to the decoder, `decoder_only_raw` feeds raw phases directly.

## Run on one 8-GPU host

    export WANDB_API_KEY=...
    ablation_study/run_ablation.sh --wandb-project un0-ablations

Phase 1 runs a short LR sweep per experiment (one LR per GPU via
`lr_sweep.sh`) and ranks by FID; Phase 2 runs the best LR per experiment at
full length, one experiment per GPU. Use `--dry-run` to print the commands
without launching, and `--no-sync` if the devbox is already synced.

## Results

The best LR per experiment is written to `outputs/dynamics/best_lr.json`, and
each sweep run's FID lands in its own `fid.json` under the run directory.
