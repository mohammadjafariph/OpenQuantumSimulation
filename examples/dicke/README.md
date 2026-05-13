# Dicke MI Runbook

This is the first practical workflow for the two-ensemble Dicke mutual
information simulations.

## Benchmark Snapshot

Small benchmarks on this machine showed:

| Case | Command shape | Result |
| --- | --- | --- |
| `N=4`, serial | `6` trajectories, `31` time points, `t_final=0.5` | `131.3 traj/s` |
| `N=4`, `n_jobs=2` | same short benchmark | dominated by Julia worker startup |
| `N=6`, serial | `4` trajectories, `41` time points, `t_final=0.5` | `35.7 traj/s` |

For now, use `--n-jobs 1` for production MI runs. Process workers are correct,
but each worker starts its own Julia runtime, so they only make sense after we
add persistent workers or much larger per-worker batches.

## First Pilot Run

Use this before launching the full roadmap grid:

```bash
python examples/dicke/run_mi_distribution.py \
    --n-values 4,6 \
    --kappa-values 0.1 \
    --n-traj 200 \
    --time-points 401 \
    --t-final 20 \
    --max-step 0.02 \
    --checkpoint-every 20 \
    --batch-size 20 \
    --n-jobs 1 \
    --output runs/dicke_mi_pilot.h5
```

Then analyze:

```bash
python examples/dicke/analyze_mi_distribution.py \
    --input runs/dicke_mi_pilot.h5 \
    --output-dir runs/dicke_mi_pilot_analysis
```

## Scaling Up

After the pilot looks stable, expand in this order:

1. Increase `n_traj` to `1000` for `N=4,6`.
2. Add nearby coupling values, for example `--kappa-values 0.05,0.1,0.2`.
3. Add larger `N` one at a time and benchmark first:

```bash
python examples/dicke/bench_mi.py \
    --N 8 \
    --kappa 0.1 \
    --n-traj 4 \
    --time-points 41 \
    --t-final 0.5 \
    --max-step 0.02 \
    --batch-size 2 \
    --n-jobs 1 \
    --repeats 2 \
    --target-n-traj 1000
```

Keep `--checkpoint-every` equal to `--batch-size` for clean restart behavior.
