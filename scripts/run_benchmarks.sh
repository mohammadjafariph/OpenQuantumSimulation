#!/usr/bin/env bash
# Re-run the benchmarks reported on docs/performance.rst and redraw the
# summary figure. Results go to runs/benchmarks/<date>/ (ignored by Git).
#
#   bash scripts/run_benchmarks.sh            # everything except the sysimage build
#   bash scripts/run_benchmarks.sh --sysimage # also time startup with a sysimage
#
# Needs QuTiP (pip install -e ".[validation]") for the cross-library runs.
# Close other heavy programs first: the numbers are wall-clock times.
set -euo pipefail

out="runs/benchmarks/$(date +%Y-%m-%d)"
mkdir -p "$out"
export PYTHON_JULIACALL_HANDLE_SIGNALS=yes MPLBACKEND=Agg

echo "== mesolve vs QuTiP"
python benchmarks/bench_vs_qutip.py \
    --repeats 9 --time-points 81 --t-final 6.0 \
    --cases qubit jc5 jc10 jc20 jc40 --oqs-methods auto krylov ode \
    --json "$out/bench_vs_qutip.json"

echo "== mesolve integrator choices"
python benchmarks/bench_mesolve.py --cases jc5 jc10 jc20 --json "$out/bench_mesolve.json"

echo "== mcsolve vs QuTiP (4 Julia threads)"
JULIA_NUM_THREADS=4 python benchmarks/bench_mcsolve_vs_qutip.py \
    --n-traj 200 1000 5000 --repeats 5 --json "$out/bench_mcsolve_vs_qutip.json"

echo "== mcsolve thread scaling"
JULIA_NUM_THREADS=4 python benchmarks/bench_mcsolve.py \
    --n-traj 5000 --time-points 41 --t-final 4.0 --repeats 5 --n-jobs 1 2 4 \
    --json "$out/bench_mcsolve.json"

echo "== Dicke mutual-information batches"
JULIA_NUM_THREADS=1 python examples/dicke/bench_mi.py \
    --N 4 --kappa 0.1 --n-traj 40 --time-points 21 --t-final 0.2 \
    --batch-size 10 --repeats 3 --n-jobs 1 2 --target-n-traj 1000 \
    --json "$out/bench_dicke_mi.json"

echo "== sparse spin chains (several minutes)"
python benchmarks/bench_sparse_scaling.py \
    --cases spin9 spin10 --time-points 21 --t-final 5 --repeats 3 \
    --json "$out/sparse_scaling.json"

if [[ "${1:-}" == "--sysimage" ]]; then
    echo "== startup with and without a sysimage (builds a sysimage, ~15 min)"
    python benchmarks/bench_startup_sysimage.py \
        --install-local . --repeats 2 --json "$out/bench_startup_sysimage.json"
fi

echo "== summary figure"
python benchmarks/plot_benchmark_summary.py "$out" \
    --output docs/_static/benchmarks/readme_benchmark_summary.png
