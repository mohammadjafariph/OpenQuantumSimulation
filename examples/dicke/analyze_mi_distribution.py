"""Analyze two-ensemble Dicke mutual-information distribution outputs."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FloatArray = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class SummaryRow:
    """Summary statistics for one ``(N, kappa)`` point."""

    N: int
    kappa: float
    completed: int
    MI_mean_A: float
    MI_std_A: float
    MI_stderr_A: float
    MI_median_A: float
    MI_q05_A: float
    MI_q95_A: float
    MI_mean_B: float
    MI_std_B: float
    MI_stderr_B: float
    MI_median_B: float
    MI_q05_B: float
    MI_q95_B: float
    MI_mean_AB: float
    MI_stderr_AB: float


def main() -> None:
    args = _parse_args()
    rows = analyze(
        args.input,
        args.output_dir,
        make_plots=not args.no_plots,
        max_boxplots=args.max_boxplots,
    )
    print(f"Wrote summary: {args.output_dir / 'summary.csv'}")
    print(f"Wrote summary: {args.output_dir / 'summary.h5'}")
    if not args.no_plots:
        print(f"Wrote plots in: {args.output_dir}")
    print(f"Analyzed {len(rows)} completed points.")


def analyze(
    input_path: Path,
    output_dir: Path,
    *,
    make_plots: bool = True,
    max_boxplots: int = 30,
) -> list[SummaryRow]:
    """Analyze a Dicke MI distribution HDF5 file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_summary_rows(input_path)
    _write_csv(output_dir / "summary.csv", rows)
    _write_hdf5(output_dir / "summary.h5", rows)
    if make_plots and rows:
        _plot_mean_summary(output_dir / "steady_mi_mean.png", rows)
        _plot_distribution_boxes(
            input_path,
            output_dir / "steady_mi_boxplot.png",
            max_boxplots=max_boxplots,
        )
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize and plot Dicke mutual-information distributions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/dicke_mi_distribution.h5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/dicke_mi_analysis"),
    )
    parser.add_argument(
        "--max-boxplots",
        type=int,
        default=30,
        help="Maximum number of parameter points shown in the boxplot figure.",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if args.max_boxplots <= 0:
        parser.error("--max-boxplots must be positive")
    return args


def _load_summary_rows(input_path: Path) -> list[SummaryRow]:
    with h5py.File(input_path, "r") as handle:
        _validate_input_file(handle)
        rows = [
            _summarize_group(group)
            for group in _iter_point_groups(handle)
            if int(group.attrs.get("completed", 0)) > 0
        ]
    return sorted(rows, key=lambda row: (row.N, row.kappa))


def _validate_input_file(handle: h5py.File) -> None:
    if handle.attrs.get("format") != "openquantumsim.dicke_mi_distribution":
        msg = "Input is not an OpenQuantumSim Dicke MI distribution file."
        raise ValueError(msg)


def _iter_point_groups(handle: h5py.File) -> list[h5py.Group]:
    groups: list[h5py.Group] = []
    for kappa_name in sorted(handle):
        if not kappa_name.startswith("kappa_"):
            continue
        kappa_group = handle[kappa_name]
        if not isinstance(kappa_group, h5py.Group):
            continue
        for n_name in sorted(kappa_group):
            point = kappa_group[n_name]
            if isinstance(point, h5py.Group):
                groups.append(point)
    return groups


def _summarize_group(group: h5py.Group) -> SummaryRow:
    completed = int(group.attrs["completed"])
    steady_a = _steady_values(group, "A")[:completed]
    steady_b = _steady_values(group, "B")[:completed]
    combined = 0.5 * (steady_a + steady_b)

    stats_a = _stats(steady_a)
    stats_b = _stats(steady_b)
    stats_ab = _stats(combined)
    return SummaryRow(
        N=int(group.attrs["N"]),
        kappa=float(group.attrs["kappa"]),
        completed=completed,
        MI_mean_A=stats_a["mean"],
        MI_std_A=stats_a["std"],
        MI_stderr_A=stats_a["stderr"],
        MI_median_A=stats_a["median"],
        MI_q05_A=stats_a["q05"],
        MI_q95_A=stats_a["q95"],
        MI_mean_B=stats_b["mean"],
        MI_std_B=stats_b["std"],
        MI_stderr_B=stats_b["stderr"],
        MI_median_B=stats_b["median"],
        MI_q05_B=stats_b["q05"],
        MI_q95_B=stats_b["q95"],
        MI_mean_AB=stats_ab["mean"],
        MI_stderr_AB=stats_ab["stderr"],
    )


def _steady_values(group: h5py.Group, side: str) -> FloatArray:
    steady_name = f"MI_steady_{side}"
    if steady_name in group:
        return np.asarray(group[steady_name], dtype=np.float64)

    time_name = f"MI_time_{side}"
    if time_name not in group:
        msg = f"Missing {steady_name} and {time_name} in {group.name}."
        raise ValueError(msg)
    time_values = np.asarray(group[time_name], dtype=np.float64)
    start = int(group.attrs.get("n_transient", 0))
    if start < 0 or start >= time_values.shape[1]:
        msg = f"Invalid n_transient={start} in {group.name}."
        raise ValueError(msg)
    return time_values[:, start:].mean(axis=1)


def _stats(values: FloatArray) -> dict[str, float]:
    if len(values) == 0:
        msg = "Cannot summarize an empty distribution."
        raise ValueError(msg)
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "stderr": std / float(np.sqrt(len(values))),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _write_csv(path: Path, rows: list[SummaryRow]) -> None:
    fieldnames = _fieldnames()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_hdf5(path: Path, rows: list[SummaryRow]) -> None:
    fieldnames = _fieldnames()
    with h5py.File(path, "w") as handle:
        handle.attrs["format"] = "openquantumsim.dicke_mi_analysis"
        handle.attrs["format_version"] = "1"
        for field in fieldnames:
            if field in {"N", "completed"}:
                data = np.array(
                    [int(asdict(row)[field]) for row in rows],
                    dtype=np.int64,
                )
            else:
                data = np.array(
                    [float(asdict(row)[field]) for row in rows],
                    dtype=np.float64,
                )
            handle.create_dataset(field, data=data)


def _plot_mean_summary(path: Path, rows: list[SummaryRow]) -> None:
    _configure_plot_cache(path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for n_spins in sorted({row.N for row in rows}):
        subset = sorted(
            (row for row in rows if row.N == n_spins),
            key=lambda r: r.kappa,
        )
        x = np.array([row.kappa for row in subset], dtype=np.float64)
        y = np.array([row.MI_mean_AB for row in subset], dtype=np.float64)
        yerr = np.array([row.MI_stderr_AB for row in subset], dtype=np.float64)
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, label=f"N={n_spins}")

    ax.set_xlabel("kappa")
    ax.set_ylabel("steady mutual information")
    ax.set_title("Mean steady Dicke mutual information")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_distribution_boxes(
    input_path: Path,
    path: Path,
    *,
    max_boxplots: int,
) -> None:
    _configure_plot_cache(path.parent)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels: list[str] = []
    values: list[FloatArray] = []
    with h5py.File(input_path, "r") as handle:
        for group in _iter_point_groups(handle)[:max_boxplots]:
            completed = int(group.attrs.get("completed", 0))
            if completed <= 0:
                continue
            steady_a = _steady_values(group, "A")[:completed]
            steady_b = _steady_values(group, "B")[:completed]
            values.append(0.5 * (steady_a + steady_b))
            labels.append(f"N={int(group.attrs['N'])}\nk={float(group.attrs['kappa']):g}")

    if not values:
        return

    fig_width = max(7.0, 0.55 * len(values))
    fig, ax = plt.subplots(figsize=(fig_width, 4.4))
    ax.boxplot(values, tick_labels=labels, showfliers=False)
    ax.set_ylabel("steady mutual information")
    ax.set_title("Steady MI distributions")
    ax.tick_params(axis="x", labelrotation=45)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _fieldnames() -> list[str]:
    return list(SummaryRow.__dataclass_fields__)


def _configure_plot_cache(output_dir: Path) -> None:
    cache_dir = output_dir / ".cache"
    matplotlib_dir = cache_dir / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))


if __name__ == "__main__":
    main()
