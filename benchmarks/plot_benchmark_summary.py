"""Draw the benchmark summary figure from the benchmark JSON reports.

Reads the reports written by ``scripts/run_benchmarks.sh`` from one directory
and draws one panel per benchmark, each with its own axis, showing how many
times faster OpenQuantumSim is than QuTiP (1x means equal speed):

* ``bench_vs_qutip.json``         Lindblad solves of Jaynes-Cummings models
* ``sparse_scaling.json``         Lindblad solves of sparse spin chains
* ``bench_mcsolve_vs_qutip.json`` Monte Carlo trajectories, 1 and N threads

A missing report leaves its panel out.

    python benchmarks/plot_benchmark_summary.py runs/benchmarks/2026-11-16 \
        --output docs/_static/benchmarks/readme_benchmark_summary.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Validated categorical slots 1-2 and chart chrome (light surface).
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

BAR_WIDTH = 0.28
SINGLE_BAR_WIDTH = 0.38


def main() -> None:
    """Draw the figure from the reports in the given directory."""
    args = _parse_args()
    panels: list[Callable[[Axes], None]] = []

    dense = _load(args.reports / "bench_vs_qutip.json")
    if dense is not None:
        panels.append(lambda ax: _dense_panel(ax, dense))
    sparse = _load(args.reports / "sparse_scaling.json")
    if sparse is not None:
        panels.append(lambda ax: _sparse_panel(ax, sparse))
    monte_carlo = _load(args.reports / "bench_mcsolve_vs_qutip.json")
    if monte_carlo is not None:
        panels.append(lambda ax: _monte_carlo_panel(ax, monte_carlo))
    if not panels:
        msg = f"no benchmark reports found in {args.reports}"
        raise SystemExit(msg)

    plt.rcParams.update(
        {
            "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_MUTED,
        },
    )
    widths = [1.25, 0.7, 1.0][: len(panels)] if len(panels) == 3 else None
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(12.5, 4.4),
        gridspec_kw={"width_ratios": widths} if widths else None,
        squeeze=False,
    )
    figure.patch.set_facecolor(SURFACE)
    for ax, draw in zip(axes[0], panels, strict=True):
        ax.set_facecolor(SURFACE)
        draw(ax)
        _style(ax)
    # Both Lindblad panels measure the same thing: give them one scale so bar
    # heights can be compared across them.
    if dense is not None and sparse is not None:
        shared = max(axes[0][0].get_ylim()[1], axes[0][1].get_ylim()[1])
        axes[0][0].set_ylim(0, shared)
        axes[0][1].set_ylim(0, shared)

    figure.suptitle(
        "How many times faster OpenQuantumSim is than QuTiP",
        x=0.012,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    note = args.note or _environment_note([dense, sparse, monte_carlo])
    figure.text(0.012, 0.875, note, ha="left", fontsize=9, color=INK_SECONDARY)
    figure.tight_layout(rect=(0, 0, 1, 0.87), w_pad=3.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, facecolor=SURFACE)
    print(f"Wrote {args.output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("reports", type=Path, help="Directory with the JSON reports.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/_static/benchmarks/readme_benchmark_summary.png"),
    )
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--note",
        default=None,
        help="Subtitle; defaults to the environment recorded in the reports.",
    )
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _dense_panel(ax: Axes, report: dict[str, Any]) -> None:
    rows = [
        row
        for row in report["rows"]
        if row["engine"] == "openquantumsim" and row["method"] == "auto"
    ]
    rows.sort(key=lambda row: row["dim"])
    labels = [str(row["dim"]) for row in rows]
    values = [float(row["speedup_vs_qutip"]) for row in rows]
    _bars(ax, range(len(rows)), values, SERIES_1, width=SINGLE_BAR_WIDTH)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("Hilbert dimension")
    _title(ax, "Lindblad master equation", "Jaynes-Cummings models, mesolve")
    _parity(ax, len(rows))


def _sparse_panel(ax: Axes, report: dict[str, Any]) -> None:
    rows = sorted(
        (row for row in report["rows"] if "speedup_vs_qutip" in row),
        key=lambda row: row["dim"],
    )
    labels = [str(row["dim"]) for row in rows]
    values = [float(row["speedup_vs_qutip"]) for row in rows]
    _bars(ax, range(len(rows)), values, SERIES_1, width=SINGLE_BAR_WIDTH)
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel("Hilbert dimension")
    _title(ax, "Large sparse systems", "Spin chains, mesolve")
    _parity(ax, len(rows))


def _monte_carlo_panel(ax: Axes, report: dict[str, Any]) -> None:
    rows = sorted(report["rows"], key=lambda row: row["n_traj"])
    positions = range(len(rows))
    threads = max(int(row["oqs_workers"]) for row in rows)
    serial = [float(row["oqs_serial_speedup_vs_qutip"]) for row in rows]
    threaded = [float(row["oqs_threaded_speedup_vs_qutip"]) for row in rows]
    gap = 0.02
    _bars(
        ax,
        [p - BAR_WIDTH / 2 - gap for p in positions],
        serial,
        SERIES_1,
        width=BAR_WIDTH,
        label="1 thread",
    )
    _bars(
        ax,
        [p + BAR_WIDTH / 2 + gap for p in positions],
        threaded,
        SERIES_2,
        width=BAR_WIDTH,
        label=f"{threads} threads",
    )
    ax.set_xticks(list(positions), [f"{int(row['n_traj']):,}" for row in rows])
    ax.set_xlabel("Trajectories")
    _title(ax, "Monte Carlo trajectories", "Qubit decay, mcsolve")
    _parity(ax, len(rows))
    legend = ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=1.0,
        borderaxespad=0.2,
    )
    for handle in legend.legend_handles:
        if handle is not None:
            handle.set_edgecolor("none")


def _bars(
    ax: Axes,
    positions: Any,
    values: list[float],
    color: str,
    *,
    width: float,
    label: str | None = None,
) -> None:
    positions = list(positions)
    top = max(values) if values else 1.0
    for index, (x, value) in enumerate(zip(positions, values, strict=True)):
        # Square at the baseline, rounded at the data end.
        radius = min(0.06, width / 3)
        ax.add_patch(
            FancyBboxPatch(
                (x - width / 2, 0.0),
                width,
                value,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=top / 6,
                facecolor=color,
                edgecolor="none",
                label=label if index == 0 else None,
                zorder=3,
            ),
        )
        ax.add_patch(
            plt.Rectangle(
                (x - width / 2, 0.0),
                width,
                min(value, top * 0.05),
                facecolor=color,
                edgecolor="none",
                zorder=3,
            ),
        )
        ax.annotate(
            f"{value:.1f}x",
            (x, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK_PRIMARY,
            zorder=4,
        )
    ax.set_ylim(0, max(top * 1.18, 1.3))
    ax.set_xlim(min(positions) - 0.6, max(positions) + 0.6)


def _parity(ax: Axes, n_groups: int) -> None:
    del n_groups
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=1.0, zorder=2)


def _title(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_title(
        f"{title}\n",
        loc="left",
        fontsize=11,
        fontweight="semibold",
        color=INK_PRIMARY,
    )
    ax.text(
        0.0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=INK_SECONDARY,
    )


def _style(ax: Axes) -> None:
    ax.grid(axis="y", color=GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, pad=6)
    ax.yaxis.set_major_formatter(lambda value, _pos: f"{value:g}x")


def _environment_note(reports: list[dict[str, Any] | None]) -> str:
    """Versions recorded by any of the reports, for the figure subtitle."""
    versions: dict[str, str] = {}
    for report in reports:
        if not report:
            continue
        for key, value in report.get("environment", {}).items():
            versions.setdefault(key, str(value))
        for key, value in report.get("versions", {}).items():
            label = {"openquantumsim": "OpenQuantumSim", "qutip": "QuTiP"}.get(key, key)
            if value is not None:
                versions.setdefault(label, str(value))
    parts = [
        f"{name} {versions[name]}"
        for name in ("OpenQuantumSim", "QuTiP", "Julia")
        if name in versions
    ]
    parts += ["same tolerances for both", "dark line at 1x = equal speed"]
    return " · ".join(parts)


if __name__ == "__main__":
    main()
