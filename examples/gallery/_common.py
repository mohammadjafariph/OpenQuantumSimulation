"""Shared helpers for public example-gallery scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, TypeAlias

Summary: TypeAlias = dict[str, float | int | str]


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add shared command-line options to an example parser."""
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run a smaller problem suitable for smoke tests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for generated figures and data files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing PNG figures.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a machine-readable summary.",
    )


def example_output_dir(name: str, output_dir: Path | None) -> Path:
    """Return and create the output directory for one example."""
    target = (
        Path("runs") / "example_gallery" / name
        if output_dir is None
        else output_dir
    )
    target.mkdir(parents=True, exist_ok=True)
    return target


def emit_summary(summary: Summary, *, as_json: bool = False) -> None:
    """Print a compact example summary."""
    if as_json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return
    print(f"{summary['example']}:")
    for key, value in summary.items():
        if key == "example":
            continue
        print(f"  {key}: {_format_value(value)}")


def configure_matplotlib() -> Any:
    """Import matplotlib with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
